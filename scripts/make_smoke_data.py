#!/usr/bin/env python3
"""Generate a tiny local dataset for CPU smoke-training SenseVoiceSmall.

The jsonl files shipped in ``data/`` reference audio under ``/cpfs01/...`` which
does not exist outside the original training cluster.  This script slices the one
real wav in the repository into several shorter clips of *differing* lengths and
emits jsonl manifests using exactly the same schema as
``data/train_example.jsonl``.

The varying clip lengths are deliberate: they force the batch collator to pad,
which is what exercises the dynamic chunk-mask code paths during a smoke run.

This produces data for *pipeline plumbing checks only* -- the transcripts are
dummies, so nothing about the resulting model is meaningful.

The script is idempotent: re-running it overwrites the generated wavs and
manifests in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE_WAV = REPO_ROOT / "runtime" / "llama.cpp" / "tests" / "sample.wav"
DEFAULT_WAV_DIR = REPO_ROOT / "data" / "smoke"
DEFAULT_TRAIN_JSONL = REPO_ROOT / "data" / "smoke_train.jsonl"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "smoke_val.jsonl"

# One encoder input frame is 10 ms, matching ``source_len`` in the shipped jsonl.
FRAME_MS = 10

# Dummy transcripts.  Must never be empty: CTC cannot handle a zero-length
# target and the trainer crashes rather than skipping the sample.
DUMMY_TEXTS = (
    "this is a smoke test utterance",
    "dynamic chunk masking pipeline check",
    "hello world from the smoke dataset",
    "the quick brown fox jumps over the lazy dog",
    "streaming encoder padding path exercise",
    "one two three four five six seven",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-wav",
        type=Path,
        default=DEFAULT_SOURCE_WAV,
        help="Source wav to slice (default: %(default)s)",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=DEFAULT_WAV_DIR,
        help="Directory for the generated clips (default: %(default)s)",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=DEFAULT_TRAIN_JSONL,
        help="Output train manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=DEFAULT_VAL_JSONL,
        help="Output validation manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=12,
        help="Number of training clips (default: %(default)s)",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=4,
        help="Number of validation clips (default: %(default)s)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1.0,
        help="Shortest clip duration (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=4.0,
        help="Longest clip duration (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the clip start offsets (default: %(default)s)",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_wav.is_file():
        raise SystemExit(f"source wav not found: {args.source_wav}")
    if args.num_train < 1 or args.num_val < 1:
        raise SystemExit("--num-train and --num-val must both be >= 1")
    if args.min_seconds <= 0:
        raise SystemExit("--min-seconds must be > 0")
    if args.max_seconds < args.min_seconds:
        raise SystemExit("--max-seconds must be >= --min-seconds")


def clip_durations(count: int, min_seconds: float, max_seconds: float) -> list[float]:
    """Spread ``count`` durations evenly over the requested range.

    Using a deterministic spread rather than random draws guarantees that every
    batch really does contain mixed lengths, however small the batch is.
    """
    if count == 1:
        return [min_seconds]
    step = (max_seconds - min_seconds) / (count - 1)
    return [round(min_seconds + step * i, 3) for i in range(count)]


def take_slice(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    """Return ``length`` samples starting at ``start``, wrapping if needed.

    The source wav is only a few seconds long, so wrapping lets us cut more (and
    longer) clips than the source could otherwise provide.
    """
    if length <= audio.shape[0]:
        end = start + length
        if end <= audio.shape[0]:
            return audio[start:end]
        return np.concatenate((audio[start:], audio[: end - audio.shape[0]]))
    repeats = int(np.ceil((start + length) / audio.shape[0]))
    tiled = np.tile(audio, repeats)
    return tiled[start : start + length]


def write_split(
    audio: np.ndarray,
    samplerate: int,
    wav_dir: Path,
    jsonl_path: Path,
    prefix: str,
    count: int,
    min_seconds: float,
    max_seconds: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, seconds in enumerate(clip_durations(count, min_seconds, max_seconds)):
        length = max(1, int(round(seconds * samplerate)))
        start = int(rng.integers(0, audio.shape[0]))
        clip = take_slice(audio, start, length)

        key = f"{prefix}_{index:03d}"
        wav_path = (wav_dir / f"{key}.wav").resolve()
        sf.write(wav_path, clip, samplerate, subtype="PCM_16")

        text = DUMMY_TEXTS[index % len(DUMMY_TEXTS)]
        records.append(
            {
                "key": key,
                "text_language": "<|en|>",
                "emo_target": "<|NEUTRAL|>",
                "event_target": "<|Speech|>",
                "with_or_wo_itn": "<|woitn|>",
                "target": text,
                "source": str(wav_path),
                "target_len": len(text.split()),
                "source_len": int(clip.shape[0] / samplerate * 1000 / FRAME_MS),
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    audio, samplerate = sf.read(args.source_wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:  # downmix to mono
        audio = audio.mean(axis=1)

    args.wav_dir.mkdir(parents=True, exist_ok=True)
    args.train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.val_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train = write_split(
        audio, samplerate, args.wav_dir, args.train_jsonl,
        "smoke_train", args.num_train, args.min_seconds, args.max_seconds, rng,
    )
    val = write_split(
        audio, samplerate, args.wav_dir, args.val_jsonl,
        "smoke_val", args.num_val, args.min_seconds, args.max_seconds, rng,
    )

    print(f"wav dir : {args.wav_dir.resolve()}")
    print(f"train   : {args.train_jsonl.resolve()} ({len(train)} clips)")
    print(f"val     : {args.val_jsonl.resolve()} ({len(val)} clips)")
    print(f"lengths : {sorted({record['source_len'] for record in train})} frames (train)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
