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

Emotion targets are a constant ``<|NEUTRAL|>`` by default, which is all a
padding/chunk-mask check ever needed.  ``--emo-mix`` exists because the emotion
slot itself is now under test: the manifest may carry the ``<|SER|>`` sentinel
for a clip with no reliable label, and ``model.py`` maps that token to
``ignore_id`` so the slot drops out of the rich cross-entropy loss while the
clip still trains CTC.  A constant target cannot exercise any of that -- it
saturates ``acc_emo`` instantly, which is the exact defect the sentinel was
introduced to fix -- so ``--emo-mix`` spreads the seven real emotion tokens and
a realistic share of sentinels across the generated records.  That makes the
smoke run the cheap proof that the masking works end to end, before a real run
spends about two days of GPU time.

The script is idempotent: re-running it overwrites the generated wavs and
manifests in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

# The emotion target every record carries unless --emo-mix is passed.  Existing
# smoke runs and the checked-in data/smoke_*.jsonl are built on this constant,
# so the default output must stay byte-identical.
EMO_TARGET = "<|NEUTRAL|>"

# The "no reliable emotion label" sentinel.  Deliberately not <|NEUTRAL|>:
# stamping neutral on an unlabelled clip asserts an emotion nothing measured.
# model.py rewrites this single token (id 24991) to ignore_id just before the
# rich CE loss, so a masked clip trains CTC on its transcript as usual and drops
# out of the emotion head's numerator and denominator entirely.  Matches
# EMO_MASK_TARGET in scripts/prepare_vn_data.py -- the two generators feed the
# same trainer and have to agree.
EMO_MASK_TARGET = "<|SER|>"

# The seven emotion tokens SenseVoice can be trained on.  <|EMO_UNKNOWN|> is
# deliberately absent: it is the "no prediction" token, and a clip carrying it
# as a *target* teaches the model to predict "unknown".  Unlabelled clips belong
# behind EMO_MASK_TARGET instead.
EMO_LABEL_TARGETS: tuple[str, ...] = (
    "<|HAPPY|>",
    "<|SAD|>",
    "<|ANGRY|>",
    "<|NEUTRAL|>",
    "<|FEARFUL|>",
    "<|DISGUSTED|>",
    "<|SURPRISED|>",
)
EMO_ALL_TARGETS: tuple[str, ...] = EMO_LABEL_TARGETS + (EMO_MASK_TARGET,)

# Share of records that get the sentinel under --emo-mix.  Round 3 expects
# roughly 80-85% of the real corpus to be masked, and the interesting failure
# modes only appear at a realistic rate: a batch that comes out entirely masked,
# and a rich loss whose denominator has shrunk a long way.  A token-batched
# smoke run packs 2-4 clips per batch, so at this rate a fully masked batch
# happens on its own rather than having to be manufactured.
DEFAULT_EMO_MASK_RATE = 0.82

# Stream index used to expand --seed into an independent generator for the
# emotion assignment; see emotion_rng().
EMO_STREAM_ID = 1


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
        help="Seed for the clip start offsets and the emotion assignment (default: %(default)s)",
    )
    parser.add_argument(
        "--emo-mix",
        action="store_true",
        help=(
            "Vary emo_target across the generated records instead of writing a "
            f"constant {EMO_TARGET}: the seven SenseVoice emotion tokens plus a "
            f"--emo-mask-rate share of the {EMO_MASK_TARGET} mask sentinel. "
            "Needed to prove the emotion-slot loss mask works end to end, "
            "because a constant target saturates acc_emo. Off by default, so "
            "the default output is unchanged."
        ),
    )
    parser.add_argument(
        "--emo-mask-rate",
        type=float,
        default=None,
        help=(
            f"Share of records that carry the {EMO_MASK_TARGET} sentinel under "
            f"--emo-mix, between 0 and 1 (default: {DEFAULT_EMO_MASK_RATE}, "
            "close to what round 3 expects in production). Requires --emo-mix."
        ),
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
    # Silently ignoring a rate the caller asked for is how a smoke run succeeds
    # at the wrong thing: the log would claim a mask rate nobody applied.
    if args.emo_mask_rate is not None and not args.emo_mix:
        raise SystemExit(
            f"--emo-mask-rate {args.emo_mask_rate} was given without --emo-mix, "
            f"so every record would still carry {EMO_TARGET}; pass --emo-mix"
        )
    if args.emo_mask_rate is not None and not 0.0 <= args.emo_mask_rate <= 1.0:
        raise SystemExit(
            f"--emo-mask-rate must be between 0 and 1, got {args.emo_mask_rate}"
        )


def clip_durations(count: int, min_seconds: float, max_seconds: float) -> list[float]:
    """Spread ``count`` durations evenly over the requested range.

    Using a deterministic spread rather than random draws guarantees that every
    batch really does contain mixed lengths, however small the batch is.
    """
    if count == 1:
        return [min_seconds]
    step = (max_seconds - min_seconds) / (count - 1)
    return [round(min_seconds + step * i, 3) for i in range(count)]


def emotion_rng(seed: int) -> np.random.Generator:
    """An independent random stream derived from the *same* ``--seed``.

    The emotion assignment must not perturb the clip start offsets: a run with
    ``--emo-mix`` and a run without it should slice identical audio, so that a
    smoke failure can be blamed on the labels rather than on the wavs.  Drawing
    from the generator the offsets already use would shift every subsequent
    draw, so ``--seed`` is expanded into a second, independent stream instead
    (``np.random.default_rng`` accepts a sequence as its entropy).  There is
    still exactly one seed: the same ``--seed`` reproduces the same manifests.
    """
    return np.random.default_rng([seed, EMO_STREAM_ID])


def emotion_targets(
    count: int, mask_rate: float, rng: np.random.Generator
) -> list[str]:
    """``count`` emotion targets, ``mask_rate`` of them the mask sentinel.

    The masked share is a quota rather than an independent coin flip per record
    so that a small split cannot come out with no sentinels (or no real labels)
    at all by chance -- both splits have to exercise both paths.  The quota is
    clamped to leave at least one of each whenever the rate is strictly between
    0 and 1; an explicit rate of exactly 0 or 1 is honoured as asked, because
    "every record masked" is a crash case worth being able to force.

    The real labels cycle through EMO_LABEL_TARGETS instead of being sampled
    from a realistic, neutral-heavy prior: a smoke split leaves only a handful
    of unmasked clips, and sampling would routinely hand all of them the same
    token -- reintroducing the constant target this flag exists to avoid.

    Positions are then shuffled rather than striped.  Consecutive runs of masked
    records are the point: a token-batched smoke run packs 2-4 neighbouring
    clips into a batch, so shuffling lets a fully masked batch occur naturally,
    without forcing an arrangement that no real corpus would produce.
    """
    masked = int(round(count * mask_rate))
    if 0.0 < mask_rate < 1.0 and count >= 2:
        masked = min(max(masked, 1), count - 1)
    targets = [EMO_MASK_TARGET] * masked
    targets += [
        EMO_LABEL_TARGETS[index % len(EMO_LABEL_TARGETS)]
        for index in range(count - masked)
    ]
    return [targets[index] for index in rng.permutation(count)]


def emotion_counts(records: list[dict[str, object]]) -> str:
    """Report the emotion distribution actually present in ``records``.

    Counted from the emitted records, not from the requested rate, because this
    project has repeatedly been bitten by runs that succeeded at the wrong
    thing.  The log should state what the manifest contains rather than what it
    was asked for.
    """
    counts = Counter(str(record["emo_target"]) for record in records)
    total = len(records)
    masked = counts.get(EMO_MASK_TARGET, 0)
    percent = 100.0 * masked / total if total else 0.0
    labelled = [target for target in EMO_LABEL_TARGETS if counts.get(target)]
    # Anything outside the eight legal tokens is a bug, but print it rather than
    # hide it -- a silent drop here is what makes such a bug hard to notice.
    labelled += [
        target
        for target in sorted(counts)
        if target != EMO_MASK_TARGET and target not in EMO_LABEL_TARGETS
    ]
    spread = " ".join(f"{target}x{counts[target]}" for target in labelled) or "none"
    return f"{masked}/{total} masked ({percent:.1f}%) {spread}"


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
    emo_targets: list[str] | None = None,
) -> list[dict[str, object]]:
    """Write ``count`` clips and their manifest; return the records written.

    ``emo_targets`` defaults to ``None``, meaning the historical constant
    ``<|NEUTRAL|>`` for every record.  The default has to stay reachable without
    the caller opting in, because the checked-in smoke manifests were generated
    that way.
    """
    if emo_targets is not None and len(emo_targets) != count:
        raise ValueError(
            f"emo_targets has {len(emo_targets)} entries for {count} clips"
        )
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
                "emo_target": EMO_TARGET if emo_targets is None else emo_targets[index],
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
    # Each split is drawn separately from the emotion stream, so the sentinels
    # cannot all land on one side of the split: an all-masked val set would make
    # the validation emotion accuracy undefined, and an all-masked train set
    # would leave the emotion head with no supervision at all -- both look like
    # a working run in the log.
    emo_rng = emotion_rng(args.seed) if args.emo_mix else None
    mask_rate = (
        DEFAULT_EMO_MASK_RATE if args.emo_mask_rate is None else args.emo_mask_rate
    )
    train = write_split(
        audio, samplerate, args.wav_dir, args.train_jsonl,
        "smoke_train", args.num_train, args.min_seconds, args.max_seconds, rng,
        emo_targets=(
            None if emo_rng is None
            else emotion_targets(args.num_train, mask_rate, emo_rng)
        ),
    )
    val = write_split(
        audio, samplerate, args.wav_dir, args.val_jsonl,
        "smoke_val", args.num_val, args.min_seconds, args.max_seconds, rng,
        emo_targets=(
            None if emo_rng is None
            else emotion_targets(args.num_val, mask_rate, emo_rng)
        ),
    )

    print(f"wav dir : {args.wav_dir.resolve()}")
    print(f"train   : {args.train_jsonl.resolve()} ({len(train)} clips)")
    print(f"val     : {args.val_jsonl.resolve()} ({len(val)} clips)")
    print(f"lengths : {sorted({record['source_len'] for record in train})} frames (train)")
    print(f"emotion : train {emotion_counts(train)} | val {emotion_counts(val)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
