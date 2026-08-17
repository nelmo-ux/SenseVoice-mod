#!/usr/bin/env python3
"""Score SenseVoice's emotion head on JVNV, an external human-labelled corpus.

Why a second SER benchmark exists
---------------------------------

``scripts/eval_chunk_gap.py`` also reports emotion accuracy, but it scores
against ``emo_target`` in ``data/vn/val.jsonl``, and those labels were produced
by a SenseVoice model in the first place.  Scoring a SenseVoice checkpoint
against SenseVoice-generated labels is partly circular: it measures agreement
with the labeller, and a checkpoint can move on that axis for reasons that have
nothing to do with recognising emotion.  This script is the independent
measurement - the labels here were assigned by the people who recorded the
corpus, not by any model in this repository.

Corpus
------

`litagin/jvnv_corpus_v1_no_nv <https://huggingface.co/datasets/litagin/jvnv_corpus_v1_no_nv>`_
- Japanese, four speakers (two female, two male), six emotions.  It is licensed
**CC BY-SA**, which means anything redistributed from it (including a derived
dataset) carries the same licence forward; nothing here redistributes it.

**JVNV is an evaluation corpus only.  It is never mixed into training.**  It is
the only emotion measurement in this repo that is not downstream of the model's
own labels, and folding it into a training set would destroy that property
permanently and silently - the resulting numbers would look fine.

The six JVNV emotions map onto SenseVoice's tokens one-for-one.  JVNV has **no
neutral class**, so ``<|NEUTRAL|>`` is always a wrong answer here, and it is
counted rather than dropped: the round-2 failure mode was a collapsed head that
answers ``<|NEUTRAL|>`` for every input, and that model has to score near zero
on this benchmark instead of quietly having all its predictions discarded.  The
per-model prediction distribution is in the JSON for the same reason - "it
predicts one class for everything" should be visible directly, not inferred.

Reading the output
------------------

Pass every checkpoint in one run so the comparison shares one corpus, one
device and one decode setting::

    .venv/bin/python scripts/eval_ser_jvnv.py \\
        --corpus-dir /data/jvnv_corpus_v1_no_nv \\
        --base \\
        --checkpoint round2-ep3=outputs/chunk/model.pt.ep3 \\
        --checkpoint round3-ep3=outputs/chunk_r3/model.pt.ep3 \\
        --out outputs/ser_jvnv.json

The shape a successful repair produces is ``round-2 << base <= round-3``: round
2 collapsed the head, so it must score far below the published model, and round
3 must have got back to at least parity.  A round-3 number that merely beats
round 2 proves nothing - the bar is the base model.

Expect the headline to be uncomfortable: round-2 epoch 3 is the **best** ASR
checkpoint measured so far (chunk CER 0.1623) and is expected to be the **worst**
emotion checkpoint here.  That is not a contradiction, it is the finding - the
transcription objective improved while the emotion head was being trained
against a constant target.  Which is why the prediction distribution is printed
next to the accuracy for every model: "collapsed onto one class" and "merely
inaccurate" produce similar accuracies and completely different diagnoses, and
only the distribution tells them apart.

Decoding is full-attention, in fp32 with TF32 explicitly off, matching
``eval_chunk_gap.py`` so the two scripts' numbers are comparable.  The metric
arithmetic is imported from that script rather than reimplemented, so accuracy
and macro-F1 cannot come to mean different things in the two reports.
"""

from __future__ import annotations

import os
import sys

# Set before torch is imported, exactly as ``eval_chunk_gap`` does and for the
# same reason - several SenseVoice ops have no MPS kernel.  It has to happen
# here too because this module is an entry point: whichever of the two is
# imported first must have already set it.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import textwrap  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

# ``scripts/`` is not a package, so the sibling script is imported by putting
# its directory on the path.  The repo root goes on too, for ``streaming``.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _entry in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from eval_chunk_gap import (  # noqa: E402
    DEFAULT_BASE_DIR,
    EMOTION_TOKENS,
    NO_PREDICTION,
    CheckpointRecogniser,
    classification_metrics,
    configure_precision,
    default_device,
    extract_emotion_tag,
    load_audio,
    precision_report,
    release,
)
from streaming.config import StreamingConfig  # noqa: E402

__all__ = ["main"]

#: The corpus this script scores, recorded in the report so a JSON file names
#: its own provenance.
JVNV_DATASET_ID = "litagin/jvnv_corpus_v1_no_nv"

JVNV_LICENCE = "CC BY-SA (share-alike: any redistributed derivative inherits it)"

JVNV_USAGE_NOTE = (
    "evaluation only - JVNV is never mixed into training; it is the only "
    "emotion measurement in this repo that is not downstream of the model's "
    "own labels, and training on it would silently destroy that"
)

#: JVNV's six emotion names to SenseVoice's tokens.  There is deliberately no
#: entry for neutral: JVNV has no neutral recordings, so a ``<|NEUTRAL|>``
#: prediction is always wrong here and is scored as such.
JVNV_EMOTION_TO_TOKEN: Dict[str, str] = {
    "anger": "<|ANGRY|>",
    "disgust": "<|DISGUSTED|>",
    "fear": "<|FEARFUL|>",
    "happy": "<|HAPPY|>",
    "sad": "<|SAD|>",
    "surprise": "<|SURPRISED|>",
}

#: Spellings seen in the wild for the same six classes.  Accepting them costs
#: nothing and stops a re-export of the corpus under slightly different
#: directory names from aborting the run.
_EMOTION_ALIASES: Dict[str, str] = {
    "angry": "anger",
    "disgusted": "disgust",
    "fearful": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprised": "surprise",
}

_EMOTION_LOOKUP: Dict[str, str] = {
    **{name: name for name in JVNV_EMOTION_TO_TOKEN},
    **_EMOTION_ALIASES,
}

#: JVNV speaker ids: ``F1``/``F2``/``M1``/``M2``.
_SPEAKER_RE = re.compile(r"^[fm]\d+$")

#: Split a path component into the fields that might name an emotion.
_FIELD_RE = re.compile(r"[_\-\s.]+")

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

#: What a repaired emotion head has to produce, stated in the report so the
#: reader does not have to remember it.
EXPECTED_SHAPE_NOTE = (
    "a successful round-3 repair looks like 'round-2 << base <= round-3': the "
    "collapsed round-2 head must score far below the published base model, and "
    "round 3 must reach at least parity with it.  Beating round 2 alone is not "
    "evidence of anything"
)

#: Why a bad number here is compatible with a good ASR checkpoint.  Stated in
#: the report because the two measurements will be read side by side.
ASR_SER_DIVERGENCE_NOTE = (
    "SER quality here is independent of chunk CER: round-2 epoch 3 is the best "
    "ASR checkpoint measured so far (chunk CER 0.1623) and is expected to be "
    "the worst model in this table, because its emotion head was trained "
    "against a constant target while its transcription improved.  Read the "
    "prediction distribution alongside the accuracy - a head collapsed onto one "
    "class and a head that is merely inaccurate score alike and mean different "
    "things"
)


# ---------------------------------------------------------------- corpus walk


@dataclass(frozen=True)
class JvnvClip:
    """One JVNV recording.

    Attributes:
        key: Path relative to the corpus root, the stable identifier used in
            the report and to align predictions across models.
        path: Absolute path to the audio.
        emotion: The SenseVoice emotion token this clip is labelled with.
        speaker: JVNV speaker id (``"F1"``...), or ``None`` when the layout does
            not name one.
    """

    key: str
    path: Path
    emotion: str
    speaker: Optional[str]


def _fields(part: str) -> List[str]:
    """Split one path component into lowercase fields.

    Args:
        part: A directory name or a file stem.

    Returns:
        Its ``_``/``-``/``.``/space-separated fields, lowercased.
    """
    return [field for field in _FIELD_RE.split(part.lower()) if field]


def _relative_parts(path: Path, root: Optional[Path]) -> Tuple[str, ...]:
    """The path components to search, relative to the corpus root when given.

    Restricting the search to the part of the path *below* the corpus root
    matters: a corpus staged under ``/mnt/sad-machine/jvnv`` must not have every
    clip in it labelled sad.

    Args:
        path: The clip path.
        root: Corpus root, or ``None`` to search the whole path.

    Returns:
        The components, file stem last.

    Raises:
        ValueError: If ``path`` is not under ``root``.
    """
    relative = path.relative_to(root) if root is not None else path
    return (*relative.parts[:-1], relative.stem)


def emotion_from_path(path: Path, root: Optional[Path] = None) -> str:
    """Read a clip's labelled emotion out of its path.

    JVNV encodes the label in the layout (``F1/anger/F1_anger_1.wav`` and the
    like), so both the directory names and the filename fields are searched.
    Kept a pure function of the path so it is unit-testable without the corpus.

    Args:
        path: The clip path.
        root: Corpus root; components above it are ignored.  See
            :func:`_relative_parts`.

    Returns:
        The SenseVoice emotion token.

    Raises:
        ValueError: If no emotion can be determined, or if the path names two
            different ones.  Both are raised rather than skipped: a clip
            silently dropped shrinks the population behind the accuracy without
            saying so, which is the one failure mode this whole benchmark
            exists to avoid.
    """
    names = set()
    for part in _relative_parts(path, root):
        for field in _fields(part):
            canonical = _EMOTION_LOOKUP.get(field)
            if canonical is not None:
                names.add(canonical)

    if not names:
        raise ValueError(
            f"cannot determine the emotion of {path}: no path component names "
            f"one of {sorted(JVNV_EMOTION_TO_TOKEN)}.  Point --corpus-dir at "
            "the JVNV root, whose layout is <speaker>/<emotion>/<clip>.wav"
        )
    if len(names) > 1:
        raise ValueError(
            f"ambiguous emotion for {path}: the path names {sorted(names)}"
        )
    return JVNV_EMOTION_TO_TOKEN[names.pop()]


def speaker_from_path(path: Path, root: Optional[Path] = None) -> Optional[str]:
    """Read the JVNV speaker id out of a clip's path.

    Args:
        path: The clip path.
        root: Corpus root; components above it are ignored.

    Returns:
        The speaker id upper-cased (``"F1"``), or ``None`` when the layout names
        none.  Unlike the emotion this is not fatal - the speaker is only used
        for ``--speakers`` filtering and for the report.
    """
    for part in _relative_parts(path, root):
        for field in _fields(part):
            if _SPEAKER_RE.match(field):
                return field.upper()
    return None


def balanced_subset(clips: Sequence[JvnvClip], limit: int) -> List[JvnvClip]:
    """Take ``limit`` clips spread evenly over speaker and emotion.

    A plain ``clips[:limit]`` over a path-sorted corpus would take one speaker's
    anger recordings and nothing else, and macro-F1 over a single class is not a
    number worth printing.  Round-robin over the ``(speaker, emotion)`` groups
    instead, in sorted group order, so a truncated run is still a comparison.

    Args:
        clips: Candidates, in a deterministic order.
        limit: How many to keep.

    Returns:
        The subset, restored to the input order so the report reads naturally.
    """
    groups: Dict[Tuple[str, str], List[JvnvClip]] = {}
    for clip in clips:
        groups.setdefault((clip.speaker or "", clip.emotion), []).append(clip)

    ordered = [groups[key] for key in sorted(groups)]
    keep: List[JvnvClip] = []
    index = 0
    while len(keep) < limit and any(index < len(group) for group in ordered):
        for group in ordered:
            if index < len(group):
                keep.append(group[index])
                if len(keep) == limit:
                    break
        index += 1

    chosen = {clip.key for clip in keep}
    return [clip for clip in clips if clip.key in chosen]


def discover_clips(
    corpus_dir: Path,
    speakers: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[JvnvClip]:
    """Walk a locally staged JVNV tree.

    The corpus is read from disk and never from the network: an evaluation that
    downloads is an evaluation that fails differently on a cluster node than on
    a laptop.

    Args:
        corpus_dir: Root of the staged corpus.
        speakers: Keep only these speaker ids (case-insensitive), or ``None``
            for all.
        limit: Keep at most this many clips, chosen by :func:`balanced_subset`.

    Returns:
        The clips, sorted by path.

    Raises:
        FileNotFoundError: If ``corpus_dir`` does not exist.
        ValueError: If it holds no audio, if a clip's emotion cannot be
            determined, or if ``--speakers`` selected nothing.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"JVNV corpus directory not found: {corpus_dir}.  Stage "
            f"{JVNV_DATASET_ID} locally first; this script never downloads"
        )

    wanted = {name.upper() for name in speakers} if speakers else None
    clips: List[JvnvClip] = []
    found_audio = False
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        found_audio = True
        speaker = speaker_from_path(path, corpus_dir)
        if wanted is not None and (speaker or "").upper() not in wanted:
            continue
        clips.append(
            JvnvClip(
                key=path.relative_to(corpus_dir).as_posix(),
                path=path,
                # Raises on an unreadable layout, which aborts the run.
                emotion=emotion_from_path(path, corpus_dir),
                speaker=speaker,
            )
        )

    if not found_audio:
        raise ValueError(
            f"no audio under {corpus_dir} (looked for {', '.join(AUDIO_SUFFIXES)})"
        )
    if not clips:
        raise ValueError(f"--speakers {sorted(wanted or [])} matched no clip")
    if limit is not None and len(clips) > limit:
        clips = balanced_subset(clips, limit)
    return clips


# ------------------------------------------------------------------- decoding


@dataclass(frozen=True)
class ModelSpec:
    """One model to evaluate.

    Attributes:
        label: Column name in the comparison, from the command line.
        model_dir: Directory supplying the config, CMVN and BPE.
        checkpoint: Weights to load, or ``None`` for the published base model.
    """

    label: str
    model_dir: Path
    checkpoint: Optional[Path]


def parse_checkpoint_spec(value: str) -> Tuple[str, Path]:
    """Parse one ``LABEL=PATH`` ``--checkpoint`` argument.

    Args:
        value: The raw argument.

    Returns:
        ``(label, path)``.  The path is not required to exist yet; ``main``
        checks that, so every bad path in a multi-checkpoint run is reported at
        once instead of one per re-run.

    Raises:
        argparse.ArgumentTypeError: If the argument is not ``LABEL=PATH`` with
            both sides non-empty.  A bare path is rejected rather than given a
            generated label, because the labels are what the comparison table is
            read by.
    """
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            f"--checkpoint expects LABEL=PATH, got {value!r} "
            "(e.g. --checkpoint round3-ep3=outputs/chunk_r3/model.pt.ep3)"
        )
    return label.strip(), Path(path.strip())


def predict_emotions(
    spec: ModelSpec,
    config: StreamingConfig,
    clips: Sequence[JvnvClip],
    audio: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Load one model, decode every clip with it, then release it.

    Only one model is in memory at a time; a three-point comparison would
    otherwise hold three ~234M-parameter models at once for no reason.

    Args:
        spec: Which weights to load, from where, and what to call the result.
        config: Decode settings; shared by every model in the run.
        clips: Clips to decode.
        audio: Pre-decoded waveforms keyed by clip key, shared across models so
            each one sees bit-identical input.

    Returns:
        ``({clip_key: predicted token or None}, failures)``.  A clip whose
        decode raised gets ``None`` - the same value as "emitted no emotion" -
        rather than being dropped, so every model is scored over exactly the
        same population and the failure count is reported separately.
    """
    recogniser = CheckpointRecogniser(spec.model_dir, config, spec.checkpoint)
    predictions: Dict[str, Optional[str]] = {}
    failures: List[str] = []
    try:
        for index, clip in enumerate(clips, start=1):
            print(
                f"  [{spec.label}] {index}/{len(clips)} {clip.key}",
                end="\r",
                file=sys.stderr,
            )
            try:
                predictions[clip.key] = extract_emotion_tag(
                    recogniser.decode_full(audio[clip.key])
                )
            except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
                predictions[clip.key] = None
                failures.append(f"{spec.label}: decode failed for {clip.key}: {exc}")
        print(" " * 72, end="\r", file=sys.stderr)
    finally:
        release(recogniser)
    return predictions, failures


# --------------------------------------------------------------------- report


def _distribution(values: Iterable[str]) -> Dict[str, int]:
    """Count occurrences, sorted by key for a stable report.

    Args:
        values: The labels.

    Returns:
        ``{label: count}``.
    """
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _pct(value: Optional[float]) -> str:
    """Format a ratio as a percentage for the table.

    Args:
        value: The ratio, or ``None``.

    Returns:
        A right-aligned string; ``"-"`` for ``None``.
    """
    return f"{'-':>10}" if value is None else f"{value * 100:>9.1f}%"


def _num(value: Optional[float]) -> str:
    """Format a metric for the table.

    Args:
        value: The metric, or ``None``.

    Returns:
        A right-aligned string; ``"-"`` for ``None``.
    """
    return f"{'-':>10}" if value is None else f"{value:>10.4f}"


def print_comparison(payload: Dict[str, Any]) -> None:
    """Print the model comparison table.

    The neutral column is not decoration: JVNV has no neutral class, so it is
    the share of answers that are wrong by construction, and a collapsed
    emotion head shows up there as ~100% before any other number is read.

    Args:
        payload: The assembled report.
    """
    line = "=" * 78

    def wrapped(label: str, text: str) -> str:
        """Indent a long note under its label so the block stays scannable."""
        return textwrap.fill(
            text, width=78, initial_indent=label, subsequent_indent=" " * len(label)
        )

    print(line)
    print(f"SER on JVNV - external human-labelled benchmark ({JVNV_DATASET_ID})")
    print(line)
    print(f"corpus      : {payload['corpus']['dir']}")
    print(wrapped("licence     : ", payload["corpus"]["licence"]))
    print(wrapped("usage       : ", payload["corpus"]["usage"]))
    print(
        f"clips       : {payload['num_clips']}"
        f"   speakers: {', '.join(payload['corpus']['speakers']) or '(unnamed)'}"
    )
    print(
        f"device      : {payload['device']}"
        f"   precision: {payload['precision']['mode']}"
        f"   ban_emo_unk={payload['decode']['ban_emo_unk']}"
    )
    print(f"reference   : {payload['reference_distribution']}")
    print()

    print(
        f"{'model':<20}{'n':>6}{'accuracy':>10}{'macro-F1':>10}"
        f"{'%neutral':>10}{'%no-emo':>10}  most-predicted"
    )
    for label, metrics in payload["per_model"].items():
        num_scored = metrics["num_scored"] or 0
        neutral = metrics["prediction_distribution"].get("<|NEUTRAL|>", 0)
        no_emotion = metrics["num_pred_none"]
        dominant = metrics["dominant_prediction"]
        share = dominant["share"]
        print(
            f"{label:<20}{num_scored:>6}"
            f"{_num(metrics['accuracy'])}{_num(metrics['macro_f1'])}"
            f"{_pct(neutral / num_scored if num_scored else None)}"
            f"{_pct(no_emotion / num_scored if num_scored else None)}"
            f"  {dominant['label']}"
            f"{'' if share is None else f' ({share:.0%})'}"
        )
    print("  (%neutral is wrong by construction: JVNV has no neutral class)")

    # The full distribution, not just the headline, because the finding this
    # benchmark exists to produce is a *shape*: the checkpoint with the best
    # chunk CER is expected to answer one class for every clip here, and
    # "collapsed onto <|NEUTRAL|>" and "merely inaccurate" are the same
    # accuracy number but completely different diagnoses.
    print()
    print("prediction distributions (a single class at ~100% is a collapsed head):")
    for label, metrics in payload["per_model"].items():
        print(f"  {label:<20}{metrics['prediction_distribution']}")

    print()
    print(line)
    print(textwrap.fill(EXPECTED_SHAPE_NOTE, width=78))
    print(line)
    print(textwrap.fill(ASR_SER_DIVERGENCE_NOTE, width=78))
    print(line)

    for message in payload["warnings"]:
        print(f"[warn] {message}", file=sys.stderr)


# ----------------------------------------------------------------------- main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help=(
            "locally staged JVNV root, e.g. <speaker>/<emotion>/<clip>.wav.  "
            "Never downloaded: evaluation must not depend on the network"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=(
            "base model directory supplying the config, CMVN and BPE that every "
            f"checkpoint is loaded against (default: {DEFAULT_BASE_DIR})"
        ),
    )
    parser.add_argument(
        "--base",
        nargs="?",
        type=Path,
        const=DEFAULT_BASE_DIR,
        default=None,
        help=(
            "include the published base model in the comparison, optionally "
            "from a directory other than --model-dir.  It is the bar a repaired "
            "checkpoint has to clear, so a run without it can only show "
            "movement, not recovery"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint_spec,
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "a checkpoint to evaluate, repeatable.  All of them run against one "
            "corpus, one device and one decode setting in a single process, "
            "which is what makes the columns comparable"
        ),
    )
    parser.add_argument(
        "--speakers",
        default=None,
        help="comma-separated JVNV speaker ids to keep, e.g. 'F1,M1' (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "evaluate at most N clips, spread evenly over speaker and emotion "
            "so a truncated run still has every class in it"
        ),
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        choices=("cuda", "mps", "cpu"),
        help=(
            "torch device (default: mps on macOS, else cpu).  'cuda' is never "
            "implicit and runs in fp32 unless --allow-tf32 is given, matching "
            "eval_chunk_gap.py so the two scripts' numbers stay comparable"
        ),
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help=(
            "let CUDA use TF32.  Faster, and NOT numerically comparable to the "
            "fp32 numbers; the report records the run as such.  No effect off CUDA"
        ),
    )
    parser.add_argument(
        "--ban-emo-unk",
        action="store_true",
        help=(
            "forbid <|EMO_UNKNOWN|>, forcing the head to commit to one of the "
            "seven emotions.  Recorded in the report, since it changes what the "
            "head is able to emit and therefore what these numbers mean"
        ),
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="SenseVoice language tag for the decode (JVNV is Japanese)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return parser.parse_args(argv)


def resolve_models(args: argparse.Namespace) -> List[ModelSpec]:
    """Turn ``--base`` and ``--checkpoint`` into the models to run, in order.

    Args:
        args: Parsed command line.

    Returns:
        The specs, base first when requested so it reads as the reference column.

    Raises:
        ValueError: If no model was requested, if a label repeats, or if a
            checkpoint file or model directory is missing.  Missing paths are
            all reported at once: a three-point comparison that dies on the
            third path after decoding the first two has wasted the whole run.
    """
    specs: List[ModelSpec] = []
    if args.base is not None:
        # ``--base DIR`` evaluates the published weights *in that directory*;
        # bare ``--base`` uses the default.  Checkpoints keep loading against
        # --model-dir, which is what supplies their config and CMVN.
        specs.append(ModelSpec(label="base", model_dir=args.base, checkpoint=None))
    for label, path in args.checkpoint:
        specs.append(ModelSpec(label=label, model_dir=args.model_dir, checkpoint=path))

    if not specs:
        raise ValueError(
            "nothing to evaluate: pass --base, one or more "
            "--checkpoint LABEL=PATH, or both"
        )

    seen: Dict[str, int] = {}
    for spec in specs:
        seen[spec.label] = seen.get(spec.label, 0) + 1
    duplicates = sorted(label for label, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate model label(s): {duplicates}")

    missing = sorted(
        {str(spec.model_dir) for spec in specs if not spec.model_dir.exists()}
        | {
            str(spec.checkpoint)
            for spec in specs
            if spec.checkpoint is not None and not spec.checkpoint.exists()
        }
    )
    if missing:
        raise ValueError("not found: " + ", ".join(missing))
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` when the corpus or the model
        selection is unusable.
    """
    args = parse_args(argv)
    warnings_: List[str] = []

    try:
        specs = resolve_models(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    speakers = (
        [name.strip() for name in args.speakers.split(",") if name.strip()]
        if args.speakers
        else None
    )
    try:
        clips = discover_clips(args.corpus_dir, speakers, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    config = StreamingConfig(
        backend="chunk",
        device=args.device,
        language=args.language,
        # The references are emotion labels, not transcripts, so ITN would only
        # change punctuation nobody scores here.
        use_itn=False,
        ban_emo_unk=args.ban_emo_unk,
    )
    config.validate()
    configure_precision(config.device, args.allow_tf32)

    # Decoded once and shared by every model, so any difference between columns
    # is the weights and nothing else.
    audio: Dict[str, np.ndarray] = {}
    usable: List[JvnvClip] = []
    for clip in clips:
        try:
            audio[clip.key] = load_audio(clip.path, config.sample_rate)
        except RuntimeError as exc:
            warnings_.append(str(exc))
            continue
        usable.append(clip)
    if not usable:
        print("nothing to evaluate: no JVNV clip could be decoded", file=sys.stderr)
        for message in warnings_:
            print(f"[warn] {message}", file=sys.stderr)
        return 1

    references = [clip.emotion for clip in usable]
    per_model: Dict[str, Any] = {}
    for spec in specs:
        predictions, failures = predict_emotions(spec, config, usable, audio)
        warnings_.extend(failures)
        metrics = classification_metrics(
            references, [predictions[clip.key] for clip in usable]
        )
        per_model[spec.label] = {
            "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
            "num_decode_failures": len(failures),
            **metrics,
        }

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": config.device,
        "precision": precision_report(config.device, args.allow_tf32),
        "corpus": {
            "dataset": JVNV_DATASET_ID,
            "dir": str(args.corpus_dir),
            "licence": JVNV_LICENCE,
            "usage": JVNV_USAGE_NOTE,
            "speakers": sorted({clip.speaker or "unknown" for clip in usable}),
            "emotion_map": dict(JVNV_EMOTION_TO_TOKEN),
            "no_neutral_note": (
                "JVNV has no neutral recordings, so a <|NEUTRAL|> prediction is "
                "always wrong; such predictions are scored, never dropped"
            ),
        },
        "num_clips": len(usable),
        "decode": {
            "mode": "full-attention (SenseVoiceSmall.inference)",
            "model_dir": str(model_dir),
            "language": config.language,
            "use_itn": config.use_itn,
            "ban_emo_unk": config.ban_emo_unk,
        },
        "scoring": {
            "classes": list(EMOTION_TOKENS),
            "no_prediction_label": NO_PREDICTION,
            "note": (
                "macro-F1 averages the reference-present classes only; a class "
                "the model never predicts enters the average with F1 0.0.  "
                "Shared with scripts/eval_chunk_gap.py"
            ),
        },
        "reference_distribution": _distribution(references),
        # Keyed by the --checkpoint label, so a two-model run today and a
        # three-model run once round 3 exists produce the same file shape.
        "per_model": per_model,
        "expected_shape": EXPECTED_SHAPE_NOTE,
        "asr_ser_divergence": ASR_SER_DIVERGENCE_NOTE,
        "warnings": warnings_,
    }

    print_comparison(payload)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
