#!/usr/bin/env python3
"""Compare a chunk-finetuned SenseVoiceSmall checkpoint against the base model.

Run once per epoch to pick the best checkpoint::

    .venv/bin/python scripts/eval_chunk_gap.py \\
        --checkpoint outputs/chunk_mps/model.pt.ep2 \\
        --limit 20 --out outputs/chunk_mps/eval_ep2.json

The fine-tune is **Japanese-specialised**.  Checkpoint selection is decided by
Japanese quality alone; multilingual retention is explicitly *not* a maintained
property of this run, so the Chinese clip is measured but reported as reference
only (see :data:`REFERENCE_ONLY_NOTE`).

Three Japanese measurements
---------------------------

1.  **Chunk-mode partial quality.**  ``data/vn/val.jsonl`` decoded through the
    real streaming path (``streaming.chunk_backend.ChunkBackend`` driven by
    ``streaming.streaming_model.StreamingSenseVoice``) with the chunk geometry
    the model is finetuned with, scored against the manifest's ``target``.
    ``docs/chunk_training.md`` is explicit that validation must be chunk-mode
    *decoding*, not full-attention val loss, which is why nothing here reads a
    loss.

2.  **Forgetting check.**  The same Japanese clips decoded with **full
    attention** (``SenseVoiceSmall.inference``).  A finetune that trades offline
    quality for streaming quality shows up here as a rising full-attention CER
    relative to the base model.

3.  **Chunk-vs-full gap.**  Closing this gap is the entire point of the
    finetune, so it is reported for both models: as ``chunk CER - full CER``
    against the reference, and as a direct ``CER(full decode, chunk decode)``
    which needs no reference and so also works on the untranscribed clips.

Clips without a reference transcript
------------------------------------

``ja.mp3`` (the model-bundled Japanese sample, resolved out of the HuggingFace
snapshot cache) and ``runtime/llama.cpp/tests/sample.wav`` (Chinese) have no
ground truth.  For those the **base model's own full-attention output is the
reference**, so the number reported is *drift* from the base model rather than
accuracy.  Drift on the Chinese clip is informational only.

Two chunk decodes per clip
--------------------------

``chunk`` is the complete chunk-mode decode, tail flushed, covering every frame
- the fair thing to compare against the full-attention decode of the same
audio.  ``chunk_last_partial`` is the last partial a user would actually have
seen mid-stream; it stops short of the tail (up to ``chunk_size - 1`` buffered
frames plus ``pad_right`` withheld ones), so its CER carries deletions that are
an artefact of where the stream was cut, not a quality signal.  Both are in the
JSON; only the former is scored.

Emotion (SER)
-------------

Rounds 1 and 2 trained the emotion slot against a constant ``<|NEUTRAL|>``
target and collapsed the head, and *this script could not see it*: every CER
path runs ``strip_rich_tags`` first, so the emotion token never reached a
metric.  The ``ser`` block under ``japanese.val.metrics`` closes that hole.  It
reads ``emo_target`` from the manifest, parses the predicted emotion out of the
**full-attention** decode (in the streaming product the authoritative final
result is a full-attention pass, so that is the prediction a user actually
gets), and reports accuracy, macro-F1, per-class P/R/F1 and a confusion matrix.

Two properties of that block matter more than the numbers in it:

* It states its own ``num_scored``.  Clips whose ``emo_target`` is the
  ``<|SER|>`` mask sentinel, absent, or not an emotion token are excluded and
  counted separately, so the population behind the accuracy is never implicit.
* It refuses to report a meaningless number silently.  A manifest with no
  ``emo_target`` at all comes back ``status="na"``; one whose every reference is
  the same token - exactly the round-2 all-``<|NEUTRAL|>`` case - comes back
  ``status="degenerate"`` rather than as a healthy-looking 100% accuracy.

Because the val-set emotion labels are themselves model-generated, scoring
against them is partly circular.  ``scripts/eval_ser_jvnv.py`` is the external,
human-labelled counterpart, and it imports the metric maths from here so the two
report the same arithmetic.
"""

from __future__ import annotations

import os
import sys

# MUST be set before torch is imported, which the local imports below do
# transitively.  Several ops on the SenseVoice path have no MPS kernel -
# ``aten::_ctc_loss`` is the one that bites first - and raise
# NotImplementedError without the CPU fallback.  Setting it in-process rather
# than documenting it as a shell prerequisite means a caller cannot forget it.
#
# Kept to macOS because it is an MPS-only workaround with no business on the
# CUDA path: no CUDA build of torch runs on darwin, so this guard is exactly
# "not while running on the GPU cluster", and elsewhere the variable is inert.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import gc
import glob
import json
import re
import textwrap
import unicodedata
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# The repo root has to be importable before the local imports below: this file
# lives in ``scripts/`` but ``model`` and ``streaming`` sit at the root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streaming.backends import _SegmentState  # noqa: E402
from streaming.config import StreamingConfig  # noqa: E402
from streaming.ctc_decode import strip_rich_tags  # noqa: E402
from streaming.streaming_model import StreamingSenseVoice  # noqa: E402

#: ``main`` is the entry point; the rest is the surface
#: ``scripts/eval_ser_jvnv.py`` reuses so the two reports cannot drift into
#: computing differently-named versions of the same metric.
__all__ = [
    "main",
    "classification_metrics",
    "extract_emotion_tag",
    "leading_rich_tags",
    "summarise_ser",
]

#: The dynamic chunk-mask configuration used in training (``finetune_chunk.sh``).
#: Evaluation decodes with one of *these* geometries so the encoder is asked for
#: a computation it was actually trained on.  ``pad_right`` is implied:
#: ``chunk_size - stride - pad_left``.
TRAINING_CHUNK_CONFIG: Dict[str, Tuple[int, ...]] = {
    "chunk_size": (8, 12, 16),
    "stride": (6, 10, 14),
    "pad_left": (0, 0, 0),
    "encoder_att_look_back_factor": (1, 1, 1),
}

#: Which entry of :data:`TRAINING_CHUNK_CONFIG` to decode with by default.  The
#: middle one (720 ms window, 600 ms commit, 120 ms lookahead) is also what
#: ``StreamingConfig``'s chunk defaults mirror.
DEFAULT_GEOMETRY_INDEX = 1

DEFAULT_BASE_DIR = REPO_ROOT / "models" / "SenseVoiceSmall"
DEFAULT_VAL_JSONL = REPO_ROOT / "data" / "vn" / "val.jsonl"
DEFAULT_ZH_CLIP = REPO_ROOT / "runtime" / "llama.cpp" / "tests" / "sample.wav"

#: Where the model-bundled sample clips land.  The snapshot hash changes
#: whenever the model is re-pulled, so it is globbed rather than hardcoded.
JA_CLIP_GLOB = (
    "~/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall"
    "/snapshots/*/example/ja.mp3"
)

REFERENCE_ONLY_NOTE = (
    "reference only, not a selection criterion: this finetune is "
    "Japanese-specialised and Chinese retention is not a maintained property"
)

#: Stripped before scoring.  CER over Japanese and Chinese is conventionally
#: computed without punctuation or spacing, and the references in the manifest
#: carry neither reliably, so scoring them would measure transcription
#: convention rather than recognition.  ``--keep-punctuation`` turns this off.
_PUNCTUATION = frozenset(
    "、。，．,.!?！？;；:：…‥・「」『』()（）[]{}〈〉《》"
    "\"'`|/\\-‐—―~〜_＿*＊&＆%％#＃@＠+＋=＝<>＜＞"
)

#: The seven emotion tokens SenseVoice's emotion slot is trained over, and the
#: only values a manifest ``emo_target`` may take to be scorable.  Order is the
#: README's; report keys are sorted separately, so nothing depends on it.
EMOTION_TOKENS: Tuple[str, ...] = (
    "<|HAPPY|>",
    "<|SAD|>",
    "<|ANGRY|>",
    "<|NEUTRAL|>",
    "<|FEARFUL|>",
    "<|DISGUSTED|>",
    "<|SURPRISED|>",
)

#: Written into ``emo_target`` by ``scripts/prepare_vn_data.py`` for utterances
#: that carry no emotion label; ``model.py`` maps it to ``ignore_id`` so the
#: slot contributes no gradient.  It is not a class, so clips carrying it are
#: excluded from SER scoring rather than counted as a seventh-plus category.
EMO_MASK_TOKEN = "<|SER|>"

#: The model's "I have no emotion for this" output (``emo_dict["unk"]``).  It is
#: never a *reference* class, but it very much is a prediction, and suppressing
#: it is exactly what ``ban_emo_unk`` does - so it is recognised on the
#: prediction side and shows up in the confusion matrix as a wrong answer rather
#: than being silently reported as "no prediction".
EMO_UNKNOWN_TOKEN = "<|EMO_UNKNOWN|>"

#: Everything the emotion slot can emit, i.e. what may be parsed out of a decode.
EMOTION_PREDICTION_TOKENS: Tuple[str, ...] = EMOTION_TOKENS + (EMO_UNKNOWN_TOKEN,)

#: Confusion-matrix column for a decode that carried no emotion tag at all.  A
#: real string rather than ``None`` because JSON object keys must be strings,
#: and it is deliberately not a valid tag so it cannot collide with one.
NO_PREDICTION = "<none>"

#: One ``<|...|>`` rich tag.  ``[^|]*`` stops the match at the ``|`` of the next
#: tag, matching ``streaming.ctc_decode``'s own pattern.
_TAG_RE = re.compile(r"<\|[^|]*\|>")


# --------------------------------------------------------------------- scoring


def levenshtein(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Edit distance between two sequences.

    Args:
        reference: The ground-truth sequence (characters or word tokens).
        hypothesis: The predicted sequence.

    Returns:
        The minimum number of insertions, deletions and substitutions turning
        ``hypothesis`` into ``reference``.
    """
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion from the reference
                    current[j - 1] + 1,  # insertion into the reference
                    previous[j - 1] + (ref_item != hyp_item),  # substitution
                )
            )
        previous = current
    return previous[-1]


def normalize_chars(text: str, keep_punctuation: bool = False) -> str:
    """Reduce decoded or reference text to the character sequence CER scores.

    Rich tags go first (``<|ja|><|NEUTRAL|><|Speech|><|woitn|>`` and friends are
    model metadata, not transcription), then NFKC folds the full-width/half-width
    distinction, then whitespace is dropped entirely - Japanese and Chinese
    references are not consistently spaced and the model does not emit spaces
    the same way twice.

    Args:
        text: Raw decoded or reference text.
        keep_punctuation: Keep punctuation instead of stripping it.

    Returns:
        The normalised character string.
    """
    text = strip_rich_tags(text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if not ch.isspace())
    if not keep_punctuation:
        text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    return text


def normalize_words(text: str, keep_punctuation: bool = False) -> List[str]:
    """Tokenise text for WER on whitespace.

    Note:
        WER is near-meaningless for Japanese and Chinese, which are unsegmented -
        a whole utterance is usually one "word", so WER saturates near 1.0 and
        moves in steps of 1/n.  It is reported because it is cheap and because
        the manifest may one day carry spaced text, but **CER is the metric to
        read** and the only one selection keys off.

    Args:
        text: Raw decoded or reference text.
        keep_punctuation: Keep punctuation instead of treating it as a separator.

    Returns:
        The whitespace-separated tokens.
    """
    text = strip_rich_tags(text)
    text = unicodedata.normalize("NFKC", text)
    if not keep_punctuation:
        text = "".join(" " if ch in _PUNCTUATION else ch for ch in text)
    return text.split()


def pair_cer(reference: str, hypothesis: str, keep_punctuation: bool = False) -> float:
    """CER of one hypothesis against one reference.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.
        keep_punctuation: Passed to :func:`normalize_chars`.

    Returns:
        ``edits / len(reference)``, or ``0.0`` when both normalise to empty and
        ``1.0`` when only the reference does (any output against no reference is
        wholly insertion).  Not clamped to 1.0 otherwise: a hypothesis longer
        than its reference can legitimately exceed it.
    """
    ref = normalize_chars(reference, keep_punctuation)
    hyp = normalize_chars(hypothesis, keep_punctuation)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def corpus_metrics(
    pairs: Sequence[Tuple[str, str]],
    keep_punctuation: bool = False,
) -> Optional[Dict[str, Any]]:
    """Aggregate CER/WER over a set of (reference, hypothesis) pairs.

    Args:
        pairs: ``(reference, hypothesis)`` per clip.
        keep_punctuation: Passed through to the normalisers.

    Returns:
        ``None`` for an empty input, else a dict with ``cer`` (corpus-level:
        total edits over total reference characters - the headline number, since
        it weights clips by length), ``mean_cer`` (unweighted mean over clips,
        which a single short clip can dominate), ``wer``, and the counts behind
        them.
    """
    if not pairs:
        return None

    char_edits = char_len = word_edits = word_len = 0
    per_clip: List[float] = []
    for reference, hypothesis in pairs:
        ref_chars = normalize_chars(reference, keep_punctuation)
        hyp_chars = normalize_chars(hypothesis, keep_punctuation)
        edits = levenshtein(ref_chars, hyp_chars)
        char_edits += edits
        char_len += len(ref_chars)
        per_clip.append(pair_cer(reference, hypothesis, keep_punctuation))

        ref_words = normalize_words(reference, keep_punctuation)
        hyp_words = normalize_words(hypothesis, keep_punctuation)
        word_edits += levenshtein(ref_words, hyp_words)
        word_len += len(ref_words)

    return {
        "cer": char_edits / char_len if char_len else 0.0,
        "mean_cer": sum(per_clip) / len(per_clip),
        "wer": word_edits / word_len if word_len else 0.0,
        "num_clips": len(pairs),
        "ref_chars": char_len,
        "char_edits": char_edits,
        "ref_words": word_len,
        "word_edits": word_edits,
    }


# --------------------------------------------------------------- SER scoring


def leading_rich_tags(text: str) -> List[str]:
    """Return the unbroken run of ``<|...|>`` tags at the start of ``text``.

    SenseVoice emits its four metadata slots as a tag block before the
    transcript (``<|ja|><|HAPPY|><|Speech|><|woitn|>...``).  Only that leading
    block is metadata: an identical-looking token later in the string is
    transcript content, which is why this stops at the first character that is
    not part of a tag instead of scanning the whole string.  Leading whitespace
    is tolerated, tag *order* is not assumed.

    Args:
        text: Raw decoded text, tags intact.

    Returns:
        The tags in the order they appear, ``[]`` if the text does not start
        with one.
    """
    tags: List[str] = []
    position = len(text) - len(text.lstrip())
    while True:
        match = _TAG_RE.match(text, position)
        if match is None:
            return tags
        tags.append(match.group(0))
        position = match.end()


def extract_emotion_tag(text: str) -> Optional[str]:
    """Parse the predicted emotion out of a raw decode.

    Args:
        text: Raw decoded text with rich tags intact - i.e. the output of
            :meth:`CheckpointRecogniser.decode_full`, *before* any
            normalisation, since ``strip_rich_tags`` destroys exactly this.

    Returns:
        The emotion token, or ``None`` when the leading tag block holds none.
        ``None`` is a real outcome, not an error: it is scored as a wrong
        answer rather than dropped, because dropping it would let a model that
        emits no emotion at all inflate its own accuracy.
    """
    for tag in leading_rich_tags(text):
        if tag in EMOTION_PREDICTION_TOKENS:
            return tag
    return None


def classification_metrics(
    references: Sequence[str],
    predictions: Sequence[Optional[str]],
) -> Dict[str, Any]:
    """Accuracy, macro-F1, per-class P/R/F1 and a confusion matrix.

    Shared by the val-set SER block here and by ``scripts/eval_ser_jvnv.py``, so
    the in-domain and the external benchmark cannot drift into reporting
    differently-computed numbers under the same names.  Implemented directly
    rather than via scikit-learn: this repo's dependency set is pinned against a
    numpy ceiling, and the arithmetic is a dozen lines.

    Macro-F1 averages over the classes **present in the reference** only.  A
    class the model never predicts still enters that average with F1 ``0.0``
    (its precision is ``0.0`` by the empty-denominator convention below), which
    is the whole point: a model that answers ``<|NEUTRAL|>`` for everything must
    be punished for the six classes it abandoned, not scored on the one it kept.
    Averaging over predicted classes instead would hand that model a high
    number.

    Args:
        references: Ground-truth label per clip; every entry must be a real
            label, so masked and unlabelled clips are filtered out by the
            caller and counted there.
        predictions: Predicted label per clip, ``None`` where none could be
            parsed.  Aligned with ``references``.

    Returns:
        A metric dict.  ``accuracy`` and ``macro_f1`` are ``None`` for an empty
        population rather than ``0.0``, so "no data" cannot be misread as "the
        model got everything wrong".  ``num_scored`` is always present: every
        metric block in this report states the population it was computed over.

    Raises:
        ValueError: If the two sequences have different lengths, which would
            silently misalign every pair.
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"references ({len(references)}) and predictions "
            f"({len(predictions)}) must be the same length"
        )

    labels = sorted(set(references))
    distribution: Dict[str, int] = {}
    counts: Dict[str, Dict[str, int]] = {label: {} for label in labels}
    num_correct = 0
    for reference, prediction in zip(references, predictions):
        column = NO_PREDICTION if prediction is None else prediction
        counts[reference][column] = counts[reference].get(column, 0) + 1
        distribution[column] = distribution.get(column, 0) + 1
        if prediction is not None and prediction == reference:
            num_correct += 1

    # Zero-filled so every row has the same columns: a confusion matrix with
    # ragged rows is unreadable in a diff and awkward to load.  ``<none>`` is
    # pinned last, after the sorted labels, because it is not a class.
    columns = sorted(set(labels) | (set(distribution) - {NO_PREDICTION}))
    if NO_PREDICTION in distribution:
        columns.append(NO_PREDICTION)
    confusion = {
        label: {column: counts[label].get(column, 0) for column in columns}
        for label in labels
    }

    per_class: Dict[str, Dict[str, Any]] = {}
    f1_scores: List[float] = []
    for label in labels:
        support = sum(counts[label].values())
        num_predicted = distribution.get(label, 0)
        true_positives = counts[label].get(label, 0)
        # Empty denominator -> 0.0, not undefined: a class that was never
        # predicted has to score, or it would drop out of the macro average.
        precision = true_positives / num_predicted if num_predicted else 0.0
        recall = true_positives / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "num_predicted": num_predicted,
        }
        f1_scores.append(f1)

    num_scored = len(references)
    # The diagnosis for a collapsed emotion head is "it answers one class for
    # everything", which is a property of the predictions alone - a low accuracy
    # does not distinguish it from a model that is merely wrong in varied ways.
    # Ties break on the label so the field is deterministic across runs.
    dominant_label, dominant_count = (
        max(distribution.items(), key=lambda item: (item[1], item[0]))
        if distribution
        else (None, 0)
    )
    return {
        "num_scored": num_scored,
        "num_correct": num_correct,
        "accuracy": num_correct / num_scored if num_scored else None,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "macro_f1_classes": labels,
        "num_pred_none": distribution.get(NO_PREDICTION, 0),
        "dominant_prediction": {
            "label": dominant_label,
            "count": dominant_count,
            "share": dominant_count / num_scored if num_scored else None,
        },
        "per_class": per_class,
        "confusion": confusion,
        "prediction_distribution": dict(sorted(distribution.items())),
    }


# ------------------------------------------------------------------- clip I/O


@dataclass
class EvalClip:
    """One audio file to decode.

    Attributes:
        key: Short identifier, used in the report.
        path: Absolute path to the audio.
        reference: Ground-truth transcript, or ``None`` when the clip has none
            (in which case the base model's full-attention decode stands in).
        language: SenseVoice language tag for the decode prompt.
        use_itn: Whether to ask for inverse text normalisation.
        scope: ``"japanese"`` for clips on the selection axis, ``"reference"``
            for informational ones.
        emo_target: The manifest's emotion label, or ``None`` when the record
            carries no ``emo_target`` field at all.  ``<|SER|>`` (the mask
            sentinel) and anything that is not an emotion token are kept
            verbatim here and classified by :func:`summarise_ser`, so the
            report can distinguish "masked" from "missing" from "junk".
    """

    key: str
    path: Path
    reference: Optional[str]
    language: str
    use_itn: bool
    scope: str = "japanese"
    emo_target: Optional[str] = None


@dataclass
class ClipDecode:
    """Everything one model produced for one clip.

    Attributes:
        full: Full-attention decode (``SenseVoiceSmall.inference``), raw.
        chunk: Complete chunk-mode decode, tail flushed, raw.
        chunk_last_partial: The last partial emitted before the stream ended;
            display text, tags already stripped by the backend.
    """

    full: str
    chunk: str
    chunk_last_partial: str


_LANGUAGE_FIELDS = ("text_language", "lang")


def _manifest_language(record: Dict[str, Any], default: str) -> str:
    """Read the language tag out of a manifest record.

    Args:
        record: One decoded JSONL line.
        default: Fallback when the record carries no usable tag.

    Returns:
        A bare SenseVoice language code such as ``"ja"``.
    """
    for field_name in _LANGUAGE_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value:
            return value.strip().strip("<>|") or default
    return default


def load_val_clips(
    path: Path,
    limit: Optional[int],
    default_language: str,
    default_use_itn: bool,
) -> Tuple[List[EvalClip], List[str]]:
    """Read the held-out manifest.

    Decoding follows the manifest's own conventions where it states them
    (``text_language``, ``with_or_wo_itn``), so the hypothesis is produced under
    the same convention the ``target`` was written under.

    Args:
        path: Path to ``val.jsonl``.
        limit: Keep only the first ``limit`` usable clips, or ``None`` for all.
        default_language: Language tag for records that do not state one.
        default_use_itn: ITN setting for records that do not state one.

    Returns:
        ``(clips, warnings)``.  ``clips`` is empty and ``warnings`` explains why
        if the file is missing or unusable - a parallel job generates it, so its
        absence is expected rather than an error.
    """
    if not path.exists():
        return [], [
            f"val manifest not found: {path} - Japanese val metrics skipped, "
            "so this run produces NO checkpoint-selection signal"
        ]

    clips: List[EvalClip] = []
    notes: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                notes.append(f"{path}:{lineno}: unparseable JSON ({exc.msg}), skipped")
                continue

            source = record.get("source")
            target = record.get("target")
            if not source or target is None:
                notes.append(f"{path}:{lineno}: missing 'source' or 'target', skipped")
                continue
            audio = Path(source)
            if not audio.is_absolute():
                audio = (REPO_ROOT / audio).resolve()
            if not audio.exists():
                notes.append(f"{path}:{lineno}: audio not found ({audio}), skipped")
                continue

            itn = record.get("with_or_wo_itn")
            # Read verbatim, not validated here: an unexpected value is a
            # finding the SER block reports, not a reason to drop the clip from
            # the CER metrics it has nothing to do with.
            emo_target = record.get("emo_target")
            clips.append(
                EvalClip(
                    key=str(record.get("key") or f"val_{lineno:05d}"),
                    path=audio,
                    reference=str(target),
                    language=_manifest_language(record, default_language),
                    use_itn=(
                        "withitn" in itn if isinstance(itn, str) else default_use_itn
                    ),
                    emo_target=None if emo_target is None else str(emo_target),
                )
            )
            if limit is not None and len(clips) >= limit:
                break

    if not clips and not notes:
        notes.append(f"{path} contained no usable records")
    return clips, notes


def resolve_ja_clip(explicit: Optional[Path]) -> Tuple[Optional[Path], List[str]]:
    """Locate the model-bundled Japanese sample.

    Args:
        explicit: A path given on the command line, which wins if it exists.

    Returns:
        ``(path, warnings)``; ``path`` is ``None`` when the clip cannot be found,
        which happens when the HuggingFace snapshot has not been pulled.
    """
    if explicit is not None:
        if explicit.exists():
            return explicit, []
        return None, [f"--ja-clip not found: {explicit}, skipped"]

    matches = sorted(glob.glob(os.path.expanduser(JA_CLIP_GLOB)))
    if not matches:
        return None, [
            "bundled ja.mp3 not found under "
            f"{JA_CLIP_GLOB} - skipped; pull the HuggingFace snapshot or pass "
            "--ja-clip to include it"
        ]
    return Path(matches[-1]), []


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode an audio file to mono float32 at ``sample_rate``.

    librosa is used rather than soundfile so that mp3 works alongside wav.  Its
    PySoundFile/audioread deprecation warning on mp3 is suppressed: it is
    harmless and this script's stdout is meant to be scanned and diffed across
    epochs.

    Args:
        path: Audio file, any format librosa can open.
        sample_rate: Target rate; the model expects 16 kHz.

    Returns:
        A 1-D float32 array.

    Raises:
        RuntimeError: If the file cannot be decoded, naming the file - a silent
            skip would quietly shrink the evaluation set.
    """
    import librosa

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with the filename
        raise RuntimeError(f"could not decode {path}: {exc}") from exc
    return np.ascontiguousarray(samples, dtype=np.float32)


# ------------------------------------------------------------------ recogniser


class CheckpointRecogniser(StreamingSenseVoice):
    """``StreamingSenseVoice`` that can load weights from an arbitrary file.

    The base ``__init__`` hardcodes ``from_pretrained(model=model_dir)``, which
    always picks up ``<model_dir>/model.pt``; a finetuned checkpoint lives
    somewhere else entirely.  funasr's loader honours an explicit ``init_param``
    over the one it derives from the model directory, so this overrides
    ``__init__`` - the extension point the base class documents for a subclass
    that replaces weight loading - to pass it through.  Everything after the
    load is the base class's own code, so the frontend, the backend selection
    and the streaming path are the production ones.

    Args:
        model_dir: Base model directory, source of the config, CMVN and BPE.
        config: Streaming tunables; ``backend`` must be ``"chunk"`` for the
            chunk measurements to mean anything.
        checkpoint: Finetuned ``model.pt``; ``None`` loads the base weights.
    """

    def __init__(
        self,
        model_dir: Path,
        config: StreamingConfig,
        checkpoint: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.config.validate()

        from model import SenseVoiceSmall

        load_kwargs: Dict[str, Any] = {
            "model": str(model_dir),
            "device": config.device,
        }
        if checkpoint is not None:
            load_kwargs["init_param"] = str(checkpoint)
        model, kwargs = SenseVoiceSmall.from_pretrained(**load_kwargs)
        model.eval()

        self.model = model
        self.kwargs: Dict[str, Any] = kwargs
        self.frontend = kwargs["frontend"]
        self.tokenizer = kwargs["tokenizer"]

        # funasr silently downgrades to CPU when the requested accelerator is
        # unavailable.  Believe the loader, not the request, so the config, the
        # backend's device and the report all agree on where this actually ran.
        resolved = str(kwargs.get("device", config.device))
        if resolved != config.device:
            print(
                f"[warn] requested device {config.device!r} unavailable; "
                f"running on {resolved!r}",
                file=sys.stderr,
            )
            self.config.device = resolved
        self.device = torch.device(resolved)

        torch.set_num_threads(self.config.num_threads)
        self._online_frontend = self._build_online_frontend()
        self._state = _SegmentState()
        self.reset()

    # ------------------------------------------------------------------ decode

    def decode_full(self, samples: np.ndarray) -> str:
        """Decode with full attention, the offline path.

        Mirrors ``StreamingSenseVoice._full_inference``: the build kwargs go in
        *under* the explicit arguments so a checkpoint that ships e.g.
        ``language`` in its config cannot collide with the one chosen here.

        Args:
            samples: 1-D float32 16 kHz audio.

        Returns:
            Raw text with rich tags intact; the scorer strips them.
        """
        call_kwargs: Dict[str, Any] = {
            **self.kwargs,
            "data_in": [torch.from_numpy(samples)],
            "language": self.config.language,
            "use_itn": self.config.use_itn,
            "ban_emo_unk": self.config.ban_emo_unk,
            "key": ["eval"],
            "fs": self.config.sample_rate,
        }
        with torch.inference_mode():
            results, _ = self.model.inference(**call_kwargs)
        return results[0].get("text", "") if results else ""

    def decode_chunk(self, samples: np.ndarray) -> Tuple[str, str]:
        """Decode through the streaming chunk path.

        Audio is pushed one emission cadence at a time, exactly as a live
        session would, so the partial schedule is the real one.  The stream is
        deliberately **not** ended with ``push_audio(is_last=True)``: that runs
        the backend-independent full-quality pass, which is the very
        full-attention decode this is being compared against.  Instead the
        frontend is flushed directly and the backend asked for one last partial,
        which releases its buffered frames and withheld lookahead.

        Args:
            samples: 1-D float32 16 kHz audio.

        Returns:
            ``(complete decode, last mid-stream partial)``.  The first covers
            every frame; the second is what a user would have seen at the moment
            the audio stopped, and stops short of the tail.
        """
        self.reset()
        last_partial = ""
        block = self.config.chunk_samples
        for start in range(0, len(samples), block):
            for result in self.push_audio(samples[start : start + block], is_last=False):
                last_partial = result.text

        self._extract_frames(np.zeros(0, dtype=np.float32), is_last=True)
        self._state.finished = True
        complete, _ = self._backend.emit_partial()
        return complete, last_partial

    def decode_clip(self, clip: EvalClip, samples: np.ndarray) -> ClipDecode:
        """Run both decodes for one clip under that clip's conventions.

        Args:
            clip: The clip, whose ``language`` and ``use_itn`` are applied to
                both decodes so the hypothesis matches the reference convention.
            samples: Its audio.

        Returns:
            The decodes.
        """
        self.config.language = clip.language
        self.config.use_itn = clip.use_itn
        chunk, last_partial = self.decode_chunk(samples)
        return ClipDecode(
            full=self.decode_full(samples),
            chunk=chunk,
            chunk_last_partial=last_partial,
        )


def release(recogniser: Optional[CheckpointRecogniser]) -> None:
    """Drop a loaded model and its accelerator memory.

    Both models are ~234M parameters; they are loaded one at a time so a run
    never needs to hold two.

    Args:
        recogniser: The recogniser to release; ``None`` is a no-op.
    """
    if recogniser is None:
        return
    # Read before the del: which allocator to drain is a property of where this
    # model ran, not of what the machine happens to offer.  Branching on
    # availability would have a --device cpu run create a CUDA context on a
    # shared cluster, which is precisely what declaring 'cpu' promised not to do.
    device = recogniser.config.device
    del recogniser
    gc.collect()
    if device.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


# -------------------------------------------------------------------- geometry


def build_config(args: argparse.Namespace) -> Tuple[StreamingConfig, Dict[str, Any]]:
    """Build the streaming config for the chosen training geometry.

    Args:
        args: Parsed command line.

    Returns:
        ``(config, description)`` where ``description`` records the geometry in
        the report, including the training configuration it was taken from.

    Raises:
        SystemExit: Via ``argparse`` bounds, not here; an out-of-range index is
            rejected by the parser.
    """
    index = args.geometry_index
    width = TRAINING_CHUNK_CONFIG["chunk_size"][index]
    stride = TRAINING_CHUNK_CONFIG["stride"][index]
    pad_left = TRAINING_CHUNK_CONFIG["pad_left"][index]
    look_back = TRAINING_CHUNK_CONFIG["encoder_att_look_back_factor"][index]
    pad_right = width - stride - pad_left

    config = StreamingConfig(
        backend="chunk",
        device=args.device,
        language=args.language,
        use_itn=args.use_itn,
        # Emission cadence, not the encoder window: partials come out every
        # ``chunk_size`` new frames.  Tying it to the training window width
        # keeps one partial per encoder window.
        chunk_size=width,
        chunk_pad_left=pad_left,
        chunk_stride=stride,
        chunk_pad_right=pad_right,
        chunk_encoder_look_back=look_back,
        # Off by default, which is what every run before the SER metrics existed
        # did.  It changes only which token the emotion slot may emit, never the
        # transcript, so the CER numbers are identical either way - but the SER
        # numbers are not, which is why the report records the setting.
        ban_emo_unk=getattr(args, "ban_emo_unk", False),
    )
    config.validate()

    description = {
        "geometry_index": index,
        "window_frames": width,
        "stride_frames": stride,
        "pad_left_frames": pad_left,
        "pad_right_frames": pad_right,
        "encoder_look_back": look_back,
        "lookahead_ms": config.chunk_lookahead_ms,
        "training_config": {k: list(v) for k, v in TRAINING_CHUNK_CONFIG.items()},
    }
    return config, description


# --------------------------------------------------------------------- report


@dataclass
class Report:
    """Collects non-fatal problems so they reach both stdout and the JSON.

    A clip that cannot be read or decoded is skipped rather than aborting the
    run - a per-epoch harness that dies on one bad file is worse than one that
    reports 6 clips out of 7 and says so.
    """

    warnings: List[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a message for both the JSON and stdout.

        Args:
            message: What went wrong or was skipped.
        """
        self.warnings.append(message)


def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Fine-tuned minus base, guarding missing values.

    Args:
        new: Fine-tuned value.
        old: Base value.

    Returns:
        The difference, or ``None`` if either side is missing.  Negative is an
        improvement for every metric in this report.
    """
    if new is None or old is None:
        return None
    return new - old


def summarise_ser(
    clips: Sequence[EvalClip],
    decodes: Dict[str, Dict[str, ClipDecode]],
    ban_emo_unk: bool,
) -> Dict[str, Any]:
    """Score the emotion slot against the manifest's ``emo_target``.

    The prediction is parsed from each model's **full-attention** decode.  That
    is the product-relevant one: the streaming path's authoritative final result
    for a segment is a full-attention pass, so the emotion a user ends up seeing
    is this one, not whatever a mid-stream partial guessed.

    The population is stated rather than implied.  Only clips whose
    ``emo_target`` is one of :data:`EMOTION_TOKENS` are scored; masked, missing
    and unparseable ones are excluded and counted.  A clip the model produced no
    emotion for *is* scored, as a miss - see :func:`extract_emotion_tag`.

    Args:
        clips: The val clips, in manifest order.
        decodes: ``{model_name: {clip_key: ClipDecode}}``; every clip must be
            present for every model, which the caller guarantees.
        ban_emo_unk: The setting the decodes actually ran under, recorded
            because banning ``<|EMO_UNKNOWN|>`` changes what the head is able to
            emit and therefore what these numbers mean.

    Returns:
        A block whose ``status`` is one of:

        ``"na"``
            Nothing scorable.  Either no clip carries ``emo_target`` at all, or
            every one that does is masked/unparseable.  No numbers are reported,
            because there are none to report.
        ``"degenerate"``
            Every reference is the same token, which makes accuracy meaningless
            - always answering that token scores 1.0.  The numbers *are*
            reported, but under a status that stops them being read as quality.
            The round-2 manifest, whose ``emo_target`` was a constant
            ``<|NEUTRAL|>``, lands here.
        ``"ok"``
            At least two reference classes are present.
    """
    num_missing = num_mask = num_unparseable = 0
    scored: List[EvalClip] = []
    for clip in clips:
        target = clip.emo_target
        if target is None or not target.strip():
            num_missing += 1
        elif target == EMO_MASK_TOKEN:
            num_mask += 1
        elif target in EMOTION_TOKENS:
            scored.append(clip)
        else:
            num_unparseable += 1

    block: Dict[str, Any] = {
        "status": "ok",
        "reason": None,
        "prediction_source": "full-attention decode (the streaming path's final result)",
        "ban_emo_unk": ban_emo_unk,
        "classes": list(EMOTION_TOKENS),
        "population": {
            "num_val_clips": len(clips),
            "num_scored": len(scored),
            "num_excluded_mask": num_mask,
            "num_excluded_missing": num_missing,
            "num_excluded_unparseable": num_unparseable,
        },
    }

    if not scored:
        block["status"] = "na"
        block["reason"] = (
            "manifest has no emo_target field"
            if num_mask == num_unparseable == 0
            else (
                f"no scorable emo_target: {num_mask} masked ({EMO_MASK_TOKEN}), "
                f"{num_missing} missing, {num_unparseable} not an emotion token"
            )
        )
        return block

    references = [clip.emo_target or "" for clip in scored]
    reference_classes = sorted(set(references))
    block["reference_classes"] = reference_classes
    if len(reference_classes) == 1:
        block["status"] = "degenerate"
        block["reason"] = (
            f"every scored clip carries the same reference emotion "
            f"{reference_classes[0]}, so accuracy is not a quality signal - a "
            "model that always answers that one token scores 1.0.  This is the "
            "shape of the round-2 manifest, whose emo_target was a constant "
            "<|NEUTRAL|>; the numbers below are reported but must not be read "
            "as emotion recognition quality"
        )

    per_model: Dict[str, Any] = {}
    for name, by_key in decodes.items():
        predictions = [extract_emotion_tag(by_key[clip.key].full) for clip in scored]
        per_model[name] = classification_metrics(references, predictions)
    block["per_model"] = per_model

    if "base" in per_model and "finetuned" in per_model:
        base, tuned = per_model["base"], per_model["finetuned"]
        # Higher is better here, unlike every CER delta in this report, so the
        # sign convention is spelled out in the key rather than assumed.
        block["delta"] = {
            "accuracy_finetuned_minus_base": _delta(
                tuned["accuracy"], base["accuracy"]
            ),
            "macro_f1_finetuned_minus_base": _delta(
                tuned["macro_f1"], base["macro_f1"]
            ),
            "note": "positive is an improvement (unlike the CER deltas)",
        }
    return block


def summarise_val(
    clips: Sequence[EvalClip],
    decodes: Dict[str, Dict[str, ClipDecode]],
    keep_punctuation: bool,
    ban_emo_unk: bool = False,
) -> Dict[str, Any]:
    """Score the referenced Japanese val clips for every loaded model.

    Args:
        clips: The val clips, each carrying a reference transcript.
        decodes: ``{model_name: {clip_key: ClipDecode}}``.
        keep_punctuation: Passed to the scorers.
        ban_emo_unk: The decode setting, recorded in the ``ser`` block.

    Returns:
        Per-model ``chunk`` / ``full`` / gap metrics, the base-vs-finetuned
        deltas, and the ``ser`` emotion block.  Empty dict when there are no
        clips.
    """
    if not clips:
        return {}

    per_model: Dict[str, Any] = {}
    for name, by_key in decodes.items():
        chunk_pairs = [(c.reference or "", by_key[c.key].chunk) for c in clips]
        partial_pairs = [
            (c.reference or "", by_key[c.key].chunk_last_partial) for c in clips
        ]
        full_pairs = [(c.reference or "", by_key[c.key].full) for c in clips]
        # Reference-free view of the same gap: how far the chunk decode strays
        # from this model's own full-attention decode.
        gap_pairs = [(by_key[c.key].full, by_key[c.key].chunk) for c in clips]

        chunk = corpus_metrics(chunk_pairs, keep_punctuation)
        full = corpus_metrics(full_pairs, keep_punctuation)
        per_model[name] = {
            "chunk": chunk,
            "chunk_last_partial": corpus_metrics(partial_pairs, keep_punctuation),
            "full": full,
            "chunk_minus_full_cer": _delta(chunk["cer"], full["cer"]),
            "chunk_vs_full_cer": corpus_metrics(gap_pairs, keep_punctuation)["cer"],
        }

    result: Dict[str, Any] = {"per_model": per_model}
    if "base" in per_model and "finetuned" in per_model:
        base, tuned = per_model["base"], per_model["finetuned"]
        result["delta"] = {
            "chunk_cer": _delta(tuned["chunk"]["cer"], base["chunk"]["cer"]),
            "full_cer": _delta(tuned["full"]["cer"], base["full"]["cer"]),
            "chunk_minus_full_cer": _delta(
                tuned["chunk_minus_full_cer"], base["chunk_minus_full_cer"]
            ),
            "chunk_vs_full_cer": _delta(
                tuned["chunk_vs_full_cer"], base["chunk_vs_full_cer"]
            ),
        }
    # A sibling of the CER metrics, never a component of them: nothing above
    # reads it and the selection metric is untouched.
    result["ser"] = summarise_ser(clips, decodes, ban_emo_unk)
    return result


def summarise_unreferenced(
    clip: EvalClip,
    decodes: Dict[str, Dict[str, ClipDecode]],
    keep_punctuation: bool,
) -> Dict[str, Any]:
    """Score one clip that has no ground truth.

    The base model's full-attention decode stands in for the reference, so
    ``drift_cer`` measures how far the finetune moved this clip's offline
    transcript - not how correct either transcript is.

    Args:
        clip: The clip.
        decodes: ``{model_name: {clip_key: ClipDecode}}``.
        keep_punctuation: Passed to the scorers.

    Returns:
        Per-model chunk-vs-full gaps, the drift, and the decodes themselves.
    """
    base = decodes["base"][clip.key]
    entry: Dict[str, Any] = {
        "key": clip.key,
        "path": str(clip.path),
        "language": clip.language,
        "reference_source": "base model full-attention decode (no ground truth)",
        "per_model": {},
    }
    for name, by_key in decodes.items():
        decode = by_key[clip.key]
        entry["per_model"][name] = {
            "full": decode.full,
            "chunk": decode.chunk,
            "chunk_last_partial": decode.chunk_last_partial,
            "chunk_vs_full_cer": pair_cer(decode.full, decode.chunk, keep_punctuation),
        }

    if "finetuned" in decodes:
        tuned = decodes["finetuned"][clip.key]
        entry["drift_cer_vs_base_full"] = pair_cer(base.full, tuned.full, keep_punctuation)
        entry["gap_delta"] = _delta(
            entry["per_model"]["finetuned"]["chunk_vs_full_cer"],
            entry["per_model"]["base"]["chunk_vs_full_cer"],
        )
    else:
        entry["drift_cer_vs_base_full"] = 0.0
        entry["gap_delta"] = None
    return entry


def build_examples(
    clips: Sequence[EvalClip],
    decodes: Dict[str, Dict[str, ClipDecode]],
    count: int,
) -> List[Dict[str, Any]]:
    """Pick a few clips to show side by side for eyeballing.

    Args:
        clips: Candidate clips, in manifest order.
        decodes: ``{model_name: {clip_key: ClipDecode}}``.
        count: How many to include.

    Returns:
        One row per clip with the reference and both models' decodes.
    """
    examples: List[Dict[str, Any]] = []
    for clip in list(clips)[:count]:
        row: Dict[str, Any] = {
            "key": clip.key,
            "scope": clip.scope,
            "reference": clip.reference,
        }
        for name, by_key in decodes.items():
            decode = by_key[clip.key]
            row[f"{name}_full"] = decode.full
            row[f"{name}_chunk"] = decode.chunk
            row[f"{name}_chunk_last_partial"] = decode.chunk_last_partial
        examples.append(row)
    return examples


# --------------------------------------------------------------------- stdout


def _fmt(value: Optional[float], width: int = 9) -> str:
    """Format a metric for the fixed-width summary.

    Args:
        value: The metric, or ``None`` when it was not computed.
        width: Column width.

    Returns:
        A right-aligned string; ``"-"`` for ``None``.
    """
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.4f}"


def _signed(value: Optional[float], width: int = 9) -> str:
    """Format a delta with an explicit sign.

    Args:
        value: The delta, or ``None``.
        width: Column width.

    Returns:
        A right-aligned signed string; ``"-"`` for ``None``.
    """
    return f"{'-':>{width}}" if value is None else f"{value:>+{width}.4f}"


def _ser_lines(ser: Optional[Dict[str, Any]]) -> List[str]:
    """Render the SER block for the terminal summary.

    The headline carries ``num_scored`` and the exclusion counts, because an
    accuracy whose population is not on the same screen is the kind of number
    this repo has already published wrongly once.  A non-``ok`` status is shouted
    rather than mentioned: a degenerate reference set produces a *high* accuracy,
    so nothing about the numbers themselves warns the reader.

    Args:
        ser: The block from :func:`summarise_ser`, or ``None`` when the report
            predates it.

    Returns:
        Lines to print, empty when there is no block.
    """
    if not ser:
        return []
    population = ser.get("population", {})
    lines = [
        "SER (full-attn)       "
        f"n={population.get('num_scored', 0)} scored of "
        f"{population.get('num_val_clips', 0)} val clips  "
        f"(excluded: mask {population.get('num_excluded_mask', 0)}, "
        f"missing {population.get('num_excluded_missing', 0)}, "
        f"unparseable {population.get('num_excluded_unparseable', 0)}; "
        f"ban_emo_unk={ser.get('ban_emo_unk')})"
    ]
    for name, metrics in (ser.get("per_model") or {}).items():
        dominant = metrics.get("dominant_prediction") or {}
        share = dominant.get("share")
        lines.append(
            f"  {name:<20}accuracy={_fmt(metrics['accuracy'], 0)}  "
            f"macro-F1={_fmt(metrics['macro_f1'], 0)}  "
            f"no-emotion-emitted={metrics['num_pred_none']}  "
            # A collapsed head is diagnosed here, not by the accuracy: "answers
            # one class for everything" is a fact about the predictions.
            f"most-predicted={dominant.get('label')}"
            f"{'' if share is None else f' ({share:.0%})'}"
        )
    if ser.get("status") != "ok":
        lines.append(f"  *** SER {str(ser.get('status')).upper()} ***")
        lines.append(
            textwrap.fill(
                str(ser.get("reason")),
                width=68,
                initial_indent="      ",
                subsequent_indent="      ",
            )
        )
    return lines


def _clip_line(clip: Dict[str, Any]) -> str:
    """One summary line for a clip that has no ground truth.

    Args:
        clip: An entry produced by :func:`summarise_unreferenced`.

    Returns:
        A line giving the drift from the base model's offline transcript and the
        chunk-vs-full gap of each model that was run.
    """
    gaps = " ".join(
        f"{name}={_fmt(values['chunk_vs_full_cer'], 0)}"
        for name, values in clip["per_model"].items()
    )
    return (
        f"{clip['key']:<22}drift vs base full = "
        f"{_fmt(clip['drift_cer_vs_base_full'], 0)}   chunk-vs-full gap: {gaps}"
    )


def print_summary(payload: Dict[str, Any], warnings_: Sequence[str]) -> None:
    """Print the compact per-epoch summary.

    Leads with the Japanese metrics, because those alone decide which checkpoint
    wins, and ends with a single named selection number so successive epochs can
    be diffed by eye or by ``grep``.

    Args:
        payload: The assembled report.
        warnings_: Messages collected during the run.
    """
    geometry = payload["geometry"]
    models = payload["models"]
    line = "=" * 68

    print(line)
    print("chunk-gap eval - Japanese-specialised finetune")
    print(line)
    # The bare line is kept verbatim off CUDA: MPS and CPU have no TF32 mode,
    # so there is nothing to disclose and the summary stays diffable against
    # every report produced before precision was recorded.  On CUDA the
    # arithmetic is the first thing a reader must check, so it rides along.
    precision = payload.get("precision") or {}
    device_line = f"device      : {payload['device']}"
    if str(payload["device"]).startswith("cuda"):
        device_line += (
            f"   precision: {precision.get('mode')}"
            f" (cudnn.allow_tf32={precision.get('cudnn_allow_tf32')}"
            f" matmul.allow_tf32={precision.get('cuda_matmul_allow_tf32')}"
            f" float32_matmul={precision.get('float32_matmul_precision')}"
            f" deterministic={precision.get('cudnn_deterministic')})"
        )
    print(device_line)
    if not precision.get("comparable_to_fp32_baseline", True):
        print("*** TF32 ON - NOT comparable to the fp32 reference numbers ***")
    print(
        "geometry    : idx{geometry_index} window={window_frames} "
        "stride={stride_frames} pad_left={pad_left_frames} "
        "pad_right={pad_right_frames} look_back={encoder_look_back} "
        "({lookahead_ms:.0f} ms lookahead)".format(**geometry)
    )
    print(f"base        : {models['base']['dir']}")
    print(f"finetuned   : {models['finetuned']['checkpoint'] or '(none - base only)'}")

    val = payload["japanese"]["val"]
    print()
    print("-- JAPANESE (selection axis) " + "-" * 39)
    metrics = val.get("metrics", {}).get("per_model")
    if not metrics:
        print(f"val set     : {val['jsonl']} - UNAVAILABLE, no selection signal")
    else:
        print(f"val set     : {val['jsonl']}  ({val['num_clips']} clips)")
        delta = val["metrics"].get("delta", {})
        base = metrics["base"]
        # Left as ``None`` when no checkpoint was given, so the column reads "-"
        # rather than repeating the base numbers under a "finetuned" heading.
        tuned = metrics.get("finetuned")
        print(f"{'':22}{'base':>9}{'finetuned':>11}{'delta':>10}")
        rows = (
            ("chunk CER", lambda m: m["chunk"]["cer"], "chunk_cer"),
            ("full-attn CER", lambda m: m["full"]["cer"], "full_cer"),
            (
                "chunk-full gap",
                lambda m: m["chunk_minus_full_cer"],
                "chunk_minus_full_cer",
            ),
            (
                "chunk vs full CER",
                lambda m: m["chunk_vs_full_cer"],
                "chunk_vs_full_cer",
            ),
            ("chunk WER", lambda m: m["chunk"]["wer"], None),
        )
        for label, pick, delta_key in rows:
            delta_value = delta.get(delta_key) if delta_key else None
            print(
                f"{label:<22}{_fmt(pick(base))}"
                f"{_fmt(pick(tuned) if tuned else None, 11)}"
                f"{_signed(delta_value, 10)}"
            )
        print("  (full-attn CER is the forgetting check: Japanese only)")
        ser_lines = _ser_lines(val["metrics"].get("ser"))
        if ser_lines:
            print()
            for ser_line in ser_lines:
                print(ser_line)

    for clip in payload["japanese"]["clips"]:
        print(_clip_line(clip) + "   (informational)")

    reference_only = payload.get("reference_only", {}).get("clips", [])
    if reference_only:
        print()
        print("-- REFERENCE ONLY (not a selection criterion) " + "-" * 22)
        print(f"   {REFERENCE_ONLY_NOTE}")
        for clip in reference_only:
            print(_clip_line(clip))

    selection = payload["selection"]
    print()
    print(line)
    if selection["value"] is None:
        print(f"BEST-CHECKPOINT SIGNAL: unavailable - {selection['note']}")
    else:
        print(
            f"BEST-CHECKPOINT SIGNAL: {selection['metric']} = "
            f"{selection['value']:.4f}  (lower is better)"
        )
    print(line)

    for message in warnings_:
        print(f"[warn] {message}", file=sys.stderr)


# ----------------------------------------------------------------------- main


def _backend_flag(*path: str) -> Optional[bool]:
    """Read a ``torch.backends`` boolean without assuming it exists.

    The TF32 switches have moved around between torch releases and some are
    absent from CPU-only builds, so every read is best-effort: a missing or
    unreadable flag is reported as unknown rather than crashing a run or, worse,
    being reported as False when nobody checked.

    Args:
        *path: Attribute names to walk from ``torch.backends``.

    Returns:
        The flag as a bool, or ``None`` if it does not exist on this build.
    """
    node: Any = torch.backends
    for name in path:
        try:
            node = getattr(node, name)
        except (AttributeError, RuntimeError):
            return None
    return bool(node) if isinstance(node, bool) else None


def configure_precision(device: str, allow_tf32: bool) -> None:
    """Pin CUDA arithmetic to fp32 so runs stay comparable to the baseline.

    Only touches global torch state on the CUDA path; MPS and CPU have no TF32
    mode, and leaving their state alone is what keeps the baseline machine
    byte-identical to before this function existed.

    On CUDA, torch would otherwise default ``cudnn.allow_tf32`` to True and run
    the subsampling convolutions in TF32 - roughly 10 bits of mantissa against
    fp32's 23 - which is a different computation from the one that produced the
    reference CER, not merely a faster one.  ``cudnn.deterministic`` is pinned
    for the same reason: a run that cannot be reproduced cannot be compared.

    Args:
        device: The resolved device string, e.g. ``"cuda"`` or ``"cuda:0"``.
        allow_tf32: Opt into TF32 anyway.  Faster, and explicitly *not*
            comparable to the fp32 baseline; the report says so when set.
    """
    if not device.startswith("cuda"):
        return

    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        if hasattr(cudnn, "allow_tf32"):
            cudnn.allow_tf32 = allow_tf32
        # Pinned regardless of the TF32 choice: this is about a run being
        # repeatable, which the escape hatch is not meant to trade away.
        if hasattr(cudnn, "deterministic"):
            cudnn.deterministic = True

    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = allow_tf32

    # The modern spelling of the same switch.  Set as well as - not instead of -
    # the flags above, so the arithmetic is pinned even on a build where only
    # one of the two spellings is present.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def precision_report(device: str, allow_tf32: bool) -> Dict[str, Any]:
    """Describe the arithmetic a run actually used, for the JSON report.

    Read after the models are loaded so it reflects the device funasr resolved,
    not the one requested.  The point is that someone holding two reports can
    tell whether they are comparable without knowing which machine produced
    either one.

    Args:
        device: The resolved device string.
        allow_tf32: Whether ``--allow-tf32`` was passed.

    Returns:
        A metadata block; the flags read ``None`` where this torch build does
        not expose them.
    """
    on_cuda = device.startswith("cuda")
    matmul_precision: Optional[str] = None
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            matmul_precision = str(torch.get_float32_matmul_precision())
        except RuntimeError:
            matmul_precision = None

    if not on_cuda:
        note = (
            f"{device} has no TF32 path; all matmuls and convolutions ran in "
            "fp32, as they did for the reference numbers"
        )
    elif allow_tf32:
        note = (
            "TF32 ENABLED via --allow-tf32: convolutions and matmuls ran at "
            "reduced precision.  These numbers are NOT comparable to the fp32 "
            "baseline"
        )
    else:
        note = (
            "TF32 disabled explicitly; CUDA ran in fp32 to stay comparable to "
            "the fp32 baseline"
        )

    return {
        "mode": "tf32" if (on_cuda and allow_tf32) else "fp32",
        "comparable_to_fp32_baseline": not (on_cuda and allow_tf32),
        "allow_tf32_requested": allow_tf32,
        "cudnn_allow_tf32": _backend_flag("cudnn", "allow_tf32"),
        "cuda_matmul_allow_tf32": _backend_flag("cuda", "matmul", "allow_tf32"),
        "cudnn_deterministic": _backend_flag("cudnn", "deterministic"),
        "float32_matmul_precision": matmul_precision,
        "note": note,
    }


def default_device() -> str:
    """Pick the device to decode on when ``--device`` is not given.

    Deliberately never ``"cuda"``.  The published numbers were measured on
    Apple MPS in fp32, and CUDA changes the arithmetic under you: torch
    defaults ``cudnn.allow_tf32`` to True, so SenseVoice's subsampling
    convolutions would run in TF32 where the Mac ran fp32.  If merely landing
    on a CUDA box flipped the default, the exact command line that produced
    the baseline would silently produce numbers that are not comparable to it.
    So CUDA is opt-in via ``--device cuda``, which is also the point where
    :func:`configure_precision` can pin the arithmetic back to fp32.

    ``"mps"`` is kept as the darwin default - unchanged from before, so the
    baseline machine is unaffected - and everything else gets ``"cpu"``, the
    only device those machines can actually offer now that CUDA is explicit.

    Returns:
        ``"mps"`` on macOS, otherwise ``"cpu"``.
    """
    return "mps" if sys.platform == "darwin" else "cpu"


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
        "--checkpoint",
        type=Path,
        default=None,
        help="finetuned model.pt; omit to evaluate the base model alone",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"base model directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=DEFAULT_VAL_JSONL,
        help=f"held-out Japanese manifest (default: {DEFAULT_VAL_JSONL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N val clips, for fast per-epoch runs",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        choices=("cuda", "mps", "cpu"),
        help=(
            "torch device (default: mps on macOS, else cpu).  'cuda' is never "
            "implicit - ask for it - and when asked for it runs in fp32 unless "
            "--allow-tf32 is given.  Bare 'cuda' means device 0; select another "
            "with CUDA_VISIBLE_DEVICES, as this is a single-process evaluation"
        ),
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help=(
            "let CUDA use TF32 for convolutions and matmuls.  Faster, and NOT "
            "numerically comparable to the fp32 reference numbers; the report "
            "records the run as such.  No effect off CUDA"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument(
        "--geometry-index",
        type=int,
        default=DEFAULT_GEOMETRY_INDEX,
        choices=range(len(TRAINING_CHUNK_CONFIG["chunk_size"])),
        help=(
            "which finetune_chunk.sh chunk geometry to decode with "
            f"(default: {DEFAULT_GEOMETRY_INDEX}, the 720 ms window)"
        ),
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="SenseVoice language tag for clips whose manifest omits one",
    )
    parser.add_argument(
        "--use-itn",
        action="store_true",
        help=(
            "request inverse text normalisation; off by default because the "
            "manifest targets are plain text and ITN punctuation would be "
            "scored as insertions"
        ),
    )
    parser.add_argument(
        "--ban-emo-unk",
        action="store_true",
        help=(
            "forbid the emotion slot from emitting <|EMO_UNKNOWN|>, forcing it "
            "to commit to one of the seven emotions.  Affects the SER metrics "
            "only - the transcript, and therefore every CER number, is "
            "unchanged - and the report records which setting it ran under"
        ),
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="score punctuation instead of stripping it before CER/WER",
    )
    parser.add_argument(
        "--ja-clip",
        type=Path,
        default=None,
        help="Japanese sample clip; defaults to the bundled ja.mp3 if present",
    )
    parser.add_argument(
        "--zh-clip",
        type=Path,
        default=DEFAULT_ZH_CLIP,
        help="Chinese reference-only clip (informational, never a selection criterion)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="side-by-side example decodes to include in the report (default: 5)",
    )
    return parser.parse_args(argv)


def collect_clips(
    args: argparse.Namespace, report: Report
) -> Tuple[List[EvalClip], List[EvalClip], List[EvalClip]]:
    """Assemble every clip to decode.

    Args:
        args: Parsed command line.
        report: Collects warnings about anything skipped.

    Returns:
        ``(val_clips, ja_clips, reference_clips)``.
    """
    val_clips, notes = load_val_clips(
        args.val_jsonl, args.limit, args.language, args.use_itn
    )
    for note in notes:
        report.warn(note)

    ja_clips: List[EvalClip] = []
    ja_path, ja_notes = resolve_ja_clip(args.ja_clip)
    for note in ja_notes:
        report.warn(note)
    if ja_path is not None:
        ja_clips.append(
            EvalClip(
                key="ja.mp3",
                path=ja_path,
                reference=None,
                language="ja",
                use_itn=args.use_itn,
            )
        )

    reference_clips: List[EvalClip] = []
    if args.zh_clip is not None and args.zh_clip.exists():
        reference_clips.append(
            EvalClip(
                key="zh_sample.wav",
                path=args.zh_clip,
                reference=None,
                language="auto",
                use_itn=args.use_itn,
                scope="reference",
            )
        )
    elif args.zh_clip is not None:
        report.warn(f"Chinese reference clip not found: {args.zh_clip}, skipped")

    return val_clips, ja_clips, reference_clips


def decode_all(
    model_dir: Path,
    checkpoint: Optional[Path],
    config: StreamingConfig,
    clips: Sequence[EvalClip],
    audio: Dict[str, np.ndarray],
    report: Report,
) -> Dict[str, ClipDecode]:
    """Load one model, decode every clip with it, then release it.

    Args:
        model_dir: Base model directory.
        checkpoint: Finetuned weights, or ``None`` for the base model.
        config: Streaming configuration; mutated per clip for language/ITN.
        clips: Clips to decode.
        audio: Pre-decoded waveforms keyed by clip key.
        report: Collects per-clip failures.

    Returns:
        ``{clip_key: ClipDecode}``, omitting clips whose decode raised.
    """
    recogniser = CheckpointRecogniser(model_dir, config, checkpoint)
    decodes: Dict[str, ClipDecode] = {}
    try:
        for index, clip in enumerate(clips, start=1):
            label = "base" if checkpoint is None else "finetuned"
            print(
                f"  [{label}] {index}/{len(clips)} {clip.key}",
                end="\r",
                file=sys.stderr,
            )
            try:
                decodes[clip.key] = recogniser.decode_clip(clip, audio[clip.key])
            except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
                report.warn(f"{label}: decode failed for {clip.key}: {exc}")
        print(" " * 72, end="\r", file=sys.stderr)
    finally:
        release(recogniser)
    return decodes


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evaluation.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` when nothing could be
        evaluated at all.
    """
    args = parse_args(argv)
    report = Report()

    if not args.base.exists():
        print(f"base model directory not found: {args.base}", file=sys.stderr)
        return 1
    if args.checkpoint is not None and not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    config, geometry = build_config(args)
    # Before any model is built, since these are global torch switches that the
    # first forward pass would otherwise bake in at their permissive defaults.
    configure_precision(config.device, args.allow_tf32)
    val_clips, ja_clips, reference_clips = collect_clips(args, report)
    all_clips = [*val_clips, *ja_clips, *reference_clips]
    if not all_clips:
        print("nothing to evaluate: no val manifest and no sample clips", file=sys.stderr)
        for message in report.warnings:
            print(f"[warn] {message}", file=sys.stderr)
        return 1

    # Decoded once and shared by both models, so the two see bit-identical
    # input and any difference is the weights.
    audio: Dict[str, np.ndarray] = {}
    usable: List[EvalClip] = []
    for clip in all_clips:
        try:
            audio[clip.key] = load_audio(clip.path, config.sample_rate)
        except RuntimeError as exc:
            report.warn(str(exc))
            continue
        usable.append(clip)
    val_clips = [c for c in val_clips if c.key in audio]
    ja_clips = [c for c in ja_clips if c.key in audio]
    reference_clips = [c for c in reference_clips if c.key in audio]
    if not usable:
        print("nothing to evaluate: no clip could be decoded", file=sys.stderr)
        for message in report.warnings:
            print(f"[warn] {message}", file=sys.stderr)
        return 1

    decodes: Dict[str, Dict[str, ClipDecode]] = {
        "base": decode_all(args.base, None, config, usable, audio, report)
    }
    if args.checkpoint is not None:
        decodes["finetuned"] = decode_all(
            args.base, args.checkpoint, config, usable, audio, report
        )

    # A clip only counts if every loaded model decoded it, so the columns of the
    # comparison are always over the same set.
    complete = {c.key for c in usable if all(c.key in d for d in decodes.values())}
    val_clips = [c for c in val_clips if c.key in complete]
    ja_clips = [c for c in ja_clips if c.key in complete]
    reference_clips = [c for c in reference_clips if c.key in complete]

    val_metrics = summarise_val(
        val_clips, decodes, args.keep_punctuation, config.ban_emo_unk
    )
    selection_value = (
        val_metrics["per_model"]
        .get("finetuned", val_metrics["per_model"].get("base", {}))
        .get("chunk", {})
        .get("cer")
        if val_metrics
        else None
    )

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": config.device,
        # Read here rather than at configure time so it reports the device
        # funasr resolved, and so a report is self-describing: two CER numbers
        # are only comparable if this block matches.
        "precision": precision_report(config.device, args.allow_tf32),
        "geometry": geometry,
        "scoring": {
            "cer": "corpus-level Levenshtein over NFKC-normalised characters",
            "punctuation": "kept" if args.keep_punctuation else "stripped",
            "whitespace": "stripped",
            "rich_tags": "stripped",
            "wer_note": (
                "whitespace tokens; near-meaningless for unsegmented Japanese "
                "and Chinese - read CER"
            ),
            "ser": (
                "emotion token parsed from the leading rich-tag block of the "
                "full-attention decode and compared with the manifest's "
                "emo_target; macro-F1 over the reference-present classes, "
                "unpredicted classes entering the average with F1 0.0"
            ),
        },
        "models": {
            "base": {"dir": str(args.base), "checkpoint": None},
            "finetuned": {
                "dir": str(args.base),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            },
        },
        "japanese": {
            "note": "the selection axis: checkpoint choice keys off these numbers only",
            "val": {
                "jsonl": str(args.val_jsonl),
                "num_clips": len(val_clips),
                "metrics": val_metrics,
            },
            "clips": [
                summarise_unreferenced(c, decodes, args.keep_punctuation)
                for c in ja_clips
            ],
        },
        "reference_only": {
            "note": REFERENCE_ONLY_NOTE,
            "clips": [
                summarise_unreferenced(c, decodes, args.keep_punctuation)
                for c in reference_clips
            ],
        },
        "selection": {
            "metric": "ja_val_chunk_cer",
            "value": selection_value,
            "model": "finetuned" if args.checkpoint else "base",
            "note": (
                "corpus CER of the complete chunk-mode decode on the held-out "
                "Japanese val set; lower is better"
                if selection_value is not None
                else "no Japanese val set was available"
            ),
        },
        "examples": build_examples(
            [*val_clips, *ja_clips, *reference_clips], decodes, args.num_examples
        ),
        "warnings": report.warnings,
    }

    print_summary(payload, report.warnings)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
