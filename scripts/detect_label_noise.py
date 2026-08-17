#!/usr/bin/env python3
"""Flag manifest entries whose *ground truth* disagrees with a teacher ASR model.

The VisualNovel corpus takes its transcripts from game script text, which is
imperfect: ``scripts/prepare_vn_data.py`` already drops thousands of entries for
missing audio and for duplicate keys carrying conflicting text, and the upstream
dataset card warns of "some few wrong transcriptions due to extraction error".
At 1,000 hours those survivors are worth finding.

This script decodes clips with a strong in-domain teacher
(``litagin/anime-whisper`` via CTranslate2) and ranks clips by how badly the
teacher disagrees with the stored ``target``.  A high score is a candidate
**mislabelled transcript**, not a candidate bad recording -- the audio is
assumed good and the text is what is on trial.

    .venv/bin/python scripts/detect_label_noise.py \\
        --manifest data/vn/train.jsonl \\
        --include-titles configs/uncontaminated_titles.txt \\
        --device cuda --out outputs/label_noise.jsonl

**It never deletes or edits anything.**  The manifest, the audio and the corpus
are opened read-only; the only paths written are ``--out`` and its summary
sidecar.  A human reads the ranked report and decides.

Why the comparison is normalised
--------------------------------

A raw CER between teacher output and our transcripts measures *punctuation
convention*, not correctness.  Our corpus and this teacher disagree
systematically on orthography, measured over ``data/vn/*.jsonl``:

======  ==============  ===========================
char    our corpus      teacher emits
======  ==============  ===========================
``！``   6,534           0 (ASCII ``!``)
``？``   7,980           0 (ASCII ``?``)
``。``   11,158          0
``……``   very common     0 (collapses to one ``…``)
``―``    3,502           not in its charset
``～``    914             not in its charset
``♪``    304             not in its charset
======  ==============  ===========================

Applying the teacher's measured emission rates to our corpus puts a **6.18
CER-point floor** on the comparison from orthography alone -- 39% of the
student's entire 15.84% CER budget.  Worse, ``……`` / ``！`` / ``♪`` density is
highest on emotive and NSFW lines, so an unnormalised ranking would
preferentially flag exactly those, producing a content-correlated bias in what
gets called "noisy".

So both sides are compared through a normalised projection, and *both raw
strings are kept in every report row* so the reviewer reads what was actually
said rather than what the scorer saw.

The projection is :func:`normalize_for_comparison`, which is
``scripts/eval_chunk_gap.py``'s frozen :func:`~eval_chunk_gap.normalize_chars`
(rich-tag strip, NFKC, whitespace strip, punctuation strip -- which already
covers every row of the table above except the last) plus one addition,
:data:`_DECORATIVE_SYMBOLS`.  Reusing that function rather than writing a second
normaliser is deliberate: two normalisers for one corpus drift apart, and
``eval_chunk_gap.py``'s is the one the reported CER numbers are defined by.  It
is imported, never modified -- its scoring path is a frozen baseline.

Title filtering is load-bearing
-------------------------------

The teacher was trained on roughly a third of our archives.  On those titles it
has memorised the audio/transcript pairs, so it reproduces our errors instead of
detecting them and its agreement means nothing.  ``--include-titles`` takes the
caller's non-contaminated subset; without it every clip is processed and the
scores over contaminated titles are not interpretable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_chunk_gap.py"

__all__ = [
    "ManifestClip",
    "ReportRow",
    "TeacherConfig",
    "load_manifest",
    "load_title_prefixes",
    "normalize_for_comparison",
    "normalized_cer",
    "select_clips",
    "sort_rows",
    "summarise",
    "teacher_transcribe_options",
    "main",
]


def _load_eval_module() -> ModuleType:
    """Import ``scripts/eval_chunk_gap.py`` for its frozen scoring helpers.

    ``scripts/`` is not a package, so the module is loaded by path.  Only
    :func:`normalize_chars` and :func:`levenshtein` are used from it, and
    neither is touched -- see the module docstring for why the normaliser is
    shared rather than reimplemented.

    Returns:
        The loaded ``eval_chunk_gap`` module.

    Raises:
        ImportError: If the file is missing or its own imports fail.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("eval_chunk_gap", _EVAL_SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the scoring helpers from {_EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("eval_chunk_gap", module)
    spec.loader.exec_module(module)
    return module


_EVAL = _load_eval_module()

#: Stripped *in addition to* ``eval_chunk_gap._PUNCTUATION``.
#:
#: Decorative and musical annotation marks.  They are script-editor notation for
#: how a line is delivered (``♪`` marks singing or a sing-song tone), carry no
#: phonetic content, and are absent from the teacher's output charset -- so they
#: are pure convention noise in a comparison.  ``♪`` alone occurs 304 times in
#: ``data/vn/*.jsonl`` and is concentrated on emotive lines, which is exactly the
#: content-correlated bias this projection exists to remove.
#:
#: Note what is deliberately *not* here: ``ー`` (U+30FC, 5,994 occurrences) is the
#: katakana prolonged sound mark and is phonemic -- stripping it would erase real
#: transcription differences.  ``〇`` (51) is used to censor names and is content.
_DECORATIVE_SYMBOLS = frozenset("♪♫♬♩♭♯★☆♡♥❤※")

#: Report rows are bucketed against these normalised-CER thresholds in the
#: summary.  Chosen to bracket the interesting range rather than to be a
#: decision rule: below ~0.2 the disagreement is usually the teacher's, above
#: ~0.5 the two strings are rarely the same utterance at all.  The script does
#: not act on these numbers; a reviewer does.
CER_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

# ------------------------------------------------------------------ teacher IDs

#: ``litagin/anime-whisper`` converted to CTranslate2 fp16.  In-domain for this
#: corpus (it was trained on visual-novel and anime speech), which is the whole
#: reason it can be trusted to disagree with our transcripts.
DEFAULT_MODEL_ID = "quantumcookie/anime-whisper-ct2-fp16"
DEFAULT_BACKEND = "faster-whisper"
DEFAULT_LANGUAGE = "ja"

#: The model card documents repetition artifacts on emotive and NSFW content --
#: the model loops a syllable until it hits the token limit.  Blocking repeated
#: 5-grams suppresses the loop.  A looped hypothesis would otherwise score a
#: huge CER and flood the top of a report that is meant to rank *our* errors.
DEFAULT_NO_REPEAT_NGRAM_SIZE = 5


@dataclass(frozen=True)
class TeacherConfig:
    """How to instantiate and drive the teacher ASR model.

    Attributes:
        model_id: HuggingFace id or local directory of the CT2 model.
        backend: Key into :data:`BACKENDS`.
        device: ``"cuda"``, ``"cpu"`` or ``"auto"``, passed to the backend.
        device_index: GPU ordinal, or several for multi-GPU data parallelism.
        compute_type: CTranslate2 quantisation, e.g. ``float16``/``int8``.
        beam_size: Decoder beam width.
        batch_size: >1 selects faster-whisper's batched pipeline.
        num_workers: Backend loader threads.
        language: Forced decode language; never autodetected, the corpus is
            wholly Japanese and detection only adds a failure mode.
        no_repeat_ngram_size: See :data:`DEFAULT_NO_REPEAT_NGRAM_SIZE`.
    """

    model_id: str = DEFAULT_MODEL_ID
    backend: str = DEFAULT_BACKEND
    device: str = "auto"
    device_index: Sequence[int] = (0,)
    compute_type: str = "float16"
    beam_size: int = 5
    batch_size: int = 1
    num_workers: int = 1
    language: str = DEFAULT_LANGUAGE
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE


def teacher_transcribe_options(config: TeacherConfig) -> Dict[str, Any]:
    """Build the per-clip decode kwargs.

    Pure and free of any backend import so the non-negotiable settings can be
    asserted in tests on a machine without faster-whisper.  Each of the four
    fixed values is fixed for a measured reason:

    ``word_timestamps=False``
        CTranslate2 segfaults inside ``align()`` on this model's 2-layer
        decoder (Whisper-WebUI issue #609).  Not a slow path -- a crash.
    ``initial_prompt=None`` and ``condition_on_previous_text=False``
        The model card documents that prompt conditioning induces
        hallucination.  Both routes into the decoder's prompt are closed, so a
        clip's hypothesis depends on that clip's audio and nothing else --
        also what makes the report reproducible regardless of clip ordering.
    ``no_repeat_ngram_size``
        See :data:`DEFAULT_NO_REPEAT_NGRAM_SIZE`.
    ``language``
        Forced, never autodetected.

    Args:
        config: The teacher configuration.

    Returns:
        Keyword arguments for the backend's ``transcribe``.
    """
    return {
        "language": config.language,
        "beam_size": config.beam_size,
        "word_timestamps": False,
        "initial_prompt": None,
        "condition_on_previous_text": False,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
    }


class FasterWhisperTeacher:
    """faster-whisper/CTranslate2 teacher, loaded on first use.

    The import and the model load are deferred to :meth:`transcribe` so that
    ``--help``, the manifest reader and the whole normalisation path stay
    importable and testable where faster-whisper is not installed.
    """

    def __init__(self, config: TeacherConfig) -> None:
        self._config = config
        self._model: Any = None
        self._pipeline: Any = None
        self._options = teacher_transcribe_options(config)

    def _ensure_loaded(self) -> None:
        """Import faster-whisper and construct the model, once."""
        if self._model is not None:
            return
        try:
            import faster_whisper  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the faster-whisper backend needs the 'faster-whisper' package; "
                "install it or pass --backend with another teacher"
            ) from exc

        config = self._config
        self._model = faster_whisper.WhisperModel(
            config.model_id,
            device=config.device,
            device_index=list(config.device_index),
            compute_type=config.compute_type,
            num_workers=config.num_workers,
        )
        if config.batch_size > 1:
            # Present from faster-whisper 1.1; without it batching silently
            # degrades to sequential rather than failing the run.
            pipeline_cls = getattr(faster_whisper, "BatchedInferencePipeline", None)
            if pipeline_cls is not None:
                self._pipeline = pipeline_cls(model=self._model)

    def transcribe(self, path: Path) -> str:
        """Decode one clip.

        Args:
            path: Audio file to decode.

        Returns:
            The concatenated segment text, whitespace-stripped.  Empty when the
            teacher emits nothing, which is itself a signal worth ranking.
        """
        self._ensure_loaded()
        options = dict(self._options)
        if self._pipeline is not None:
            # The batched pipeline drives its own windowing and rejects the
            # sequential-only conditioning flag.
            options.pop("condition_on_previous_text", None)
            segments, _info = self._pipeline.transcribe(
                str(path), batch_size=self._config.batch_size, **options
            )
        else:
            segments, _info = self._model.transcribe(str(path), **options)
        return "".join(segment.text for segment in segments).strip()


#: Backend name -> factory.  Swapping in another teacher means adding an entry
#: whose object exposes ``transcribe(Path) -> str``; nothing else in the script
#: knows what a teacher is.
BACKENDS: Dict[str, Callable[[TeacherConfig], Any]] = {
    DEFAULT_BACKEND: FasterWhisperTeacher,
}


def build_teacher(config: TeacherConfig) -> Any:
    """Construct the teacher named by ``config.backend``.

    Args:
        config: The teacher configuration.

    Returns:
        An object with ``transcribe(Path) -> str``.

    Raises:
        ValueError: If the backend name is unknown.
    """
    try:
        factory = BACKENDS[config.backend]
    except KeyError:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown backend {config.backend!r}; known: {known}") from None
    return factory(config)


# ----------------------------------------------------------------- normalising


def normalize_for_comparison(text: str) -> str:
    """Project text onto the form the CER is computed over.

    Delegates to ``eval_chunk_gap.normalize_chars`` (default
    ``keep_punctuation=False``), which strips rich tags, applies NFKC, drops all
    whitespace and drops punctuation.  That already collapses every orthographic
    difference in the module docstring's table:

    * ``！``/``？`` fold to ASCII under NFKC and are then stripped as punctuation;
    * ``。`` ``、`` ``―`` ``～`` ``〜`` are stripped as punctuation;
    * ``…`` decomposes to ``...`` under NFKC and the dots are stripped, so a run
      of any length and a single ``…`` both reduce to nothing -- a stronger
      guarantee than collapsing runs to one token, and it removes the
      ``……``-vs-``…`` axis entirely;
    * spacing is gone.

    The only thing layered on top is :data:`_DECORATIVE_SYMBOLS`, which
    ``normalize_chars`` does not cover.  Applied after, so the shared function
    stays untouched.

    Args:
        text: Raw ground truth or raw teacher output.

    Returns:
        The normalised character string, possibly empty.
    """
    reduced = _EVAL.normalize_chars(text)
    return "".join(ch for ch in reduced if ch not in _DECORATIVE_SYMBOLS)


def normalized_cer(reference: str, hypothesis: str) -> float:
    """CER between one ground truth and one teacher hypothesis.

    Same contract as ``eval_chunk_gap.pair_cer`` -- and the same
    :func:`~eval_chunk_gap.levenshtein` -- but over
    :func:`normalize_for_comparison` instead of ``normalize_chars`` alone.

    Args:
        reference: Stored ground-truth text.
        hypothesis: Teacher output.

    Returns:
        ``edits / len(reference)``.  ``0.0`` when both sides normalise to empty
        (a purely decorative line the teacher also declined to transcribe is not
        evidence of a bad label), ``1.0`` when only the reference does.  Not
        clamped above: a hypothesis far longer than its reference should rank
        above one that is merely wrong, and that is the ordering a reviewer
        wants.
    """
    ref = normalize_for_comparison(reference)
    hyp = normalize_for_comparison(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _EVAL.levenshtein(ref, hyp) / len(ref)


# ----------------------------------------------------------------- manifest I/O


@dataclass(frozen=True)
class ManifestClip:
    """One manifest entry considered for scoring.

    Attributes:
        key: Manifest ``key``; its leading component is the title, which is what
            ``--include-titles`` matches against.
        source: Path to the audio, as recorded in the manifest.
        target: The stored ground-truth transcript on trial.
    """

    key: str
    source: str
    target: str


@dataclass
class ReportRow:
    """One scored clip.

    Attributes:
        clip: The manifest entry.
        teacher_text: Raw teacher output, or ``""`` when decoding failed.
        cer: Normalised CER, or ``None`` when decoding failed.
        error: Decode failure message, else ``None``.
    """

    clip: ManifestClip
    teacher_text: str = ""
    cer: Optional[float] = None
    error: Optional[str] = None
    ground_truth_normalized: str = ""
    teacher_normalized: str = ""

    def to_json(self) -> Dict[str, Any]:
        """Render the row for the jsonl report.

        Both raw strings are kept alongside the normalised ones: the CER is
        computed on the projection, but a reviewer judging whether the label is
        wrong has to see what was actually written and actually heard.
        """
        return {
            "key": self.clip.key,
            "source": self.clip.source,
            "ground_truth": self.clip.target,
            "teacher_text": self.teacher_text,
            "ground_truth_normalized": self.ground_truth_normalized,
            "teacher_normalized": self.teacher_normalized,
            "cer": self.cer,
            "ref_chars": len(self.ground_truth_normalized),
            "error": self.error,
        }


def load_manifest(path: Path) -> Iterator[ManifestClip]:
    """Stream ``key``/``source``/``target`` records out of a manifest jsonl.

    Opened read-only and yielded lazily -- ``data/vn/train.jsonl`` is large and
    ``--limit`` should not pay for the whole file.

    Args:
        path: Manifest jsonl in the ``data/vn/*.jsonl`` schema.

    Yields:
        One :class:`ManifestClip` per non-blank line.

    Raises:
        ValueError: On malformed JSON or a record missing a required field --
            silently skipping would understate how much of the corpus was
            actually examined.
    """
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            missing = [f for f in ("key", "source", "target") if f not in record]
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: record missing {', '.join(missing)}"
                )
            yield ManifestClip(
                key=str(record["key"]),
                source=str(record["source"]),
                target=str(record["target"]),
            )


def load_title_prefixes(path: Path) -> List[str]:
    """Read the allowed title prefixes, one per line.

    Blank lines and ``#`` comments are skipped so the file can carry a note
    about *why* a title is on the uncontaminated list.

    Args:
        path: The prefix file.

    Returns:
        The prefixes, in file order.

    Raises:
        ValueError: If the file contains no usable prefix.  An empty include
            file would otherwise silently select nothing and produce a clean
            empty report, which looks like "no label noise found".
    """
    prefixes = [
        stripped
        for line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]
    if not prefixes:
        raise ValueError(f"{path} lists no title prefixes")
    return prefixes


def select_clips(
    clips: Iterable[ManifestClip],
    prefixes: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[ManifestClip]:
    """Apply the title filter and the sampling limit, in that order.

    Order matters: ``--limit`` after filtering samples the titles the caller
    asked for, whereas limiting first would usually return nothing at all,
    since the manifest is grouped by title.

    Args:
        clips: Manifest entries, typically the lazy :func:`load_manifest`.
        prefixes: Keep only clips whose ``key`` starts with one of these. ``None``
            keeps everything -- see the module docstring on contamination.
        limit: Stop after this many surviving clips.

    Returns:
        The selected clips, in manifest order.
    """
    selected: List[ManifestClip] = []
    for clip in clips:
        if prefixes is not None and not clip.key.startswith(tuple(prefixes)):
            continue
        selected.append(clip)
        if limit is not None and len(selected) >= limit:
            break
    return selected


# -------------------------------------------------------------------- scoring


def score_clip(clip: ManifestClip, teacher: Any) -> ReportRow:
    """Decode one clip and score it against its stored transcript.

    A decode failure (unreadable file, backend error) is recorded on the row
    rather than raised: one bad clip must not lose a multi-hour sweep, and an
    unreadable clip is a fact the reviewer wants in the report.

    Args:
        clip: The manifest entry.
        teacher: Anything with ``transcribe(Path) -> str``.

    Returns:
        The scored row.
    """
    row = ReportRow(clip=clip)
    row.ground_truth_normalized = normalize_for_comparison(clip.target)
    try:
        row.teacher_text = teacher.transcribe(Path(clip.source))
    except Exception as exc:  # noqa: BLE001 - any backend failure is data
        row.error = f"{type(exc).__name__}: {exc}"
        return row
    row.teacher_normalized = normalize_for_comparison(row.teacher_text)
    row.cer = normalized_cer(clip.target, row.teacher_text)
    return row


def sort_rows(rows: Sequence[ReportRow]) -> List[ReportRow]:
    """Order the report worst-first, deterministically.

    Descending CER puts the strongest mislabelling candidates at the top, which
    is where a reviewer's attention runs out.  ``key`` breaks ties so that two
    runs over the same manifest produce byte-identical reports.  Failed decodes
    have no CER and sort last: they need a different kind of follow-up.

    Args:
        rows: Scored rows in any order.

    Returns:
        A new list, sorted.
    """
    return sorted(
        rows,
        key=lambda row: (row.cer is None, -(row.cer or 0.0), row.clip.key),
    )


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated quantile of an already-sorted sequence.

    Implemented here rather than pulled from numpy so the summary is defined by
    this file and does not shift with a dependency's default ``method``.
    Matches numpy's ``linear`` default.

    Args:
        sorted_values: Values in ascending order; must be non-empty.
        fraction: Quantile in ``[0, 1]``.

    Returns:
        The interpolated value.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def summarise(rows: Sequence[ReportRow]) -> Dict[str, Any]:
    """Aggregate the scored rows into the report's summary block.

    Reports both means, because they answer different questions.  ``mean_cer``
    is the unweighted per-clip mean -- the one to read here, since every clip is
    one candidate label regardless of length.  ``corpus_cer`` weights by
    reference length and is included only so this run can be compared against
    the corpus-level numbers ``eval_chunk_gap.py`` reports.

    Args:
        rows: Scored rows, in any order.

    Returns:
        Clip counts, the CER deciles, the mean and per-threshold counts.  CER
        statistics are computed over successfully decoded clips only; failures
        are counted separately rather than folded in as some sentinel value.
    """
    scored = [row for row in rows if row.cer is not None]
    values = sorted(float(row.cer) for row in scored)  # type: ignore[arg-type]
    failed = len(rows) - len(scored)

    summary: Dict[str, Any] = {
        "num_clips": len(rows),
        "num_scored": len(scored),
        "num_failed": failed,
        "mean_cer": (sum(values) / len(values)) if values else None,
        "corpus_cer": None,
        "deciles": None,
        "counts_above": {f"{t:g}": 0 for t in CER_THRESHOLDS},
        "cer_thresholds": list(CER_THRESHOLDS),
    }
    if not values:
        return summary

    ref_chars = sum(len(row.ground_truth_normalized) for row in scored)
    edits = sum(
        float(row.cer) * len(row.ground_truth_normalized)  # type: ignore[arg-type]
        for row in scored
    )
    summary["corpus_cer"] = (edits / ref_chars) if ref_chars else 0.0
    summary["deciles"] = {
        f"p{decile * 10}": _quantile(values, decile / 10.0) for decile in range(11)
    }
    summary["counts_above"] = {
        f"{threshold:g}": sum(1 for value in values if value >= threshold)
        for threshold in CER_THRESHOLDS
    }
    return summary


# --------------------------------------------------------------------- output


def write_report(rows: Sequence[ReportRow], path: Path) -> None:
    """Write the ranked jsonl report.

    Args:
        rows: Rows, already sorted by :func:`sort_rows`.
        path: Destination; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def format_summary(summary: Dict[str, Any], manifest: Path, config: TeacherConfig) -> str:
    """Render the summary for a log.

    Args:
        summary: Output of :func:`summarise`.
        manifest: The manifest that was read.
        config: The teacher that produced the hypotheses.

    Returns:
        A multi-line plain-text block.
    """
    lines = [
        "label-noise scan",
        f"  manifest    : {manifest}",
        f"  teacher     : {config.model_id} ({config.backend}, {config.device})",
        f"  clips       : {summary['num_clips']}"
        f" (scored {summary['num_scored']}, failed {summary['num_failed']})",
    ]
    if summary["mean_cer"] is None:
        lines.append("  no clips scored")
        return "\n".join(lines)

    lines.append(f"  mean CER    : {summary['mean_cer']:.4f}")
    lines.append(f"  corpus CER  : {summary['corpus_cer']:.4f}")
    deciles = summary["deciles"]
    lines.append("  deciles     : " + "  ".join(f"{k}={v:.3f}" for k, v in deciles.items()))
    counts = summary["counts_above"]
    lines.append("  at or above : " + "  ".join(f"{k}:{v}" for k, v in counts.items()))
    return "\n".join(lines)


# ------------------------------------------------------------------------ CLI


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
        "--manifest",
        type=Path,
        required=True,
        help="manifest jsonl to audit (key/source/target schema); read-only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="destination jsonl report, one row per clip, worst CER first",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="summary JSON destination (default: <out>.summary.json)",
    )
    parser.add_argument(
        "--include-titles",
        type=Path,
        default=None,
        help=(
            "file of title prefixes, one per line; only clips whose key starts "
            "with one are processed. Pass the subset the teacher was NOT trained "
            "on -- on its training titles it reproduces our errors instead of "
            "detecting them"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N clips (applied after --include-titles), for sampling runs",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=sorted(BACKENDS),
        help=f"teacher backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"teacher model id or local dir (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="teacher device: auto, cpu or cuda (default: auto)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        nargs="+",
        default=[0],
        help="GPU ordinal(s) for the teacher (default: 0)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type, e.g. float16, int8_float16, float32",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="teacher decoder beam width (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="clips per batched-pipeline call; 1 decodes sequentially (default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="backend loader workers (default: 1)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="log progress to stderr every N clips; 0 disables (default: 100)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Scan a manifest for mislabelled ground truth and write the ranked report.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when the selection is empty or an input is
        unusable.  Never modifies the corpus or the manifest.
    """
    args = parse_args(argv)

    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    prefixes: Optional[List[str]] = None
    if args.include_titles is not None:
        if not args.include_titles.is_file():
            print(f"title list not found: {args.include_titles}", file=sys.stderr)
            return 1
        try:
            prefixes = load_title_prefixes(args.include_titles)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        print(
            "warning: no --include-titles; clips from titles the teacher was "
            "trained on will be scored, and its agreement on those is memorised "
            "rather than earned",
            file=sys.stderr,
        )

    try:
        clips = select_clips(load_manifest(args.manifest), prefixes, args.limit)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not clips:
        print("no clips selected; check --include-titles", file=sys.stderr)
        return 1

    config = TeacherConfig(
        model_id=args.model_id,
        backend=args.backend,
        device=args.device,
        device_index=tuple(args.device_index),
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    try:
        teacher = build_teacher(config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"scoring {len(clips)} clips with {config.model_id}", file=sys.stderr)
    rows: List[ReportRow] = []
    for index, clip in enumerate(clips, start=1):
        try:
            rows.append(score_clip(clip, teacher))
        except RuntimeError as exc:
            # Backend construction failures surface on the first transcribe and
            # will repeat for every clip; stop rather than emit N copies.
            print(str(exc), file=sys.stderr)
            return 1
        if args.progress_every and index % args.progress_every == 0:
            print(f"  {index}/{len(clips)}", file=sys.stderr)

    ordered = sort_rows(rows)
    write_report(ordered, args.out)

    summary = summarise(ordered)
    summary_path = args.summary_out or args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(args.manifest),
        "report": str(args.out),
        "include_titles": str(args.include_titles) if args.include_titles else None,
        "teacher": {
            "model_id": config.model_id,
            "backend": config.backend,
            "device": config.device,
            "compute_type": config.compute_type,
            "beam_size": config.beam_size,
            "batch_size": config.batch_size,
        },
        "transcribe_options": teacher_transcribe_options(config),
        "summary": summary,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(format_summary(summary, args.manifest, config), file=sys.stderr)
    print(f"  report      : {args.out}", file=sys.stderr)
    print(f"  summary     : {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
