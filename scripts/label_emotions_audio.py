#!/usr/bin/env python3
"""Pseudo-label every clip's *acoustic* emotion with emotion2vec+ large.

Rounds 1 and 2 wrote ``emo_target="<|NEUTRAL|>"`` for all ~550k clips
(``scripts/prepare_vn_data.py``'s fixed ``EMO_TARGET``).  A constant target is
not supervision: the emotion head's optimum under it is to emit ``<|NEUTRAL|>``
unconditionally, which is exactly what it learned.  Round 3 replaces the
constant with two *independent* pseudo-labellers -- this one, from the audio,
and ``scripts/label_emotions_text.py``, from the transcript -- and keeps only
the labels they agree on (``scripts/merge_emo_labels.py``).

Agreement is the point.  Either labeller alone is noisy enough that training on
it would trade one bias for another; two labellers that see *different
evidence* (waveform vs. words) make their errors largely independently, so
their intersection is far cleaner than either input.  The cost is coverage, and
that cost is affordable here because a clip whose labellers disagree is not
dropped -- it is masked, still training the ASR/CTC branch.

    .venv/bin/python scripts/label_emotions_audio.py \\
        --manifest data/vn/train.jsonl \\
        --model-dir /staged/models/emotion2vec_plus_large \\
        --device cuda --batch-size 32 \\
        --out outputs/emo_audio.jsonl

**Read-only with respect to the corpus.**  The manifest and the audio are
opened for reading; the only path written is ``--out``.

Why emotion2vec+ large
----------------------

It is Apache-2.0 and FunASR-native, so it loads through the same stack the
student model already depends on and adds no new licence to the project.  That
constraint is not incidental: several stronger-looking SER checkpoints and
corpora are GPL-derived or research-only, which would contaminate a model we
intend to ship.  Anything proposed as a replacement here has to clear the same
bar.

Operational notes for the cluster
---------------------------------

FunASR's model loader does two things that fail on a compute node: it phones
home for a version check, and it runs ``pip install`` against the
``requirements.txt`` bundled inside the model directory.  On a node with no
egress that is a hang, and on a read-only ``site-packages`` it is a crash --
either way after the job has already been scheduled.  So this script always
passes ``disable_update=True``, and ``--model-dir`` is expected to be a
**locally staged path** rather than a hub id.  The model's own dependencies
must be baked into the image beforehand; see the note at the bottom of this
docstring.

Resumability is not optional at this scale.  ~550k clips / ~900 audio-hours is
a multi-hour job on one H100 and cluster jobs get pre-empted.  ``--out`` is
opened in append mode and every key already present in it is skipped, with a
flush after each batch, so a killed job loses at most one batch of work.

Extra dependencies beyond the repo's ``requirements.txt``: ``funasr`` (already
a dependency), plus emotion2vec+'s own stack -- ``torch``, ``torchaudio`` and
``modelscope`` are already implied by FunASR; ``soundfile`` is used here only
to read clip durations from the file header and is treated as optional.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

__all__ = [
    "AUDIO_CLASSES",
    "ManifestClip",
    "Emotion2vecLabeller",
    "clip_duration_seconds",
    "load_manifest",
    "looks_like_hub_id",
    "parse_class_label",
    "read_done_keys",
    "select_clips",
    "main",
]

#: ``iic/emotion2vec_plus_large`` emits exactly these nine classes.  Kept as an
#: ordered tuple because it is also the contract ``scripts/merge_emo_labels.py``
#: maps from -- ``AUDIO_LABEL_TO_TOKEN`` there must cover every entry here, and
#: ``tests/test_emo_labels.py`` asserts that it does.  Note that ``other`` and
#: ``unknown`` are real emissions of this model, not error states.
AUDIO_CLASSES: tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

#: The FunASR model id.  Apache-2.0; see the module docstring on why the licence
#: is a hard filter here rather than a preference.
DEFAULT_MODEL_ID = "iic/emotion2vec_plus_large"

#: Batches may fail individually -- one truncated wav, one unreadable file --
#: and the sweep should survive that.  A *run* of failures is a different
#: animal: it means the fault is systemic and every remaining batch will fail
#: the same way, so continuing walks the whole corpus to produce nothing and
#: exit 0.  Five is enough to ride out a couple of genuinely bad clips at any
#: sane batch size and short enough to fail within seconds of a bad mount.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


# ----------------------------------------------------------------- label parsing


def parse_class_label(raw: str) -> str:
    """Reduce one raw emotion2vec label to its bare English class name.

    The model returns bilingual labels -- ``"生气/angry"``, ``"<unk>/unknown"``
    -- and the Chinese half has changed between checkpoint revisions while the
    English half has not, so only the English half is trusted.

    Args:
        raw: A label string as emitted by the model.

    Returns:
        The lowercased class name, guaranteed to be in :data:`AUDIO_CLASSES`.

    Raises:
        ValueError: If the class is not one this script knows.  Failing loudly
            is deliberate: the alternative is bucketing an unrecognised class
            into ``other`` (and therefore into the mask), which would silently
            delete supervision for a whole emotion if a future checkpoint
            renamed one.  A crash on clip 1 is cheap; discovering this after a
            550k-clip sweep is not.
    """
    name = raw.split("/")[-1].strip().lower()
    if name not in AUDIO_CLASSES:
        known = ", ".join(AUDIO_CLASSES)
        raise ValueError(
            f"unrecognised emotion2vec class {raw!r} (parsed as {name!r}); known: {known}"
        )
    return name


# ----------------------------------------------------------------- manifest I/O


@dataclass(frozen=True)
class ManifestClip:
    """One manifest entry to label.

    Attributes:
        key: The join identifier (``<title>__<speaker>__<voice>``).  Everything
            downstream joins on this and nothing else.
        source: Absolute path to the wav.
    """

    key: str
    source: str


def load_manifest(path: Path) -> Iterator[ManifestClip]:
    """Stream ``key``/``source`` out of a manifest jsonl.

    Lazy: ``data/vn/train.jsonl`` is large and a ``--limit`` run should not pay
    to parse all of it.

    Args:
        path: Manifest jsonl in the ``data/vn/*.jsonl`` schema.

    Yields:
        One :class:`ManifestClip` per non-blank line.

    Raises:
        ValueError: On malformed JSON or a record missing ``key``/``source``.
            Skipping bad records would understate how much of the corpus was
            actually labelled, which is indistinguishable downstream from
            "these clips were masked on purpose".
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
            missing = [f for f in ("key", "source") if f not in record]
            if missing:
                raise ValueError(f"{path}:{lineno}: record missing {', '.join(missing)}")
            yield ManifestClip(key=str(record["key"]), source=str(record["source"]))


def read_done_keys(path: Path) -> Set[str]:
    """Collect the keys already written to an output file.

    Tolerant of a truncated final line, which is the normal shape of a file
    left behind by a pre-empted job: that line is dropped and its clip is
    relabelled.

    Args:
        path: An existing ``--out`` file, or a path that does not exist yet.

    Returns:
        The set of keys to skip.  Empty when the file is absent.
    """
    if not path.is_file():
        return set()
    done: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail of a killed job
            key = record.get("key")
            if key is not None:
                done.add(str(key))
    return done


def select_clips(
    clips: Iterable[ManifestClip],
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 0,
) -> List[ManifestClip]:
    """Apply the pilot's ``--limit`` / ``--sample`` selection.

    Applied *before* the resume filter, never after.  A run that sampled from
    the not-yet-done clips would draw a different subset every restart, so a
    pre-empted pilot would end up covering more clips than it was asked for and
    its statistics would not describe any single sample.

    Args:
        clips: Manifest entries, typically the lazy :func:`load_manifest`.
        limit: Take the first N in manifest order.  Streams, so it does not
            read the rest of the file.
        sample: Take N uniformly at random.  Requires reading the whole
            manifest, which is the price of an unbiased pilot -- manifest order
            is grouped by title, so ``--limit`` samples titles, not the corpus.
        seed: Seed for ``--sample``, so a pilot is reproducible.

    Returns:
        The selected clips, in manifest order in both cases.

    Raises:
        ValueError: If both ``limit`` and ``sample`` are given; they answer
            different questions and silently composing them would hide which
            one shaped the result.
    """
    if limit is not None and sample is not None:
        raise ValueError("--limit and --sample are mutually exclusive")

    if limit is not None:
        selected: List[ManifestClip] = []
        for clip in clips:
            selected.append(clip)
            if len(selected) >= limit:
                break
        return selected

    if sample is None:
        return list(clips)

    everything = list(clips)
    if sample >= len(everything):
        return everything
    indices = sorted(random.Random(seed).sample(range(len(everything)), sample))
    return [everything[i] for i in indices]


def looks_like_hub_id(model_dir: str) -> bool:
    """Guess whether ``--model-dir`` names a hub repo rather than a staged path.

    A hub id is ``owner/name`` with no such directory on disk.  Deliberately
    conservative: an existing directory is never flagged, so the false-positive
    case is a warning about a path that has not been staged *yet*, which is
    itself worth saying.

    Args:
        model_dir: The value of ``--model-dir``.

    Returns:
        ``True`` when it looks like a hub id.
    """
    return "/" in model_dir and not Path(model_dir).is_dir()


def clip_duration_seconds(source: str) -> Optional[float]:
    """Read a clip's duration from its file header.

    Header-only, so it costs no decode.  Reported per clip because the pilot
    needs to know whether the labeller's confidence tracks clip length -- very
    short backchannels are where a frame-level SER model is least reliable, and
    the merge's confidence threshold should be read with that in mind.

    Args:
        source: Path to the wav.

    Returns:
        The duration in seconds, or ``None`` if soundfile is unavailable or the
        header is unreadable.  Never raises: a missing duration is a missing
        diagnostic, not a reason to lose the label.
    """
    try:
        import soundfile  # noqa: PLC0415 - optional, and only for diagnostics
    except ImportError:
        return None
    try:
        info = soundfile.info(source)
    except Exception:  # noqa: BLE001 - unreadable header is not fatal here
        return None
    return float(info.frames) / float(info.samplerate) if info.samplerate else None


# -------------------------------------------------------------------- labelling


class Emotion2vecLabeller:
    """emotion2vec+ large behind a ``label(paths) -> list[dict]`` interface.

    The FunASR import and the model load happen on first use, not at
    construction, so that ``--help``, the manifest reader and the whole label
    parsing path stay importable and testable on a machine with no funasr, no
    torch and no GPU.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
    ) -> None:
        self._model_dir = model_dir
        self._device = device
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """Import FunASR and construct the model, once."""
        if self._model is not None:
            return
        try:
            from funasr import AutoModel  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the audio labeller needs the 'funasr' package; install it in "
                "the image, or stage the model and run on a node that has it"
            ) from exc

        # disable_update=True is mandatory, not a tuning choice -- see the
        # module docstring: without it FunASR reaches for the network and for
        # pip on a node that has neither.
        self._model = AutoModel(
            model=self._model_dir,
            device=self._device,
            disable_update=True,
        )

    def label(self, paths: Sequence[str]) -> List[Dict[str, Any]]:
        """Label a batch of clips.

        Args:
            paths: Audio paths, in the order results are wanted.

        Returns:
            One dict per input path, in the same order, with ``label`` (the
            argmax class), ``score`` (its probability) and ``probs`` (the full
            distribution over :data:`AUDIO_CLASSES`).

        Raises:
            RuntimeError: If the backend returns a different number of results
                than it was given -- the results are matched to inputs
                positionally, so a length mismatch would silently attach every
                label to the wrong clip.
        """
        self._ensure_loaded()
        results = self._model.generate(
            list(paths),
            granularity="utterance",
            extract_embedding=False,
        )
        if len(results) != len(paths):
            raise RuntimeError(
                f"emotion2vec returned {len(results)} results for {len(paths)} clips; "
                "results are matched positionally and cannot be trusted"
            )
        return [self._to_record(result) for result in results]

    @staticmethod
    def _to_record(result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one FunASR result dict into this script's output shape.

        Args:
            result: A ``{"labels": [...], "scores": [...]}`` dict.

        Returns:
            ``{"label", "score", "probs"}``.

        Raises:
            ValueError: On a malformed result or an unknown class name.
        """
        labels = result.get("labels")
        scores = result.get("scores")
        if not labels or scores is None or len(labels) != len(scores):
            raise ValueError(f"malformed emotion2vec result: {result!r}")
        probs = {parse_class_label(name): float(score) for name, score in zip(labels, scores)}
        best = max(probs.items(), key=lambda item: item[1])
        return {"label": best[0], "score": best[1], "probs": probs}


# --------------------------------------------------------------------- progress


class Progress:
    """Periodic stderr progress with a throughput rate and an ETA.

    At ~550k clips the only way to tell a slow job from a stuck one is a rate,
    so the rate is reported rather than a bare counter.
    """

    def __init__(self, total: int, every: int, stream: Any = sys.stderr) -> None:
        self._total = total
        self._every = every
        self._stream = stream
        self._done = 0
        self._start = time.monotonic()

    def advance(self, count: int = 1) -> None:
        """Record progress and log if the interval has elapsed."""
        previous = self._done
        self._done += count
        if not self._every:
            return
        if self._done // self._every > previous // self._every:
            self.log()

    def log(self) -> None:
        """Emit one progress line."""
        elapsed = max(time.monotonic() - self._start, 1e-9)
        rate = self._done / elapsed
        remaining = max(self._total - self._done, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"  {self._done}/{self._total}  {rate:.1f} clips/s  "
            f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m",
            file=self._stream,
            flush=True,
        )


# ------------------------------------------------------------------------- CLI


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
        help="manifest jsonl to label (key/source schema); read-only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "destination jsonl, one row per clip. Appended to, never truncated: "
            "keys already present are skipped so a pre-empted job resumes"
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_ID,
        help=(
            f"local staged model directory, or a hub id (default: {DEFAULT_MODEL_ID}). "
            "Prefer a staged path on the cluster -- see the module docstring"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="device for emotion2vec (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="clips per backend call, and per output flush (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="label only the first N clips in manifest order (pilot runs)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "label N clips drawn uniformly at random; use with --seed. The SAME "
            "--sample/--seed must be passed to scripts/label_emotions_text.py, "
            "or the merge sees two different subsets and reports the non-overlap "
            "as disagreement"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "seed for --sample, so a pilot is reproducible (default: 0). Must "
            "match the text labeller's --seed"
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=(
            "abort after this many batches fail in a row, which indicates a "
            "systemic fault rather than bad clips (default: "
            f"{DEFAULT_MAX_CONSECUTIVE_FAILURES})"
        ),
    )
    parser.add_argument(
        "--no-duration",
        action="store_true",
        help="skip the per-clip duration header read",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="log progress to stderr every N clips; 0 disables (default: 1000)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Label a manifest's audio with emotion2vec+ and append the results.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when an input is unusable.
    """
    args = parse_args(argv)

    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1
    if args.max_consecutive_failures < 1:
        print("--max-consecutive-failures must be at least 1", file=sys.stderr)
        return 1
    if looks_like_hub_id(args.model_dir):
        # A warning, not an error: pulling from the hub is the right thing on a
        # workstation. On a cluster node it is the wrong thing and fails late,
        # after the job has been scheduled and the queue time spent.
        print(
            f"warning: --model-dir {args.model_dir!r} looks like a hub id rather "
            "than a locally staged directory. On a compute node FunASR will try "
            "to reach the network to fetch it and to pip-install the model's "
            "bundled requirements, which hangs with no egress and fails on a "
            "read-only site-packages. Stage the model and pass its path.",
            file=sys.stderr,
        )

    try:
        clips = select_clips(
            load_manifest(args.manifest),
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not clips:
        print("no clips selected", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done_keys(args.out)
    pending = [clip for clip in clips if clip.key not in done]
    if done:
        print(
            f"resuming: {len(done)} keys already in {args.out}, "
            f"{len(pending)} of {len(clips)} selected clips left",
            file=sys.stderr,
        )
    if not pending:
        print("nothing to do; every selected clip is already labelled", file=sys.stderr)
        return 0

    labeller = Emotion2vecLabeller(model_dir=args.model_dir, device=args.device)
    progress = Progress(len(pending), args.progress_every)
    print(
        f"labelling {len(pending)} clips with {args.model_dir} on {args.device}",
        file=sys.stderr,
    )

    rows_written = 0
    batches_skipped = 0
    consecutive_failures = 0
    aborted = False

    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            try:
                records = labeller.label([clip.source for clip in batch])
            except RuntimeError as exc:
                # A backend failure repeats for every subsequent batch; stop
                # rather than burn the queue. The work so far is already on disk
                # and the next invocation resumes from it.
                print(f"backend failure after {rows_written} rows: {exc}", file=sys.stderr)
                aborted = True
                break
            except Exception as exc:  # noqa: BLE001 - one bad batch is data, not a crash
                batches_skipped += 1
                consecutive_failures += 1
                print(
                    f"batch at offset {start} failed ({type(exc).__name__}: {exc}); skipping",
                    file=sys.stderr,
                )
                if consecutive_failures >= args.max_consecutive_failures:
                    # The failures worth tolerating are per-clip: one truncated
                    # wav, one unreadable file. Those do not come in a run. A run
                    # of them means the fault is systemic -- wrong audio mount,
                    # corrupt staged model, unusable device -- and every
                    # remaining batch would fail identically, so continuing walks
                    # the whole corpus to write nothing and exit clean.
                    print(
                        f"aborting: {consecutive_failures} consecutive batch failures. "
                        "That is a systemic fault (wrong --manifest source paths, "
                        "unmounted audio, corrupt staged model, unusable device), "
                        "not bad data. Fix the cause and rerun -- the rows already "
                        "written are kept and will be skipped on resume.",
                        file=sys.stderr,
                    )
                    aborted = True
                    break
                progress.advance(len(batch))
                continue

            consecutive_failures = 0
            for clip, record in zip(batch, records):
                payload = {
                    "key": clip.key,
                    "label": record["label"],
                    "score": record["score"],
                    "probs": record["probs"],
                    "duration": None if args.no_duration else clip_duration_seconds(clip.source),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rows_written += 1
            handle.flush()
            progress.advance(len(batch))

    progress.log()
    # Always stated, success or failure: "how much did this job actually
    # produce" must never have to be inferred from the exit code alone.
    print(f"  rows written    : {rows_written}/{len(pending)}", file=sys.stderr)
    print(f"  batches skipped : {batches_skipped}", file=sys.stderr)
    print(f"  labels: {args.out}", file=sys.stderr)

    if aborted:
        return 1
    if rows_written == 0:
        print(
            "no rows were written; treating as a failure rather than a clean "
            "empty run",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
