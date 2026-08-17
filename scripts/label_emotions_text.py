#!/usr/bin/env python3
"""Pseudo-label every clip's emotion from its *transcript* with a local LLM.

The second of the two independent labellers behind round 3's emotion targets;
see ``scripts/label_emotions_audio.py`` for why there are two and what the
constant-``<|NEUTRAL|>`` collapse was.  This one reads only the ground-truth
Japanese text and never hears the audio, which is precisely what makes its
errors independent of the acoustic labeller's and therefore makes their
agreement (``scripts/merge_emo_labels.py``) worth more than either alone.

    .venv/bin/python scripts/label_emotions_text.py \\
        --manifest data/vn/train.jsonl \\
        --model /staged/models/Qwen2.5-14B-Instruct \\
        --batch-size 512 \\
        --out outputs/emo_text.jsonl

**Read-only with respect to the corpus.**  Only ``--out`` is written.

Why Qwen2.5-14B-Instruct through vLLM
-------------------------------------

Apache-2.0, and strong on Japanese.  A larger Japanese-tuned candidate was
rejected on licensing -- it is research/non-commercial only, and a
non-commercial labeller taints every downstream weight it supervises even
though it never ships itself.  The licence filter here is the same one applied
to the acoustic labeller and it is not negotiable.

vLLM in *offline batch* mode (not a served endpoint) because this is a single
pass over ~550k short lines with no latency requirement: continuous batching
gives the throughput, and there is no server to keep alive across a
pre-emption.

Why ``sexual`` and ``embarrassed`` are classes
----------------------------------------------

The corpus is Japanese visual-novel dialogue.  A large fraction of it is
neither plainly neutral nor cleanly one of SenseVoice's seven emotions --
flustered lines and sexual lines are both frequent, and forcing them into the
seven is how the seven get poisoned: a classifier with no bucket for "flustered"
scatters those lines across ``happy``, ``fearful`` and ``surprised``, and the
emotion head learns that noise.  Giving them their own buckets keeps the seven
clean.  Downstream both map to the mask, so these clips contribute nothing to
the emotion head -- but they stay in the corpus and keep training the ASR
branch.  ``sexual`` is a content category to be labelled, never a filter.

Extra dependencies beyond the repo's ``requirements.txt``: ``vllm`` (with a
matching ``torch``) and ``transformers`` for the chat template.  Both are
imported lazily, inside the backend, so this module -- and its unit tests --
run without either.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

__all__ = [
    "TEXT_CLASSES",
    "CLASS_GLOSS_JA",
    "ManifestClip",
    "VllmLabeller",
    "build_prompt",
    "load_manifest",
    "parse_response",
    "read_done_keys",
    "select_clips",
    "main",
]

#: The label set this labeller may emit.  The first seven are SenseVoice's own
#: emotions; ``other`` is the honest "none of these"; ``embarrassed`` and
#: ``sexual`` are the two auxiliary buckets described in the module docstring.
#:
#: This tuple is the contract ``scripts/merge_emo_labels.py``'s
#: ``TEXT_LABEL_TO_TOKEN`` maps from, and ``tests/test_emo_labels.py`` asserts
#: the two stay in step -- along with the glossary below, which is the third
#: place the same list appears and the one most likely to drift.
TEXT_CLASSES: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "neutral",
    "fearful",
    "disgusted",
    "surprised",
    "other",
    "embarrassed",
    "sexual",
)

#: Japanese gloss per class, used both to build the prompt's glossary and to
#: accept a Japanese answer in :func:`parse_response`.  A Japanese-prompted
#: model answers in Japanese often enough that rejecting those would throw away
#: correct labels as if they were refusals.
CLASS_GLOSS_JA: Dict[str, str] = {
    "happy": "喜び",
    "sad": "悲しみ",
    "angry": "怒り",
    "neutral": "平静",
    "fearful": "恐れ",
    "disgusted": "嫌悪",
    "surprised": "驚き",
    "other": "その他",
    "embarrassed": "照れ",
    "sexual": "性的",
}

#: One short description per class, shown to the model.  Written to separate the
#: classes that are actually confusable in this corpus: 照れ vs 喜び, 性的 vs 喜び,
#: 驚き vs 恐れ.  Without these the model collapses the auxiliary buckets into
#: ``happy``, which is the failure the buckets exist to prevent.
CLASS_HINT_JA: Dict[str, str] = {
    "happy": "楽しさ・嬉しさ・上機嫌",
    "sad": "悲しみ・落胆・寂しさ",
    "angry": "怒り・苛立ち・叱責",
    "neutral": "感情の色が薄い平常の発話。説明・相槌・事務的なやり取り",
    "fearful": "恐怖・不安・怯え",
    "disgusted": "嫌悪・侮蔑・生理的な拒否",
    "surprised": "驚き・意外・動揺",
    "other": "上のどれにも当てはまらない",
    "embarrassed": "照れ・恥じらい・きまり悪さ。嬉しさとは区別する",
    "sexual": "性的な文脈の発話・喘ぎ・情事の描写",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

#: See ``scripts/label_emotions_audio.py`` for the reasoning; a run of batch
#: failures is systemic, and continuing through it produces an empty output file
#: and a zero exit code.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

#: Above this fraction of unparsed responses the run is treated as failed.
#:
#: An unparsed response is not an error anywhere -- it becomes ``label: null``,
#: the merge masks the clip, and every downstream step behaves correctly. That
#: is what makes it dangerous: a broken prompt or a mismatched chat template
#: produces a complete-looking label file that supervises nothing, which is the
#: precise failure round 3 exists to undo. Half is deliberately loose; a healthy
#: run sits near zero, so anything approaching this is already broken.
DEFAULT_MAX_UNPARSED_FRAC = 0.5

_SYSTEM_PROMPT_HEADER = (
    "あなたは日本語のビジュアルノベルの台詞を感情分類する専門家です。\n"
    "与えられた台詞ひとつを、次のラベルのうちちょうど一つに分類してください。\n"
)

_SYSTEM_PROMPT_FOOTER = (
    "\n規則:\n"
    "- 出力はラベル名の英単語ひとつだけ。説明・理由・記号・句読点を一切付けないこと。\n"
    "- 判断に迷う場合や感情が読み取れない場合は neutral ではなく other を選ぶこと。\n"
    "- 台詞の文字面だけで判断すること。前後の文脈は与えられない。\n"
)


def build_prompt(transcript: str) -> List[Dict[str, str]]:
    """Build the chat messages that classify one transcript.

    The glossary is generated from :data:`TEXT_CLASSES` rather than written out
    by hand, so the prompt cannot drift away from the label set the merge step
    maps from.  ``tests/test_emo_labels.py`` pins that every class name reaches
    the prompt.

    The instruction to prefer ``other`` over ``neutral`` when unsure is
    load-bearing.  ``neutral`` is the class that already swallowed the entire
    emotion head once; a model that resolves its uncertainty towards
    ``neutral`` would reproduce that collapse through the agreement filter,
    whereas uncertainty routed to ``other`` becomes a mask and costs only
    coverage.

    Args:
        transcript: The ground-truth Japanese line.

    Returns:
        Chat messages in the OpenAI/vLLM ``{"role", "content"}`` shape.
    """
    glossary = "\n".join(
        f"- {name}（{CLASS_GLOSS_JA[name]}）: {CLASS_HINT_JA[name]}" for name in TEXT_CLASSES
    )
    system = _SYSTEM_PROMPT_HEADER + glossary + _SYSTEM_PROMPT_FOOTER
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"台詞: {transcript}\nラベル:"},
    ]


# ---------------------------------------------------------------- response parse

#: Wrapper characters a chatty model puts around a one-word answer: markdown
#: emphasis and code fences, ASCII and Japanese quotes and brackets.
_WRAPPERS = "*_`\"'「」『』“”‘’()（）[]［］【】<>《》 \t\r\n"

#: Trailing punctuation in either script.
_TRAILING = "。．.、,!！?？:：;；　 \t\r\n"

_LABEL_RE = re.compile(r"^[A-Za-z぀-ヿ一-鿿]+$")

#: Reverse of :data:`CLASS_GLOSS_JA`, built once.
_GLOSS_TO_CLASS: Dict[str, str] = {gloss: name for name, gloss in CLASS_GLOSS_JA.items()}


def parse_response(text: str) -> Optional[str]:
    """Extract the class name from a model response, or give up.

    Tolerant of the ways an instruct model decorates a one-word answer --
    ``"**happy**"``, ``"「happy」"``, ``" Happy. "``, the Japanese gloss -- and
    intolerant of everything else.

    It never guesses.  A response like ``"I think this is happy or maybe sad"``
    contains a valid class name, and a substring search would happily return
    ``happy``; but that answer is the model refusing to commit, and recording it
    as a confident label injects exactly the noise the two-labeller design
    exists to keep out.  ``None`` propagates to the merge as a missing label,
    which masks the clip -- the cheap failure.

    One deliberate exception: only the **first line** is considered, so a model
    that answers ``"happy"`` and then explains itself on the following lines is
    accepted.  That is a formatting deviation from the "one word" instruction,
    not a hedge -- the model did commit, on the line where the answer goes.  The
    distinction that matters is *within* the first line: a first line containing
    anything besides the bare class name is rejected, which is why the hedged
    sentence above still returns ``None``.

    Args:
        text: The raw generated text.

    Returns:
        A member of :data:`TEXT_CLASSES`, or ``None``.
    """
    if not text:
        return None
    candidate = text.strip()
    # A model that ignores the "one word" instruction usually still puts the
    # answer on its own first line.
    candidate = candidate.splitlines()[0] if candidate.splitlines() else ""

    previous = None
    while candidate != previous:
        previous = candidate
        candidate = candidate.strip(_WRAPPERS).strip(_TRAILING)

    if not candidate or not _LABEL_RE.match(candidate):
        return None

    lowered = candidate.lower()
    if lowered in TEXT_CLASSES:
        return lowered
    return _GLOSS_TO_CLASS.get(candidate)


# ----------------------------------------------------------------- manifest I/O


@dataclass(frozen=True)
class ManifestClip:
    """One manifest entry to label.

    Attributes:
        key: The join identifier shared with the audio labeller.
        target: The ground-truth Japanese transcript, the only input here.
    """

    key: str
    target: str


def load_manifest(path: Path) -> Iterator[ManifestClip]:
    """Stream ``key``/``target`` out of a manifest jsonl.

    Args:
        path: Manifest jsonl in the ``data/vn/*.jsonl`` schema.

    Yields:
        One :class:`ManifestClip` per non-blank line.

    Raises:
        ValueError: On malformed JSON or a record missing ``key``/``target``.
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
            missing = [f for f in ("key", "target") if f not in record]
            if missing:
                raise ValueError(f"{path}:{lineno}: record missing {', '.join(missing)}")
            yield ManifestClip(key=str(record["key"]), target=str(record["target"]))


def read_done_keys(path: Path) -> Set[str]:
    """Collect the keys already written to an output file.

    Args:
        path: An existing ``--out`` file, or a path that does not exist yet.

    Returns:
        The set of keys to skip; empty when the file is absent.  A truncated
        final line -- the normal shape of a pre-empted job's output -- is
        dropped and its clip relabelled.
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
                continue
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

    Applied before the resume filter for the same reason as in the audio
    labeller: sampling the not-yet-done clips would redraw the subset on every
    restart.  Note also that a pilot is only interpretable if *both* labellers
    are run over the same subset, so ``--sample``/``--seed`` must match between
    the two scripts.

    Args:
        clips: Manifest entries.
        limit: Take the first N in manifest order.
        sample: Take N uniformly at random.
        seed: Seed for ``--sample``.

    Returns:
        The selected clips, in manifest order.

    Raises:
        ValueError: If both ``limit`` and ``sample`` are given.
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


# -------------------------------------------------------------------- labelling


class VllmLabeller:
    """Qwen2.5-Instruct behind a ``label(transcripts) -> list[dict]`` interface.

    vLLM and torch are imported on first use, never at construction, so the
    prompt builder and the response parser -- the two parts with any real logic
    -- stay unit-testable on a laptop.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
        max_tokens: int = 8,
        seed: int = 0,
    ) -> None:
        self._model_id = model
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._max_tokens = max_tokens
        self._seed = seed
        self._llm: Any = None
        self._params: Any = None

    def _ensure_loaded(self) -> None:
        """Import vLLM and construct the engine, once."""
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the text labeller needs the 'vllm' package; install it in the "
                "image and run on a GPU node"
            ) from exc

        self._llm = LLM(
            model=self._model_id,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            dtype=self._dtype,
            seed=self._seed,
        )
        # temperature=0 removes sampling noise, but it does NOT make this run
        # bit-reproducible. vLLM's continuous batching means a request's
        # numerics depend on which other requests share its batch, and resuming
        # a pre-empted job always changes batch composition -- so a resumed run
        # can differ from an uninterrupted one on a small number of borderline
        # lines. Greedy decoding makes that difference rare and confined to
        # near-ties, which is as much determinism as this backend offers; the
        # merge's agreement filter is what actually absorbs the residue.
        # max_tokens is tiny because the answer is one word and anything longer
        # is a response parse_response will reject anyway. logprobs=1 gives the
        # first-token probability used as a confidence.
        self._params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self._max_tokens,
            logprobs=1,
        )

    def label(self, transcripts: Sequence[str]) -> List[Dict[str, Any]]:
        """Classify a batch of transcripts.

        Args:
            transcripts: Ground-truth lines, in the order results are wanted.

        Returns:
            One dict per input, in the same order, with ``label`` (``None`` when
            the response did not parse), ``score`` and ``raw``.

        Raises:
            RuntimeError: On a result-count mismatch; results are matched
                positionally and a mismatch would attach labels to the wrong
                clips.
        """
        self._ensure_loaded()
        conversations = [build_prompt(text) for text in transcripts]
        outputs = self._llm.chat(conversations, self._params, use_tqdm=False)
        if len(outputs) != len(transcripts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} results for {len(transcripts)} prompts; "
                "results are matched positionally and cannot be trusted"
            )
        return [self._to_record(output) for output in outputs]

    @staticmethod
    def _to_record(output: Any) -> Dict[str, Any]:
        """Convert one vLLM ``RequestOutput`` into this script's output shape.

        Args:
            output: A vLLM request output.

        Returns:
            ``{"label", "score", "raw"}``.  ``score`` is the probability of the
            *first generated token*, which is a proxy for label confidence
            rather than the probability of the whole label string -- good
            enough to rank answers, and ``None`` whenever logprobs are absent.
        """
        completion = output.outputs[0]
        raw = completion.text
        return {
            "label": parse_response(raw),
            "score": _first_token_probability(completion),
            "raw": raw,
        }


def _first_token_probability(completion: Any) -> Optional[float]:
    """Recover ``exp(logprob)`` of the first generated token.

    Defensive throughout: the logprobs payload has changed shape across vLLM
    releases, and a confidence is a diagnostic -- losing it must never lose the
    label it belongs to.

    Args:
        completion: A vLLM ``CompletionOutput``.

    Returns:
        A probability in ``(0, 1]``, or ``None`` when unavailable.
    """
    logprobs = getattr(completion, "logprobs", None)
    if not logprobs:
        return None
    first = logprobs[0]
    if not first:
        return None
    token_ids = getattr(completion, "token_ids", None)
    entry = None
    if token_ids:
        entry = first.get(token_ids[0])
    if entry is None:
        entry = max(first.values(), key=lambda item: getattr(item, "logprob", float("-inf")))
    value = getattr(entry, "logprob", None)
    if value is None:
        return None
    try:
        return math.exp(float(value))
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------- progress


class Progress:
    """Periodic stderr progress with a throughput rate and an ETA."""

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
            f"  {self._done}/{self._total}  {rate:.1f} lines/s  "
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
        help="manifest jsonl to label (key/target schema); read-only",
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
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"local staged model directory, or a hub id (default: {DEFAULT_MODEL}). "
            "Must be Apache/MIT-licensed -- see the module docstring"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="prompts per vLLM call, and per output flush (default: 512)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel degree, i.e. GPUs per engine (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="fraction of GPU memory vLLM may claim (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM context length; VN lines are short (default: 2048)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="vLLM weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="generation cap; the answer is one word (default: 8)",
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
        help="label N clips drawn uniformly at random; must match the audio run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "seed for --sample and for the engine (default: 0). Must match the "
            "audio labeller's --seed"
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=(
            "abort after this many batches fail in a row, which indicates a "
            "systemic fault rather than bad input (default: "
            f"{DEFAULT_MAX_CONSECUTIVE_FAILURES})"
        ),
    )
    parser.add_argument(
        "--max-unparsed-frac",
        type=float,
        default=DEFAULT_MAX_UNPARSED_FRAC,
        help=(
            "fail the run when more than this fraction of responses do not "
            "parse; an unparsed response masks its clip, so a high rate means "
            f"the run supervised nothing (default: {DEFAULT_MAX_UNPARSED_FRAC})"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="log progress to stderr every N clips; 0 disables (default: 1000)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Label a manifest's transcripts with the LLM and append the results.

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
    if not 0.0 <= args.max_unparsed_frac <= 1.0:
        print("--max-unparsed-frac must be between 0 and 1", file=sys.stderr)
        return 1

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

    labeller = VllmLabeller(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    progress = Progress(len(pending), args.progress_every)
    print(f"labelling {len(pending)} transcripts with {args.model}", file=sys.stderr)

    unparsed = 0
    rows_written = 0
    batches_skipped = 0
    consecutive_failures = 0
    aborted = False

    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            try:
                records = labeller.label([clip.target for clip in batch])
            except RuntimeError as exc:
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
                    # See the audio labeller: an isolated bad batch is data, a
                    # run of them is a systemic fault that would otherwise walk
                    # the whole corpus and exit 0 with an empty output file.
                    print(
                        f"aborting: {consecutive_failures} consecutive batch failures. "
                        "That is a systemic fault (missing chat template, corrupt "
                        "staged model, engine out of memory), not bad data. Fix the "
                        "cause and rerun -- the rows already written are kept and "
                        "will be skipped on resume.",
                        file=sys.stderr,
                    )
                    aborted = True
                    break
                progress.advance(len(batch))
                continue

            consecutive_failures = 0
            for clip, record in zip(batch, records):
                if record["label"] is None:
                    unparsed += 1
                handle.write(
                    json.dumps(
                        {
                            "key": clip.key,
                            "label": record["label"],
                            "score": record["score"],
                            "raw": record["raw"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows_written += 1
            handle.flush()
            progress.advance(len(batch))

    progress.log()
    unparsed_frac = (unparsed / rows_written) if rows_written else 0.0
    print(f"  rows written    : {rows_written}/{len(pending)}", file=sys.stderr)
    print(f"  batches skipped : {batches_skipped}", file=sys.stderr)
    print(
        f"  unparsed        : {unparsed}/{rows_written} ({unparsed_frac:.1%})",
        file=sys.stderr,
    )
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
    if unparsed_frac > args.max_unparsed_frac:
        # This is the failure mode with no other symptom. A broken prompt or a
        # missing chat template yields label=null for every line; the merge then
        # correctly masks all of them, so no *wrong* label is ever produced and
        # nothing downstream complains -- the emotion head simply gets no
        # supervision from the entire corpus, exactly as in rounds 1 and 2.
        print(
            f"aborting: {unparsed_frac:.1%} of responses did not parse, above "
            f"--max-unparsed-frac {args.max_unparsed_frac:.2f}. Every unparsed "
            "response becomes a masked clip, so this run would contribute no "
            "emotion supervision at all. Inspect the 'raw' field in the output "
            "and check the prompt and the model's chat template.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
