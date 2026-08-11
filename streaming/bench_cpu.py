#!/usr/bin/env python3
"""Measure SenseVoiceSmall encoder latency on CPU to size the streaming history.

Streaming decoding re-runs the encoder over ``[4 query frames] + [T history
frames]`` every time a new chunk arrives, so the wall-clock cost of one encoder
pass is what caps ``max_history``.  This script measures that cost directly.

Latency budget (defaults, all tunable from the CLI)::

    total budget            1500 ms
    - chunk accumulation     480 ms
    = available for inference 1020 ms
    x safety factor           0.7  -> 714 ms recommended ceiling

The script reports, for each history length ``T``:

* median / mean / p95 encoder wall time
* the largest ``T`` whose p95 fits in the inference budget
* the recommended ``max_history`` using the safety factor

One encoder frame is 60 ms (10 ms frame shift x LFR n=6), so ``T`` maps
directly to seconds of retained audio.

Scope
-----
This is a **development tool, not part of the package's public API**: nothing
in ``streaming/`` imports it, and it exposes no API meant to be called from
other modules - only a ``main()`` behind ``__main__``.  It lives inside the
package because what it measures is the package's own tuning parameter
(``StreamingConfig.max_history``), and it should be re-run whenever that
default is revisited on new hardware.  For the same reason it deliberately
keeps its own copies of ``MS_PER_FRAME`` / ``NUM_QUERY_FRAMES`` and imports
nothing from its siblings: it must stay runnable as a plain script
(``python streaming/bench_cpu.py``), which relative imports would break.

Usage
-----
    ./.venv/bin/python streaming/bench_cpu.py
    ./.venv/bin/python -m streaming.bench_cpu --frames 17,50,100 --json b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: One encoder frame after LFR: frame_shift(10 ms) * lfr_n(6).
MS_PER_FRAME = 60.0

#: The 4 prompt embeddings prepended in ``SenseVoiceSmall.inference``:
#: language, event, emotion, textnorm.
NUM_QUERY_FRAMES = 4

DEFAULT_FRAMES: Tuple[int, ...] = (17, 50, 83, 100, 167, 250, 333, 500)
DEFAULT_TOTAL_BUDGET_MS = 1500.0
DEFAULT_CHUNK_MS = 480.0
DEFAULT_SAFETY_FACTOR = 0.7


@dataclass
class FrameResult:
    """Timing statistics for one history length."""

    frames: int
    total_frames: int
    audio_sec: float
    median_ms: float
    mean_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples_ms: List[float] = field(default_factory=list)


@dataclass
class Recommendation:
    """Largest history length that fits a latency ceiling."""

    label: str
    budget_ms: float
    max_measured_frames: Optional[int]
    max_measured_audio_sec: Optional[float]
    interpolated_frames: Optional[int]
    note: str


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_model(model_id: str, device: str) -> Tuple[Any, Dict[str, Any]]:
    """Load SenseVoiceSmall in eval mode.

    Returns ``(model, build_kwargs)``.  ``build_kwargs`` carries ``frontend``
    and ``tokenizer`` as produced by ``AutoModel.build_model``.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from model import SenseVoiceSmall  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit(
            f"could not import SenseVoiceSmall from {REPO_ROOT}/model.py ({exc}).\n"
            "torch and funasr must be installed in the active interpreter."
        ) from exc

    try:
        model, kwargs = SenseVoiceSmall.from_pretrained(model=model_id, device=device)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"failed to load '{model_id}': {type(exc).__name__}: {exc}\n"
            "The first run downloads the weights - check network access and the "
            "modelscope/huggingface cache."
        ) from exc

    model = model.to(device)
    model.eval()
    return model, kwargs


def resolve_input_size(model: Any, kwargs: Dict[str, Any]) -> int:
    """Determine the encoder input dimension (LFR-stacked mel bins).

    ``model.embed`` is ``Embedding(vocab, input_size)`` where ``input_size`` is
    exactly the encoder input dim, so it is the most reliable source.  The
    frontend config is used as a fallback.
    """
    embed = getattr(model, "embed", None)
    if embed is not None and getattr(embed, "embedding_dim", None):
        return int(embed.embedding_dim)

    frontend = kwargs.get("frontend")
    if frontend is not None:
        n_mels = int(getattr(frontend, "n_mels", 80))
        lfr_m = int(getattr(frontend, "lfr_m", 7))
        return n_mels * lfr_m
    raise SystemExit("could not determine the encoder input size from the model")


# --------------------------------------------------------------------------- #
# benchmarking
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]) without numpy."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bench_frames(
    model: Any,
    input_size: int,
    frames: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> FrameResult:
    """Time ``model.encoder`` over ``NUM_QUERY_FRAMES + frames`` inputs.

    A fresh tensor is used for every call: ``SenseVoiceEncoderSmall.forward``
    scales its input in-place (``xs_pad *= output_size ** 0.5``), so reusing one
    buffer would keep multiplying the same memory and change what is measured.
    The clone happens outside the timed region.
    """
    import statistics  # noqa: WPS433
    import time  # noqa: WPS433

    import torch  # noqa: WPS433

    total_frames = NUM_QUERY_FRAMES + frames
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(1, total_frames, input_size, generator=generator)
    lengths = torch.tensor([total_frames], dtype=torch.int32)

    timings: List[float] = []
    with torch.inference_mode():
        for iteration in range(warmup + repeats):
            speech = base.clone()
            start = time.perf_counter()
            model.encoder(speech, lengths)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if iteration >= warmup:
                timings.append(elapsed_ms)

    return FrameResult(
        frames=frames,
        total_frames=total_frames,
        audio_sec=round(frames * MS_PER_FRAME / 1000.0, 3),
        median_ms=round(statistics.median(timings), 2),
        mean_ms=round(statistics.fmean(timings), 2),
        p95_ms=round(percentile(timings, 95.0), 2),
        min_ms=round(min(timings), 2),
        max_ms=round(max(timings), 2),
        samples_ms=[round(value, 3) for value in timings],
    )


def recommend(
    results: Sequence[FrameResult],
    budget_ms: float,
    label: str,
) -> Recommendation:
    """Find the largest history length whose p95 stays under *budget_ms*."""
    ordered = sorted(results, key=lambda item: item.frames)
    fitting = [item for item in ordered if item.p95_ms <= budget_ms]

    if not fitting:
        return Recommendation(
            label=label,
            budget_ms=round(budget_ms, 1),
            max_measured_frames=None,
            max_measured_audio_sec=None,
            interpolated_frames=None,
            note=(
                f"even the smallest measured T={ordered[0].frames} "
                f"({ordered[0].p95_ms:.1f} ms p95) exceeds the budget"
            ),
        )

    best = fitting[-1]
    interpolated: Optional[int] = None
    note = f"largest measured T within budget (p95 {best.p95_ms:.1f} ms)"

    remaining = [item for item in ordered if item.frames > best.frames]
    if remaining:
        nxt = remaining[0]
        span_ms = nxt.p95_ms - best.p95_ms
        if span_ms > 0:
            ratio = (budget_ms - best.p95_ms) / span_ms
            interpolated = int(best.frames + ratio * (nxt.frames - best.frames))
            note = (
                f"interpolated between T={best.frames} ({best.p95_ms:.1f} ms) and "
                f"T={nxt.frames} ({nxt.p95_ms:.1f} ms)"
            )
    else:
        note += "; the budget may allow more than the largest T measured"

    return Recommendation(
        label=label,
        budget_ms=round(budget_ms, 1),
        max_measured_frames=best.frames,
        max_measured_audio_sec=best.audio_sec,
        interpolated_frames=interpolated,
        note=note,
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_table(results: Sequence[FrameResult]) -> None:
    """Print the timing table."""
    print()
    print("=" * 78)
    print("ENCODER LATENCY (CPU)")
    print("=" * 78)
    header = (
        f"{'T':>6}  {'audio':>8}  {'median':>10}  {'mean':>10}  "
        f"{'p95':>10}  {'min':>9}  {'max':>9}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.frames:>6}  {item.audio_sec:>7.2f}s  {item.median_ms:>9.2f}ms  "
            f"{item.mean_ms:>9.2f}ms  {item.p95_ms:>9.2f}ms  "
            f"{item.min_ms:>8.2f}ms  {item.max_ms:>8.2f}ms"
        )


def print_budget(
    total_budget_ms: float,
    chunk_ms: float,
    inference_budget_ms: float,
    safety_factor: float,
    hard: Recommendation,
    safe: Recommendation,
) -> None:
    """Print the derived ``max_history`` recommendations."""
    print()
    print("=" * 78)
    print("LATENCY BUDGET")
    print("=" * 78)
    print(f"  total latency budget       : {total_budget_ms:.0f} ms")
    print(f"  chunk accumulation         : {chunk_ms:.0f} ms")
    print(f"  available for inference    : {inference_budget_ms:.0f} ms")
    print(
        f"  safe ceiling (x{safety_factor:.2f})       : "
        f"{inference_budget_ms * safety_factor:.0f} ms"
    )

    for rec in (hard, safe):
        print(f"\n  [{rec.label}] p95 <= {rec.budget_ms:.0f} ms")
        if rec.max_measured_frames is None:
            print(f"    no measured T fits - {rec.note}")
            continue
        print(
            f"    max measured T   : {rec.max_measured_frames} frames "
            f"({rec.max_measured_audio_sec:.2f}s of audio)"
        )
        if rec.interpolated_frames is not None:
            print(
                f"    interpolated T   : ~{rec.interpolated_frames} frames "
                f"(~{rec.interpolated_frames * MS_PER_FRAME / 1000.0:.2f}s)"
            )
        print(f"    note             : {rec.note}")

    if safe.max_measured_frames is not None:
        chosen = safe.interpolated_frames or safe.max_measured_frames
        print()
        print("-" * 78)
        print(
            f"  RECOMMENDED max_history = {chosen} frames "
            f"(~{chosen * MS_PER_FRAME / 1000.0:.1f}s of audio history)"
        )
        print("-" * 78)


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def parse_frames(raw: str) -> List[int]:
    """Parse a comma-separated list of positive frame counts."""
    frames: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not an integer: {token!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"frame count must be positive: {value}")
        frames.append(value)
    if not frames:
        raise argparse.ArgumentTypeError("at least one frame count is required")
    return sorted(set(frames))


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(
        description="Benchmark the SenseVoiceSmall encoder on CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="iic/SenseVoiceSmall", help="model id or local path")
    parser.add_argument("--device", default="cpu", help="torch device (this bench targets CPU)")
    parser.add_argument(
        "--frames",
        type=parse_frames,
        default=list(DEFAULT_FRAMES),
        help="comma-separated encoder history lengths (60 ms per frame)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="untimed warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="timed iterations per length")
    parser.add_argument("--seed", type=int, default=0, help="seed for the dummy features")
    parser.add_argument(
        "--total-budget-ms",
        type=float,
        default=DEFAULT_TOTAL_BUDGET_MS,
        help="end-to-end latency budget",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=DEFAULT_CHUNK_MS,
        help="audio accumulated before each inference",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=DEFAULT_SAFETY_FACTOR,
        help="fraction of the inference budget to actually target",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="override torch.set_num_threads"
    )
    parser.add_argument(
        "--input-size", type=int, default=None, help="override the encoder input dim"
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark and print the recommendation."""
    args = build_parser().parse_args(argv)

    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 < args.safety_factor <= 1.0:
        print("--safety-factor must be in (0, 1]", file=sys.stderr)
        return 2

    try:
        import torch  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        print(f"torch is required: {exc}", file=sys.stderr)
        return 2

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print(f"loading '{args.model}' on {args.device} (first run downloads the weights)")
    model, build_kwargs = load_model(args.model, args.device)
    input_size = args.input_size or resolve_input_size(model, build_kwargs)

    num_threads = torch.get_num_threads()
    print(f"  torch {torch.__version__}  threads={num_threads} "
          f"(interop={torch.get_num_interop_threads()})")
    print(f"  encoder input size : {input_size}")
    print(f"  query frames       : {NUM_QUERY_FRAMES} (language/event/emotion/textnorm)")
    print(f"  warmup={args.warmup}  repeats={args.repeats}")

    results: List[FrameResult] = []
    for frames in args.frames:
        try:
            result = bench_frames(
                model=model,
                input_size=input_size,
                frames=frames,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n[error] T={frames} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        results.append(result)
        print(
            f"  T={frames:>4} ({result.audio_sec:>5.2f}s) -> "
            f"median {result.median_ms:>8.2f} ms, p95 {result.p95_ms:>8.2f} ms"
        )

    inference_budget_ms = args.total_budget_ms - args.chunk_ms
    hard = recommend(results, inference_budget_ms, "hard limit")
    safe = recommend(results, inference_budget_ms * args.safety_factor, "recommended")

    print_table(results)
    print_budget(
        total_budget_ms=args.total_budget_ms,
        chunk_ms=args.chunk_ms,
        inference_budget_ms=inference_budget_ms,
        safety_factor=args.safety_factor,
        hard=hard,
        safe=safe,
    )

    if args.json:
        payload = {
            "model": args.model,
            "device": args.device,
            "torch_version": torch.__version__,
            "num_threads": num_threads,
            "num_interop_threads": torch.get_num_interop_threads(),
            "input_size": input_size,
            "query_frames": NUM_QUERY_FRAMES,
            "ms_per_frame": MS_PER_FRAME,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "total_budget_ms": args.total_budget_ms,
            "chunk_ms": args.chunk_ms,
            "inference_budget_ms": inference_budget_ms,
            "safety_factor": args.safety_factor,
            "results": [asdict(item) for item in results],
            "recommendations": [asdict(hard), asdict(safe)],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nJSON report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
