#!/usr/bin/env python3
"""Copy the model weights out of a training checkpoint, dropping the rest.

::

    .venv/bin/python scripts/extract_weights.py \\
        outputs/chunk_mps_run1/model.pt.ep4 \\
        outputs/chunk_mps_run1/weights_ep4.pt

Why
---

funasr's trainer writes ``{"state_dict", "optimizer", "scheduler",
"scaler_state", ...}``, so a checkpoint on disk is ~2.62 GiB of which only
~893 MiB is the model.  The remainder exists to *resume* training and is dead
weight for anything else, which matters twice:

*   **Archiving epochs on the cluster.**  Disk quota is the binding constraint
    and the trainer prunes old checkpoints on its own schedule; a weights-only
    copy parked outside that path keeps an epoch around for later comparison at
    a third of the cost.
*   **Pulling the best checkpoint to a laptop.**  Three times less to download,
    and the optimizer moments were never going to be used there.

Output shape
------------

The result is written **flat**: a bare mapping of parameter name to tensor,
byte-for-byte the same shape as the published ``models/SenseVoiceSmall/
model.pt`` (an ``OrderedDict`` of 917 tensors / 233,999,167 parameters), so it
is a drop-in replacement for that file.

An earlier revision wrapped the weights as ``{"state_dict": ...}``.  That form
happens to survive ``funasr.train_utils.load_pretrained_model``, which unwraps
``state_dict`` / ``model_state_dict`` / ``model``, so ``--init_param`` and
``scripts/eval_chunk_gap.py --checkpoint`` kept working - but anything that
loads a ``model.pt`` the way funasr's own model directory is loaded sees a
one-key dict and matches no parameter at all.  Since the point of the script is
to produce a stand-in for ``model.pt``, the container now matches it, and the
extra nesting the loader would have tolerated is simply not created.  The
source container is preserved, so an ``OrderedDict`` in gives an
``OrderedDict`` out with its key order intact.

DDP prefixes
------------

Under ``torchrun`` the trained object is a ``DistributedDataParallel``, whose
``state_dict()`` prefixes every key with ``module.``.  funasr unwraps ``.module``
before saving on most paths, but not all of them, and inference builds a bare
``SenseVoiceSmall`` that expects unprefixed keys.  The prefix is therefore
stripped here rather than left as a trap at load time.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import pickle
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

__all__ = ["main"]

#: Keys a checkpoint may hide its weights under, in the order
#: ``funasr.train_utils.load_pretrained_model`` tries them, so that anything
#: this script accepts is also something the loader would have accepted.
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")

#: Prefix ``DistributedDataParallel.state_dict()`` adds to every key.
_DDP_PREFIX = "module."


#: Exception types a *safe-load rejection* has been observed to surface as.
#: torch does not promise one: 2.6 raises ``pickle.UnpicklingError`` ("Weights
#: only load failed"), earlier versions raise ``RuntimeError`` and the
#: restricted unpickler itself can raise ``AttributeError: Unsupported global``.
#: Catching only ``UnpicklingError`` therefore missed the funasr checkpoints -
#: the exact files the unrestricted fallback exists for.
_SAFE_LOAD_REJECTIONS = (pickle.UnpicklingError, RuntimeError, AttributeError)

#: Substrings that identify one of the above as "the restricted unpickler
#: refused a *named object*" rather than "this file is corrupt".  All three
#: types are also what a truncated or garbage checkpoint raises, and re-reading
#: *that* unrestricted would both fail again and needlessly execute whatever the
#: broken file contains.  Matched case-insensitively.
#:
#: Deliberately none of the generic wrapper text: torch 2.6 prefixes every
#: safe-load failure with "Weights only load failed" and a paragraph about the
#: ``weights_only`` argument, including for a file of pure garbage (which fails
#: at the opcode level with "Unsupported operand 110").  Only the specific
#: cause - a global/class the allowlist does not cover, which is exactly what
#: funasr's scheduler and scaler state trip - distinguishes the two.
_SAFE_LOAD_REJECTION_MARKERS = (
    "unsupported global",
    "unsupported class",
    "not an allowed global",
    "add_safe_globals",
    "safe_globals",
)

#: Leading bytes of a zip-format (torch >= 1.6) checkpoint.  ``mmap=True``
#: only works on these; the legacy tar-ish format must be read normally.
_ZIP_MAGIC = b"PK\x03\x04"


class MalformedCheckpoint(Exception):
    """A file loaded fine but holds nothing that looks like model weights."""


# ---------------------------------------------------------------------- loading


def _torch_load_accepts(name: str) -> bool:
    """Whether the installed ``torch.load`` takes a named parameter.

    Real feature detection, replacing a ``except TypeError`` around the call.
    That ``except`` could not tell "this torch has no such parameter" from a
    ``TypeError`` raised deep inside deserialising a corrupt file, and answered
    both by retrying with unrestricted unpickling.

    A ``**kwargs`` catch-all deliberately does not count: old ``torch.load``
    signatures end in ``**pickle_load_args``, which is forwarded to
    ``Unpickler`` and raises there rather than being honoured.

    Args:
        name: Parameter to look for, e.g. ``"weights_only"`` or ``"mmap"``.

    Returns:
        ``True`` if it is an explicitly named parameter.
    """
    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _is_zip_checkpoint(path: Path) -> bool:
    """Whether the file is a zip-format checkpoint, so ``mmap=True`` applies.

    Args:
        path: The checkpoint to sniff.

    Returns:
        ``True`` for the zip container torch has written since 1.6.  Any read
        error answers ``False``: the caller only uses this to *enable* an
        optimisation, and a file that cannot be opened will report its real
        error from ``torch.load`` a moment later.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(len(_ZIP_MAGIC)) == _ZIP_MAGIC
    except OSError:
        return False


def _is_safe_load_rejection(exc: BaseException) -> bool:
    """Whether an exception means "the restricted unpickler refused an object".

    Args:
        exc: What ``torch.load(..., weights_only=True)`` raised.

    Returns:
        ``True`` if the message carries one of
        :data:`_SAFE_LOAD_REJECTION_MARKERS`, i.e. re-reading unrestricted is
        likely to succeed.  ``False`` keeps a corrupt-file error distinguishable
        so it propagates with its original message instead of being retried.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _SAFE_LOAD_REJECTION_MARKERS)


def _load_options(path: Path) -> Dict[str, Any]:
    """Keyword arguments for reading ``path`` with the smallest footprint.

    ``mmap=True`` is the point of this: without it ``torch.load`` materialises
    the whole ~2.6 GiB checkpoint - optimizer moments, scheduler and scaler
    state included - before this script discards all but the weights, which is
    the run-time peak that can OOM a memory-capped login node.  Mapped storages
    are paged in on demand, so the optimizer state is never resident.

    Args:
        path: The checkpoint to read.

    Returns:
        Options accepted by the installed torch, always including
        ``map_location``.
    """
    options: Dict[str, Any] = {"map_location": "cpu"}
    if _torch_load_accepts("mmap") and _is_zip_checkpoint(path):
        options["mmap"] = True
    return options


def load_checkpoint(path: Path) -> Any:
    """Read a checkpoint onto the CPU.

    ``weights_only=True`` is used where it exists: it refuses to execute the
    arbitrary code a pickle can carry, which is the right default for a file
    that may have come off a shared cluster.  Two things make it fall back:

    *   torch is old enough not to have the parameter at all, detected from the
        signature rather than by catching the resulting ``TypeError``;
    *   the checkpoint carries a non-tensor object the restricted unpickler
        will not build - scheduler and scaler state legitimately do - in which
        case the file is re-read unrestricted and a warning says so, because
        refusing to extract weights from the user's own training output would
        be the less useful failure.  Only errors that *look* like that
        rejection fall back; a corrupt file propagates as itself.

    Args:
        path: The checkpoint to read.

    Returns:
        Whatever the file contained, tensors on the CPU.
    """
    options = _load_options(path)

    if not _torch_load_accepts("weights_only"):
        # torch predates the parameter; there is no safe mode to ask for.
        return torch.load(path, **options)

    try:
        return torch.load(path, weights_only=True, **options)
    except _SAFE_LOAD_REJECTIONS as exc:
        if not _is_safe_load_rejection(exc):
            raise
        print(
            f"[warn] {path}: safe load rejected this checkpoint ({exc}); "
            "re-reading with weights_only=False, which executes pickled code - "
            "only do this for checkpoints you produced",
            file=sys.stderr,
        )
        return torch.load(path, weights_only=False, **options)


def _is_tensor_mapping(value: Any) -> bool:
    """Whether a value looks like a state dict.

    Args:
        value: Candidate object pulled out of a checkpoint.

    Returns:
        ``True`` for a non-empty mapping of ``str`` to tensor.  Buffers saved as
        Python scalars are rare in this model but would fail this test, which is
        the conservative direction: a false negative names the key it rejected,
        a false positive writes a file that only breaks at load time.
    """
    if not isinstance(value, Mapping) or not value:
        return False
    return all(
        isinstance(key, str) and isinstance(item, torch.Tensor)
        for key, item in value.items()
    )


def _copy_like(
    source: Mapping[str, torch.Tensor],
    items: Iterable[Tuple[str, torch.Tensor]] = (),
) -> Dict[str, torch.Tensor]:
    """A fresh mapping of the same container type as ``source``.

    The published ``models/SenseVoiceSmall/model.pt`` is an ``OrderedDict``, and
    the file this script writes is meant to stand in for it, so the container
    type and the key order that comes with it are carried through rather than
    flattened to a plain ``dict`` on the way past.

    Args:
        source: The mapping whose type to mirror.
        items: Pairs to fill the copy with; empty for an empty container the
            caller fills itself.

    Returns:
        An ``OrderedDict`` if ``source`` is one, otherwise a ``dict``.  (Both
        preserve insertion order on every supported Python; the distinction kept
        here is the type ``torch.save`` records.)
    """
    factory = OrderedDict if isinstance(source, OrderedDict) else dict
    return factory(items)


def extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    """Find the model weights inside a loaded checkpoint.

    Handles both shapes seen in this project: the trainer's wrapper dict, where
    the weights sit under one of :data:`_STATE_DICT_KEYS` next to the optimizer
    state, and a bare state dict saved by hand.

    Args:
        checkpoint: The object :func:`load_checkpoint` returned.

    Returns:
        ``(weights, source)`` where ``source`` names the key the weights came
        from, or ``"<bare state_dict>"``.  The weights keep the container type
        and key order they had in the file (see :func:`_copy_like`).

    Raises:
        MalformedCheckpoint: If nothing in the file looks like model weights.
    """
    if not isinstance(checkpoint, Mapping):
        raise MalformedCheckpoint(
            f"expected a dict at the top level, got {type(checkpoint).__name__}"
        )

    for key in _STATE_DICT_KEYS:
        if key in checkpoint:
            if not _is_tensor_mapping(checkpoint[key]):
                raise MalformedCheckpoint(
                    f"{key!r} is not a mapping of names to tensors "
                    f"(got {type(checkpoint[key]).__name__})"
                )
            weights = checkpoint[key]
            return _copy_like(weights, weights.items()), key

    if _is_tensor_mapping(checkpoint):
        return _copy_like(checkpoint, checkpoint.items()), "<bare state_dict>"

    raise MalformedCheckpoint(
        "no model weights found: the file has none of "
        f"{_STATE_DICT_KEYS} and is not itself a state dict "
        f"(top-level keys: {sorted(map(str, checkpoint))[:10]})"
    )


def strip_ddp_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Drop the ``module.`` prefix ``DistributedDataParallel`` adds.

    Args:
        state_dict: Weights, possibly prefixed.

    Returns:
        ``(weights, stripped)``; ``stripped`` is how many keys were renamed, so
        the caller can report that it happened.  The container type and key
        order are preserved.

    Raises:
        MalformedCheckpoint: If stripping would collide two keys, which means
            the checkpoint mixes wrapped and unwrapped names and no rename is
            safe.
    """
    prefixed = [key for key in state_dict if key.startswith(_DDP_PREFIX)]
    if not prefixed:
        return state_dict, 0

    stripped: Dict[str, torch.Tensor] = _copy_like(state_dict)
    for key, tensor in state_dict.items():
        name = key[len(_DDP_PREFIX) :] if key.startswith(_DDP_PREFIX) else key
        if name in stripped:
            raise MalformedCheckpoint(
                f"stripping {_DDP_PREFIX!r} would collide on {name!r}: the "
                "checkpoint carries both the prefixed and unprefixed key"
            )
        stripped[name] = tensor
    return stripped, len(prefixed)


# ----------------------------------------------------------------------- report


def _mib(num_bytes: int) -> str:
    """Format a byte count for the summary.

    Args:
        num_bytes: The count.

    Returns:
        The count in MiB to one decimal place.
    """
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def print_summary(
    source: str,
    input_path: Path,
    output_path: Path,
    state_dict: Mapping[str, torch.Tensor],
    num_stripped: int,
) -> None:
    """Report what was written, in what shape, and what it saved.

    The ``output shape`` line exists because the size lines alone cannot catch a
    wrong container: a revision that wrapped the weights as ``{"state_dict":
    ...}`` wrote a file of the right size, full of the right tensors, that no
    plain ``model.pt`` consumer could load - and every line of this summary
    still read as a success.  The shape line is stated in terms a reader can
    compare against the published model (``top-level tensors`` and
    ``parameters`` both match ``models/SenseVoiceSmall/model.pt``).

    Args:
        source: Where in the checkpoint the weights were found.
        input_path: The checkpoint read.
        output_path: The file written.
        state_dict: Exactly what was written, for its shape and size.
        num_stripped: How many keys lost a DDP prefix.
    """
    input_bytes = input_path.stat().st_size
    output_bytes = output_path.stat().st_size
    ratio = output_bytes / input_bytes if input_bytes else 0.0
    num_parameters = sum(tensor.numel() for tensor in state_dict.values())

    print(f"weights source : {source}")
    print(f"tensors kept   : {len(state_dict)}")
    print(
        f"output shape   : flat {type(state_dict).__name__} of "
        f"{len(state_dict)} top-level tensors "
        f"({num_parameters:,} parameters), no wrapper key"
    )
    if num_stripped:
        print(f"ddp prefix     : stripped {_DDP_PREFIX!r} from {num_stripped} keys")
    print(f"input          : {input_path}  ({input_bytes} B, {_mib(input_bytes)})")
    print(f"output         : {output_path}  ({output_bytes} B, {_mib(output_bytes)})")
    print(
        f"reduction      : {ratio:.3f} of the original "
        f"({_mib(input_bytes - output_bytes)} saved)"
    )


# ------------------------------------------------------------------------- main


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
    parser.add_argument("checkpoint", type=Path, help="training checkpoint to read")
    parser.add_argument("output", type=Path, help="weights-only file to write")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace an existing output file; without this an existing path is "
            "refused, so a batch archiving loop cannot silently eat an epoch"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Extract the weights.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a missing, refused or
        malformed file.
    """
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if args.output.exists() and not args.overwrite:
        print(
            f"output already exists: {args.output} (pass --overwrite to replace)",
            file=sys.stderr,
        )
        return 1
    if args.output.resolve() == args.checkpoint.resolve():
        print("refusing to write the weights over the checkpoint itself", file=sys.stderr)
        return 1

    try:
        checkpoint = load_checkpoint(args.checkpoint)
    except Exception as exc:  # noqa: BLE001 - re-reported with the filename
        print(f"could not read {args.checkpoint}: {exc}", file=sys.stderr)
        return 1

    try:
        state_dict, source = extract_state_dict(checkpoint)
        state_dict, num_stripped = strip_ddp_prefix(state_dict)
    except MalformedCheckpoint as exc:
        print(f"malformed checkpoint {args.checkpoint}: {exc}", file=sys.stderr)
        return 1
    finally:
        # Release the optimizer and scheduler tensors before writing, so peak
        # memory is one model rather than a whole checkpoint plus a copy of its
        # weights.  ``state_dict`` holds the tensors it kept alive by itself.
        del checkpoint
        gc.collect()

    # Written beside the target and renamed into place: ``torch.save`` running
    # out of quota half way through would otherwise leave a truncated file that
    # looks like a finished extraction.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    try:
        torch.save(state_dict, partial)
        partial.replace(args.output)
    except Exception as exc:  # noqa: BLE001 - re-reported with the filename
        partial.unlink(missing_ok=True)
        print(f"could not write {args.output}: {exc}", file=sys.stderr)
        return 1

    print_summary(source, args.checkpoint, args.output, state_dict, num_stripped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
