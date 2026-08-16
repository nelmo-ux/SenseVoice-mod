"""Unit tests for ``scripts/extract_weights.py`` -- the shape of what it writes.

The script's job is to turn a ~2.6 GiB trainer checkpoint into a file that can
stand in for ``models/SenseVoiceSmall/model.pt``.  Getting the *tensors* right
is not enough for that, and the tensors were never the part that broke: an
earlier revision extracted all 917 of them correctly and then saved them as
``{"state_dict": <the 917 tensors>}``.  The published ``model.pt`` is a flat
``OrderedDict`` mapping parameter name to tensor, so the wrapped file was a
one-key dict that matches no parameter at all -- while the script's own summary
reported ``tensors kept : 917`` and a healthy size reduction, because it
described the extraction and never the file.

Everything here therefore asserts against the *loaded artifact*, never against
what the script says it did:

Container
    :func:`test_output_is_a_flat_state_dict` and
    :func:`test_output_loads_into_a_module_with_all_keys_matched` pin the flat
    shape from both ends -- structurally (no wrapper key, values are tensors)
    and behaviourally (``load_state_dict(..., strict=True)`` matches every key,
    which is precisely what the wrapped form could not do).

``OrderedDict`` and key order
    The published file is an ``OrderedDict``; a copy that degrades to a plain
    ``dict``, or reorders, is no longer byte-comparable with it.

Summary
    :func:`test_summary_states_the_output_shape` exists because the missing
    shape line is *why* the wrapper survived review.  It fails if the summary
    goes back to describing only the extraction.

The checkpoints are built in ``tmp_path`` from a four-parameter
:class:`TinyModel`; nothing here reads a real checkpoint or the 893 MiB model.
"""

import importlib.util
import sys
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

SCRIPT_PATH = ROOT / "scripts" / "extract_weights.py"


def _load_extract_weights():
    """Import ``scripts/extract_weights.py`` by path.

    ``scripts/`` is not a package, so there is nothing to ``import``.  The
    module is registered in ``sys.modules`` before ``exec_module`` runs, for the
    same reason ``tests/test_vn_data_prep.py`` does it.
    """
    spec = importlib.util.spec_from_file_location("extract_weights", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ew = _load_extract_weights() if SCRIPT_PATH.is_file() else None

pytestmark = pytest.mark.skipif(
    ew is None, reason=f"{SCRIPT_PATH} does not exist yet"
)


# --- fixtures & helpers ------------------------------------------------------


class TinyModel(torch.nn.Module):
    """Four parameters under dotted names, standing in for SenseVoiceSmall.

    Its ``state_dict()`` is a real ``OrderedDict`` of ``str -> Tensor`` with
    submodule-qualified keys, which is the only property of the 917-tensor
    original these tests depend on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.ctc = torch.nn.Linear(2, 4)


def write_checkpoint(path: Path, weights: Mapping, key: str = "state_dict") -> Path:
    """Write a trainer-shaped checkpoint: weights beside the resume state.

    Mirrors what funasr's trainer saves -- the weights under ``state_dict``,
    next to optimizer/scheduler/epoch entries that this script must drop.
    """
    payload = {
        key: weights,
        "optimizer": {"state": {0: {"exp_avg": torch.zeros(2, 3)}}},
        "scheduler": {"last_epoch": 4},
        "epoch": 4,
    }
    torch.save(payload, path)
    return path


def extract(tmp_path: Path, weights: Mapping, key: str = "state_dict"):
    """Run the script end to end; return what came back off disk.

    Returns ``(exit_code, output_path, loaded_object)``.  The output is read
    back with ``torch.load`` rather than inspected in memory, because the defect
    this file guards against lived entirely in the save call.
    """
    checkpoint = write_checkpoint(tmp_path / "model.pt.ep4", weights, key=key)
    output = tmp_path / "weights_ep4.pt"
    code = ew.main([str(checkpoint), str(output)])
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    return code, output, loaded


@pytest.fixture
def weights():
    """The reference weights, as an ``OrderedDict`` like a real state dict."""
    return TinyModel().state_dict()


# --- the flat container ------------------------------------------------------


def test_output_is_a_flat_state_dict(tmp_path, weights):
    """No wrapper key: the top level *is* the parameter-name -> tensor mapping."""
    code, _, loaded = extract(tmp_path, weights)

    assert code == 0
    assert isinstance(loaded, Mapping)
    # The regression itself: the old revision wrote exactly {"state_dict": ...},
    # a one-key dict whose single value was the mapping below.
    assert "state_dict" not in loaded
    assert not any(isinstance(value, Mapping) for value in loaded.values())

    assert set(loaded) == set(weights)
    assert all(isinstance(value, torch.Tensor) for value in loaded.values())
    for name, tensor in weights.items():
        assert torch.equal(loaded[name], tensor)


def test_output_carries_no_optimizer_or_scheduler_state(tmp_path, weights):
    """The resume state is dropped, not just pushed down a level."""
    _, _, loaded = extract(tmp_path, weights)

    # len() is the whole assertion: 4 is TinyModel's parameter count, so any
    # surviving wrapper/optimizer/epoch entry shows up as an extra top-level key.
    assert len(loaded) == 4
    assert {"optimizer", "scheduler", "epoch"}.isdisjoint(loaded)


def test_output_loads_into_a_module_with_all_keys_matched(tmp_path, weights):
    """The behavioural half: it is a drop-in for a ``model.pt``.

    ``strict=True`` is what funasr's own load reports as ``<All keys matched
    successfully>``.  The wrapped form fails here with one unexpected key
    (``state_dict``) and every real parameter missing.
    """
    _, _, loaded = extract(tmp_path, weights)

    result = TinyModel().load_state_dict(loaded, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_output_preserves_ordereddict_and_key_order(tmp_path):
    """``models/SenseVoiceSmall/model.pt`` is an ``OrderedDict``; so is this."""
    # Deliberately not the module's natural order, and not sorted, so neither a
    # plain-dict rebuild nor an accidental sort can pass by coincidence.
    natural = TinyModel().state_dict()
    shuffled = OrderedDict(
        (name, natural[name])
        for name in ("ctc.weight", "encoder.bias", "ctc.bias", "encoder.weight")
    )

    _, _, loaded = extract(tmp_path, shuffled)

    assert isinstance(loaded, OrderedDict)
    assert list(loaded) == list(shuffled)


def test_bare_state_dict_checkpoint_is_not_wrapped(tmp_path, weights):
    """A checkpoint that is already flat stays flat rather than gaining a level."""
    checkpoint = tmp_path / "bare.pt"
    torch.save(weights, checkpoint)
    output = tmp_path / "weights_bare.pt"

    assert ew.main([str(checkpoint), str(output)]) == 0

    loaded = torch.load(output, map_location="cpu", weights_only=True)
    assert "state_dict" not in loaded
    assert set(loaded) == set(weights)


def test_ddp_prefixed_checkpoint_is_stripped_and_flat(tmp_path, weights):
    """The two transforms compose: ``module.`` goes, and so does the wrapper."""
    prefixed = OrderedDict((f"module.{name}", t) for name, t in weights.items())

    _, _, loaded = extract(tmp_path, prefixed)

    assert "state_dict" not in loaded
    assert set(loaded) == set(weights)
    assert not any(name.startswith("module.") for name in loaded)


# --- the summary -------------------------------------------------------------


def test_summary_states_the_output_shape(tmp_path, weights, capsys):
    """The summary must describe the *file*, not only the extraction.

    Reported shape is what a reader checks against the published model (917
    top-level tensors / 233,999,167 parameters); a summary that only reports
    "tensors kept" is what let the wrapped output ship.
    """
    extract(tmp_path, weights)

    summary = capsys.readouterr().out
    shape_lines = [line for line in summary.splitlines() if "shape" in line]
    assert shape_lines, f"no output-shape line in the summary:\n{summary}"

    shape = shape_lines[0]
    assert "flat" in shape
    # TinyModel: 4 top-level tensors, 3*2 + 2 + 2*4 + 4 = 20 parameters.
    assert "4 top-level tensors" in shape
    assert "20 parameters" in shape
