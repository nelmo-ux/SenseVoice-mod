"""Unit tests for streaming/streaming_model.py.

Loading SenseVoiceSmall costs ~900 MB and several seconds, which is far too
much for the logic under test here: the history cap, the ``end_ms`` clock and
the reset semantics are pure bookkeeping around the model, not the model
itself.  So almost every test drives :class:`FakeRecogniser` - a subclass that
replaces exactly two seams, the streaming frontend
(:meth:`StreamingSenseVoice._extract_frames`) and the encoder pass
(:meth:`StreamingSenseVoice._encode_and_decode`) - and leaves the real
``push_audio`` / ``_apply_history_cap`` / ``_infer_partial`` code paths
running.  Frames carry their own index as their single feature, so a test can
assert *which* frames each encoder pass saw.

The one test that does load the model is opt-in behind
``SENSEVOICE_RUN_MODEL_TESTS=1`` (same switch as
``tests/test_streaming_vad_gate.py``); point ``SENSEVOICE_MODEL_DIR`` at a
local checkout to avoid the download.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from streaming.config import StreamingConfig  # noqa: E402
from streaming.streaming_model import (  # noqa: E402
    StreamingSenseVoice,
    _SegmentState,
)

#: Samples in one encoder frame at the default 16 kHz: 60 ms.
FRAME_SAMPLES = 960


class StubFrontend:
    """Stands in for ``WavFrontendOnline``; only its cache handling is used.

    :meth:`StreamingSenseVoice.reset` is the sole caller in these tests, so the
    stub just records how often the cache was initialised and stamps it, which
    makes "reset handed the frontend a fresh cache" observable.
    """

    def __init__(self) -> None:
        self.init_calls = 0

    def init_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Reset ``cache`` in place, as the real frontend does."""
        self.init_calls += 1
        cache.clear()
        cache["stub"] = self.init_calls
        return cache


class FakeRecogniser(StreamingSenseVoice):
    """:class:`StreamingSenseVoice` with the model and frontend stubbed out.

    ``__init__`` deliberately does not call ``super().__init__``: that is the
    method that loads 900 MB of weights.  Everything the exercised code paths
    read is set up here instead.

    Args:
        config: Streaming configuration to run with.
        decodes: Text returned by successive ``_encode_and_decode`` calls; once
            exhausted, calls fall back to ``"w<n>"``.
        final_raw: Raw text the stubbed full-quality pass returns.
    """

    def __init__(
        self,
        config: StreamingConfig,
        decodes: Sequence[str] = (),
        final_raw: str = "",
    ) -> None:
        self.config = config
        self.config.validate()
        self.stub_frontend = StubFrontend()
        self._online_frontend = self.stub_frontend
        self._decodes = list(decodes)
        self._final_raw = final_raw
        #: Frame ids handed to each encoder pass, in call order.
        self.encoded_windows: List[List[int]] = []
        self._next_frame_id = 0
        self._state = _SegmentState()
        self.reset()

    def _extract_frames(self, samples: "np.ndarray", is_last: bool) -> None:
        """Emit one frame per whole ``FRAME_SAMPLES`` of audio.

        Mirrors the real frontend's contract - frames appear only once a full
        window is buffered - without any signal processing.  Each frame holds
        its own index as its only feature.
        """
        frames = int(samples.size) // FRAME_SAMPLES
        if frames <= 0:
            return
        new = torch.arange(
            self._next_frame_id,
            self._next_frame_id + frames,
            dtype=torch.float32,
        ).unsqueeze(1)
        self._next_frame_id += frames

        state = self._state
        if state.features is None:
            state.features = new
        else:
            state.features = torch.cat((state.features, new), dim=0)
        state.pending_frames += frames
        state.total_frames += frames

    def _encode_and_decode(self, features: Optional["torch.Tensor"]) -> str:
        """Record the window and return the next scripted decode."""
        ids = [] if features is None else [int(v) for v in features[:, 0].tolist()]
        self.encoded_windows.append(ids)
        if not ids:
            return ""
        index = len(self.encoded_windows) - 1
        if index < len(self._decodes):
            return self._decodes[index]
        return f"w{index}"

    def _full_inference(self) -> str:
        """Return the scripted final decode instead of running the model."""
        return self._final_raw


def audio(frames: float, extra_samples: int = 0) -> "np.ndarray":
    """Return silence worth ``frames`` encoder frames plus ``extra_samples``."""
    return np.zeros(int(frames * FRAME_SAMPLES) + extra_samples, dtype=np.float32)


def require_postprocess() -> None:
    """Skip unless funasr is importable.

    The stubs above remove every need for the *model*, but ``_infer_final``
    still runs SenseVoice's ``rich_transcription_postprocess`` on the result -
    a pure text function, yet one that lives in funasr.  Only tests that let a
    ``final`` be emitted need it.
    """
    pytest.importorskip("funasr.utils.postprocess_utils")


@pytest.fixture
def config() -> StreamingConfig:
    """A small configuration: 2-frame chunks, 4 frames of history."""
    return StreamingConfig(chunk_size=2, max_history=4)


# --- emission schedule ------------------------------------------------------


def test_partial_waits_for_a_full_chunk(config):
    model = FakeRecogniser(config)

    assert model.push_audio(audio(1)) == []
    assert [result.type for result in model.push_audio(audio(1))] == ["partial"]


def test_one_partial_per_call_however_many_chunks_arrived(config):
    model = FakeRecogniser(config)

    results = model.push_audio(audio(4))

    assert [result.type for result in results] == ["partial"]


def test_is_last_emits_only_a_final(config):
    require_postprocess()
    model = FakeRecogniser(config, final_raw="done")

    model.push_audio(audio(2))
    results = model.push_audio(audio(2), is_last=True)

    assert [result.type for result in results] == ["final"]


def test_non_1d_input_raises_value_error(config):
    model = FakeRecogniser(config)

    with pytest.raises(ValueError):
        model.push_audio(np.zeros((2, 960), dtype=np.float32))


# --- empty input ------------------------------------------------------------


def test_empty_final_push_with_nothing_accumulated_emits_nothing(config):
    model = FakeRecogniser(config)

    assert model.push_audio(np.zeros(0, dtype=np.float32), is_last=True) == []
    assert model.encoded_windows == []


def test_empty_pushes_alone_never_emit(config):
    model = FakeRecogniser(config)

    for _ in range(3):
        assert model.push_audio(np.zeros(0, dtype=np.float32)) == []
    assert model.encoded_windows == []


def test_sub_frame_audio_then_final_emits_nothing(config):
    """Audio too short to make one frame cannot produce a result."""
    model = FakeRecogniser(config)

    model.push_audio(audio(0, extra_samples=100))
    assert model.push_audio(np.zeros(0, dtype=np.float32), is_last=True) == []


# --- end_ms is a sample clock (regression for the frame/sample mix-up) ------


def test_partial_end_ms_comes_from_samples_not_frames(config):
    """Sub-frame audio must still count: 2 frames + 300 samples > 2 * 60 ms."""
    model = FakeRecogniser(config)
    samples = audio(2, extra_samples=300)

    result = model.push_audio(samples)[0]

    assert result.end_ms == pytest.approx(
        samples.size * 1000.0 / config.sample_rate
    )
    assert result.end_ms > 2 * 60.0  # what a frame-based count would have said


def test_final_end_ms_comes_from_samples(config):
    require_postprocess()
    model = FakeRecogniser(config, final_raw="done")
    samples = audio(2, extra_samples=300)

    model.push_audio(samples)
    result = model.push_audio(np.zeros(0, dtype=np.float32), is_last=True)[0]

    assert result.end_ms == pytest.approx(
        samples.size * 1000.0 / config.sample_rate
    )


def test_partial_and_final_agree_on_end_ms(config):
    """The two paths must not disagree over the very same audio."""
    require_postprocess()
    model = FakeRecogniser(config, final_raw="done")
    samples = audio(2, extra_samples=511)

    partial = model.push_audio(samples)[0]
    final = model.push_audio(np.zeros(0, dtype=np.float32), is_last=True)[0]

    assert final.end_ms == pytest.approx(partial.end_ms)


def test_end_ms_accumulates_across_pushes(config):
    model = FakeRecogniser(config)
    model.push_audio(audio(2))

    result = model.push_audio(audio(2))[0]

    assert result.end_ms == pytest.approx(4 * 60.0)


def test_end_ms_follows_a_non_default_sample_rate():
    """The clock divides by ``config.sample_rate``, not by a hard-coded 16 kHz."""
    config = StreamingConfig(chunk_size=2, max_history=4, sample_rate=8000)
    model = FakeRecogniser(config)

    result = model.push_audio(audio(2))[0]

    assert result.end_ms == pytest.approx(2 * FRAME_SAMPLES * 1000.0 / 8000)


def test_start_ms_is_always_zero(config):
    require_postprocess()
    model = FakeRecogniser(config, final_raw="done")

    partial = model.push_audio(audio(2))[0]
    final = model.push_audio(audio(2), is_last=True)[0]

    assert partial.start_ms == 0.0
    assert final.start_ms == 0.0


# --- history cap ------------------------------------------------------------


def test_history_cap_keeps_the_confirmed_prefix(config):
    """The retired window's text survives in every later partial."""
    model = FakeRecogniser(config, decodes=["A", "B", "C", "D"])

    texts = [model.push_audio(audio(2))[0].text for _ in range(4)]

    # Window 2 ("B") is retired when the 5th and 6th frames overflow the cap,
    # so from then on partials read as "B" + the current window's decode.
    assert texts == ["A", "B", "BC", "BD"]
    assert model._state.confirmed_text == "B"


def test_history_cap_bounds_the_window(config):
    model = FakeRecogniser(config)

    for _ in range(6):
        model.push_audio(audio(2))
        assert model._state.features.shape[0] <= config.max_history


def test_history_cap_drops_exactly_the_decoded_frames(config):
    """Windows tile the stream: no frame is decoded twice and none is skipped."""
    model = FakeRecogniser(config)

    for _ in range(4):
        model.push_audio(audio(2))

    assert model.encoded_windows == [[0, 1], [0, 1, 2, 3], [4, 5], [4, 5, 6, 7]]


def test_a_push_larger_than_the_cap_keeps_the_newest_frames(config):
    model = FakeRecogniser(config)

    model.push_audio(audio(10))

    assert model.encoded_windows == [[6, 7, 8, 9]]
    assert model._state.features.shape[0] == config.max_history


def test_history_cap_preserves_tags_in_the_confirmed_raw_text(config):
    model = FakeRecogniser(config, decodes=["<|zh|>你", "<|zh|>好", "世"])

    for _ in range(3):
        result = model.push_audio(audio(2))[0]

    assert model._state.confirmed_raw == "<|zh|>好"
    assert model._state.confirmed_text == "好"
    assert result.text == "好世"
    assert result.raw_text == "<|zh|>好世"


def test_history_cap_is_a_no_op_below_the_limit(config):
    model = FakeRecogniser(config)

    model.push_audio(audio(2))
    model.push_audio(audio(2))

    assert model._state.confirmed_text == ""
    assert model._state.features.shape[0] == 4


def test_the_full_waveform_survives_the_history_cap(config):
    """Only encoder frames are cut; the final pass still sees all the audio."""
    require_postprocess()
    model = FakeRecogniser(config, final_raw="done")

    for _ in range(4):
        model.push_audio(audio(2))
    result = model.push_audio(np.zeros(0, dtype=np.float32), is_last=True)[0]

    assert model._state.total_samples == 8 * FRAME_SAMPLES
    assert result.end_ms == pytest.approx(8 * 60.0)


# --- reset ------------------------------------------------------------------


def test_reset_clears_every_accumulator(config):
    require_postprocess()
    model = FakeRecogniser(config, decodes=["A", "B", "C"])
    for _ in range(3):
        model.push_audio(audio(2))
    model.push_audio(audio(2), is_last=True)

    model.reset()

    state = model._state
    assert state.features is None
    assert state.decoded_frames == 0
    assert state.last_raw_text == ""
    assert state.confirmed_text == ""
    assert state.confirmed_raw == ""
    assert state.pending_frames == 0
    assert state.total_frames == 0
    assert state.total_samples == 0
    assert state.waveform == []
    assert state.finished is False


def test_reset_hands_the_frontend_a_fresh_cache(config):
    model = FakeRecogniser(config)
    model.push_audio(audio(2))
    first_cache = model._state.frontend_cache

    model.reset()

    assert model._state.frontend_cache is not first_cache
    assert model._state.frontend_cache == {"stub": model.stub_frontend.init_calls}


def test_reset_restarts_the_end_ms_clock(config):
    model = FakeRecogniser(config)
    model.push_audio(audio(4))

    model.reset()
    result = model.push_audio(audio(2))[0]

    assert result.end_ms == pytest.approx(2 * 60.0)


def test_reset_lets_a_second_segment_start_clean(config):
    require_postprocess()
    model = FakeRecogniser(config, final_raw="first")
    model.push_audio(audio(2))
    model.push_audio(np.zeros(0, dtype=np.float32), is_last=True)

    model.reset()

    assert model.push_audio(np.zeros(0, dtype=np.float32), is_last=True) == []


# --- keyword merge in the full-quality pass ---------------------------------


class RecordingModel:
    """Captures the kwargs ``_full_inference`` passes to ``model.inference``."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def inference(self, **kwargs: Any):
        self.calls.append(kwargs)
        return [{"text": "<|zh|>ok"}], {}


def test_explicit_arguments_win_over_the_build_kwargs(config):
    """Build kwargs must never shadow - or collide with - the explicit ones."""
    model = FakeRecogniser(config)
    model.model = RecordingModel()
    # A checkpoint whose config happens to carry these keys used to raise
    # "got multiple values for keyword argument".
    model.kwargs = {
        "language": "en",
        "use_itn": False,
        "key": ["from_config"],
        "fs": 8000,
        "tokenizer": "TOKENIZER",
    }
    model.push_audio(audio(2))

    raw = StreamingSenseVoice._full_inference(model)

    call = model.model.calls[0]
    assert raw == "<|zh|>ok"
    assert call["language"] == config.language
    assert call["use_itn"] == config.use_itn
    assert call["key"] == ["stream"]
    assert call["fs"] == config.sample_rate
    assert call["tokenizer"] == "TOKENIZER"  # untouched build kwarg


# --- real model -------------------------------------------------------------

SAMPLE_WAV = ROOT / "runtime" / "llama.cpp" / "tests" / "sample.wav"

#: Opt-in switch for the end-to-end test.  It is off by default because it
#: loads ~900 MB of weights (downloading them on the first run).
RUN_MODEL_TESTS = os.environ.get("SENSEVOICE_RUN_MODEL_TESTS") == "1"

#: Local checkout to test against; falls back to the ModelScope id.
MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set SENSEVOICE_RUN_MODEL_TESTS=1")
def test_real_model_transcribes_sample_wav():
    """End-to-end check against the real model (loads it on first run)."""
    pytest.importorskip("funasr")
    soundfile = pytest.importorskip("soundfile")
    if not SAMPLE_WAV.exists():
        pytest.skip(f"{SAMPLE_WAV} is not available")

    samples, sample_rate = soundfile.read(
        str(SAMPLE_WAV), dtype="float32", always_2d=True
    )
    config = StreamingConfig(sample_rate=sample_rate)
    model = StreamingSenseVoice(MODEL_DIR, config)

    mono = samples[:, 0]
    step = config.chunk_samples
    results = []
    for offset in range(0, len(mono), step):
        results.extend(model.push_audio(mono[offset : offset + step]))
    results.extend(model.push_audio(np.zeros(0, dtype=np.float32), is_last=True))

    finals = [result for result in results if result.type == "final"]
    assert any(result.type == "partial" for result in results)
    assert len(finals) == 1
    assert finals[0].text == "我想问我在滨海新区有房。"
    assert finals[0].start_ms == 0.0
    assert finals[0].end_ms == pytest.approx(len(mono) * 1000.0 / sample_rate)
