"""Unit tests for streaming/vad_gate.py.

The gate accepts an injected VAD, so every test here drives it with
:class:`StubVad` - a few lines that replay a scripted list of raw fsmn-vad
answers.  That keeps the state machine under test and keeps FunASR and torch
out of the suite; only numpy (the audio container) is required, and it is
guarded with ``importorskip`` like the other ML-adjacent tests here.
"""

import os
import sys
from pathlib import Path
from typing import Any, List, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from streaming.config import StreamingConfig  # noqa: E402
from streaming.vad_gate import VadEvent, VadGate  # noqa: E402


class StubVad:
    """Replays scripted fsmn-vad answers, one per ``generate`` call.

    Args:
        scripted: One ``value`` list per expected call, e.g.
            ``[[[750, -1]], [], [[-1, 4480]]]``.  Calls beyond the script get
            an empty ``value``, which is what a real VAD returns most of the
            time.
    """

    def __init__(self, scripted: Sequence[Sequence[Sequence[float]]] = ()) -> None:
        self.scripted = list(scripted)
        self.calls: List[dict] = []

    def generate(self, **kwargs: Any) -> List[dict]:
        index = len(self.calls)
        self.calls.append(kwargs)
        value = self.scripted[index] if index < len(self.scripted) else []
        return [{"key": f"rand_key_{index}", "value": value}]


@pytest.fixture
def config() -> StreamingConfig:
    return StreamingConfig()


def chunk(config: StreamingConfig, ms: float = 480.0):
    """Return ``ms`` of silence as a float32 mono array."""
    return np.zeros(int(ms * config.sample_rate / 1000), dtype=np.float32)


def kinds(events: Sequence[VadEvent]) -> List[str]:
    return [event.kind for event in events]


# --- sentinel decoding ------------------------------------------------------


def test_open_start_sentinel_emits_single_start(config):
    gate = VadGate(config=config, vad_model=StubVad([[[750, -1]]]))

    events = gate.push(chunk(config))

    assert kinds(events) == ["start"]
    assert events[0].time_ms == 750
    assert gate.in_speech


def test_open_end_sentinel_emits_single_end(config):
    gate = VadGate(config=config, vad_model=StubVad([[[750, -1]], [[-1, 4480]]]))
    gate.push(chunk(config))

    events = gate.push(chunk(config))

    assert kinds(events) == ["end"]
    assert events[0].time_ms == 4480
    assert not gate.in_speech


def test_complete_interval_emits_start_then_end(config):
    gate = VadGate(config=config, vad_model=StubVad([[[100, 900]]]))

    events = gate.push(chunk(config, ms=1000))

    assert kinds(events) == ["start", "end"]
    assert [event.time_ms for event in events] == [100, 900]
    assert not gate.in_speech


def test_empty_value_emits_nothing(config):
    gate = VadGate(config=config, vad_model=StubVad([[]]))

    assert gate.push(chunk(config)) == []
    assert not gate.in_speech


def test_several_intervals_in_one_call_are_all_reported(config):
    gate = VadGate(config=config, vad_model=StubVad([[[100, 900], [1200, -1]]]))

    events = gate.push(chunk(config, ms=1500))

    assert kinds(events) == ["start", "end", "start"]
    assert [event.time_ms for event in events] == [100, 900, 1200]
    assert gate.in_speech


def test_times_are_reported_as_the_vad_stated_them(config):
    """fsmn-vad already counts from the start of the stream; no offset is added."""
    stub = StubVad([[], [], [[750, -1]]])
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))
    gate.push(chunk(config))

    events = gate.push(chunk(config))

    assert events[0].time_ms == 750


# --- contradictory events ---------------------------------------------------


def test_start_while_in_speech_is_suppressed(config):
    gate = VadGate(config=config, vad_model=StubVad([[[750, -1]], [[900, -1]]]))
    gate.push(chunk(config))

    assert gate.push(chunk(config)) == []
    assert gate.in_speech


def test_end_while_not_in_speech_is_suppressed(config):
    gate = VadGate(config=config, vad_model=StubVad([[[-1, 4480]]]))

    assert gate.push(chunk(config)) == []
    assert not gate.in_speech


def test_complete_interval_while_in_speech_only_closes(config):
    gate = VadGate(config=config, vad_model=StubVad([[[750, -1]], [[900, 1800]]]))
    gate.push(chunk(config))

    events = gate.push(chunk(config))

    assert kinds(events) == ["end"]
    assert events[0].time_ms == 1800


def test_malformed_intervals_are_skipped(config):
    gate = VadGate(config=config, vad_model=StubVad([[[-1, -1], [1, 2, 3], [750, -1]]]))

    assert kinds(gate.push(chunk(config))) == ["start"]


# --- final flush ------------------------------------------------------------


def test_is_final_closes_an_open_segment(config):
    gate = VadGate(config=config, vad_model=StubVad([[[750, -1]]]))
    gate.push(chunk(config))

    events = gate.push(chunk(config, ms=520), is_final=True)

    assert kinds(events) == ["end"]
    assert events[0].time_ms == pytest.approx(1000.0)
    assert not gate.in_speech


def test_is_final_emits_nothing_when_not_in_speech(config):
    gate = VadGate(config=config, vad_model=StubVad())

    assert gate.push(chunk(config), is_final=True) == []


def test_is_final_does_not_duplicate_an_end_the_vad_reported(config):
    gate = VadGate(config=config, vad_model=StubVad([[[750, -1]], [[-1, 900]]]))
    gate.push(chunk(config))

    events = gate.push(chunk(config), is_final=True)

    assert kinds(events) == ["end"]
    assert events[0].time_ms == 900


def test_is_final_is_forwarded_to_the_vad(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))
    gate.push(chunk(config), is_final=True)

    assert [call["is_final"] for call in stub.calls] == [False, True]


def test_empty_chunk_still_closes_an_open_segment(config):
    stub = StubVad([[[750, -1]]])
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))

    events = gate.push(np.zeros(0, dtype=np.float32), is_final=True)

    assert kinds(events) == ["end"]


def test_an_empty_final_chunk_still_flushes_the_vad(config):
    """Regression: ``{"type": "eof"}`` carries no audio but must reach the VAD.

    fsmn-vad keeps the trailing ``end`` in its cache until a final call, and it
    ignores an input of zero samples, so the gate pads the flush to a single
    sample.  Without that, the end below is never reported.
    """
    stub = StubVad([[[750, -1]], [[-1, 4480]]])
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))

    events = gate.push(np.zeros(0, dtype=np.float32), is_final=True)

    assert len(stub.calls) == 2
    assert stub.calls[1]["is_final"] is True
    assert stub.calls[1]["input"].size == 1
    assert kinds(events) == ["end"]
    assert events[0].time_ms == 4480


def test_an_empty_non_final_chunk_does_not_call_the_vad(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))

    assert gate.push(np.zeros(0, dtype=np.float32)) == []
    assert len(stub.calls) == 1


def test_an_empty_final_chunk_is_dropped_when_no_audio_was_pushed(config):
    """Flushing a VAD that holds nothing makes its frontend raise."""
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)

    assert gate.push(np.zeros(0, dtype=np.float32), is_final=True) == []
    assert stub.calls == []


def test_an_empty_final_chunk_is_dropped_after_an_earlier_flush(config):
    """A second flush would hit the same empty-cache crash as the first push."""
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))
    gate.push(np.zeros(0, dtype=np.float32), is_final=True)

    assert gate.push(np.zeros(0, dtype=np.float32), is_final=True) == []
    assert len(stub.calls) == 2


def test_a_vad_failure_does_not_kill_the_stream(config):
    class ExplodingVad:
        def generate(self, **kwargs):
            raise RuntimeError("stack expects a non-empty TensorList")

    gate = VadGate(config=config, vad_model=ExplodingVad())

    assert gate.push(chunk(config, ms=10), is_final=True) == []


# --- forced segment cap -----------------------------------------------------


def test_segment_cap_forces_an_end(config):
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[[0, -1]]]))
    gate.push(chunk(config, ms=500))

    events = gate.push(chunk(config, ms=500))

    assert kinds(events) == ["forced_end"]
    assert events[0].time_ms == pytest.approx(1000.0)
    assert not gate.in_speech


def test_segment_cap_does_not_fire_early(config):
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[[0, -1]]]))
    gate.push(chunk(config, ms=500))

    assert gate.push(chunk(config, ms=480)) == []
    assert gate.in_speech


def test_segment_cap_is_measured_from_the_speech_start(config):
    """A segment that opens late gets the full budget, not the stream's."""
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[], [[500, -1]]]))
    gate.push(chunk(config, ms=500))
    gate.push(chunk(config, ms=500))

    assert gate.push(chunk(config, ms=400)) == []
    assert kinds(gate.push(chunk(config, ms=200))) == ["forced_end"]


def test_segment_cap_fires_when_the_vad_reports_nothing_at_all(config):
    """The cut comes from the pushed sample count, not from the VAD."""
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[[0, -1]]]))

    events: List[VadEvent] = []
    for _ in range(4):
        events.extend(gate.push(chunk(config, ms=500)))

    assert kinds(events) == ["start", "forced_end"]


def test_a_new_segment_can_open_after_a_forced_end(config):
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[[0, -1]], [], [[2000, -1]]]))
    gate.push(chunk(config, ms=500))
    gate.push(chunk(config, ms=500))

    events = gate.push(chunk(config, ms=500))

    assert kinds(events) == ["start"]
    assert gate.in_speech


def test_the_end_trailing_a_forced_end_is_suppressed(config):
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[[0, -1]], [], [[-1, 1400]]]))
    gate.push(chunk(config, ms=500))
    gate.push(chunk(config, ms=500))

    assert gate.push(chunk(config, ms=500)) == []
    assert not gate.in_speech


# --- reset ------------------------------------------------------------------


def test_reset_clears_speech_state_and_the_clock(config):
    config.max_segment_sec = 1.0
    gate = VadGate(config=config, vad_model=StubVad([[[0, -1]], [], [[100, -1]]]))
    gate.push(chunk(config, ms=900))
    assert gate.in_speech

    gate.reset()

    assert not gate.in_speech
    # The 900 ms pushed before the reset must not count towards the new cap.
    assert gate.push(chunk(config, ms=500)) == []
    assert kinds(gate.push(chunk(config, ms=400))) == ["start"]


def test_reset_hands_the_vad_a_fresh_cache(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))
    first_cache = stub.calls[0]["cache"]
    first_cache["stats"] = "stale"

    gate.reset()
    gate.push(chunk(config))

    assert stub.calls[1]["cache"] == {}
    assert stub.calls[1]["cache"] is not first_cache


def test_the_cache_dict_is_reused_across_pushes(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)
    gate.push(chunk(config))
    gate.push(chunk(config))

    assert stub.calls[0]["cache"] is stub.calls[1]["cache"]


# --- input handling ---------------------------------------------------------


def test_non_1d_input_raises_value_error(config):
    gate = VadGate(config=config, vad_model=StubVad())

    with pytest.raises(ValueError):
        gate.push(np.zeros((2, 100), dtype=np.float32))


def test_input_is_cast_to_float32(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)

    gate.push(np.zeros(160, dtype=np.float64))

    assert stub.calls[0]["input"].dtype == np.float32


def test_the_vad_is_told_the_configured_sample_rate(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)

    gate.push(chunk(config))

    assert stub.calls[0]["fs"] == config.sample_rate


def test_the_vad_is_told_the_configured_vad_chunk_length(config):
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)

    gate.push(chunk(config))

    assert stub.calls[0]["chunk_size"] == int(config.vad_chunk_ms)


def test_the_vad_chunk_is_independent_of_the_recogniser_chunk(config):
    """The endpointer's granularity is not the recogniser's latency tradeoff."""
    config.chunk_size = 8  # 480 ms, the value the VAD used to inherit
    config.vad_chunk_ms = 300.0
    stub = StubVad()
    gate = VadGate(config=config, vad_model=stub)

    gate.push(chunk(config))

    assert stub.calls[0]["chunk_size"] == 300


@pytest.mark.parametrize("vad_chunk_ms", [0.0, -480.0, 15000.0, 60000.0])
def test_an_unusable_vad_chunk_is_rejected_at_construction(vad_chunk_ms):
    """The gate validates its config, so a bad chunk fails before any audio."""
    config = StreamingConfig(vad_chunk_ms=vad_chunk_ms)

    with pytest.raises(ValueError):
        VadGate(config=config, vad_model=StubVad())


def test_a_default_config_is_used_when_none_is_given():
    gate = VadGate(vad_model=StubVad())

    assert gate.config == StreamingConfig()


# --- real fsmn-vad ----------------------------------------------------------

SAMPLE_WAV = ROOT / "runtime" / "llama.cpp" / "tests" / "sample.wav"

#: Opt-in switch for the end-to-end test.  It is off by default because it
#: downloads the fsmn-vad weights from ModelScope, which is slow and flaky.
RUN_MODEL_TESTS = os.environ.get("SENSEVOICE_RUN_MODEL_TESTS") == "1"


def _read_sample_wav():
    """Return ``(mono_float32, sample_rate)`` for the shared sample clip."""
    pytest.importorskip("funasr")
    soundfile = pytest.importorskip("soundfile")
    if not SAMPLE_WAV.exists():
        pytest.skip(f"{SAMPLE_WAV} is not available")
    audio, sample_rate = soundfile.read(
        str(SAMPLE_WAV), dtype="float32", always_2d=True
    )
    return audio[:, 0], sample_rate


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set SENSEVOICE_RUN_MODEL_TESTS=1")
def test_real_fsmn_vad_endpoints_sample_wav():
    """End-to-end check against the real model (downloads it on first run)."""
    mono, sample_rate = _read_sample_wav()
    config = StreamingConfig(sample_rate=sample_rate)
    gate = VadGate(config=config)

    step = config.chunk_samples
    events: List[VadEvent] = []
    for offset in range(0, len(mono), step):
        block = mono[offset : offset + step]
        events.extend(gate.push(block, is_final=offset + step >= len(mono)))

    assert kinds(events) == ["start", "end"]
    assert events[0].time_ms == pytest.approx(770, abs=100)
    assert events[1].time_ms == pytest.approx(5980, abs=200)


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set SENSEVOICE_RUN_MODEL_TESTS=1")
def test_real_fsmn_vad_flushes_on_an_empty_final_chunk():
    """The eof path: all audio pushed non-final, then flushed with no samples.

    This is the shape the WebSocket server produces, and before the flush pad
    it silently dropped the closing ``end``.
    """
    mono, sample_rate = _read_sample_wav()
    config = StreamingConfig(sample_rate=sample_rate)
    gate = VadGate(config=config)

    step = config.chunk_samples
    events: List[VadEvent] = []
    for offset in range(0, len(mono), step):
        events.extend(gate.push(mono[offset : offset + step]))

    assert kinds(events) == ["start"]

    events.extend(gate.push(np.zeros(0, dtype=np.float32), is_final=True))

    assert kinds(events) == ["start", "end"]
    assert events[1].time_ms == pytest.approx(5980, abs=200)
    assert not gate.in_speech
