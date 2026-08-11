"""Unit tests for streaming/config.py (no ML dependencies required)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streaming.config import (  # noqa: E402
    MS_PER_FRAME,
    NUM_QUERY_FRAMES,
    VAD_STREAMING_CHUNK_LIMIT_MS,
    StreamingConfig,
)


def test_frame_constants():
    assert MS_PER_FRAME == 60.0
    assert NUM_QUERY_FRAMES == 4


def test_vad_streaming_chunk_limit_constant():
    # FunASR's fsmn_vad_streaming defaults ``is_streaming_input`` to False once
    # ``chunk_size >= 15000`` ms: every call is then treated as a complete
    # utterance, so incremental endpointing stops working.  The constant is the
    # exclusive upper bound for ``vad_chunk_ms``.
    assert VAD_STREAMING_CHUNK_LIMIT_MS == 15000.0


def test_defaults():
    config = StreamingConfig()
    assert config.chunk_size == 12
    assert config.max_history == 167
    assert config.sample_rate == 16000
    assert config.device == "cpu"
    assert config.language == "auto"
    assert config.use_itn is True
    assert config.ban_emo_unk is False
    assert config.max_segment_sec == 30.0
    assert config.num_threads == 4
    assert config.vad_model == "fsmn-vad"
    # 480 ms is the measured working point for fsmn-vad: on the reference clip
    # it places ``start`` at 770 ms and ``end`` at 5980 ms, matching the
    # offline decision.
    assert config.vad_chunk_ms == 480.0


def test_default_derived_values():
    config = StreamingConfig()
    assert config.chunk_ms == 720.0
    assert config.chunk_samples == 11520
    assert config.max_history_ms == pytest.approx(10020.0)


def test_derived_values_track_overrides():
    config = StreamingConfig(chunk_size=8, max_history=100, sample_rate=8000)
    assert config.chunk_ms == 480.0
    assert config.chunk_samples == 3840
    assert config.max_history_ms == 6000.0


def test_chunk_samples_is_an_int():
    config = StreamingConfig(chunk_size=1, sample_rate=44100)
    assert isinstance(config.chunk_samples, int)
    assert config.chunk_samples == 2646


def test_validate_accepts_defaults():
    StreamingConfig().validate()


def test_validate_accepts_history_equal_to_chunk():
    StreamingConfig(chunk_size=12, max_history=12).validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 12, "max_history": 11},
        {"sample_rate": 0},
        {"sample_rate": -16000},
        {"max_segment_sec": 0.0},
        {"max_segment_sec": -1.0},
        {"num_threads": 0},
        {"device": ""},
        {"language": ""},
        {"vad_model": ""},
    ],
)
def test_validate_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        StreamingConfig(**overrides).validate()


@pytest.mark.parametrize("vad_chunk_ms", [0.0, -1.0, -480.0])
def test_validate_rejects_non_positive_vad_chunk_ms(vad_chunk_ms):
    with pytest.raises(ValueError):
        StreamingConfig(vad_chunk_ms=vad_chunk_ms).validate()


@pytest.mark.parametrize("vad_chunk_ms", [15000.0, 15000.1, 30000.0])
def test_validate_rejects_vad_chunk_ms_at_or_above_streaming_limit(vad_chunk_ms):
    # The bound is exclusive: at ``chunk_size >= 15000`` ms FunASR turns
    # ``is_streaming_input`` off and treats each call as a complete utterance,
    # so the VAD no longer works as a streaming endpointer.
    with pytest.raises(ValueError):
        StreamingConfig(vad_chunk_ms=vad_chunk_ms).validate()


@pytest.mark.parametrize("vad_chunk_ms", [0.1, 480.0, 14999.0])
def test_validate_accepts_vad_chunk_ms_below_streaming_limit(vad_chunk_ms):
    StreamingConfig(vad_chunk_ms=vad_chunk_ms).validate()


def test_vad_chunk_ms_is_independent_of_chunk_size():
    # The VAD chunk sets endpoint granularity; the recogniser chunk trades
    # latency against encoder occupancy.  Neither may be derived from the other.
    config = StreamingConfig(chunk_size=8, max_history=100)
    assert config.chunk_ms == 480.0
    assert config.vad_chunk_ms == 480.0

    config = StreamingConfig(chunk_size=50, max_history=167)
    assert config.chunk_ms == 3000.0
    assert config.vad_chunk_ms == 480.0


def test_chunk_size_is_unaffected_by_vad_chunk_ms():
    config = StreamingConfig(vad_chunk_ms=1000.0)
    assert config.vad_chunk_ms == 1000.0
    assert config.chunk_size == StreamingConfig().chunk_size
    assert config.chunk_ms == StreamingConfig().chunk_ms
    assert config.chunk_samples == StreamingConfig().chunk_samples
