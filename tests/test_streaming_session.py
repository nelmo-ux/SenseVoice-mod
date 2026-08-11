"""Unit tests for streaming/session.py.

The session takes both of its collaborators by injection, so every test here
drives it with :class:`StubModel` and :class:`StubVad`: the recogniser is
replaced by a recorder that keeps the exact samples it was handed, and the gate
by a script of endpoints.  Neither torch nor funasr is imported, which keeps the
suite fast and makes the assertions about *which samples reach the recogniser*
exact rather than statistical.

The audio in these tests is a ramp (``0, 1, 2, ...``), so a sample carries its
own absolute index: comparing what the recogniser received against a slice of
the source proves there is neither a gap nor an overlap at a segment boundary.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from streaming.config import StreamingConfig  # noqa: E402
from streaming.session import StreamingSession  # noqa: E402
from streaming.vad_gate import VadEvent  # noqa: E402

SAMPLE_RATE = 16000


@dataclass
class FakeResult:
    """Stand-in for ``streaming.streaming_model.StreamingResult``.

    It mirrors the real dataclass field for field - the session rebases the
    timestamps with :func:`dataclasses.replace`, so the stub has to be a
    dataclass with the same names for the test to exercise the real code path.
    """

    type: str
    text: str
    raw_text: str
    start_ms: float
    end_ms: float


class StubModel:
    """Recogniser stub that records the audio it is fed, segment by segment.

    It reproduces the timing contract of the real recogniser: every segment
    starts at :meth:`reset`, so the results it returns carry ``start_ms=0.0``
    and an ``end_ms`` counted from the first sample of *that* segment.

    Args:
        sample_rate: Used to turn the samples received into milliseconds.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.resets = 0
        #: Every array handed to :meth:`push_audio`, un-copied, in order.
        self.received: List[Any] = []
        #: ``(is_last)`` flag of each call, in order.
        self.flags: List[bool] = []
        #: Every result object returned, as the stub built it.
        self.emitted: List[FakeResult] = []
        self._segments: List[List[Any]] = []
        self._current: List[Any] = []

    def reset(self) -> None:
        self.resets += 1
        self._current = []
        self._segments.append(self._current)

    def push_audio(self, samples: Any, is_last: bool = False) -> List[FakeResult]:
        samples = np.asarray(samples)
        self.received.append(samples)
        self.flags.append(is_last)
        self._current.append(samples)
        elapsed_ms = self.segment_samples() * 1000.0 / self.sample_rate
        if is_last:
            produced = [FakeResult("final", "final-text", "raw", 0.0, elapsed_ms)]
        elif samples.size:
            produced = [FakeResult("partial", "partial-text", "raw", 0.0, elapsed_ms)]
        else:
            produced = []
        self.emitted.extend(produced)
        return produced

    # -- inspection ---------------------------------------------------------

    def segment_samples(self, index: int = -1) -> int:
        """Number of samples fed into one segment (the current one by default)."""
        return int(sum(chunk.size for chunk in self._segments[index]))

    def segment_audio(self, index: int = -1) -> Any:
        """Concatenation of every chunk fed into one segment."""
        if not self._segments:
            return np.zeros(0, dtype=np.float32)
        chunks = self._segments[index]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    @property
    def all_audio(self) -> Any:
        """Concatenation of every chunk ever fed, across all segments."""
        if not self.received:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.received)


class StubVad:
    """Gate stub replaying one list of :class:`VadEvent` per :meth:`push` call.

    Args:
        script: Events to return from successive pushes.  Calls beyond the
            script return nothing, which is what silence looks like.
    """

    def __init__(self, script: Sequence[Sequence[VadEvent]] = ()) -> None:
        self.script = [list(events) for events in script]
        self.resets = 0
        #: ``(sample_count, is_final)`` of each push, in order.
        self.calls: List[Any] = []

    def reset(self) -> None:
        self.resets += 1

    def push(self, samples: Any, is_final: bool = False) -> List[VadEvent]:
        index = len(self.calls)
        self.calls.append((int(np.asarray(samples).size), is_final))
        return list(self.script[index]) if index < len(self.script) else []


# --- helpers ----------------------------------------------------------------


def ramp(n: int, start: int = 0) -> Any:
    """Return ``n`` samples whose value is their absolute index."""
    return np.arange(start, start + n, dtype=np.float32)


def samples(ms: float) -> int:
    """Samples in ``ms`` milliseconds at :data:`SAMPLE_RATE`."""
    return int(ms * SAMPLE_RATE / 1000)


def build(
    script: Sequence[Sequence[VadEvent]] = (),
    look_back_sec: float = 1.0,
    config: Optional[StreamingConfig] = None,
) -> Any:
    """Build a session over stubs and return ``(session, model, vad)``."""
    model = StubModel()
    vad = StubVad(script)
    session = StreamingSession(
        model=model,
        vad=vad,
        config=config if config is not None else StreamingConfig(),
        look_back_sec=look_back_sec,
    )
    return session, model, vad


def start_at(ms: float) -> VadEvent:
    return VadEvent(kind="start", time_ms=ms)


def end_at(ms: float, kind: str = "end") -> VadEvent:
    return VadEvent(kind=kind, time_ms=ms)


# --- gating -----------------------------------------------------------------


def test_a_start_event_resets_the_recogniser():
    session, model, _ = build([[], [start_at(480.0)]])
    session.push_audio(ramp(samples(480)))
    assert model.resets == 0

    session.push_audio(ramp(samples(480), start=samples(480)))

    assert model.resets == 1
    assert session.in_speech


def test_silence_never_reaches_the_recogniser():
    session, model, _ = build()

    results = []
    for index in range(4):
        block = ramp(samples(480), start=index * samples(480))
        results.extend(session.push_audio(block))

    assert results == []
    assert model.received == []
    assert not session.in_speech


def test_audio_after_an_end_is_not_forwarded():
    session, model, _ = build([[start_at(0.0), end_at(240.0)]])
    audio = ramp(samples(480))
    session.push_audio(audio)
    fed_before = model.segment_samples()

    session.push_audio(ramp(samples(480), start=samples(480)))

    assert model.segment_samples() == fed_before
    assert not session.in_speech


# --- look-back --------------------------------------------------------------


def test_look_back_rewinds_to_before_the_start_event():
    """The onset predates the VAD's report; the audio before it must be fed."""
    session, model, _ = build([[start_at(500.0)]], look_back_sec=1.0)
    audio = ramp(samples(1000))

    session.push_audio(audio)

    fed = model.segment_audio()
    assert np.array_equal(fed, audio[samples(500) :])
    assert fed[0] == samples(500)


def test_an_onset_older_than_the_window_is_clamped():
    """A start older than the retained window opens at the oldest sample kept."""
    session, model, _ = build([[], [], [start_at(0.0)]], look_back_sec=0.05)
    block = samples(480)
    for index in range(3):
        session.push_audio(ramp(block, start=index * block))

    oldest_kept = 3 * block - samples(50) - block  # buffer_start after push 2
    fed = model.segment_audio()
    assert fed[0] == oldest_kept
    assert fed[-1] == 3 * block - 1


def test_look_back_zero_starts_at_the_reported_onset():
    session, model, _ = build([[start_at(500.0)]], look_back_sec=0.0)
    audio = ramp(samples(1000))

    session.push_audio(audio)

    assert np.array_equal(model.segment_audio(), audio[samples(500) :])


def test_a_negative_look_back_is_rejected():
    with pytest.raises(ValueError):
        build(look_back_sec=-0.1)


# --- sample-exact boundaries ------------------------------------------------


def test_segment_audio_matches_the_source_samples_exactly():
    """The audio fed for a segment is exactly ``audio[start:end]``."""
    block = samples(480)
    session, model, _ = build(
        [[], [start_at(500.0)], [end_at(1300.0)]],
        look_back_sec=0.1,
    )
    audio = ramp(3 * block)

    for index in range(3):
        session.push_audio(audio[index * block : (index + 1) * block])

    assert np.array_equal(
        model.segment_audio(), audio[samples(500) : samples(1300)]
    )


def test_two_segments_neither_overlap_nor_share_audio():
    block = samples(480)
    session, model, _ = build(
        [
            [start_at(100.0), end_at(400.0)],
            [],
            [start_at(1000.0), end_at(1300.0)],
        ]
    )
    audio = ramp(3 * block)
    for index in range(3):
        session.push_audio(audio[index * block : (index + 1) * block])

    first = model.segment_audio(0)
    second = model.segment_audio(1)
    assert np.array_equal(first, audio[samples(100) : samples(400)])
    assert np.array_equal(second, audio[samples(1000) : samples(1300)])
    assert model.resets == 2


def test_a_short_utterance_inside_one_block_is_one_segment():
    session, model, _ = build([[start_at(100.0), end_at(300.0)]])
    audio = ramp(samples(480))

    results = session.push_audio(audio)

    assert [result.type for result in results] == ["final"]
    assert np.array_equal(
        model.segment_audio(), audio[samples(100) : samples(300)]
    )
    assert model.resets == 1
    assert not session.in_speech


def test_partials_are_emitted_while_the_segment_stays_open():
    session, _, _ = build([[start_at(0.0)], [], []])
    block = samples(480)

    first = session.push_audio(ramp(block))
    second = session.push_audio(ramp(block, start=block))

    assert [result.type for result in first] == ["partial"]
    assert [result.type for result in second] == ["partial"]


# --- closing ----------------------------------------------------------------


def test_is_final_closes_an_open_segment():
    session, model, vad = build([[start_at(0.0)]])
    session.push_audio(ramp(samples(480)))

    results = session.push_audio(np.zeros(0, dtype=np.float32), is_final=True)

    assert [result.type for result in results] == ["final"]
    assert model.flags[-1] is True
    assert not session.in_speech
    assert vad.calls[-1] == (0, True)


def test_is_final_flushes_the_audio_of_the_block_it_carries():
    session, model, _ = build([[start_at(0.0)], []])
    block = samples(480)
    session.push_audio(ramp(block))

    session.push_audio(ramp(block, start=block), is_final=True)

    assert np.array_equal(model.segment_audio(), ramp(2 * block))


def test_forced_end_is_treated_like_end():
    session, model, _ = build([[start_at(100.0), end_at(300.0, "forced_end")]])
    audio = ramp(samples(480))

    results = session.push_audio(audio)

    assert [result.type for result in results] == ["final"]
    assert np.array_equal(
        model.segment_audio(), audio[samples(100) : samples(300)]
    )
    assert not session.in_speech


def test_an_end_before_the_audio_already_fed_flushes_without_resending():
    """The hangover is normally forwarded before the ``end`` arrives."""
    block = samples(480)
    session, model, _ = build([[start_at(0.0)], [end_at(200.0)]])
    session.push_audio(ramp(block))
    fed_before = model.segment_samples()

    results = session.push_audio(ramp(block, start=block))

    assert model.segment_samples() == fed_before
    assert [result.type for result in results] == ["final"]


def test_is_final_in_silence_produces_nothing():
    session, model, _ = build()

    results = session.push_audio(ramp(samples(480)), is_final=True)

    assert results == []
    assert model.received == []


# --- absolute timestamps (regression: rebasing) ------------------------------


def test_result_times_are_rebased_onto_the_stream_clock():
    block = samples(480)
    session, _, _ = build([[], [start_at(500.0)], [end_at(1300.0)]])
    audio = ramp(3 * block)

    session.push_audio(audio[:block])
    partials = session.push_audio(audio[block : 2 * block])
    finals = session.push_audio(audio[2 * block :])

    # The segment opens at 500 ms; the first partial covers up to 960 ms of
    # stream time (960 - 500 = 460 ms of audio), the final up to 1300 ms.
    assert partials[0].start_ms == pytest.approx(500.0)
    assert partials[0].end_ms == pytest.approx(960.0)
    assert finals[-1].start_ms == pytest.approx(500.0)
    assert finals[-1].end_ms == pytest.approx(1300.0)


def test_a_second_segment_is_timed_from_its_own_onset():
    block = samples(480)
    session, _, _ = build(
        [
            [start_at(100.0), end_at(400.0)],
            [],
            [start_at(1000.0), end_at(1300.0)],
        ]
    )
    audio = ramp(3 * block)
    first = session.push_audio(audio[:block])
    session.push_audio(audio[block : 2 * block])
    second = session.push_audio(audio[2 * block :])

    assert first[-1].start_ms == pytest.approx(100.0)
    assert first[-1].end_ms == pytest.approx(400.0)
    assert second[-1].start_ms == pytest.approx(1000.0)
    assert second[-1].end_ms == pytest.approx(1300.0)


def test_rebasing_accounts_for_a_clamped_onset():
    """A clamped start moves the origin, and the times must follow it."""
    session, _, _ = build([[], [], [start_at(0.0), end_at(1440.0)]], look_back_sec=0.05)
    block = samples(480)
    results = []
    for index in range(3):
        results = session.push_audio(ramp(block, start=index * block))

    # Two blocks pushed, 50 ms of look-back kept: the oldest retained sample
    # sits at 910 ms, so that is where the segment can start.
    assert results[-1].start_ms == pytest.approx(910.0)
    assert results[-1].end_ms == pytest.approx(1440.0)


def test_rebasing_leaves_the_recognisers_own_results_alone():
    """Rebasing returns copies, so the recogniser's bookkeeping is untouched."""
    session, model, _ = build([[start_at(500.0), end_at(900.0)]])

    results = session.push_audio(ramp(samples(1000)))

    assert model.emitted
    assert [result.start_ms for result in model.emitted] == [0.0]
    assert results[-1] is not model.emitted[-1]
    assert results[-1].start_ms == pytest.approx(500.0)
    assert results[-1].text == model.emitted[-1].text


# --- buffering (regression: _slice must copy) --------------------------------


def test_slice_returns_a_copy_not_a_view():
    session, _, _ = build()
    session.push_audio(ramp(samples(480)))

    chunk = session._slice(0, samples(100))

    assert not np.shares_memory(chunk, session._buffer)


def test_the_audio_handed_to_the_recogniser_does_not_alias_the_buffer():
    session, model, _ = build([[start_at(0.0)]])
    session.push_audio(ramp(samples(480)))

    assert model.received
    for chunk in model.received:
        assert not np.shares_memory(chunk, session._buffer)


def test_an_empty_slice_is_empty():
    session, _, _ = build()
    session.push_audio(ramp(samples(480)))

    assert session._slice(100, 100).size == 0
    assert session._slice(200, 100).size == 0


# --- reset ------------------------------------------------------------------


def test_reset_clears_state_and_the_clock():
    session, model, vad = build([[start_at(0.0)], [start_at(0.0)]])
    session.push_audio(ramp(samples(480)))
    assert session.in_speech

    session.reset()

    assert not session.in_speech
    assert vad.resets == 1
    assert model.resets == 2  # the onset, then the reset
    assert session._total_samples == 0
    assert session._buffer.size == 0
    assert session._fed_upto == 0
    assert session._segment_start == 0


def test_after_reset_the_stream_is_timed_from_zero_again():
    session, _, _ = build([[start_at(400.0)], [start_at(100.0), end_at(300.0)]])
    session.push_audio(ramp(samples(480)))

    session.reset()
    results = session.push_audio(ramp(samples(480)))

    assert results[-1].start_ms == pytest.approx(100.0)
    assert results[-1].end_ms == pytest.approx(300.0)


# --- input handling ---------------------------------------------------------


def test_non_1d_input_raises_value_error():
    session, _, _ = build()

    with pytest.raises(ValueError):
        session.push_audio(np.zeros((2, 100), dtype=np.float32))


def test_input_is_cast_to_float32():
    session, model, _ = build([[start_at(0.0)]])

    session.push_audio(np.ones(samples(480), dtype=np.float64))

    assert model.received[0].dtype == np.float32


def test_a_default_config_is_used_when_none_is_given():
    session = StreamingSession(model=StubModel(), vad=StubVad())

    assert session.config == StreamingConfig()


def test_the_vad_sees_every_block_and_the_final_flag():
    session, _, vad = build()
    block = samples(480)
    session.push_audio(ramp(block))
    session.push_audio(np.zeros(0, dtype=np.float32), is_final=True)

    assert vad.calls == [(block, False), (0, True)]
