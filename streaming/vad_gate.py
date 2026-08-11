"""Endpointing gate around FunASR's streaming FSMN-VAD.

``fsmn-vad`` is a *streaming* voice-activity detector: it is fed successive
audio chunks together with a ``cache`` dict that carries its state, and it
answers with raw intervals::

    [{'key': 'rand_key_xxx', 'value': [[750, -1]]}]

The ``value`` list is empty most of the time, and when it is not, a ``-1`` on
either side is a sentinel meaning "this side of the interval is still open":

===================  ==========================================================
``[[750, -1]]``      speech started at 750 ms, the end is not known yet
``[[-1, 4480]]``     the speech segment that was open ended at 4480 ms
``[[100, 900]]``     a complete segment, both edges reported in one call
===================  ==========================================================

The times are milliseconds measured from the *start of the stream*, not from
the start of the chunk that triggered them - the frame counters live in
``cache`` - so :class:`VadGate` reports them unchanged, except across a final
call, where FunASR re-initialises ``cache`` and its clock restarts.  A single
call can carry several intervals.

This module turns that raw, sentinel-encoded stream into a flat sequence of
:class:`VadEvent` objects with a guaranteed alternation of ``start`` and
``end``/``forced_end``, so the caller can drive a segment buffer without
re-implementing the sentinel logic or worrying about the VAD contradicting
itself.  It also owns the hard segment cap (:attr:`StreamingConfig.max_segment_sec`),
which is enforced from the sample count pushed in - never from the VAD's own
answers, since a VAD that has stopped reporting endpoints is exactly the
failure the cap exists to contain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import StreamingConfig

__all__ = ["VadEvent", "VadGate"]

logger = logging.getLogger(__name__)

#: Value used by fsmn-vad for "this edge of the interval is not known yet".
_OPEN_SENTINEL = -1

#: Audio handed to the VAD when the caller flushes the stream with no samples
#: of its own.  See :meth:`VadGate._vad_intervals` for why a single sample -
#: rather than the empty array the caller passed - is what reaches FunASR.
_FLUSH_PAD_SAMPLES = 1


@dataclass
class VadEvent:
    """A single endpointing decision.

    Attributes:
        kind: ``"start"``, ``"end"`` or ``"forced_end"``.  ``forced_end`` is
            emitted by the segment cap rather than by the VAD; callers should
            treat it exactly like ``end`` (close the segment, flush the
            decoder) - the distinct name only exists so that a caller can
            report or count truncated segments.
        time_ms: Absolute time from the start of the stream, in milliseconds.
    """

    kind: str
    time_ms: float


class VadGate:
    """Stateful endpointing gate over a streaming FSMN-VAD.

    One instance owns one stream.  Audio is handed over with :meth:`push` in
    the order it was captured; the gate keeps the VAD's ``cache`` dict, the
    running sample count and the speech/non-speech state, and answers with the
    events that this chunk produced.

    The state machine has exactly two states, ``non-speech`` (initial) and
    ``speech``.  Transitions that would repeat the current state are dropped
    rather than reported, so the emitted sequence always alternates:

    * ``start`` while already in speech -> suppressed (the VAD re-announcing a
      segment it already opened, e.g. after a forced cut has been decided by
      the model but not by us).
    * ``end`` while not in speech -> suppressed (typically the trailing edge of
      a segment this gate already closed with ``forced_end``).
    * ``[[a, b]]`` with both edges present -> ``start`` then ``end``, unless
      speech was already open, in which case only the ``end`` survives.

    The model is loaded **eagerly** in :meth:`__init__` when it is not
    injected, so the download/warm-up cost is paid at construction time
    instead of stalling the first audio chunk of a live stream.

    Args:
        config: Streaming configuration.  A default :class:`StreamingConfig`
            is used when omitted.
        vad_model: A preloaded VAD exposing
            ``generate(input=..., cache=..., is_final=..., ...) -> list[dict]``
            - normally a ``funasr.AutoModel``.  When ``None``, one is built
            from ``config.vad_model`` / ``config.device``.  Injecting a stub
            here is how the gate is tested without FunASR.

    Example:
        >>> gate = VadGate(vad_model=stub)              # doctest: +SKIP
        >>> for chunk, last in stream:                  # doctest: +SKIP
        ...     for event in gate.push(chunk, is_final=last):
        ...         print(event.kind, event.time_ms)
    """

    def __init__(
        self,
        config: StreamingConfig | None = None,
        vad_model: Any | None = None,
    ) -> None:
        self.config = config if config is not None else StreamingConfig()
        self.config.validate()
        self._model = vad_model if vad_model is not None else self._load_model()

        self._cache: dict[str, Any] = {}
        self._in_speech: bool = False
        self._speech_start_ms: float = 0.0
        self._total_samples: int = 0
        self._session_start_ms: float = 0.0
        #: Samples handed to the VAD since the last flush (or since the last
        #: :meth:`reset`).  Zero means the VAD holds no un-flushed audio, which
        #: is the one state in which it must not be asked to flush.
        self._samples_since_flush: int = 0

    def _load_model(self) -> Any:
        """Build the FunASR VAD described by :attr:`config`.

        ``funasr`` is imported here rather than at module scope so that this
        module - and the pure state-machine logic in it - stays importable on a
        checkout without the ML dependencies installed.

        Returns:
            A ``funasr.AutoModel`` for ``config.vad_model`` on ``config.device``.
        """
        from funasr import AutoModel

        return AutoModel(
            model=self.config.vad_model,
            disable_update=True,
            disable_pbar=True,
            device=self.config.device,
        )

    @property
    def in_speech(self) -> bool:
        """Whether a speech segment is currently open."""
        return self._in_speech

    @property
    def _stream_ms(self) -> float:
        """Absolute time of the end of the audio pushed so far, in ms."""
        return self._total_samples * 1000.0 / self.config.sample_rate

    @property
    def _max_segment_ms(self) -> float:
        """Hard cap on a segment's duration, in milliseconds."""
        return self.config.max_segment_sec * 1000.0

    def reset(self) -> None:
        """Drop all stream state and start over at time zero.

        Clears the VAD cache (a fresh dict makes FunASR re-initialise its
        internal state on the next call), the speech flag and the sample
        counter.  The model itself is kept, so resetting between utterances or
        sessions is cheap.
        """
        self._cache = {}
        self._in_speech = False
        self._speech_start_ms = 0.0
        self._total_samples = 0
        self._session_start_ms = 0.0
        self._samples_since_flush = 0

    def push(self, samples: "np.ndarray", is_final: bool = False) -> list[VadEvent]:
        """Feed one chunk of audio and collect the endpointing events.

        Args:
            samples: 1-D array of mono PCM at ``config.sample_rate``.  It is
                cast to ``float32`` when needed.  An empty array is accepted:
                on a non-final push it skips the VAD call (FunASR returns
                nothing for empty input anyway), and on a final one it still
                flushes the VAD - see :meth:`_vad_intervals`.
            is_final: Mark this as the last chunk.  The VAD flushes its
                pending decision, and any segment still open afterwards is
                closed with an ``end`` at the current stream time.

        Returns:
            The events triggered by this chunk, in chronological order.
            Empty when nothing changed.

        Raises:
            ValueError: If ``samples`` is not 1-D.
        """
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError(
                f"samples must be a 1-D mono array, got shape {samples.shape}"
            )
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        self._total_samples += int(samples.size)

        events: list[VadEvent] = []
        for start_ms, end_ms in self._vad_intervals(samples, is_final):
            events.extend(self._apply_interval(start_ms, end_ms))

        forced = self._enforce_segment_cap()
        if forced is not None:
            events.append(forced)

        if is_final and self._in_speech:
            closing = self._close("end", self._stream_ms)
            if closing is not None:
                events.append(closing)

        return events

    def _vad_intervals(
        self,
        samples: "np.ndarray",
        is_final: bool,
    ) -> list[tuple[float, float]]:
        """Run the VAD on one chunk and return its raw ``(start, end)`` pairs.

        FunASR counts its milliseconds from the start of a *VAD session*, and a
        session ends on a final call - ``inference`` re-initialises ``cache``
        once it has flushed.  A stream that keeps going after that (a caller
        that finalises an utterance without calling :meth:`reset`) would
        otherwise see the clock jump back to zero, so the times are rebased on
        :attr:`_session_start_ms`, which is nought for the common single-session
        stream.

        **Flushing on an empty final chunk.**  A caller that ends the stream
        with no audio left over (the WebSocket ``{"type": "eof"}`` frame) still
        has to reach the VAD, or the trailing ``end`` of the last utterance is
        never reported.  Handing the empty array straight to FunASR does *not*
        work: ``FsmnVADStreaming.inference`` returns ``{"value": []}`` for an
        empty input *before* it looks at ``is_final``, so the pending decision
        stays buried in the cache (measured: streaming
        ``runtime/llama.cpp/tests/sample.wav`` and then flushing with an empty
        array yields ``[]``, losing the ``end`` at 5980 ms).  The chunk is
        therefore padded to a single zero sample, which is enough to get past
        that guard and flushes the exact same endpoint the reference run
        reports (``[[-1, 5980]]``).  One sample rather than a longer pad
        because the pad is audio the caller never sent: at 16 kHz it moves the
        VAD's clock by 0.0625 ms, whereas a 10 ms pad already shifts the
        reported end to 5990 ms and a 100 ms one to 6080 ms.

        The pad is only safe once the VAD holds audio: flushing a session that
        has been fed nothing makes FunASR's frontend raise
        ``RuntimeError: stack expects a non-empty TensorList`` (its LFR splice
        cache is empty), so :attr:`_samples_since_flush` gates it, and the
        residual case of a stream shorter than one fbank window is caught and
        logged rather than allowed to kill the stream.

        Args:
            samples: ``float32`` mono chunk.  An empty array short-circuits
                unless it is flushing a VAD that holds un-flushed audio.
            is_final: Forwarded to the VAD so it can flush.

        Returns:
            The intervals from ``value``, as absolute stream times, still
            carrying ``-1`` sentinels.
        """
        if samples.size == 0:
            if not is_final or self._samples_since_flush == 0:
                return []
            samples = np.zeros(_FLUSH_PAD_SAMPLES, dtype=np.float32)

        try:
            result = self._model.generate(
                input=samples,
                cache=self._cache,
                is_final=is_final,
                chunk_size=int(self.config.vad_chunk_ms),
                fs=self.config.sample_rate,
            )
        except RuntimeError:
            # FunASR's frontend cannot flush a session shorter than one fbank
            # window.  Such a stream carries no speech to report, and a live
            # stream must not die on it.
            logger.warning(
                "VAD produced no result for %d samples (is_final=%s)",
                samples.size,
                is_final,
                exc_info=True,
            )
            result = None

        if is_final:
            self._samples_since_flush = 0
        else:
            self._samples_since_flush += int(samples.size)

        intervals = [
            (self._to_stream_ms(start_ms), self._to_stream_ms(end_ms))
            for start_ms, end_ms in _parse_intervals(result)
        ]
        if is_final:
            self._session_start_ms = self._stream_ms
        return intervals

    def _to_stream_ms(self, session_ms: float) -> float:
        """Rebase a VAD-session timestamp on the stream clock, sentinels apart."""
        if session_ms == _OPEN_SENTINEL:
            return session_ms
        return self._session_start_ms + session_ms

    def _apply_interval(self, start_ms: float, end_ms: float) -> list[VadEvent]:
        """Turn one raw VAD interval into zero, one or two events.

        Args:
            start_ms: Left edge, or ``-1`` when the segment opened earlier.
            end_ms: Right edge, or ``-1`` when the segment is still open.

        Returns:
            The events that survive the state machine.
        """
        events: list[VadEvent] = []
        if start_ms != _OPEN_SENTINEL:
            opening = self._open(start_ms)
            if opening is not None:
                events.append(opening)
        if end_ms != _OPEN_SENTINEL:
            closing = self._close("end", end_ms)
            if closing is not None:
                events.append(closing)
        return events

    def _open(self, time_ms: float) -> VadEvent | None:
        """Enter the speech state, or drop the transition if already there."""
        if self._in_speech:
            logger.debug(
                "VAD reported start at %.0f ms while already in speech", time_ms
            )
            return None
        self._in_speech = True
        self._speech_start_ms = time_ms
        return VadEvent(kind="start", time_ms=time_ms)

    def _close(self, kind: str, time_ms: float) -> VadEvent | None:
        """Leave the speech state, or drop the transition if not in speech."""
        if not self._in_speech:
            logger.debug("VAD reported end at %.0f ms while not in speech", time_ms)
            return None
        self._in_speech = False
        return VadEvent(kind=kind, time_ms=time_ms)

    def _enforce_segment_cap(self) -> VadEvent | None:
        """Cut the current segment when it has run past ``max_segment_sec``.

        The elapsed time is measured from the samples pushed in, so the cap
        still fires when the VAD has stopped reporting endpoints altogether -
        continuous speech, sustained noise, or a VAD failure.  The cut is
        stamped at the current stream time (never before), which keeps the
        emitted segments contiguous, and it returns the gate to non-speech so
        the next ``start`` opens a fresh segment.

        Returns:
            A ``forced_end`` event, or ``None`` when the cap does not apply.
        """
        if not self._in_speech:
            return None
        if self._stream_ms - self._speech_start_ms < self._max_segment_ms:
            return None
        return self._close("forced_end", self._stream_ms)


def _parse_intervals(result: Any) -> list[tuple[float, float]]:
    """Extract the ``(start, end)`` pairs from a FunASR VAD result.

    The expected shape is ``[{'key': ..., 'value': [[start, end], ...]}]``, but
    a streaming loop must not die on a malformed answer, so anything that does
    not look like a pair is logged and skipped.

    Args:
        result: Whatever ``generate`` returned.

    Returns:
        The well-formed intervals, in the order the VAD listed them.
    """
    if not result:
        return []

    first = result[0]
    value = first.get("value") if isinstance(first, dict) else None
    if not value:
        return []

    intervals: list[tuple[float, float]] = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            logger.warning("ignoring malformed VAD interval: %r", entry)
            continue
        start_ms, end_ms = float(entry[0]), float(entry[1])
        if start_ms == _OPEN_SENTINEL and end_ms == _OPEN_SENTINEL:
            logger.warning("ignoring VAD interval with no known edge: %r", entry)
            continue
        intervals.append((start_ms, end_ms))
    return intervals
