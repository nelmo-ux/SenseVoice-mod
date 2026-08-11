"""Wiring between the endpointing gate and the streaming recogniser.

:class:`StreamingSession` is the arbiter that both front-ends (the WebSocket
server and the microphone demo) drive.  It owns the answer to one question:
*which samples reach the recogniser, and when*.

Four behaviours justify it existing rather than letting each front-end call
:class:`~streaming.vad_gate.VadGate` and
:class:`~streaming.streaming_model.StreamingSenseVoice` directly:

**Silence is never decoded.**
    The encoder costs ~430 ms of CPU per pass whatever it is fed (see
    :class:`~streaming.config.StreamingConfig`), so decoding non-speech would
    burn most of the real-time budget on nothing.  Audio pushed outside a
    speech segment is buffered and dropped, never forwarded.

**Sample-exact segment boundaries.**
    :class:`~streaming.vad_gate.VadGate` reports endpoints as absolute
    *milliseconds* while the recogniser consumes *samples*.  The session keeps
    a single absolute sample counter and converts every event time through it,
    so consecutive segments neither overlap nor drop a sample: the audio fed
    for a segment is exactly ``audio[start_index:end_index]``.

**Look-back over the VAD's reaction delay.**
    FSMN-VAD only recognises speech once it has heard some, so its ``start``
    timestamp points into audio that has *already been pushed* - measured at
    roughly 200 ms behind the true onset for this model, and further behind
    whenever the caller pushes large blocks.  Feeding the recogniser from
    "now" would clip the first phoneme of every utterance.  The session
    therefore retains a rolling window of recent audio (``look_back_sec``,
    default 1.0 s - a 5x margin over the measured delay, and cheap at 16 kHz
    mono: ~64 KB) and rewinds into it when a segment opens.  The window is a
    hard bound: an onset older than it is clamped to the oldest retained
    sample rather than silently mis-timed.

**Stream-absolute result times.**
    :class:`~streaming.streaming_model.StreamingSenseVoice` is reset at every
    speech onset and therefore times its results from the start of the
    *segment* (``start_ms`` is always ``0.0``).  The session owns the only
    clock that spans segments, so it rebases every result onto it: ``start_ms``
    becomes the absolute time the segment opened and ``end_ms`` the absolute
    time recognition has reached.  That is the contract the WebSocket protocol
    documents, and it is what makes two consecutive finals comparable.

Typical use::

    session = StreamingSession(model, vad, config)
    for block in microphone:                     # float32, 16 kHz, mono
        for result in session.push_audio(block):
            print(result.type, result.text)
    for result in session.push_audio(tail, is_final=True):
        print(result.type, result.text)
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .config import StreamingConfig

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .streaming_model import StreamingResult, StreamingSenseVoice
    from .vad_gate import VadGate

__all__ = ["DEFAULT_LOOK_BACK_SEC", "StreamingSession"]

#: ``VadEvent.kind`` opening a segment.  The kinds are spelled out here rather
#: than imported so this module depends on the gate's *contract* (the three
#: strings) and not on which symbols it happens to export.
_EVENT_START = "start"

#: Rolling window of past audio kept so a segment can start *before* the VAD
#: noticed it, in seconds.  See the module docstring for the 200 ms measurement
#: this is sized against.
DEFAULT_LOOK_BACK_SEC: float = 1.0

#: ``VadEvent.kind`` values that close a segment.  ``forced_end`` is the gate
#: cutting an over-long segment itself; the session treats it exactly like
#: ``end``.
_CLOSING_EVENTS = frozenset({"end", "forced_end"})


class StreamingSession:
    """Drive a recogniser from VAD endpoints, one speech segment at a time.

    The session is a small state machine over an absolute sample clock:

    ``idle``
        Audio is appended to the look-back window and discarded as it ages
        out.  The recogniser is not called.
    ``speaking``
        Entered on a ``start`` event, which also resets the recogniser.  Audio
        from the event's timestamp onwards is forwarded, and every partial the
        recogniser produces is returned.  Left on ``end`` / ``forced_end`` (or
        on ``is_final``), which flushes the remaining audio with
        ``is_last=True`` and yields the final result.

    A single :meth:`push_audio` call may carry a whole short utterance, so both
    transitions can fire within one block; events are applied in the order the
    gate reports them and the audio is sliced at each boundary.

    The session is **not** thread-safe and, like the recogniser it drives,
    represents exactly one audio stream.  Give each concurrent stream its own
    instance.

    Args:
        model: Recogniser exposing ``reset()`` and
            ``push_audio(samples, is_last=False) -> list[StreamingResult]``.
        vad: Endpointing gate exposing ``reset()``, ``in_speech`` and
            ``push(samples, is_final=False) -> list[VadEvent]``.
        config: Streaming configuration; a default :class:`StreamingConfig` is
            used when omitted.  Only ``sample_rate`` is read here - the sample
            clock that converts event times to sample indices.
        look_back_sec: Seconds of past audio kept available for rewinding into
            when a segment opens.  ``0`` disables look-back (segments then
            start at the VAD's timestamp only if it is still in the current
            block).  It is a constructor argument rather than a
            :class:`StreamingConfig` field because it belongs to *this* wiring
            between the two components, not to either of them: neither the
            recogniser nor the gate ever reads it.

    Raises:
        ValueError: If ``look_back_sec`` is negative.
    """

    def __init__(
        self,
        model: "StreamingSenseVoice",
        vad: "VadGate",
        config: Optional[StreamingConfig] = None,
        look_back_sec: float = DEFAULT_LOOK_BACK_SEC,
    ) -> None:
        if look_back_sec < 0:
            raise ValueError(f"look_back_sec must be >= 0, got {look_back_sec}")

        self.config = config if config is not None else StreamingConfig()
        self.config.validate()
        self.model = model
        self.vad = vad
        self.look_back_sec = look_back_sec

        self._look_back_samples = int(
            round(look_back_sec * self.config.sample_rate)
        )
        self._buffer = np.zeros(0, dtype=np.float32)
        #: Absolute index of ``self._buffer[0]`` in the stream.
        self._buffer_start = 0
        #: Absolute index one past the last sample ever pushed.
        self._total_samples = 0
        #: Absolute index up to which the current segment has been forwarded.
        self._fed_upto = 0
        #: Absolute index of the first sample of the current segment - the
        #: origin the recogniser's segment-relative times are rebased on.
        self._segment_start = 0
        self._in_speech = False

    # ------------------------------------------------------------------ state

    @property
    def in_speech(self) -> bool:
        """Whether the session currently sits inside a speech segment."""
        return self._in_speech

    def reset(self) -> None:
        """Drop all stream state and start a new stream at time zero.

        Resets the gate and the recogniser, clears the look-back window and
        rewinds the sample clock.  An open segment is abandoned *without*
        producing a final result - push with ``is_final=True`` first if the
        caller needs it closed.
        """
        self.vad.reset()
        self.model.reset()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0
        self._total_samples = 0
        self._fed_upto = 0
        self._segment_start = 0
        self._in_speech = False

    # ------------------------------------------------------------------- push

    def push_audio(
        self,
        samples: np.ndarray,
        is_final: bool = False,
    ) -> List["StreamingResult"]:
        """Feed one block of audio through the gate and the recogniser.

        Args:
            samples: 1-D array of mono samples at ``config.sample_rate``.  It
                is converted to ``float32`` when needed; may be empty (useful
                for flushing with ``is_final=True``).
            is_final: Marks the end of the stream.  The flag is passed to the
                gate, and an open segment is guaranteed to be closed with a
                final result even if the gate reports no endpoint.

        Returns:
            The results produced by this block, in chronological order: any
            number of ``"partial"`` entries, plus one ``"final"`` per segment
            that closed here.  Their ``start_ms`` / ``end_ms`` are stream
            times, not segment-relative ones.  Empty while the stream sits in
            silence.

        Raises:
            ValueError: If ``samples`` is not 1-D.
        """
        block = np.asarray(samples)
        if block.ndim != 1:
            raise ValueError(
                f"samples must be a 1-D array of mono audio, got shape "
                f"{tuple(block.shape)}"
            )
        block = block.astype(np.float32, copy=False)

        self._append(block)
        results: List["StreamingResult"] = []

        for event in self.vad.push(block, is_final=is_final):
            index = self._to_index(event.time_ms)
            if event.kind == _EVENT_START:
                self._open_segment(index)
            elif event.kind in _CLOSING_EVENTS:
                results.extend(self._close_segment(index))

        if self._in_speech:
            if is_final:
                # The gate reported no endpoint for a segment that is still
                # open at end of stream; close it at the last sample so the
                # caller always sees a final for audio it pushed.
                results.extend(self._close_segment(self._total_samples))
            else:
                results.extend(self._feed(self._total_samples, is_last=False))

        self._trim()
        return results

    # -------------------------------------------------------------- internals

    def _append(self, block: np.ndarray) -> None:
        """Add a block to the look-back window and advance the sample clock."""
        if block.size:
            self._buffer = np.concatenate((self._buffer, block))
        self._total_samples += int(block.size)

    def _trim(self) -> None:
        """Drop window samples that can no longer be needed.

        Two claims hold on the window: the look-back margin must survive for a
        segment that has not opened yet, and audio not yet forwarded to the
        recogniser must survive for the segment that is open.  The window is
        cut at the earlier of the two, so it stays bounded at roughly
        ``look_back_sec`` regardless of how long a segment runs.
        """
        keep_from = self._total_samples - self._look_back_samples
        if self._in_speech:
            keep_from = min(keep_from, self._fed_upto)
        keep_from = max(keep_from, self._buffer_start)
        if keep_from <= self._buffer_start:
            return
        self._buffer = self._buffer[keep_from - self._buffer_start :]
        self._buffer_start = keep_from

    def _to_index(self, time_ms: float) -> int:
        """Convert an absolute event time to an absolute sample index.

        Rounding (rather than truncation) keeps the boundary within half a
        sample of the reported time, and the result is clamped to the stream so
        a timestamp beyond the last pushed sample cannot slice past the buffer.
        """
        index = int(round(time_ms * self.config.sample_rate / 1000.0))
        return max(0, min(index, self._total_samples))

    def _open_segment(self, index: int) -> None:
        """Start a segment at ``index``, rewinding into the look-back window.

        ``index`` is where the gate says speech began, which is normally
        behind the audio already pushed.  It is clamped forward to the oldest
        retained sample: with the default window that only bites for an onset
        more than a second old, which the gate does not produce.
        """
        self.model.reset()
        self._in_speech = True
        self._fed_upto = max(index, self._buffer_start)
        self._segment_start = self._fed_upto

    def _close_segment(self, index: int) -> List["StreamingResult"]:
        """Flush a segment up to ``index`` and leave the speech state.

        The endpoint may land *before* ``_fed_upto`` - the gate reports the end
        of speech, while the trailing hangover has usually been forwarded as
        part of an earlier partial.  Re-sending audio would corrupt the
        recogniser's history, so the slice is clamped to empty and the flush
        simply closes the segment.
        """
        if not self._in_speech:
            return []
        results = self._feed(index, is_last=True)
        self._in_speech = False
        return results

    def _feed(self, upto: int, is_last: bool) -> List["StreamingResult"]:
        """Forward ``audio[_fed_upto:upto]`` to the recogniser.

        Args:
            upto: Absolute index one past the last sample to forward.  Values
                at or below ``_fed_upto`` forward nothing.
            is_last: Passed through to the recogniser; ``True`` ends the
                segment and asks for a final result.

        Returns:
            Whatever the recogniser returned, rebased onto the stream clock.  A
            no-op slice with ``is_last=False`` skips the call entirely, so
            silence never reaches the encoder.
        """
        upto = max(upto, self._fed_upto)
        chunk = self._slice(self._fed_upto, upto)
        self._fed_upto = upto
        if chunk.size == 0 and not is_last:
            return []
        return [
            self._rebase(result)
            for result in self.model.push_audio(chunk, is_last=is_last)
        ]

    def _rebase(self, result: "StreamingResult") -> "StreamingResult":
        """Shift a result's times from segment-relative to stream-absolute.

        The recogniser is reset at every onset, so it reports ``start_ms=0.0``
        and an ``end_ms`` counted from the first sample of the segment.  Adding
        the segment's own start turns both into the stream times the protocol
        promises: ``start_ms`` is where this segment began, ``end_ms`` where
        recognition has reached within it.

        Args:
            result: A result as the recogniser produced it.

        Returns:
            A copy with both timestamps shifted; the original is left alone so
            the recogniser's own bookkeeping cannot be disturbed.
        """
        offset_ms = self._segment_start * 1000.0 / self.config.sample_rate
        return replace(
            result,
            start_ms=offset_ms + result.start_ms,
            end_ms=offset_ms + result.end_ms,
        )

    def _slice(self, start: int, stop: int) -> np.ndarray:
        """Return a *copy* of ``audio[start:stop]``, in absolute indices.

        The copy is deliberate.  A numpy slice is a view that keeps its whole
        base array alive, and the recogniser retains every chunk it is fed for
        the final full-quality pass - so views would pin one look-back buffer
        per chunk for the lifetime of the segment.  At 16 kHz that is tens of
        megabytes for a 30 s segment, all of it audio that has already been
        superseded.  Copying costs one memcpy of the chunk and lets each buffer
        be freed as soon as it is trimmed.
        """
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        lo = max(start - self._buffer_start, 0)
        hi = max(stop - self._buffer_start, 0)
        return self._buffer[lo:hi].copy()
