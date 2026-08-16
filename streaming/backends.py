"""Pluggable per-segment recognition strategies.

:class:`~streaming.streaming_model.StreamingSenseVoice` owns everything that is
the same whatever strategy is in use - the streaming frontend, the retained
waveform, the ``end_ms`` sample clock and the final full-quality pass - and
delegates the one question that genuinely differs to a *backend*: given the
encoder frames extracted so far, what is the current partial hypothesis?

Two strategies answer it very differently.  :class:`AccumulateBackend`
re-encodes a growing window from scratch on every partial; ``ChunkBackend``
(``streaming.chunk_backend``) feeds each frame into the encoder's streaming
cache exactly once.  :class:`StreamingBackend` is the narrow surface they share.

The module targets CPU: no CUDA-specific code paths are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from .config import StreamingConfig
from .ctc_decode import ctc_greedy_decode, strip_rich_tags

__all__ = [
    "AccumulateBackend",
    "StreamingBackend",
    "build_prompt_frames",
    "encode_window",
]


@dataclass
class _SegmentState:
    """Mutable per-segment state, reset by :meth:`StreamingSenseVoice.reset`.

    The recogniser and its backend share one instance of this: the recogniser
    owns the waveform, the sample clock, the frontend cache and
    ``total_frames``, while the fields marked below belong to
    :class:`AccumulateBackend` and are meaningless under any other strategy.
    They live here rather than inside the backend because the frames and the
    audio they came from must be dropped together, by one :meth:`reset`.
    """

    #: *Accumulate only.* Encoder frames of the *current window* only, shape ``(T, D)``.
    features: Optional[torch.Tensor] = None
    #: *Accumulate only.* Frames at the front of ``features`` already covered by ``last_raw_text``.
    decoded_frames: int = 0
    #: *Accumulate only.* Raw decode of the current window as of the last inference.
    last_raw_text: str = ""
    #: *Accumulate only.* Display text of every window already retired by the history cap.
    confirmed_text: str = ""
    #: *Accumulate only.* Raw text (tags kept) of every window already retired by the history cap.
    confirmed_raw: str = ""
    #: *Accumulate only.* New frames accumulated since the last inference.
    pending_frames: int = 0
    #: Total encoder frames produced in this segment.
    total_frames: int = 0
    #: Raw waveform of the segment, kept for the final full-quality pass.
    waveform: List[np.ndarray] = field(default_factory=list)
    #: Total samples pushed in this segment.
    total_samples: int = 0
    #: ``WavFrontendOnline`` streaming cache.
    frontend_cache: Dict[str, Any] = field(default_factory=dict)
    #: Set once ``is_last=True`` has flushed the frontend.
    finished: bool = False


@runtime_checkable
class StreamingBackend(Protocol):
    """The per-segment recognition strategy behind ``StreamingSenseVoice``.

    The protocol is deliberately five methods wide, because that is what the
    accumulate and chunk strategies genuinely have in common.  Everything else
    they do is either shared (and therefore stayed in the recogniser) or
    private to one of them:

    *Shared, so not here.*  Feature extraction, waveform retention, the
    ``end_ms`` clock and the ``final`` full-quality pass are identical under
    both strategies; the recogniser keeps them and a backend never sees audio,
    only encoder frames.

    *Private, so not here.*  How many encoder passes a partial costs, whether
    any history is retired, and whether a cache is carried between calls are
    exactly the decisions a backend exists to make.  Nothing in the protocol
    mentions windows, caches or encoder passes, so neither strategy leaks into
    the recogniser: the accumulate backend re-encodes ``max_history`` frames
    inside :meth:`emit_partial` while the chunk backend does most of its work
    inside :meth:`accept_frames`, and the recogniser cannot tell.

    The split between :meth:`should_emit_partial` and :meth:`emit_partial` is
    load-bearing rather than a convenience: the recogniser must be able to ask
    whether a partial is due *without* producing one, because ``is_last``
    suppresses a due partial (:meth:`discard_pending`) while the fallback path
    inside the final pass forces one that is not due.  A single
    ``emit_partial() -> Optional[...]`` could express neither.

    Implementations are single-stream and not thread-safe, like the recogniser
    that drives them.
    """

    def reset(self, state: _SegmentState) -> None:
        """Start a new segment against ``state``.

        Args:
            state: The recogniser's fresh per-segment state.  Backends that
                keep their bookkeeping in it (see :class:`_SegmentState`) must
                latch this object, since the recogniser replaces it on every
                reset rather than clearing it in place.
        """

    def accept_frames(self, frames: torch.Tensor) -> None:
        """Take newly extracted encoder frames.

        Args:
            frames: ``(T, D)`` float32 frames from the streaming frontend, in
                order, never overlapping a previous call.  May be called with
                any ``T >= 1``, several times between two partials or not at
                all between them.
        """

    def should_emit_partial(self) -> bool:
        """Whether enough new frames have arrived to justify a partial."""

    def emit_partial(self) -> Tuple[str, str]:
        """Produce the current hypothesis and consume the pending frames.

        Returns:
            ``(text, raw_text)``: the display text (rich tags stripped) and the
            raw decode with its tags intact.  The recogniser wraps them in a
            :class:`~streaming.streaming_model.StreamingResult` and adds the
            timestamps.
        """

    def discard_pending(self) -> None:
        """Drop the pending frames without emitting anything.

        Called when the segment ends: the ``final`` pass covers the same audio
        with a better decode, so the partial that was about to be emitted must
        not surface later.
        """


def build_prompt_frames(
    model: Any,
    config: StreamingConfig,
    device: torch.device,
) -> torch.Tensor:
    """Build the four query embeddings SenseVoice prepends to its input.

    The four query embeddings are prepended exactly as
    ``SenseVoiceSmall.inference`` does: ``textnorm`` first (+1), then
    ``language`` and the event/emotion pair (+3), giving the frame order
    ``[language, event, emotion, textnorm, speech...]``.

    Args:
        model: The loaded ``SenseVoiceSmall``.
        config: Streaming configuration; ``language`` and ``use_itn`` select
            the query ids.
        device: Device to build the embeddings on.

    Returns:
        A ``(1, 4, D)`` tensor in that frame order, ready to concatenate in
        front of the speech features.
    """
    language_id = model.lid_dict.get(config.language, 0)
    language_query = model.embed(torch.LongTensor([[language_id]]).to(device))
    textnorm = "withitn" if config.use_itn else "woitn"
    textnorm_query = model.embed(
        torch.LongTensor([[model.textnorm_dict[textnorm]]]).to(device)
    )
    event_emo_query = model.embed(torch.LongTensor([[1, 2]]).to(device))
    return torch.cat((language_query, event_emo_query, textnorm_query), dim=1)


def encode_window(
    model: Any,
    tokenizer: Any,
    config: StreamingConfig,
    features: Optional[torch.Tensor],
    device: torch.device,
) -> str:
    """Encode one feature window with the offline encoder and greedily decode it.

    This is the whole of :class:`AccumulateBackend`'s inference: a full,
    non-causal ``SenseVoiceEncoderSmall.forward`` over the window, exactly the
    computation the offline model performs.

    Args:
        model: The loaded ``SenseVoiceSmall``.
        tokenizer: Its tokenizer, as returned by ``from_pretrained``.
        config: Streaming configuration (``language``, ``use_itn``,
            ``ban_emo_unk``).
        features: Encoder frames of the current window, shape ``(T, D)``.
        device: Device to run on.

    Returns:
        The raw decoded text, rich tags included; ``""`` for an empty window.
    """
    if features is None or features.shape[0] == 0:
        return ""

    with torch.inference_mode():
        speech = features.unsqueeze(0).to(device)

        # SenseVoiceEncoderSmall.forward scales its input in place
        # (``xs_pad *= output_size ** 0.5``).  torch.cat allocates a fresh
        # tensor, so the accumulated ``features`` are never touched - feeding
        # the cached tensor directly would inflate it by 22.6x per chunk.
        speech = torch.cat(
            (build_prompt_frames(model, config, speech.device), speech), dim=1
        )
        speech_lengths = torch.tensor(
            [speech.shape[1]], dtype=torch.int32, device=speech.device
        )

        encoder_out, encoder_out_lens = model.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        ctc_logits = model.ctc.log_softmax(encoder_out)
        if config.ban_emo_unk:
            ctc_logits[:, :, model.emo_dict["unk"]] = -float("inf")

        logits_2d = ctc_logits[0, : int(encoder_out_lens[0].item()), :]
        return ctc_greedy_decode(logits_2d, tokenizer, blank_id=model.blank_id)


class AccumulateBackend:
    """Accumulate inside the segment and re-run the whole encoder per chunk.

    SenseVoiceSmall is a non-autoregressive CTC model whose encoder uses full
    (non-causal) SANM attention, so a checkpoint that was never trained with
    chunk masking has no incremental state to carry between chunks.  Rather
    than approximating one with truncated attention, this backend keeps the
    exact offline computation and simply repeats it as audio arrives: every
    time ``chunk_size`` new frames have accumulated, the *whole* window of
    accumulated frames - prefixed with the four query embeddings the model
    expects - is run through ``SenseVoiceEncoderSmall.forward`` and decoded
    with greedy CTC.

    The cost is quadratic-ish in segment length and bounded only by
    ``max_history`` (see :meth:`_apply_history_cap`); the benefit is that a
    partial sees precisely what the offline model would have seen for the same
    audio.  This is the default backend and the only one with measured latency
    numbers behind its defaults.

    Args:
        config: Streaming tunables; ``chunk_size`` sets the emission cadence
            and ``max_history`` the window bound.
        encode: Callable turning a ``(T, D)`` window (or ``None``) into raw
            decoded text - in practice
            :meth:`StreamingSenseVoice._encode_and_decode`, which binds
            :func:`encode_window` to the loaded model.  Injecting it keeps this
            class free of model loading and lets a test drive the window
            bookkeeping without any weights.
    """

    def __init__(
        self,
        config: StreamingConfig,
        encode: Callable[[Optional[torch.Tensor]], str],
    ) -> None:
        self._config = config
        self._encode = encode
        self._state = _SegmentState()

    # ------------------------------------------------------------- protocol

    def reset(self, state: _SegmentState) -> None:
        """Latch the recogniser's fresh state; this backend keeps no other."""
        self._state = state

    def accept_frames(self, frames: torch.Tensor) -> None:
        """Append ``frames`` to the accumulated window."""
        state = self._state
        if state.features is None:
            state.features = frames.clone()
        else:
            state.features = torch.cat((state.features, frames), dim=0)
        state.pending_frames += frames.shape[0]

    def should_emit_partial(self) -> bool:
        """Whether a whole ``chunk_size`` of new frames has accumulated."""
        return self._state.pending_frames >= self._config.chunk_size

    def discard_pending(self) -> None:
        """Forget the frames that would have triggered the next partial."""
        self._state.pending_frames = 0

    def emit_partial(self) -> Tuple[str, str]:
        """Run one encoder pass over the whole window and decode it.

        Returns:
            ``(text, raw_text)`` for the audio accumulated so far, with the
            confirmed prefix of any retired window prepended to both.
        """
        # The pending count is zeroed, not decremented by ``chunk_size``:
        # this pass decodes the *whole* window, so however many chunks' worth
        # of frames had piled up, they are all covered now and a second pass
        # would only reproduce the same hypothesis.
        self.discard_pending()
        self._apply_history_cap()

        state = self._state
        raw_window = self._encode(state.features)
        state.last_raw_text = raw_window
        state.decoded_frames = 0 if state.features is None else state.features.shape[0]

        raw_text = state.confirmed_raw + raw_window
        text = state.confirmed_text + strip_rich_tags(raw_window)
        return text, raw_text

    # -------------------------------------------------------------- history

    def _apply_history_cap(self) -> None:
        """Enforce ``config.max_history`` by retiring the oldest window.

        Strategy: **non-overlapping windows with a confirmed prefix.**  When the
        accumulated frames would exceed ``max_history``, the decode of the
        previous window is frozen into ``confirmed_text`` and *exactly* the
        frames that window covered are dropped; the next window restarts from
        the frames that arrived after it.  Later partials are rendered as
        ``confirmed_text + <current window decode>``.

        Every frame therefore contributes to exactly one decode - no text is
        duplicated across the boundary and none is lost.

        Limitation: because the windows do not overlap, the encoder loses all
        acoustic context across the cut.  A word straddling the boundary can be
        split into two fragments (or mis-recognised), and the rich tags of the
        retired window stay frozen in the confirmed prefix even if later audio
        would have changed them.  The ``final`` result is unaffected: it is
        produced by a full pass over the whole retained waveform.
        """
        state = self._state
        max_history = self._config.max_history
        if state.features is None or state.features.shape[0] <= max_history:
            return

        if state.decoded_frames > 0:
            state.confirmed_text += strip_rich_tags(state.last_raw_text)
            state.confirmed_raw += state.last_raw_text
            state.features = state.features[state.decoded_frames :].clone()
        state.decoded_frames = 0
        state.last_raw_text = ""

        # A single push larger than the cap (or an undecoded overflow) can still
        # leave the window too long: keep the most recent frames.
        if state.features.shape[0] > max_history:
            state.features = state.features[-max_history:].clone()
