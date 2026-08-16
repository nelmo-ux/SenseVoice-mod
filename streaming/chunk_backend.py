"""SCAMA-style chunk streaming backend.

Where :class:`~streaming.backends.AccumulateBackend` re-runs the offline
encoder over a growing window, this backend uses the encoder's own streaming
path - ``SenseVoiceEncoderSmall.init_chunk_cache`` and
``SenseVoiceEncoderSmall.forward_chunk`` - so every frame is encoded exactly
once and the cost of a partial no longer grows with the segment.

That only makes sense against a checkpoint finetuned with chunk masking
(``finetune_chunk.sh``); run against the stock non-causal checkpoint the
truncated attention is an approximation of a computation the model never saw
in training, and quality will suffer.  Nothing here can detect that, so the
backend is opt-in via ``StreamingConfig.backend``.

The module targets CPU: no CUDA-specific code paths are used.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .backends import _SegmentState, build_prompt_frames
from .config import StreamingConfig
from .ctc_decode import collapse_token_ids, strip_rich_tags

__all__ = ["ChunkBackend"]


class ChunkBackend:
    """Feed each encoder frame once, through the encoder's streaming cache.

    Frames arrive from the frontend in whatever size the caller's audio blocks
    produce, but ``forward_chunk`` commits a fixed ``chunk_stride`` per call,
    so incoming frames are buffered and the remainder below one stride is held
    back until it is complete.  A partial is emitted on the same cadence as the
    accumulate backend (``config.chunk_size`` new frames), which normally means
    several ``forward_chunk`` calls per partial - the encoder step and the
    emission cadence are independent knobs here, unlike in the accumulate
    backend where they are necessarily the same thing.

    **Rich labels are provisional.**  SenseVoice emits its ``<|en|>``,
    ``<|HAPPY|>``, ``<|Speech|>`` and ``<|withitn|>`` markers at the leading
    output positions, so under this backend the leading group is decided by the
    first chunk alone - a few hundred milliseconds of audio, before the
    utterance has a chance to contradict it.  Decoded chunk-wise the model does
    not keep the markers to the front either: it re-emits tag tokens at later
    speech frames, so a partial's ``raw_text`` accumulates *several* mutually
    inconsistent tag groups interleaved with the transcript rather than one
    group up front.  Observed on the published weights::

        <|zh|><|NEUTRAL|><|Speech|><|withitn|><|yue|><|EMO_UNKNOWN|><|Speech|>我想想问<|zh|><|NEUTRAL|>我在

    This backend revises none of them - the frames that produced each group are
    long gone from the cache - and it does not try to reconcile them.  Display
    text is unaffected: :meth:`emit_partial` runs ``strip_rich_tags``, which
    removes every group wherever it sits, so only a consumer reading
    ``raw_text`` meets the contradiction, and it should treat what it finds
    there as provisional and possibly self-contradictory.  The authoritative
    label comes from the ``final`` result, which the recogniser produces
    backend-independently with a full-quality offline pass over the whole
    retained waveform.

    Decode strategy: **keep per-frame argmax ids, collapse the whole sequence
    each time.**  The CTC head is applied frame by frame, so the argmax of a
    frame never changes once that frame has left the encoder - the ids for the
    new chunk are computed once, appended, and the log-probabilities thrown
    away.  Only the collapse (merge consecutive duplicates, drop blanks) is
    re-run over the whole segment on every partial, which is integer work over
    at most a few hundred entries.  The alternative - concatenating the encoder
    outputs and re-running the CTC head over all of them per partial - would
    cost a ``(T, 512) x (512, ~25k)`` matmul over the *whole* segment each
    time, and would produce the identical text.  Collapsing incrementally
    instead (decoding each chunk on its own and concatenating the strings) was
    rejected: CTC merges duplicates *across* a chunk boundary, so a token whose
    frames straddle one would be emitted twice.

    Args:
        config: Streaming tunables; the ``chunk_*`` fields set the geometry and
            ``chunk_size`` the emission cadence.
        model: The loaded ``SenseVoiceSmall``, already in ``eval()`` mode - the
            recogniser puts it there at load time, and the per-layer caches
            assume a single monotonic pass with dropout off.
        tokenizer: Its tokenizer, as returned by ``from_pretrained``.
        device: Device to run on.
    """

    def __init__(
        self,
        config: StreamingConfig,
        model: Any,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._cache: Optional[dict] = None
        self.reset(_SegmentState())

    # ------------------------------------------------------------- protocol

    def reset(self, state: _SegmentState) -> None:
        """Start a new segment with a fresh encoder cache.

        The cache is rebuilt rather than cleared: it carries the position
        offset, the overlap frames and the per-layer attention caches of the
        *previous* utterance, all of which would leak across the segment
        boundary.

        Args:
            state: The recogniser's fresh per-segment state.  This backend
                keeps its own bookkeeping but latches the state so it can see
                ``finished``.
        """
        self._clear_segment(state)
        self._cache = self._new_cache()

    def _clear_segment(self, state: _SegmentState) -> None:
        """Drop every per-segment field except the encoder cache.

        Args:
            state: The state object to latch; see :meth:`reset`.
        """
        self._state = state
        self._buffer: Optional[torch.Tensor] = None
        self._frame_ids: List[int] = []
        self._frames_since_emit = 0
        self._prompt_pending = True
        self._tail_flushed = False

    def _new_cache(self) -> dict:
        """Build a fresh streaming cache for the configured geometry."""
        with torch.inference_mode():
            return self._model.encoder.init_chunk_cache(
                pad_left=self._config.chunk_pad_left,
                stride=self._config.chunk_stride,
                pad_right=self._config.chunk_pad_right,
                encoder_chunk_look_back=self._config.chunk_encoder_look_back,
                batch_size=1,
                device=self._device,
                dtype=torch.float32,
            )

    def accept_frames(self, frames: torch.Tensor) -> None:
        """Buffer ``frames`` and encode every whole stride they complete.

        Args:
            frames: ``(T, D)`` float32 encoder frames, in order.  Whatever does
                not fill a stride stays in the buffer for the next call, so the
                encoder always sees exactly ``chunk_stride`` new frames per
                step (the prompt frames on the first step aside).
        """
        if frames.shape[0] == 0:
            return
        self._buffer = (
            frames if self._buffer is None else torch.cat((self._buffer, frames), dim=0)
        )
        self._frames_since_emit += int(frames.shape[0])

        stride = self._config.chunk_stride
        while self._buffer is not None and self._buffer.shape[0] >= stride:
            chunk, self._buffer = self._buffer[:stride], self._buffer[stride:]
            self._encode_chunk(chunk)

    def should_emit_partial(self) -> bool:
        """Whether ``chunk_size`` new frames have arrived since the last partial.

        Counted on *arrival*, not on commit, so the cadence matches the
        accumulate backend exactly even though some of those frames may still
        be sitting in the buffer waiting to fill a stride.
        """
        return self._frames_since_emit >= self._config.chunk_size

    def discard_pending(self) -> None:
        """Forget the frames that would have triggered the next partial.

        Only the emission counter is reset.  Frames already fed to the encoder
        stay in its cache and buffered frames stay buffered - unlike the
        accumulate backend there is nothing to undo, because the work has
        already been done.
        """
        self._frames_since_emit = 0

    def emit_partial(self) -> Tuple[str, str]:
        """Decode everything committed so far.

        Returns:
            ``(text, raw_text)``.  ``raw_text`` keeps the provisional rich
            tags, which may be several inconsistent groups scattered through
            the transcript rather than one group up front (see the class
            docstring); ``text`` has all of them stripped.
        """
        self.discard_pending()
        if self._state.finished:
            # Only reachable when the full-quality pass failed and the
            # recogniser fell back to a partial for its ``final``.  Flushing
            # makes that fallback cover every frame the frontend produced
            # rather than stopping a lookahead short.
            self._flush_tail()

        token_ids = collapse_token_ids(
            self._frame_ids, blank_id=self._model.blank_id
        )
        raw_text = self._tokenizer.decode(token_ids) if token_ids else ""
        return strip_rich_tags(raw_text), raw_text

    # -------------------------------------------------------------- encoder

    def _encode_chunk(self, frames: torch.Tensor) -> None:
        """Run one ``forward_chunk`` step and keep its per-frame argmax ids.

        Args:
            frames: ``(stride, D)`` new frames, or the shorter remainder when
                flushing the tail.
        """
        speech = frames.unsqueeze(0).to(self._device)
        if self._prompt_pending:
            # The four query embeddings go in front of the *first* chunk only,
            # so they take absolute positions 1-4 - exactly where
            # ``SenseVoiceSmall.encode`` puts them - and the position encoding
            # of the speech frames continues from 5.  Prepending them to every
            # chunk would both duplicate them in the output and shift every
            # frame's position by four.
            #
            # Built under ``inference_mode`` like ``encode_window`` does:
            # ``model.embed`` returns a tensor with ``requires_grad=True``, so
            # outside it the embedding lookup and this ``cat`` would build an
            # autograd graph on the first chunk of every segment.
            self._prompt_pending = False
            with torch.inference_mode():
                speech = torch.cat(
                    (
                        build_prompt_frames(self._model, self._config, self._device),
                        speech,
                    ),
                    dim=1,
                )
        self._forward_chunk(speech)

    def _flush_tail(self) -> None:
        """Encode the buffered remainder and release the withheld lookahead.

        ``forward_chunk`` holds its last ``chunk_pad_right`` frames back until
        their right context arrives, and up to ``chunk_stride - 1`` frames may
        still be sitting in the buffer.  Both are released here: a short final
        chunk is fed first (``forward_chunk`` accepts one), then the
        ``tail_chunk`` flag replays the cached overlap.

        Idempotency comes from the ``_tail_flushed`` guard alone, not from the
        cache being spent: ``cache["tail_chunk"]`` is left ``True`` afterwards
        and nothing clears it, so the cache is *not* usable again.  Any
        :meth:`accept_frames` after a flush is therefore silently swallowed -
        ``forward_chunk`` ignores the input of a tail call and re-emits the
        cached overlap instead.  That is acceptable only because the segment is
        over by the time this runs; :meth:`reset` is what makes the backend
        usable again.
        """
        if self._tail_flushed or self._cache is None:
            return
        self._tail_flushed = True
        if self._buffer is not None and self._buffer.shape[0] > 0:
            remainder, self._buffer = self._buffer, None
            self._encode_chunk(remainder)
        if self._prompt_pending:
            # Not one frame was ever fed, so there is no overlap to replay: the
            # cache still holds nothing but its (possibly empty) left pad, and
            # a tail call would reach the encoder's FSMN convolution with a
            # zero-length input and raise (kernel size > input size).
            # Unreachable through the recogniser, which only flushes a segment
            # once ``total_frames > 0``, but reachable by driving the backend
            # directly.
            return
        if self._config.chunk_pad_right > 0:
            self._cache["tail_chunk"] = True
            # ``forward_chunk`` ignores its input on a tail call and re-runs the
            # cached overlap instead, reading only the device and dtype off this
            # tensor - hence the empty placeholder.
            self._forward_chunk(
                torch.zeros((1, 0, 0), dtype=torch.float32, device=self._device)
            )

    def _forward_chunk(self, speech: torch.Tensor) -> None:
        """Encode one window, apply the CTC head and store the argmax ids.

        Args:
            speech: ``(1, T, D)`` input for this step.  Ignored by the encoder
                when the cache's ``tail_chunk`` flag is set.
        """
        with torch.inference_mode():
            encoder_out, self._cache = self._model.encoder.forward_chunk(
                speech, self._cache
            )
            if encoder_out.shape[1] == 0:
                return
            ctc_logits = self._model.ctc.log_softmax(encoder_out)
            if self._config.ban_emo_unk:
                ctc_logits[:, :, self._model.emo_dict["unk"]] = -float("inf")
            self._frame_ids.extend(ctc_logits[0].argmax(dim=-1).tolist())

    # ----------------------------------------------------------- inspection

    @property
    def committed_frames(self) -> int:
        """Encoder output frames decoded so far, prompt frames included.

        The first :data:`~streaming.config.NUM_QUERY_FRAMES` of them are the
        query embeddings' own outputs, which is where the *leading* provisional
        rich tag group comes from - not the only place tags appear, as the
        class docstring explains.
        """
        return len(self._frame_ids)

    @property
    def buffered_frames(self) -> int:
        """Frames waiting for a stride to complete, not yet given to the encoder."""
        return 0 if self._buffer is None else int(self._buffer.shape[0])
