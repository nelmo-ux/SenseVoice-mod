"""Chunk-driven streaming inference for SenseVoiceSmall.

This module owns everything that is the same whatever recognition strategy is
in use, and delegates the strategy itself to a *backend*
(``streaming.backends``) selected by :attr:`StreamingConfig.backend`:

1.  Audio pushed through :meth:`StreamingSenseVoice.push_audio` is fed to a
    streaming frontend (``WavFrontendOnline``) which emits LFR/CMVN encoder
    frames (60 ms each) as soon as enough samples are buffered.  The frames go
    straight to the backend.
2.  Every time ``chunk_size`` new frames have accumulated, the backend produces
    one ``partial`` result.  How it does so is its business:
    :class:`~streaming.backends.AccumulateBackend` (the default) re-runs the
    *whole* encoder over a growing window, while
    :class:`~streaming.chunk_backend.ChunkBackend` feeds each frame once into
    the encoder's SCAMA-style streaming cache.
3.  When the segment ends (``is_last=True``) the *raw waveform* of the whole
    segment is handed to ``SenseVoiceSmall.inference`` for a full-quality pass
    (ITN, rich-transcription post-processing), producing one ``final`` result.
    This pass is backend-independent and is always the authoritative result.

The streaming frontend is configured from the offline one with ``dither=0.0``;
under that setting its output is bit-identical to a single offline
``WavFrontend`` call over the concatenated audio (verified on 16 kHz speech), so
partial results see exactly the features the offline model would have seen.

The module targets CPU: no CUDA-specific code paths are used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ``_SegmentState`` moved to ``backends`` with the accumulate strategy that owns
# most of its fields; it is re-exported here because it is still *this* class's
# per-segment state, and callers (and tests) reach for it at its historical
# home.
from .backends import (  # noqa: F401 - re-export
    AccumulateBackend,
    StreamingBackend,
    _SegmentState,
    encode_window,
)
from .chunk_backend import ChunkBackend
from .config import StreamingConfig

__all__ = ["StreamingResult", "StreamingSenseVoice"]

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """One recognition event emitted by :class:`StreamingSenseVoice`.

    Attributes:
        type: ``"partial"`` for an in-progress hypothesis, ``"final"`` for the
            end-of-segment result.
        text: Display text.  Partials have their ``<|...|>`` rich tags stripped;
            finals are passed through ``rich_transcription_postprocess``.
        raw_text: Decoded text before any cleaning, still carrying rich tags.
        start_ms: Utterance start relative to the current segment.  Always
            ``0.0`` - a segment starts where the caller called :meth:`reset`.
        end_ms: Audio consumed so far in the segment, in milliseconds,
            **segment-relative** and always derived from the sample count
            (``total_samples / sample_rate * 1000``) - never from encoder
            frames, so ``partial`` and ``final`` share one unit and one origin.
            Rebasing onto an absolute stream clock is the caller's job
            (:class:`streaming.session.StreamingSession` does it).
    """

    type: str
    text: str
    raw_text: str
    start_ms: float
    end_ms: float


class StreamingSenseVoice:
    """Chunk-driven streaming recogniser around ``SenseVoiceSmall``.

    One instance handles one segment at a time; call :meth:`reset` at every VAD
    speech onset and drive it with :meth:`push_audio`.

    The recognition strategy is chosen by ``config.backend``; everything else -
    the frontend, the retained waveform, the ``end_ms`` clock and the final
    full-quality pass - is the same either way.

    The instance is *not* thread-safe: ``push_audio`` mutates the accumulated
    feature window and the frontend cache.

    Args:
        model_dir: Directory (or model id) accepted by
            ``SenseVoiceSmall.from_pretrained``.
        config: Streaming tunables; defaults to ``StreamingConfig()``.
    """

    def __init__(self, model_dir: str, config: Optional[StreamingConfig] = None) -> None:
        self.config = config or StreamingConfig()
        self.config.validate()

        # Imported lazily so that importing this module does not pull in the
        # whole funasr stack (and its logging side effects).
        from model import SenseVoiceSmall

        model, kwargs = SenseVoiceSmall.from_pretrained(
            model=model_dir, device=self.config.device
        )
        model.eval()
        self.model = model
        self.kwargs: Dict[str, Any] = kwargs
        self.frontend = kwargs["frontend"]
        self.tokenizer = kwargs["tokenizer"]
        self.device = torch.device(self.config.device)

        # MUST come after from_pretrained: funasr's AutoModel calls
        # torch.set_num_threads(ncpu) itself (ncpu defaults to 4 but is not
        # guaranteed), so setting the thread count earlier would be overwritten.
        # Four threads is the measured optimum for this encoder on CPU; six or
        # more makes it 1.6-3x slower.
        torch.set_num_threads(self.config.num_threads)

        self._online_frontend = self._build_online_frontend()
        self._state = _SegmentState()
        self.reset()

    # ------------------------------------------------------------------ setup

    def _build_online_frontend(self) -> Any:
        """Build the incremental frontend mirroring the offline one.

        Every acoustic setting is inherited from the loaded ``WavFrontend`` so
        that streaming features match the offline ones.  ``dither`` is the sole
        exception: it is forced to ``0.0`` because dithering injects per-call
        random noise, which would make chunked extraction differ from a single
        offline call (and make partial results non-reproducible).

        Returns:
            A ``funasr.frontends.wav_frontend.WavFrontendOnline`` instance.
        """
        from funasr.frontends.wav_frontend import WavFrontendOnline

        frontend = self.frontend
        return WavFrontendOnline(
            cmvn_file=frontend.cmvn_file,
            fs=frontend.fs,
            window=frontend.window,
            n_mels=frontend.n_mels,
            frame_length=frontend.frame_length,
            frame_shift=frontend.frame_shift,
            lfr_m=frontend.lfr_m,
            lfr_n=frontend.lfr_n,
            dither=0.0,
            snip_edges=frontend.snip_edges,
            upsacle_samples=frontend.upsacle_samples,
        )

    @property
    def _backend(self) -> StreamingBackend:
        """The recognition strategy, built on first use.

        Built lazily rather than in :meth:`__init__` so that construction and
        :meth:`reset` share one selection path, and so that a subclass which
        replaces the weight-loading ``__init__`` (the test suite does exactly
        that) still gets a working backend from nothing but ``config``.
        """
        backend = getattr(self, "_backend_impl", None)
        if backend is None:
            backend = self._build_backend()
            self._backend_impl = backend
        return backend

    def _build_backend(self) -> StreamingBackend:
        """Instantiate the backend named by ``config.backend``.

        The accumulate backend is handed :meth:`_encode_and_decode` rather than
        the model itself, which keeps model loading on this side of the seam
        and lets a subclass swap the encoder pass out.

        Returns:
            A :class:`~streaming.backends.StreamingBackend`.

        Raises:
            ValueError: If ``config.backend`` names no known strategy.  Normally
                unreachable, since ``config.validate()`` rejects it first.
        """
        if self.config.backend == "accumulate":
            return AccumulateBackend(self.config, self._encode_and_decode)
        if self.config.backend == "chunk":
            return ChunkBackend(
                self.config, self.model, self.tokenizer, self.device
            )
        raise ValueError(f"unknown backend: {self.config.backend!r}")

    # ------------------------------------------------------------- public API

    def reset(self) -> None:
        """Drop all segment state, ready for a new utterance.

        Clears the accumulated encoder frames, the confirmed-prefix text, the
        streaming frontend cache and the retained waveform, and hands the
        backend the fresh state so it can drop whatever it carries of its own.
        Intended to be called on every VAD speech onset.
        """
        self._state = _SegmentState()
        self._online_frontend.init_cache(self._state.frontend_cache)
        self._backend.reset(self._state)

    def push_audio(
        self, samples: "np.ndarray", is_last: bool = False
    ) -> List[StreamingResult]:
        """Feed audio to the recogniser and collect any results it produces.

        Args:
            samples: 1-D float32 array of 16 kHz PCM in ``[-1.0, 1.0]``.  May be
                empty (useful together with ``is_last=True`` to flush).
            is_last: Marks the end of the segment.  Flushes the frontend and
                always emits exactly one ``final`` result - unless no encoder
                frame was ever produced, in which case nothing is emitted.

        Returns:
            Zero or more :class:`StreamingResult`: at most one ``partial`` per
            call - emitted once ``chunk_size`` new frames have accumulated - or
            exactly one ``final`` when ``is_last`` is set.

        Raises:
            ValueError: If ``samples`` is not 1-D.
        """
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError(f"samples must be 1-D, got shape {samples.shape}")
        samples = samples.astype(np.float32, copy=False)

        state = self._state
        if samples.size:
            state.waveform.append(samples)
            state.total_samples += int(samples.size)

        self._extract_frames(samples, is_last=is_last)

        results: List[StreamingResult] = []
        if is_last:
            state.finished = True
            if state.total_frames > 0:
                # No partial here: the final pass covers the same audio with a
                # better decode, and an extra encoder pass would only delay it.
                self._backend.discard_pending()
                results.append(self._infer_final())
            return results

        if self._backend.should_emit_partial():
            # At most one partial per call, however many chunks arrived: the
            # backend consumes every pending frame in one go.
            results.append(self._infer_partial())
        return results

    # ------------------------------------------------------------- frontend

    def _extract_frames(self, samples: "np.ndarray", is_last: bool) -> None:
        """Push samples through the streaming frontend and hand on the frames.

        Frame *extraction* is backend-independent - both strategies consume the
        same 60 ms LFR/CMVN frames in the same order - so it stays here and the
        frames are forwarded to the backend, which decides what to do with them.

        Args:
            samples: 1-D float32 audio for this call (possibly empty).
            is_last: Whether the frontend should flush its internal caches.
        """
        state = self._state
        waveform = torch.from_numpy(samples).unsqueeze(0)
        lengths = torch.tensor([waveform.shape[1]], dtype=torch.int32)
        try:
            feats, _ = self._online_frontend(
                waveform, lengths, is_final=is_last, cache=state.frontend_cache
            )
        except RuntimeError:
            # WavFrontendOnline cannot flush a segment shorter than one fbank
            # window (its LFR splice cache is still empty).  Such a segment has
            # no frames to decode anyway.
            logger.warning(
                "streaming frontend produced no frames for %d samples (is_last=%s)",
                samples.size,
                is_last,
            )
            return
        if not isinstance(feats, torch.Tensor) or feats.ndim != 3 or feats.shape[1] == 0:
            return

        new_frames = feats[0].to(torch.float32)
        self._backend.accept_frames(new_frames)
        state.total_frames += new_frames.shape[0]

    # ------------------------------------------------------------- inference

    def _elapsed_ms(self) -> float:
        """Audio consumed in the current segment, in milliseconds.

        The single source of truth for :attr:`StreamingResult.end_ms`: both the
        ``partial`` and the ``final`` path call this, so the two never disagree
        on the unit.  Encoder frames are deliberately *not* used - a frame is
        only produced once the frontend has buffered a whole 60 ms window, so
        frame counts lag the audio actually pushed and would make a ``partial``
        report an earlier end time than a ``final`` over the same samples.

        Returns:
            ``total_samples / sample_rate * 1000``, relative to the last
            :meth:`reset`.
        """
        return self._state.total_samples * 1000.0 / self.config.sample_rate

    def _infer_partial(self) -> StreamingResult:
        """Ask the backend for the current hypothesis and time-stamp it.

        Returns:
            The ``partial`` result for the audio accumulated so far.
        """
        text, raw_text = self._backend.emit_partial()
        return StreamingResult(
            type="partial",
            text=text,
            raw_text=raw_text,
            start_ms=0.0,
            end_ms=self._elapsed_ms(),
        )

    def _encode_and_decode(self, features: Optional[torch.Tensor]) -> str:
        """Encode one feature window and greedily decode it.

        The seam between the loaded model and
        :class:`~streaming.backends.AccumulateBackend`: it binds
        :func:`~streaming.backends.encode_window` to this instance's model,
        tokenizer and device, so the backend can stay free of model loading.
        Overriding it replaces the encoder pass wholesale.

        Args:
            features: Encoder frames of the current window, shape ``(T, D)``.

        Returns:
            The raw decoded text, rich tags included; ``""`` for an empty
            window.
        """
        return encode_window(
            self.model, self.tokenizer, self.config, features, self.device
        )

    def _infer_final(self) -> StreamingResult:
        """Run the full-quality pass over the retained waveform.

        Falls back to the lightweight chunk path (and logs the failure) if the
        full pass raises, so a segment always yields a ``final``.

        Returns:
            The ``final`` result for the segment.
        """
        end_ms = self._elapsed_ms()
        try:
            raw_text = self._full_inference()
        except Exception:  # noqa: BLE001 - a final must always be emitted
            logger.exception("full inference failed; falling back to the chunk decode")
            fallback = self._infer_partial()
            return StreamingResult(
                type="final",
                text=fallback.text,
                raw_text=fallback.raw_text,
                start_ms=0.0,
                end_ms=end_ms,
            )

        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return StreamingResult(
            type="final",
            text=rich_transcription_postprocess(raw_text),
            raw_text=raw_text,
            start_ms=0.0,
            end_ms=end_ms,
        )

    def _full_inference(self) -> str:
        """Call ``SenseVoiceSmall.inference`` on the whole retained waveform.

        The build kwargs returned by ``from_pretrained`` (which carry the
        ``frontend`` and ``tokenizer`` the call needs, plus the whole model
        config) are merged *under* the arguments chosen here, so a build kwarg
        that happens to share a name can never collide with an explicit one -
        ``inference(**self.kwargs, language=...)`` would raise ``TypeError:
        got multiple values for keyword argument`` the moment a checkpoint
        shipped e.g. ``language`` in its config.

        Explicitly specified (these win over ``self.kwargs``):
        ``data_in``, ``language``, ``use_itn``, ``ban_emo_unk``, ``key``,
        ``fs``.  Everything else - notably ``frontend`` and ``tokenizer`` - is
        taken from ``self.kwargs`` unchanged.

        Returns:
            The raw text of the first (only) hypothesis, tags included.
        """
        state = self._state
        if not state.waveform:
            return ""
        waveform = torch.from_numpy(np.concatenate(state.waveform))
        call_kwargs: Dict[str, Any] = {
            **self.kwargs,
            "data_in": [waveform],
            "language": self.config.language,
            "use_itn": self.config.use_itn,
            "ban_emo_unk": self.config.ban_emo_unk,
            "key": ["stream"],
            "fs": self.config.sample_rate,
        }
        with torch.inference_mode():
            results, _ = self.model.inference(**call_kwargs)
        if not results:
            return ""
        return results[0].get("text", "")
