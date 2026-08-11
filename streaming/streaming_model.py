"""Chunk-driven streaming inference for SenseVoiceSmall.

Design: *accumulate inside the VAD segment, re-run the whole encoder per chunk.*

SenseVoiceSmall is a non-autoregressive CTC model whose encoder uses full
(non-causal) SANM attention, so there is no incremental state to carry between
chunks.  Rather than approximating one with truncated attention, this module
keeps the exact offline computation and simply repeats it as audio arrives:

1.  Audio pushed through :meth:`StreamingSenseVoice.push_audio` is fed to a
    streaming frontend (``WavFrontendOnline``) which emits LFR/CMVN encoder
    frames (60 ms each) as soon as enough samples are buffered.
2.  Every time ``chunk_size`` new frames have accumulated, the *whole* window of
    accumulated frames - prefixed with the four query embeddings the model
    expects - is run through ``SenseVoiceEncoderSmall.forward`` and decoded with
    greedy CTC, producing one ``partial`` result.
3.  When the segment ends (``is_last=True``) the *raw waveform* of the whole
    segment is handed to ``SenseVoiceSmall.inference`` for a full-quality pass
    (ITN, rich-transcription post-processing), producing one ``final`` result.

The streaming frontend is configured from the offline one with ``dither=0.0``;
under that setting its output is bit-identical to a single offline
``WavFrontend`` call over the concatenated audio (verified on 16 kHz speech), so
partial results see exactly the features the offline model would have seen.

The module targets CPU: no CUDA-specific code paths are used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import StreamingConfig
from .ctc_decode import ctc_greedy_decode, strip_rich_tags

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


@dataclass
class _SegmentState:
    """Mutable per-segment state, reset by :meth:`StreamingSenseVoice.reset`."""

    #: Encoder frames of the *current window* only, shape ``(T, D)``.
    features: Optional[torch.Tensor] = None
    #: Frames at the front of ``features`` already covered by ``last_raw_text``.
    decoded_frames: int = 0
    #: Raw decode of the current window as of the last inference.
    last_raw_text: str = ""
    #: Display text of every window already retired by the history cap.
    confirmed_text: str = ""
    #: Raw text (tags kept) of every window already retired by the history cap.
    confirmed_raw: str = ""
    #: New frames accumulated since the last inference.
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


class StreamingSenseVoice:
    """Chunk-driven streaming recogniser around ``SenseVoiceSmall``.

    One instance handles one segment at a time; call :meth:`reset` at every VAD
    speech onset and drive it with :meth:`push_audio`.

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

    # ------------------------------------------------------------- public API

    def reset(self) -> None:
        """Drop all segment state, ready for a new utterance.

        Clears the accumulated encoder frames, the confirmed-prefix text, the
        streaming frontend cache and the retained waveform.  Intended to be
        called on every VAD speech onset.
        """
        self._state = _SegmentState()
        self._online_frontend.init_cache(self._state.frontend_cache)

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
                state.pending_frames = 0
                results.append(self._infer_final())
            return results

        if state.pending_frames >= self.config.chunk_size:
            # One inference per call, however many chunks arrived: every pass
            # decodes the *whole* window, so running it twice back to back would
            # burn a second encoder pass on an identical hypothesis.
            state.pending_frames = 0
            results.append(self._infer_partial())
        return results

    # ------------------------------------------------------------- frontend

    def _extract_frames(self, samples: "np.ndarray", is_last: bool) -> None:
        """Push samples through the streaming frontend and accumulate frames.

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
        if state.features is None:
            state.features = new_frames.clone()
        else:
            state.features = torch.cat((state.features, new_frames), dim=0)
        state.pending_frames += new_frames.shape[0]
        state.total_frames += new_frames.shape[0]

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
        max_history = self.config.max_history
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
        """Run one chunk-driven encoder pass and build a ``partial`` result.

        Returns:
            The ``partial`` result for the audio accumulated so far.
        """
        self._apply_history_cap()
        state = self._state
        raw_window = self._encode_and_decode(state.features)
        state.last_raw_text = raw_window
        state.decoded_frames = 0 if state.features is None else state.features.shape[0]

        raw_text = state.confirmed_raw + raw_window
        text = state.confirmed_text + strip_rich_tags(raw_window)
        return StreamingResult(
            type="partial",
            text=text,
            raw_text=raw_text,
            start_ms=0.0,
            end_ms=self._elapsed_ms(),
        )

    def _encode_and_decode(self, features: Optional[torch.Tensor]) -> str:
        """Encode one feature window and greedily decode it.

        The four query embeddings are prepended exactly as
        ``SenseVoiceSmall.inference`` does: ``textnorm`` first (+1), then
        ``language`` and the event/emotion pair (+3), giving the frame order
        ``[language, event, emotion, textnorm, speech...]``.

        Args:
            features: Encoder frames of the current window, shape ``(T, D)``.

        Returns:
            The raw decoded text, rich tags included; ``""`` for an empty
            window.
        """
        if features is None or features.shape[0] == 0:
            return ""

        model = self.model
        with torch.inference_mode():
            speech = features.unsqueeze(0).to(self.device)

            language = self.config.language
            language_id = model.lid_dict.get(language, 0)
            language_query = model.embed(
                torch.LongTensor([[language_id]]).to(speech.device)
            )
            textnorm = "withitn" if self.config.use_itn else "woitn"
            textnorm_query = model.embed(
                torch.LongTensor([[model.textnorm_dict[textnorm]]]).to(speech.device)
            )
            event_emo_query = model.embed(torch.LongTensor([[1, 2]]).to(speech.device))

            # SenseVoiceEncoderSmall.forward scales its input in place
            # (``xs_pad *= output_size ** 0.5``).  torch.cat allocates a fresh
            # tensor, so the accumulated ``features`` are never touched - feeding
            # the cached tensor directly would inflate it by 22.6x per chunk.
            speech = torch.cat(
                (language_query, event_emo_query, textnorm_query, speech), dim=1
            )
            speech_lengths = torch.tensor(
                [speech.shape[1]], dtype=torch.int32, device=speech.device
            )

            encoder_out, encoder_out_lens = model.encoder(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

            ctc_logits = model.ctc.log_softmax(encoder_out)
            if self.config.ban_emo_unk:
                ctc_logits[:, :, model.emo_dict["unk"]] = -float("inf")

            logits_2d = ctc_logits[0, : int(encoder_out_lens[0].item()), :]
            return ctc_greedy_decode(
                logits_2d, self.tokenizer, blank_id=model.blank_id
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
