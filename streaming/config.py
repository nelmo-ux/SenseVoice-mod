"""Configuration for the SenseVoice streaming pipeline.

All timing in this package is expressed in *encoder frames*.  SenseVoiceSmall
stacks its 10 ms mel frames with an LFR factor of 6, so one encoder frame is
exactly 60 ms of audio (:data:`MS_PER_FRAME`).  On top of the acoustic frames
the model prepends four query embeddings (language / event / emotion /
textnorm, :data:`NUM_QUERY_FRAMES`) - they consume encoder compute but carry no
audio, which is why they are named here rather than folded into the frame math.

The defaults in :class:`StreamingConfig` come from a CPU benchmark of the
encoder (``streaming/bench_cpu.py``, Apple silicon, 4 threads); see the class
docstring for the numbers behind each one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "MS_PER_FRAME",
    "NUM_QUERY_FRAMES",
    "SUPPORTED_BACKENDS",
    "VAD_STREAMING_CHUNK_LIMIT_MS",
    "StreamingConfig",
]

#: Audio duration of a single encoder frame, in milliseconds.
#: 10 ms mel hop x LFR factor 6 = 60 ms.
MS_PER_FRAME: float = 60.0

#: Non-acoustic query frames SenseVoiceSmall prepends to every forward pass
#: (language, event, emotion, textnorm).  They add to encoder cost but not to
#: the audio duration of a chunk.
NUM_QUERY_FRAMES: int = 4

#: Chunk length, in milliseconds, at which FunASR's VAD stops behaving as a
#: streaming detector: ``fsmn_vad_streaming`` defaults ``is_streaming_input`` to
#: ``False`` for ``chunk_size >= 15000``, which makes every call a complete
#: utterance.  :attr:`StreamingConfig.vad_chunk_ms` must stay below it.
VAD_STREAMING_CHUNK_LIMIT_MS: float = 15000.0

#: Recognition strategies :class:`StreamingConfig.backend` may name.  See
#: ``streaming.backends`` for the protocol they implement and
#: ``streaming.chunk_backend`` for the SCAMA-style one.
SUPPORTED_BACKENDS: Tuple[str, ...] = ("accumulate", "chunk")


@dataclass
class StreamingConfig:
    """Tunables for streaming recognition.

    Defaults and their justification (CPU measurements, encoder forward pass):

    ``chunk_size = 12`` (720 ms)
        The encoder has a large fixed cost per pass - roughly **430 ms**
        regardless of how short the input is - so the chunk must be long enough
        to amortise it.  At a 720 ms chunk the measured p95 is **521 ms**, an
        occupancy of **72 %** of real time, which leaves headroom for the
        frontend, VAD and decoding.  A smaller ``chunk_size = 8`` (480 ms)
        pushes occupancy to **95-109 %**: the pipeline can no longer keep up
        with real time and latency grows without bound.

    ``max_history = 167`` frames (~10 s)
        Upper bound on the encoder context retained across chunks.  Ten seconds
        covers a typical utterance while keeping the attention cost bounded;
        longer segments are cut by ``max_segment_sec``.

    ``num_threads = 4``
        Measured optimum for the encoder on Apple silicon.  More threads did
        not reduce latency (the model is small enough that scheduling overhead
        dominates) and starve the audio capture thread.

    ``max_segment_sec = 30.0``
        Forced cut when VAD never reports an endpoint (continuous speech, noisy
        input, or a VAD failure), so a segment can never grow unboundedly.

    ``vad_chunk_ms = 480.0``
        Chunk length announced to fsmn-vad, deliberately **independent** of
        ``chunk_size``.  The two solve different problems: the recogniser's
        chunk trades latency against encoder occupancy, while the VAD's chunk
        only sets the granularity of its endpoint decisions (fsmn-vad is cheap
        - a few milliseconds per chunk - so it does not need amortising).
        Measured on ``runtime/llama.cpp/tests/sample.wav``, a 480 ms chunk
        places the endpoints at **770 ms** / **5980 ms**, matching the offline
        reference; tying the VAD to the recogniser's 720 ms chunk would only
        coarsen those decisions for no gain.  Note that fsmn-vad switches to
        non-streaming semantics at ``chunk_size >= 15000`` ms, so the value
        must stay well below that.

    Chunk-backend geometry (``backend = "chunk"`` only).  **These defaults are
    not measured.**  Every number above comes from a benchmark; the four fields
    below do not - no streaming WER or latency measurement of the chunk backend
    exists yet.  They mirror the middle entry of the dynamic chunk-mask
    configuration the model is finetuned with in ``finetune_chunk.sh``
    (``chunk_size=[8,12,16]``, ``stride=[6,10,14]``, ``pad_left=[0,0,0]``,
    ``encoder_att_look_back_factor=[1,1,1]``), on the reasoning that decoding
    should use a geometry the encoder was actually trained on.  Treat them as a
    starting point to benchmark, not as a tuned optimum:

    ``chunk_pad_left = 0`` / ``chunk_stride = 10`` / ``chunk_pad_right = 2``
        The training entry ``chunk_size=12, stride=10, pad_left=0`` (so
        ``pad_right = 12 - 10 - 0 = 2``).  ``pad_left = 0`` is also what
        ``SenseVoiceEncoderSmall.init_chunk_cache`` recommends whenever
        encoder look-back is on.  The two lookahead frames cost 120 ms of
        added latency (:attr:`chunk_lookahead_ms`).

    ``chunk_encoder_look_back = 1``
        One previous chunk of encoder self-attention context, matching
        ``encoder_att_look_back_factor=1`` in training.  It requires
        ``chunk_pad_right >= 1``.

    Note:
        ``chunk_size`` is *not* the encoder's chunk width.  It is the emission
        cadence - how many new encoder frames must arrive before a partial is
        produced - and it applies to both backends.  The chunk backend's
        encoder window is ``chunk_pad_left + chunk_stride + chunk_pad_right``,
        which is stepped several times per emitted partial when the two differ.

    Attributes:
        backend: Recognition strategy, one of :data:`SUPPORTED_BACKENDS`.
            ``"accumulate"`` re-runs the full encoder over a growing window
            (the default, and the only path with measured numbers);
            ``"chunk"`` feeds each frame once into the encoder's streaming
            cache.
        chunk_size: Encoder frames accumulated before each inference.
        max_history: Encoder frames retained as context across chunks.
        sample_rate: Input audio sample rate in Hz (the model expects 16 kHz).
        device: Torch device string; this pipeline targets ``"cpu"``.
        language: SenseVoice language tag (``"auto"``, ``"zh"``, ``"en"``, ...).
        use_itn: Emit inverse-text-normalised output (punctuation, numerals).
        ban_emo_unk: Suppress the ``unk`` emotion token during decoding.
        max_segment_sec: Hard cap on a segment's duration, in seconds.
        num_threads: Value passed to ``torch.set_num_threads``.
        vad_model: FunASR VAD model id used for endpointing.
        vad_chunk_ms: Chunk length passed to the VAD, in milliseconds.  It is
            *not* derived from ``chunk_size``: see above.
        chunk_pad_left: Left-context frames the chunk backend carries over from
            the previous encoder window.
        chunk_stride: Frames committed per ``forward_chunk`` call - exactly how
            many new frames the chunk backend feeds the encoder each step.
        chunk_pad_right: Lookahead frames the encoder sees but withholds from
            its output until the next step; the source of the backend's added
            latency.
        chunk_encoder_look_back: Previous chunks the encoder self-attention may
            attend to through the per-layer key/value cache.  ``0`` keeps
            attention inside the current window, ``-1`` keeps every past chunk,
            and any non-zero value requires ``chunk_pad_right >= 1``.
    """

    backend: str = "accumulate"
    chunk_size: int = 12
    max_history: int = 167
    sample_rate: int = 16000
    device: str = "cpu"
    language: str = "auto"
    use_itn: bool = True
    ban_emo_unk: bool = False
    max_segment_sec: float = 30.0
    num_threads: int = 4
    vad_model: str = "fsmn-vad"
    vad_chunk_ms: float = 480.0
    chunk_pad_left: int = 0
    chunk_stride: int = 10
    chunk_pad_right: int = 2
    chunk_encoder_look_back: int = 1

    @property
    def chunk_ms(self) -> float:
        """Audio duration of one inference chunk, in milliseconds."""
        return self.chunk_size * MS_PER_FRAME

    @property
    def chunk_samples(self) -> int:
        """Number of audio samples in one inference chunk."""
        return int(self.chunk_ms * self.sample_rate / 1000)

    @property
    def max_history_ms(self) -> float:
        """Audio duration of the retained encoder history, in milliseconds."""
        return self.max_history * MS_PER_FRAME

    @property
    def chunk_lookahead_ms(self) -> float:
        """Latency the configured chunk geometry implies, in milliseconds.

        Under the chunk backend a frame is only emitted once its
        ``chunk_pad_right`` successors have been fed, so this is how far behind
        the audio clock that backend's partials necessarily run.

        The value is pure geometry: it is derived from ``chunk_pad_right``
        alone and never consults :attr:`backend`, so it keeps describing what
        the ``chunk_*`` fields *configure* even while ``backend ==
        "accumulate"`` - a backend that reads none of them and adds no
        lookahead of its own.  It is ``0.0`` exactly when
        ``chunk_pad_right == 0``.
        """
        return self.chunk_pad_right * MS_PER_FRAME

    def validate(self) -> None:
        """Check that the configuration is internally consistent.

        Raises:
            ValueError: If any field is out of range, if ``backend`` is not one
                of :data:`SUPPORTED_BACKENDS`, if ``max_history`` is smaller
                than ``chunk_size`` (the history must hold at least the chunk
                currently being decoded), if the chunk geometry is one
                ``SenseVoiceEncoderSmall.init_chunk_cache`` would reject, or if
                ``vad_chunk_ms`` is outside the range in which FunASR treats
                the input as a stream.

        Note:
            The chunk geometry is checked here as well as in
            ``init_chunk_cache`` so that a typo fails at construction time,
            with a message naming the config field, rather than on the first
            chunk of audio.  Both checks encode the same rules.
        """
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {SUPPORTED_BACKENDS}, got "
                f"{self.backend!r}"
            )
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.max_history < self.chunk_size:
            raise ValueError(
                f"max_history ({self.max_history}) must be >= "
                f"chunk_size ({self.chunk_size})"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.max_segment_sec <= 0:
            raise ValueError(
                f"max_segment_sec must be > 0, got {self.max_segment_sec}"
            )
        if self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")
        if not self.device:
            raise ValueError("device must be a non-empty string")
        if not self.language:
            raise ValueError("language must be a non-empty string")
        if not self.vad_model:
            raise ValueError("vad_model must be a non-empty string")
        if self.vad_chunk_ms <= 0:
            raise ValueError(f"vad_chunk_ms must be > 0, got {self.vad_chunk_ms}")
        if self.vad_chunk_ms >= VAD_STREAMING_CHUNK_LIMIT_MS:
            # At or above this bound FunASR flips ``is_streaming_input`` off and
            # treats every call as a complete utterance, which breaks the
            # incremental endpointing this package is built on.
            raise ValueError(
                f"vad_chunk_ms must be < {VAD_STREAMING_CHUNK_LIMIT_MS} to keep "
                f"the VAD in streaming mode, got {self.vad_chunk_ms}"
            )
        if self.chunk_stride < 1:
            raise ValueError(f"chunk_stride must be >= 1, got {self.chunk_stride}")
        if self.chunk_pad_left < 0:
            raise ValueError(
                f"chunk_pad_left must be >= 0, got {self.chunk_pad_left}"
            )
        if self.chunk_pad_right < 0:
            raise ValueError(
                f"chunk_pad_right must be >= 0, got {self.chunk_pad_right}"
            )
        if self.chunk_encoder_look_back < -1:
            raise ValueError(
                "chunk_encoder_look_back must be >= -1 (-1 keeps every past "
                f"chunk, 0 disables look-back), got {self.chunk_encoder_look_back}"
            )
        if self.chunk_encoder_look_back != 0 and self.chunk_pad_right < 1:
            # The encoder builds its attention cache by dropping the last
            # ``pad_right`` frames of the window, so at ``pad_right == 0`` the
            # cache it would keep is empty.
            raise ValueError(
                "chunk_encoder_look_back != 0 requires chunk_pad_right >= 1, "
                f"got chunk_encoder_look_back={self.chunk_encoder_look_back} "
                f"and chunk_pad_right={self.chunk_pad_right}"
            )
