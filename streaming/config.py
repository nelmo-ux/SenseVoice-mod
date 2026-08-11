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

__all__ = [
    "MS_PER_FRAME",
    "NUM_QUERY_FRAMES",
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

    Attributes:
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
    """

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

    def validate(self) -> None:
        """Check that the configuration is internally consistent.

        Raises:
            ValueError: If any field is out of range, if ``max_history`` is
                smaller than ``chunk_size`` (the history must hold at least the
                chunk currently being decoded), or if ``vad_chunk_ms`` is
                outside the range in which FunASR treats the input as a stream.
        """
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
