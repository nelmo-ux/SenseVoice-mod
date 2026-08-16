"""Unit tests for the pluggable streaming backends.

Covers ``streaming/backends.py`` (the :class:`StreamingBackend` protocol and
:class:`AccumulateBackend`), ``streaming/chunk_backend.py``
(:class:`ChunkBackend`) and the ``backend`` / ``chunk_*`` half of
``streaming/config.py`` that selects and shapes them.

**No checkpoint is loaded.**  ``models/SenseVoiceSmall`` costs ~900 MB and
several seconds, and none of the behaviour pinned here is about the weights:
it is frame bookkeeping, window bookkeeping and configuration validation.  Two
levels of faking are used, following the existing suite:

* Where no encoder is needed at all (:class:`AccumulateBackend`'s window
  arithmetic, config validation) the encoder pass is a plain callable that
  records the window it was handed - the same trick
  ``tests/test_streaming_ws_server.py`` plays on the recogniser factory.
* Where a *real* encoder genuinely is needed - anything that depends on
  ``forward_chunk``'s emission schedule - a small randomly initialised
  ``SenseVoiceEncoderSmall`` is built from a fixed seed exactly as
  ``tests/test_chunk_streaming_equivalence.py`` does, and wrapped in
  :class:`FakeSenseVoice` so it exposes the handful of attributes the backends
  read off a real ``SenseVoiceSmall`` (``embed``, ``ctc``, ``blank_id``,
  ``lid_dict``, ``textnorm_dict``, ``emo_dict``).

Random weights make the decoded *text* meaningless, so the chunk-backend tests
assert frame counts, call schedules and cache identity rather than strings.
That is deliberate: every claim in ``ChunkBackend``'s docstring that could
regress silently is a counting claim.

The two frame-alignment invariants
----------------------------------
They are **different numbers**, and both are correct:

* Driven directly, with the tail flush invoked, the backend commits
  ``NUM_QUERY_FRAMES + N`` output frames for ``N`` input frames.  That is the
  backend's own contract.
* Driven through ``StreamingSenseVoice.push_audio(is_last=True)`` it commits
  ``NUM_QUERY_FRAMES + committed_input - chunk_pad_right``, because the
  recogniser calls ``discard_pending()`` and goes straight to the full-quality
  final pass; ``_flush_tail`` is only reached on the *fallback* path.  See
  ``test_recogniser_leaves_the_lookahead_withheld_because_the_final_pass_supersedes_it``.
"""

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from model import SenseVoiceEncoderSmall  # noqa: E402
from streaming.backends import (  # noqa: E402
    AccumulateBackend,
    StreamingBackend,
    _SegmentState,
    build_prompt_frames,
    encode_window,
)
from streaming.chunk_backend import ChunkBackend  # noqa: E402
from streaming.config import (  # noqa: E402
    MS_PER_FRAME,
    NUM_QUERY_FRAMES,
    SUPPORTED_BACKENDS,
    StreamingConfig,
)
from streaming.streaming_model import StreamingSenseVoice  # noqa: E402

# --------------------------------------------------------------------------
# A checkpoint-free stand-in for SenseVoiceSmall
# --------------------------------------------------------------------------

# Encoder geometry: small enough to keep the sweep below fast, and identical in
# spirit to tests/test_chunk_streaming_equivalence.py.  KERNEL_SIZE is
# SenseVoiceSmall's real value (11) because the FSMN half-width is what makes a
# zero-length streaming input crash - see the empty-flush test.
INPUT_SIZE = 20
OUTPUT_SIZE = 64
ATTENTION_HEADS = 4
LINEAR_UNITS = 128
NUM_BLOCKS = 3
TP_BLOCKS = 2
KERNEL_SIZE = 11
VOCAB_SIZE = 32
SEED = 0

#: Samples in one encoder frame at 16 kHz: 60 ms, i.e. ``MS_PER_FRAME``.
FRAME_SAMPLES = 960


class StubCTC:
    """The CTC head reduced to what the backends call: ``log_softmax``.

    A fixed random projection is enough - the backends only ever argmax the
    result, and with random encoder weights the resulting token ids carry no
    meaning anyway.
    """

    def __init__(self, seed: int = SEED) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(OUTPUT_SIZE, VOCAB_SIZE, generator=generator)

    def log_softmax(self, encoder_out: "torch.Tensor") -> "torch.Tensor":
        """Project the encoder output onto the vocabulary and normalise it."""
        return torch.log_softmax(encoder_out @ self.weight, dim=-1)


#: Rich tag the stub tokenizer prefixes to every decode, standing in for the
#: ``<|en|><|HAPPY|><|Speech|><|withitn|>`` markers a real decode starts with.
#: It exists so that "partials have their tags stripped" is observable at all.
STUB_RICH_TAG = "<|stub|>"


class StubTokenizer:
    """Turns token ids into a reproducible, tag-carrying string.

    No vocabulary is involved: with random encoder weights the ids are
    meaningless, so the rendering only has to be injective and to carry one rich
    tag.
    """

    def decode(self, token_ids: Sequence[int]) -> str:
        """Render ids as ``|id|`` behind a single rich tag."""
        return STUB_RICH_TAG + "".join(f"|{int(i)}|" for i in token_ids)


class FakeSenseVoice:
    """``SenseVoiceSmall``'s surface as far as the backends are concerned.

    The encoder is real (randomly initialised); everything else is the smallest
    object that answers the attribute lookups in ``build_prompt_frames``,
    ``encode_window`` and ``ChunkBackend``.
    """

    def __init__(self, seed: int = SEED) -> None:
        torch.manual_seed(seed)
        self.encoder = SenseVoiceEncoderSmall(
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            attention_heads=ATTENTION_HEADS,
            linear_units=LINEAR_UNITS,
            num_blocks=NUM_BLOCKS,
            tp_blocks=TP_BLOCKS,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            kernel_size=KERNEL_SIZE,
            sanm_shfit=0,
        )
        self.encoder.eval()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, INPUT_SIZE)
        self.embed.eval()
        self.ctc = StubCTC(seed=seed)
        self.blank_id = 0
        self.emo_dict = {"unk": 1}
        self.lid_dict = {"auto": 0, "en": 4, "zh": 5}
        self.textnorm_dict = {"withitn": 6, "woitn": 7}


@pytest.fixture(scope="module")
def fake_model() -> FakeSenseVoice:
    """One checkpoint-free model for the whole module.

    Module-scoped because building it is the expensive part and nothing here
    mutates it: both backends keep all their state in themselves or in the
    ``_SegmentState``, never in the model.
    """
    return FakeSenseVoice()


@pytest.fixture(scope="module")
def tokenizer() -> StubTokenizer:
    """The stub tokenizer, shared - it is stateless."""
    return StubTokenizer()


def make_frames(count: int, start: int = 0, seed: int = 1) -> "torch.Tensor":
    """``(count, INPUT_SIZE)`` deterministic pseudo-acoustic frames.

    ``start`` selects a slice of one fixed stream so that frames pushed across
    several calls are the same frames a single push would have produced.
    """
    generator = torch.Generator().manual_seed(seed)
    stream = torch.randn(4096, INPUT_SIZE, generator=generator)
    return stream[start : start + count].clone()


def indexed_frames(count: int, start: int = 0) -> "torch.Tensor":
    """``(count, 1)`` frames whose single feature is the frame's own index.

    Lets an accumulate-backend test assert *which* frames an encoder pass saw,
    the same device ``tests/test_streaming_model.py`` uses.
    """
    return torch.arange(start, start + count, dtype=torch.float32).unsqueeze(1)


class RecordingEncode:
    """Stands in for ``StreamingSenseVoice._encode_and_decode``.

    Records the frame indices of every window it is handed and returns scripted
    text, so a test can pin both *how many* encoder passes happened and *what
    frames* each one covered.

    Args:
        decodes: Text returned by successive calls; once exhausted, calls fall
            back to ``"w<n>"``.
        render: Overrides ``decodes`` entirely, turning the window's frame
            indices into text.  Used to make "which frames does this text stand
            for" readable straight off the emitted string.
    """

    def __init__(
        self,
        decodes: Sequence[str] = (),
        render: Optional[Callable[[List[int]], str]] = None,
    ) -> None:
        self.windows: List[List[int]] = []
        self._decodes = list(decodes)
        self._render = render

    def __call__(self, features: Optional["torch.Tensor"]) -> str:
        """Record the window and return the next scripted decode."""
        ids = [] if features is None else [int(v) for v in features[:, 0].tolist()]
        self.windows.append(ids)
        if not ids:
            return ""
        if self._render is not None:
            return self._render(ids)
        index = len(self.windows) - 1
        if index < len(self._decodes):
            return self._decodes[index]
        return f"w{index}"

    @property
    def calls(self) -> int:
        """Number of encoder passes performed."""
        return len(self.windows)


def chunk_config(
    pad_left: int,
    stride: int,
    pad_right: int,
    look_back: int,
    chunk_size: int = 12,
) -> StreamingConfig:
    """A validated ``backend="chunk"`` config for one encoder geometry."""
    config = StreamingConfig(
        backend="chunk",
        chunk_size=chunk_size,
        chunk_pad_left=pad_left,
        chunk_stride=stride,
        chunk_pad_right=pad_right,
        chunk_encoder_look_back=look_back,
    )
    config.validate()
    return config


def build_chunk_backend(
    model: FakeSenseVoice,
    tok: StubTokenizer,
    config: StreamingConfig,
) -> "tuple[ChunkBackend, _SegmentState]":
    """A fresh :class:`ChunkBackend` bound to a fresh segment state."""
    backend = ChunkBackend(config, model, tok, torch.device("cpu"))
    state = _SegmentState()
    backend.reset(state)
    return backend, state


def feed(backend: ChunkBackend, total: int, pattern: Sequence[int]) -> None:
    """Push ``total`` frames in irregularly sized calls cycling ``pattern``.

    Uneven pushes are the realistic case - the frontend emits whatever a block
    of audio yields - and they are what forces the backend's stride buffering
    to do real work.
    """
    pushed = 0
    index = 0
    while pushed < total:
        size = min(pattern[index % len(pattern)], total - pushed)
        backend.accept_frames(make_frames(size, start=pushed))
        pushed += size
        index += 1


class SpyForwardChunk:
    """Wraps ``encoder.forward_chunk`` and records every input it is given.

    ``register_forward_pre_hook`` is useless here: ``ChunkBackend`` calls
    ``encoder.forward_chunk`` directly rather than through
    ``nn.Module.__call__``, so no hook fires.  Wrapping the bound method is the
    equivalent interception point (same technique as
    ``tests/test_chunk_streaming_equivalence.py``).
    """

    def __init__(self, encoder: SenseVoiceEncoderSmall) -> None:
        self._encoder = encoder
        self._original = encoder.forward_chunk
        self.inputs: List["torch.Tensor"] = []
        self.tail_flags: List[bool] = []

    def __enter__(self) -> "SpyForwardChunk":
        def spy(xs_pad, cache=None, **kwargs):
            self.inputs.append(xs_pad.detach().clone())
            self.tail_flags.append(bool(cache and cache.get("tail_chunk", False)))
            return self._original(xs_pad, cache, **kwargs)

        self._encoder.forward_chunk = spy
        return self

    def __exit__(self, *exc: Any) -> None:
        self._encoder.forward_chunk = self._original

    @property
    def lengths(self) -> List[int]:
        """Time dimension of every ``forward_chunk`` input, in call order."""
        return [int(x.shape[1]) for x in self.inputs]


# --------------------------------------------------------------------------
# (1) Frame alignment - everything else about the chunk path depends on it
# --------------------------------------------------------------------------

# ``(pad_left, stride, pad_right, encoder_look_back)``.  Chosen to cover
# ``pad_right == 0`` (nothing withheld, so no tail replay at all),
# ``pad_left > 0`` (a non-empty left context seeded into the cache), both
# look-back settings, and the shipped default ``(0, 10, 2, 1)``.  Combinations
# with ``pad_right == 0`` must pair with ``look_back == 0``: the attention cache
# is built by dropping the last ``pad_right`` frames and would be empty.
CHUNK_GEOMETRIES = [
    (0, 10, 2, 1),  # the shipped default (finetune_chunk.sh's middle entry)
    (0, 6, 2, 1),
    (0, 10, 0, 0),
    (5, 10, 0, 0),
    (3, 8, 4, 1),
    (5, 10, 5, 0),
]

# Frame counts spanning: fewer frames than one stride (nothing is committed
# until the flush), an exact multiple of every stride above, and counts leaving
# a partial remainder in the buffer.
FRAME_COUNTS = [1, 7, 10, 23, 40]

#: Irregular push sizes, cycled, so no test pushes in stride-sized blocks.
PUSH_PATTERN = (1, 3, 7, 2)


@pytest.mark.parametrize("pad_left,stride,pad_right,look_back", CHUNK_GEOMETRIES)
@pytest.mark.parametrize("frames", FRAME_COUNTS)
def test_tail_flush_commits_exactly_one_output_frame_per_input_frame_plus_the_prompt(
    fake_model, tokenizer, pad_left, stride, pad_right, look_back, frames
):
    """The backend's central contract: committed == ``NUM_QUERY_FRAMES + N``.

    Every downstream claim about the chunk path rests on this.  CTC collapse is
    order-sensitive and position-blind, so a backend that dropped the withheld
    lookahead, double-fed the buffered remainder or re-prepended the prompt
    would still return plausible text - just text missing or repeating the end
    of every utterance.  Counting output frames is the only cheap way to catch
    all three.

    The arithmetic, for reference: the first ``forward_chunk`` emits
    ``NUM_QUERY_FRAMES + stride - pad_right`` (the prompt rides in front of the
    first chunk), later calls emit their input length, and the ``tail_chunk``
    replay emits the ``pad_right`` frames whose lookahead never arrived - so the
    ``- pad_right`` and the ``+ pad_right`` cancel exactly.

    Frames are pushed in irregularly sized calls, so this also pins that
    buffering a sub-stride remainder neither loses nor duplicates frames.
    """
    config = chunk_config(pad_left, stride, pad_right, look_back)
    backend, state = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, frames, PUSH_PATTERN)

    # ``finished`` is what the recogniser sets on ``is_last``; reaching
    # ``emit_partial`` with it set is the fallback path, and the only public
    # route to the tail flush.
    state.finished = True
    backend.emit_partial()

    assert backend.committed_frames == NUM_QUERY_FRAMES + frames
    assert backend.buffered_frames == 0


@pytest.mark.parametrize("pad_left,stride,pad_right,look_back", CHUNK_GEOMETRIES)
@pytest.mark.parametrize("frames", [7, 23, 40])
def test_committed_frames_before_the_flush_are_short_by_exactly_the_lookahead(
    fake_model, tokenizer, pad_left, stride, pad_right, look_back, frames
):
    """Mid-stream the backend is ``pad_right`` behind, plus whatever is buffered.

    This is the other half of the alignment contract and the reason
    :attr:`StreamingConfig.chunk_lookahead_ms` exists: a partial emitted now
    cannot describe the last ``pad_right`` frames, because their right context
    has not been fed yet.  Pinning it separately means a regression can be
    localised to *withholding* versus *flushing* rather than showing up as one
    confusing total.
    """
    config = chunk_config(pad_left, stride, pad_right, look_back)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, frames, PUSH_PATTERN)

    committed_input = frames - frames % stride
    assert backend.buffered_frames == frames % stride
    if committed_input == 0:
        # Not one stride completed, so ``forward_chunk`` was never called and
        # even the prompt frames are still unencoded.
        assert backend.committed_frames == 0
    else:
        assert backend.committed_frames == (
            NUM_QUERY_FRAMES + committed_input - pad_right
        )


# --------------------------------------------------------------------------
# (1b) The same invariant one level up, through the recogniser
# --------------------------------------------------------------------------


class StubFrontend:
    """Stands in for ``WavFrontendOnline``; only ``init_cache`` is exercised."""

    def __init__(self) -> None:
        self.init_calls = 0

    def init_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Reset ``cache`` in place, as the real frontend does."""
        self.init_calls += 1
        cache.clear()
        return cache


class FakeChunkRecogniser(StreamingSenseVoice):
    """``StreamingSenseVoice`` with the weights and the frontend replaced.

    ``__init__`` deliberately does not call ``super().__init__`` - that is the
    method that loads the checkpoint.  The real ``push_audio``, ``reset``,
    ``_backend`` selection and ``_infer_final`` code paths all still run, and
    the backend under them is a real :class:`ChunkBackend` over the fake model.
    """

    def __init__(self, config: StreamingConfig, model: FakeSenseVoice, tok: Any) -> None:
        self.config = config
        self.config.validate()
        self.model = model
        self.tokenizer = tok
        self.device = torch.device("cpu")
        self.stub_frontend = StubFrontend()
        self._online_frontend = self.stub_frontend
        self._next_frame_id = 0
        self._state = _SegmentState()
        self.reset()

    def _extract_frames(self, samples: "np.ndarray", is_last: bool) -> None:
        """Emit one frame per whole ``FRAME_SAMPLES`` of audio, no DSP."""
        count = int(samples.size) // FRAME_SAMPLES
        if count <= 0:
            return
        frames = make_frames(count, start=self._next_frame_id)
        self._next_frame_id += count
        self._backend.accept_frames(frames)
        self._state.total_frames += count

    def _full_inference(self) -> str:
        """Skip the full-quality pass; its text is not what these tests check."""
        return ""


def audio(frames: int) -> "np.ndarray":
    """Silence worth ``frames`` encoder frames."""
    return np.zeros(frames * FRAME_SAMPLES, dtype=np.float32)


@pytest.mark.parametrize("pad_left,stride,pad_right,look_back", CHUNK_GEOMETRIES)
@pytest.mark.parametrize("frames", [23, 40])
def test_recogniser_leaves_the_lookahead_withheld_because_the_final_pass_supersedes_it(
    fake_model, tokenizer, pad_left, stride, pad_right, look_back, frames
):
    """Through ``push_audio(is_last=True)`` the tail is *never* flushed. By design.

    ``push_audio`` calls ``discard_pending()`` and goes straight to
    ``_infer_final``, which re-decodes the whole retained waveform offline; the
    withheld ``chunk_pad_right`` frames and the sub-stride buffer remainder are
    simply abandoned, because the authoritative text for that audio is about to
    be produced by a better decode.  ``ChunkBackend._flush_tail`` is reachable
    only on the *fallback* path, when the full pass raises.

    This test exists so that the discrepancy with
    ``test_tail_flush_commits_exactly_one_output_frame_per_input_frame_plus_the_prompt``
    reads as intentional.  Measured on real weights: 102 committed frames for
    100 input frames at the default geometry, i.e. ``4 + 100 - 2`` - not 104.
    Do not "fix" this into a flush without first deciding what the extra encoder
    work would buy, given the final pass covers the same audio.
    """
    pytest.importorskip("funasr.utils.postprocess_utils")

    config = chunk_config(pad_left, stride, pad_right, look_back)
    recogniser = FakeChunkRecogniser(config, fake_model, tokenizer)

    results = recogniser.push_audio(audio(frames), is_last=True)

    assert [r.type for r in results] == ["final"]
    backend = recogniser._backend
    committed_input = frames - frames % stride
    assert backend.committed_frames == NUM_QUERY_FRAMES + committed_input - pad_right
    assert backend.buffered_frames == frames % stride


# --------------------------------------------------------------------------
# (2) Configuration
# --------------------------------------------------------------------------


def test_backend_defaults_to_accumulate():
    """The default must stay the measured path, not the experimental one.

    ``ChunkBackend`` is an approximation unless the checkpoint was finetuned
    with chunk masking, and none of its geometry defaults are benchmarked (the
    class docstring says so).  Flipping the default would silently degrade every
    caller that never sets ``backend``.
    """
    assert StreamingConfig().backend == "accumulate"
    assert SUPPORTED_BACKENDS == ("accumulate", "chunk")


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_validate_accepts_every_supported_backend(backend):
    """Each advertised name is actually constructible with default geometry."""
    StreamingConfig(backend=backend).validate()


@pytest.mark.parametrize("backend", ["", "Accumulate", "chunked", "scama", "none"])
def test_validate_rejects_an_unknown_backend_and_names_the_supported_ones(backend):
    """A typo must fail at construction with a message listing the options.

    ``_build_backend`` raises too, but only once the model is loaded - i.e.
    after ~900 MB of weights and several seconds.  Catching it in ``validate``
    keeps the failure cheap and the message actionable.
    """
    with pytest.raises(ValueError, match="backend must be one of"):
        StreamingConfig(backend=backend).validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_stride": 0},
        {"chunk_stride": -1},
        {"chunk_pad_left": -1},
        {"chunk_pad_right": -1, "chunk_encoder_look_back": 0},
        {"chunk_encoder_look_back": -2},
    ],
)
def test_validate_rejects_out_of_range_chunk_geometry(overrides):
    """Each ``chunk_*`` field has a floor, and each floor is enforced.

    ``chunk_stride >= 1`` because ``forward_chunk`` commits that many frames per
    call and zero would never advance; the two pads are counts of frames and so
    cannot be negative; ``chunk_encoder_look_back >= -1`` because ``-1`` is the
    "keep every past chunk" sentinel and anything below it is meaningless.
    """
    config = StreamingConfig(backend="chunk", **overrides)
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize("look_back", [1, 2, -1])
def test_nonzero_encoder_look_back_requires_at_least_one_lookahead_frame(look_back):
    """``chunk_encoder_look_back != 0`` needs ``chunk_pad_right >= 1``.

    Not an arbitrary bound: ``MultiHeadedAttentionSANM.forward_chunk`` builds the
    key/value cache with ``k_h[:, :, :-pad_right, :]``, which at
    ``pad_right == 0`` slices everything away and leaves an empty cache.  The
    same rule is enforced in ``SenseVoiceEncoderSmall.init_chunk_cache``; see
    ``test_config_and_init_chunk_cache_agree_on_the_look_back_rule``.
    """
    config = StreamingConfig(
        backend="chunk", chunk_pad_right=0, chunk_encoder_look_back=look_back
    )
    with pytest.raises(ValueError, match="requires chunk_pad_right >= 1"):
        config.validate()


@pytest.mark.parametrize(
    "pad_right,look_back",
    [
        (0, 0),  # no lookahead and no look-back: the one legal pad_right == 0 pair
        (1, 1),  # the minimum lookahead that supports look-back at all
        (1, -1),
        (2, 1),  # the shipped default
        (5, -1),
    ],
)
def test_validate_accepts_legal_lookahead_and_look_back_combinations(
    pad_right, look_back
):
    """The accepting side of the cross-field rule, including both boundaries.

    ``pad_right = 1`` is the smallest value that leaves a non-empty attention
    cache, and ``(0, 0)`` is the only combination in which no lookahead is legal.
    """
    StreamingConfig(
        backend="chunk", chunk_pad_right=pad_right, chunk_encoder_look_back=look_back
    ).validate()


@pytest.mark.parametrize("pad_right,look_back", [(0, 1), (0, -1)])
def test_config_and_init_chunk_cache_agree_on_the_look_back_rule(
    fake_model, pad_right, look_back
):
    """Both gates reject the same geometry - the config one is just earlier.

    ``StreamingConfig.validate`` duplicates ``init_chunk_cache``'s check so a
    typo fails at construction with a message naming the config field.  A
    duplicated rule is only useful while the two copies agree, so pin that they
    do: if the encoder's rule is ever relaxed, this test fails and points at the
    config copy that would otherwise keep rejecting a now-legal geometry.
    """
    with pytest.raises(ValueError, match="requires chunk_pad_right >= 1"):
        StreamingConfig(
            backend="chunk",
            chunk_pad_right=pad_right,
            chunk_encoder_look_back=look_back,
        ).validate()

    with pytest.raises(ValueError, match="requires pad_right >= 1"):
        fake_model.encoder.init_chunk_cache(
            pad_left=0,
            stride=10,
            pad_right=pad_right,
            encoder_chunk_look_back=look_back,
        )


@pytest.mark.parametrize("pad_right", [0, 1, 2, 5, 12])
def test_chunk_lookahead_ms_is_pad_right_times_the_frame_duration(pad_right):
    """Lookahead latency is pure geometry: ``chunk_pad_right * MS_PER_FRAME``.

    One encoder frame is 60 ms (10 ms mel hop x LFR 6), so the shipped
    ``chunk_pad_right = 2`` costs exactly 120 ms of added latency - the number
    quoted in the config docstring, and the one a caller budgets against.
    """
    config = StreamingConfig(backend="chunk", chunk_pad_right=pad_right)
    assert config.chunk_lookahead_ms == pad_right * MS_PER_FRAME
    assert (config.chunk_lookahead_ms == 0.0) is (pad_right == 0)


def test_chunk_lookahead_ms_describes_the_geometry_even_under_the_accumulate_backend():
    """It reports what the ``chunk_*`` fields configure, not what is running.

    Deliberate: the property never consults ``backend``, so it stays a pure
    description of the geometry.  A caller that wants "the latency this
    recogniser actually adds" must check ``backend`` itself - the accumulate
    backend reads none of these fields and adds no lookahead.
    """
    accumulate = StreamingConfig(chunk_pad_right=3)
    chunked = StreamingConfig(backend="chunk", chunk_pad_right=3)

    assert accumulate.backend == "accumulate"
    assert accumulate.chunk_lookahead_ms == chunked.chunk_lookahead_ms == 180.0


def test_default_chunk_geometry_matches_the_finetuning_configuration():
    """The defaults mirror ``finetune_chunk.sh``'s middle entry, not a guess.

    ``chunk_size=12, stride=10, pad_left=0`` gives ``pad_right = 12 - 10 - 0 =
    2``, and ``encoder_att_look_back_factor=1`` gives the look-back of 1.  These
    are unbenchmarked (unlike every other default in the class), and their only
    justification is that decoding should use a geometry the encoder was trained
    on - so a change here breaks that link and needs its own measurement.
    """
    config = StreamingConfig()
    assert (
        config.chunk_pad_left,
        config.chunk_stride,
        config.chunk_pad_right,
        config.chunk_encoder_look_back,
    ) == (0, 10, 2, 1)
    assert config.chunk_pad_left + config.chunk_stride + config.chunk_pad_right == 12


# --------------------------------------------------------------------------
# (3) Backend selection
# --------------------------------------------------------------------------


class FakeRecogniser(StreamingSenseVoice):
    """Enough of ``StreamingSenseVoice`` to exercise ``_build_backend``."""

    def __init__(self, config: StreamingConfig, model: Any, tok: Any) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tok
        self.device = torch.device("cpu")
        self.encode_calls: List[Optional["torch.Tensor"]] = []

    def _encode_and_decode(self, features: Optional["torch.Tensor"]) -> str:
        """Record the window instead of running the encoder."""
        self.encode_calls.append(features)
        return ""


def test_default_configuration_builds_the_accumulate_backend(fake_model, tokenizer):
    """"Default backend" must mean the accumulate one all the way down.

    ``StreamingConfig.backend`` defaulting to ``"accumulate"`` is only half the
    claim; this pins that ``_build_backend`` honours it, so the two cannot drift
    apart.
    """
    recogniser = FakeRecogniser(StreamingConfig(), fake_model, tokenizer)

    assert isinstance(recogniser._backend, AccumulateBackend)


@pytest.mark.parametrize(
    "backend,expected",
    [("accumulate", AccumulateBackend), ("chunk", ChunkBackend)],
)
def test_the_backend_named_by_the_config_is_the_one_built(
    fake_model, tokenizer, backend, expected
):
    """``config.backend`` is the single switch; nothing else selects a strategy."""
    config = StreamingConfig(backend=backend)
    config.validate()
    recogniser = FakeRecogniser(config, fake_model, tokenizer)

    assert isinstance(recogniser._backend, expected)


def test_the_backend_is_built_once_and_cached(fake_model, tokenizer):
    """The lazy property must not rebuild - a ChunkBackend rebuild loses the cache.

    ``_backend`` is read on every ``push_audio``; if it re-instantiated, the
    chunk backend's encoder cache (and its prompt-pending flag) would reset on
    every call, silently restarting the stream.
    """
    recogniser = FakeRecogniser(
        StreamingConfig(backend="chunk"), fake_model, tokenizer
    )

    assert recogniser._backend is recogniser._backend


def test_building_an_unknown_backend_raises_rather_than_falling_back(
    fake_model, tokenizer
):
    """The unreachable branch still fails loudly if ``validate`` is bypassed.

    Silently defaulting to accumulate here would hide a typo that
    ``validate()`` would have caught, and produce a recogniser that quietly
    ignores the requested strategy.
    """
    config = StreamingConfig()
    config.backend = "nonesuch"  # past validate(), as only a bug could manage
    recogniser = FakeRecogniser(config, fake_model, tokenizer)

    with pytest.raises(ValueError, match="unknown backend"):
        recogniser._build_backend()


@pytest.mark.parametrize("backend", ["accumulate", "chunk"])
def test_both_backends_satisfy_the_streaming_backend_protocol(
    fake_model, tokenizer, backend
):
    """The protocol is the recogniser's only view of a strategy.

    ``StreamingBackend`` is ``runtime_checkable``, so this checks the five
    methods exist on both implementations - i.e. that the recogniser can drive
    either one without knowing which it has.
    """
    config = StreamingConfig(backend=backend)
    config.validate()
    recogniser = FakeRecogniser(config, fake_model, tokenizer)

    assert isinstance(recogniser._backend, StreamingBackend)


# --------------------------------------------------------------------------
# (4) AccumulateBackend - behaviour the refactor could plausibly have broken
# --------------------------------------------------------------------------


def accumulate_backend(
    chunk_size: int = 2,
    max_history: int = 4,
    decodes: Sequence[str] = (),
    render: Optional[Callable[[List[int]], str]] = None,
) -> "tuple[AccumulateBackend, _SegmentState, RecordingEncode]":
    """An :class:`AccumulateBackend` over a recording encode callable."""
    config = StreamingConfig(chunk_size=chunk_size, max_history=max_history)
    config.validate()
    encode = RecordingEncode(decodes, render=render)
    backend = AccumulateBackend(config, encode)
    state = _SegmentState()
    backend.reset(state)
    return backend, state, encode


def test_one_encoder_pass_per_partial_however_many_chunks_piled_up():
    """Backlog must not cost extra encoder passes - it is the dominant cost.

    Each pass re-encodes the *whole* window, so a second pass over the same
    window would reproduce the identical hypothesis for ~430 ms of CPU (the
    encoder's measured fixed cost).  If the pipeline ever falls behind, that is
    exactly when it must not do redundant work.  Hence ``emit_partial`` zeroes
    the pending count rather than decrementing it by ``chunk_size``.
    """
    # ``max_history`` is set well clear of the window so the history cap plays
    # no part here; it has its own tests below.
    backend, state, encode = accumulate_backend(chunk_size=2, max_history=100)

    for start in range(0, 6, 2):
        backend.accept_frames(indexed_frames(2, start=start))

    # Three chunks' worth of frames are pending, but they are one window.
    assert state.pending_frames == 6
    assert backend.should_emit_partial()

    backend.emit_partial()

    assert encode.calls == 1
    assert encode.windows == [[0, 1, 2, 3, 4, 5]]
    assert state.pending_frames == 0
    assert not backend.should_emit_partial()


def test_a_partial_covers_every_frame_accepted_so_far_not_just_the_new_ones():
    """The window grows; a partial is not an incremental decode.

    This is the property that makes the accumulate backend equivalent to the
    offline model for the audio so far.  A regression to "decode only the new
    frames" would look fine on short utterances and mangle every long one.
    """
    backend, _, encode = accumulate_backend(chunk_size=2, max_history=100)

    backend.accept_frames(indexed_frames(2, start=0))
    backend.emit_partial()
    backend.accept_frames(indexed_frames(2, start=2))
    backend.emit_partial()

    assert encode.windows == [[0, 1], [0, 1, 2, 3]]


def test_discard_pending_drops_the_cadence_not_the_accumulated_frames():
    """Suppressing a partial must not throw audio away.

    ``push_audio(is_last=True)`` calls this to suppress a due partial; the
    frames it covers still belong to the segment and are still needed by the
    next window (and by the final pass).  Dropping them here would silently cut
    the end off every utterance.
    """
    backend, state, encode = accumulate_backend(chunk_size=2, max_history=100)

    backend.accept_frames(indexed_frames(3, start=0))
    backend.discard_pending()

    assert state.pending_frames == 0
    assert not backend.should_emit_partial()
    assert state.features is not None and state.features.shape[0] == 3

    backend.accept_frames(indexed_frames(1, start=3))
    backend.emit_partial()

    assert encode.windows == [[0, 1, 2, 3]]


def test_the_history_cap_retires_exactly_the_frames_the_previous_decode_covered():
    """The cut is at ``decoded_frames``, not at ``max_history``.

    The windows are non-overlapping and the retired one's text is frozen into
    the confirmed prefix, so the cut must land exactly where the frozen decode
    stopped.  Cutting anywhere else duplicates or loses whatever falls between
    the two points - and because the text still *reads* fine, that bug is
    invisible without this assertion.
    """
    backend, state, encode = accumulate_backend(
        chunk_size=2, max_history=4, decodes=["<|en|>AB", "CD"]
    )

    backend.accept_frames(indexed_frames(4, start=0))
    backend.emit_partial()
    assert state.decoded_frames == 4

    backend.accept_frames(indexed_frames(2, start=4))
    text, raw_text = backend.emit_partial()

    # Exactly the four decoded frames were retired; the new window is the rest.
    assert encode.windows == [[0, 1, 2, 3], [4, 5]]
    assert state.decoded_frames == 2
    assert state.confirmed_text == "AB"
    assert state.confirmed_raw == "<|en|>AB"
    assert text == "ABCD"
    assert raw_text == "<|en|>ABCD"


def test_no_text_is_duplicated_or_lost_across_a_history_cap_boundary():
    """Every frame contributes to exactly one decode, across four cuts.

    The single-cut case above could pass with an off-by-one that only shows up
    on the second retirement, so this drives the cap four times and reads the
    coverage straight out of the emitted text: each window decodes to the list
    of frames it saw, so the final partial spells out precisely which frames the
    displayed text stands for.

    Both failure modes are caught by the one assertion.  A cut *before*
    ``decoded_frames`` would repeat frames (``... 2 3 2 3 4 ...``), a cut after
    it would skip them (``... 2 3 6 7 ...``); only retiring exactly the decoded
    frames yields ``range(20)``.
    """
    backend, _, encode = accumulate_backend(
        chunk_size=2,
        max_history=4,
        render=lambda ids: "".join(f"<{i}>" for i in ids),
    )

    text = ""
    for start in range(0, 20, 2):
        backend.accept_frames(indexed_frames(2, start=start))
        text, _ = backend.emit_partial()

    covered = [int(frame) for frame in re.findall(r"<(\d+)>", text)]
    assert covered == list(range(20))
    # Four retirements happened, so this is not the trivial "one window" case.
    assert len(encode.windows) == 10
    assert encode.windows[-1] == [16, 17, 18, 19]


def test_an_oversized_push_keeps_the_most_recent_frames_and_drops_the_rest():
    """A single push longer than ``max_history`` is truncated, not decoded whole.

    Documented behaviour with a real cost: nothing has been decoded yet, so
    there is no confirmed prefix to freeze and the surplus frames are simply
    gone from the streaming hypothesis.  It is survivable only because the
    ``final`` pass re-decodes the whole retained waveform.  Pinned so the
    trade-off stays visible rather than being rediscovered as a bug report about
    missing words on long chunks.
    """
    backend, state, encode = accumulate_backend(chunk_size=2, max_history=4)

    backend.accept_frames(indexed_frames(10, start=0))
    backend.emit_partial()

    assert encode.windows == [[6, 7, 8, 9]]
    assert state.confirmed_text == ""
    assert state.features is not None and state.features.shape[0] == 4


def test_reset_latches_the_new_state_object_rather_than_clearing_the_old_one():
    """The recogniser *replaces* the state on reset; the backend must follow.

    ``StreamingSenseVoice.reset`` assigns a fresh ``_SegmentState`` and hands it
    over.  A backend still holding the previous object would keep appending to
    the old utterance's window and leak its confirmed prefix into the new
    segment.
    """
    backend, first, encode = accumulate_backend(chunk_size=2, max_history=100)

    backend.accept_frames(indexed_frames(2, start=0))
    backend.emit_partial()

    second = _SegmentState()
    backend.reset(second)
    backend.accept_frames(indexed_frames(2, start=100))
    text, _ = backend.emit_partial()

    assert encode.windows[-1] == [100, 101]
    assert second.confirmed_text == ""
    assert text == "w1"
    assert first.features is not None and first.features.shape[0] == 2


def test_encode_window_does_not_mutate_the_accumulated_feature_tensor(
    fake_model, tokenizer
):
    """``SenseVoiceEncoderSmall.forward`` scales its input **in place**.

    ``xs_pad *= self.output_size() ** 0.5`` (model.py) rewrites whatever tensor
    it is handed.  ``encode_window`` is safe only because ``torch.cat`` with the
    prompt frames allocates a fresh tensor; feed the cached window in directly
    and every partial would multiply the accumulated features again, inflating
    them by ``output_size ** 0.5`` per chunk (22.6x for the real 512-wide
    encoder) until the decode is noise.

    Needs the real encoder: the in-place scaling is exactly the thing a stub
    would not reproduce.
    """
    config = StreamingConfig()
    features = make_frames(12)
    snapshot = features.clone()

    encode_window(fake_model, tokenizer, config, features, torch.device("cpu"))

    assert torch.equal(features, snapshot)


def test_encode_window_returns_empty_for_an_empty_window(fake_model, tokenizer):
    """No frames means no encoder pass at all, not a decode of the prompt.

    ``AccumulateBackend.emit_partial`` can reach this before any audio has been
    extracted (a forced partial from the final fallback path), and the four
    query embeddings on their own would decode to a bare rich-tag prefix.
    """
    config = StreamingConfig()

    assert encode_window(fake_model, tokenizer, config, None, torch.device("cpu")) == ""
    assert (
        encode_window(
            fake_model,
            tokenizer,
            config,
            torch.zeros((0, INPUT_SIZE)),
            torch.device("cpu"),
        )
        == ""
    )


# --------------------------------------------------------------------------
# (5) ChunkBackend mechanics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [4, 10])
@pytest.mark.parametrize("frames", [13, 25])
def test_exactly_one_stride_is_fed_per_forward_chunk_call_and_the_rest_is_buffered(
    fake_model, tokenizer, stride, frames
):
    """``forward_chunk`` sees a constant window, whatever the caller's block size.

    Chunk boundaries are part of the computation - moving one moves every
    receptive field that straddles it (``test_resplitting_the_same_audio_...``
    in the equivalence suite measures a 2.17 deviation) - so a backend that
    forwarded whatever the frontend happened to hand it would make the output
    depend on the caller's audio block size.  Buffering the sub-stride
    remainder is what keeps the schedule constant.
    """
    config = chunk_config(0, stride, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    with SpyForwardChunk(fake_model.encoder) as spy:
        feed(backend, frames, PUSH_PATTERN)

    expected_calls = frames // stride
    # The prompt rides in front of the first chunk; every later call is a bare
    # stride.
    expected = [NUM_QUERY_FRAMES + stride] + [stride] * (expected_calls - 1)

    assert spy.lengths == expected
    assert backend.buffered_frames == frames % stride


def test_the_prompt_frames_are_prepended_to_the_first_chunk_only(
    fake_model, tokenizer
):
    """Four query embeddings, at absolute positions 1-4, exactly once per segment.

    They must land where ``SenseVoiceSmall.encode`` puts them or the model sees
    an input it was never trained on.  Prepending them to *every* chunk would
    both duplicate them in the CTC output and shift every speech frame's
    position encoding by four - and because the rich tags they produce look
    plausible either way, nothing downstream would notice.

    ``cache["start_idx"]`` is the position counter
    (``StreamSinusoidalPositionEncoder`` numbers from 1), so its value after the
    first call is the direct evidence that the prompt occupied positions 1-4.
    """
    stride = 6
    config = chunk_config(0, stride, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)
    prompt = build_prompt_frames(fake_model, config, torch.device("cpu"))

    with SpyForwardChunk(fake_model.encoder) as spy:
        backend.accept_frames(make_frames(stride, start=0))
        first_start_idx = backend._cache["start_idx"]
        backend.accept_frames(make_frames(2 * stride, start=stride))

    assert spy.lengths == [NUM_QUERY_FRAMES + stride, stride, stride]
    assert torch.equal(spy.inputs[0][:, :NUM_QUERY_FRAMES, :], prompt)
    assert torch.equal(spy.inputs[0][:, NUM_QUERY_FRAMES:, :], make_frames(stride).unsqueeze(0))
    # Positions 1-4 went to the prompt, so the first call consumed 4 + stride.
    assert first_start_idx == NUM_QUERY_FRAMES + stride
    assert backend._cache["start_idx"] == NUM_QUERY_FRAMES + 3 * stride
    # None of the later calls carries the prompt anywhere in its window.
    for chunk in spy.inputs[1:]:
        assert not any(
            torch.equal(chunk[:, i : i + NUM_QUERY_FRAMES, :], prompt)
            for i in range(chunk.shape[1] - NUM_QUERY_FRAMES + 1)
        )


def test_the_encoder_cache_is_created_once_per_segment_and_reused_across_calls(
    fake_model, tokenizer
):
    """One cache object, mutated in place, for the life of a segment.

    ``forward_chunk`` carries the position offset, the overlap frames and the
    per-layer attention caches in that dict; replacing it mid-segment would
    restart the position encoding at 1 and blank the SANM overlap, so the
    encoder would see a window with no left context and no idea where it is.
    """
    config = chunk_config(0, 6, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    cache = backend._cache
    assert cache is not None
    assert cache["start_idx"] == 0
    assert cache["chunk_size"] == [0, 6, 2]
    assert cache["encoder_chunk_look_back"] == 1

    feed(backend, 20, PUSH_PATTERN)

    assert backend._cache is cache
    assert cache["start_idx"] > 0


def test_reset_rebuilds_the_cache_so_no_state_leaks_into_the_next_segment(
    fake_model, tokenizer
):
    """A new utterance starts from nothing - position, overlap and ids alike.

    Clearing selected keys instead of rebuilding is the tempting shortcut, and
    it is how the previous utterance's tail would end up prefixed to the next
    one's transcript.
    """
    config = chunk_config(0, 6, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, 20, PUSH_PATTERN)
    first_cache = backend._cache
    assert backend.committed_frames > 0

    backend.reset(_SegmentState())

    assert backend._cache is not first_cache
    assert backend._cache["start_idx"] == 0
    assert backend.committed_frames == 0
    assert backend.buffered_frames == 0
    assert backend.emit_partial() == ("", "")


def test_reset_makes_the_prompt_pending_again_for_the_new_segment(
    fake_model, tokenizer
):
    """Each segment gets its own query embeddings at positions 1-4.

    The prompt is consumed once per segment; if the flag survived a reset, the
    second utterance would run with no language/ITN conditioning at all.
    """
    stride = 6
    config = chunk_config(0, stride, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)
    prompt = build_prompt_frames(fake_model, config, torch.device("cpu"))

    backend.accept_frames(make_frames(stride))
    backend.reset(_SegmentState())

    with SpyForwardChunk(fake_model.encoder) as spy:
        backend.accept_frames(make_frames(stride))

    assert spy.lengths == [NUM_QUERY_FRAMES + stride]
    assert torch.equal(spy.inputs[0][:, :NUM_QUERY_FRAMES, :], prompt)


@pytest.mark.parametrize("pad_left,stride,pad_right,look_back", CHUNK_GEOMETRIES)
def test_the_tail_flush_runs_at_most_once_per_segment(
    fake_model, tokenizer, pad_left, stride, pad_right, look_back
):
    """``_tail_flushed`` is the only thing making the flush idempotent.

    The cache is *not* self-limiting: ``cache["tail_chunk"]`` stays ``True``
    afterwards and nothing clears it, so a second flush would re-emit the cached
    overlap and duplicate the last ``pad_right`` frames in the CTC stream.  The
    fallback path can reach ``emit_partial`` more than once with ``finished``
    set, so this guard is load-bearing rather than defensive.
    """
    config = chunk_config(pad_left, stride, pad_right, look_back)
    backend, state = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, 25, PUSH_PATTERN)
    state.finished = True
    first_text = backend.emit_partial()
    after_first_flush = backend.committed_frames

    second_text = backend.emit_partial()

    assert after_first_flush == NUM_QUERY_FRAMES + 25
    assert backend.committed_frames == after_first_flush
    assert second_text == first_text


def test_frames_accepted_after_a_flush_are_silently_swallowed(
    fake_model, tokenizer
):
    """A flushed cache is spent; only ``reset`` makes the backend usable again.

    ``forward_chunk`` ignores its input once ``tail_chunk`` is set and re-runs
    the cached overlap instead, so post-flush frames would be replaced by a
    repeat of the previous tail rather than encoded.  Acceptable in production
    (the segment is over by then) but worth pinning, because the failure mode of
    a caller that keeps pushing is duplicated text, not an exception.
    """
    stride = 6
    config = chunk_config(0, stride, 2, 1)
    backend, state = build_chunk_backend(fake_model, tokenizer, config)

    pad_right = config.chunk_pad_right
    feed(backend, 12, PUSH_PATTERN)
    state.finished = True
    backend.emit_partial()
    flushed_frames = backend.committed_frames

    backend.accept_frames(make_frames(2 * stride, start=12))

    # Two more strides completed, so ``forward_chunk`` ran twice - but with
    # ``tail_chunk`` still set it ignored both inputs and re-emitted the spent
    # ``pad_left + pad_right`` overlap, i.e. ``pad_right`` frames each.  The new
    # audio contributed nothing.
    assert backend.committed_frames == flushed_frames + 2 * pad_right
    assert backend.buffered_frames == 0

    backend.reset(_SegmentState())
    assert backend.committed_frames == 0


@pytest.mark.parametrize("pad_left,stride,pad_right,look_back", CHUNK_GEOMETRIES)
def test_flushing_a_segment_that_never_saw_a_frame_returns_empty_instead_of_raising(
    fake_model, tokenizer, pad_left, stride, pad_right, look_back
):
    """The empty-segment guard: a tail call with no frames would crash the FSMN.

    With nothing ever fed, the cache holds only its (possibly empty) left pad,
    so a ``tail_chunk`` call would reach the encoder's FSMN convolution with a
    zero-length input and raise "kernel size can't be greater than actual input
    size".  Unreachable through the recogniser, which only flushes once
    ``total_frames > 0``, but reachable by driving the backend directly - e.g.
    a VAD segment that ends before one 60 ms frame completes.
    """
    config = chunk_config(pad_left, stride, pad_right, look_back)
    backend, state = build_chunk_backend(fake_model, tokenizer, config)

    state.finished = True
    text, raw_text = backend.emit_partial()

    assert (text, raw_text) == ("", "")
    assert backend.committed_frames == 0


@pytest.mark.parametrize("chunk_size,stride", [(12, 10), (4, 10), (12, 4)])
def test_the_emission_cadence_counts_arriving_frames_not_committed_ones(
    fake_model, tokenizer, chunk_size, stride
):
    """A partial is due after ``chunk_size`` *arrivals*, independent of ``stride``.

    The two knobs are deliberately separate here (in the accumulate backend they
    are necessarily the same number).  Counting commits instead would make the
    partial cadence lurch with the geometry, and at ``chunk_size < stride`` no
    partial would ever be due before the first full stride landed.
    """
    config = chunk_config(0, stride, 2, 1, chunk_size=chunk_size)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    backend.accept_frames(make_frames(chunk_size - 1))
    assert not backend.should_emit_partial()

    backend.accept_frames(make_frames(1, start=chunk_size - 1))
    assert backend.should_emit_partial()

    backend.emit_partial()
    assert not backend.should_emit_partial()


def test_discard_pending_resets_only_the_cadence_not_the_encoder_work(
    fake_model, tokenizer
):
    """There is nothing to undo: the frames are already inside the cache.

    Unlike the accumulate backend, this one does its work in ``accept_frames``.
    ``discard_pending`` therefore may not touch the buffer or the decoded ids -
    doing so would delete audio that has already been encoded once and can never
    be encoded again.
    """
    stride = 6
    config = chunk_config(0, stride, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, 14, PUSH_PATTERN)
    committed = backend.committed_frames
    buffered = backend.buffered_frames

    backend.discard_pending()

    assert not backend.should_emit_partial()
    assert backend.committed_frames == committed
    assert backend.buffered_frames == buffered


def test_a_partial_decodes_every_committed_frame_with_one_collapse(
    fake_model, tokenizer
):
    """Ids accumulate; only the CTC collapse is re-run, and over everything.

    Decoding each chunk on its own and concatenating the strings was rejected
    because CTC merges duplicates *across* a boundary, so a token whose frames
    straddle one would be emitted twice.  Pinning that the collapse always runs
    over the whole segment keeps that decision enforced: the id list only ever
    grows, and a later partial's text extends the earlier one's coverage rather
    than restarting it.
    """
    stride = 6
    config = chunk_config(0, stride, 2, 1, chunk_size=stride)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, 2 * stride, PUSH_PATTERN)
    backend.emit_partial()
    first_ids = list(backend._frame_ids)

    feed(backend, 2 * stride, PUSH_PATTERN)
    backend.emit_partial()

    assert backend._frame_ids[: len(first_ids)] == first_ids
    assert len(backend._frame_ids) > len(first_ids)


def test_partial_display_text_has_the_provisional_rich_tags_stripped(
    fake_model, tokenizer
):
    """Display text carries no tags, because this backend can never revise them.

    SenseVoice decides its language / emotion / event markers at the first few
    output positions, which under this backend means the first chunk alone
    decides them - a few hundred milliseconds, before the utterance can
    contradict it.  The authoritative label comes from the ``final`` pass, so a
    partial must not show one.
    """
    config = chunk_config(0, 6, 2, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    feed(backend, 24, PUSH_PATTERN)
    text, raw_text = backend.emit_partial()

    # The stub tokenizer stamps one rich tag onto every decode, so its absence
    # from ``text`` and its presence in ``raw_text`` are both meaningful.
    assert raw_text.startswith(STUB_RICH_TAG)
    assert "<|" not in text
    assert text == raw_text[len(STUB_RICH_TAG) :]
    assert text != ""


def test_committed_and_buffered_frames_account_for_every_accepted_frame(
    fake_model, tokenizer
):
    """The two inspection properties must add up, or one of them is lying.

    They are what the alignment tests above measure with, so they need their own
    check: ``committed_frames`` counts encoder *output* frames (prompt included
    and lookahead excluded) while ``buffered_frames`` counts *input* frames not
    yet handed over.
    """
    stride = 6
    pad_right = 2
    config = chunk_config(0, stride, pad_right, 1)
    backend, _ = build_chunk_backend(fake_model, tokenizer, config)

    total = 20
    feed(backend, total, PUSH_PATTERN)

    committed_input = backend.committed_frames - NUM_QUERY_FRAMES + pad_right
    assert committed_input + backend.buffered_frames == total
    assert committed_input == total - total % stride
