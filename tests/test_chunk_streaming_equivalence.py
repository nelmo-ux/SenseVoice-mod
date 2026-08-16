"""Equivalence tests between ``SenseVoiceEncoderSmall.forward`` and ``forward_chunk``.

These pin the *streaming* half of the chunk work: how much of the full-attention
computation ``forward_chunk`` reproduces, where it deliberately cannot, and one
place where we deliberately deviate from upstream funasr.

Everything here builds a small randomly-initialised encoder from a fixed seed --
no checkpoint is loaded, so the suite runs anywhere and stays fast.  Most
comparisons run in float64 so that a real algorithmic divergence cannot be
confused with float32 rounding; a float32 case is kept alongside because that is
what production runs, and the two agree to a relative 5e-7 (see
``test_streaming_deviation_is_algorithmic_not_floating_point``).

Sources of divergence between ``forward`` and a multi-call ``forward_chunk`` run
--------------------------------------------------------------------------------
Identified while measuring group (C).  A streamed frame is *not* expected to
equal its full-attention counterpart, and these are the reasons:

1. **Truncated self-attention context.**  ``forward`` lets every frame attend to
   the whole utterance.  ``forward_chunk`` restricts frame t to the current
   window ``[pad_left + chunk + pad_right]``, plus whatever the per-layer key /
   value cache carries when ``encoder_chunk_look_back != 0``.  Nothing lets a
   streamed frame see beyond ``pad_right`` frames into the future, so this
   source can never be eliminated -- only reduced by enlarging the window.
2. **No future context at all beyond the lookahead.**  Even with
   ``encoder_chunk_look_back=-1`` (unbounded history) the *right* side stays
   capped at ``pad_right``.  That is why look-back only ever narrows the gap and
   never closes it -- see
   ``test_unbounded_look_back_reduces_the_mean_deviation_versus_no_look_back``.
3. **Per-layer context re-truncation.**  The window is re-cut at *every* layer,
   so the effective receptive field does not grow with depth the way it does in
   the full-attention path.  A 5-layer encoder streamed at stride 10 therefore
   diverges much more than a 1-layer one would.
4. **The FSMN memory block is fed the window, not the utterance.**  The SANM
   memory block has no streaming state of its own; its context comes purely from
   chunk overlap.  Frames within ``kernel_size // 2 == 5`` of a window edge see
   zero padding in the streamed path where the full path sees real neighbours.
5. **Attention mask vs. no mask.**  ``forward`` passes a length mask;
   ``forward_chunk`` passes ``None``.  For a single full-length utterance the
   mask is all ones and this is provably a no-op (group A measures exactly 0.0),
   so it contributes nothing here -- but it *would* matter for a ragged batch,
   which is why streaming is single-utterance.
6. **LayerNorm is computed in float32 regardless of input dtype** (see
   ``LayerNorm.forward``), so ``.double()`` buys less precision than it looks
   like.  It is still enough: group (A) is bit-exact in both dtypes.

Sources 1-4 are structural.  Source 5 is inert.  Source 6 is bounded far below
the measured deviations.
"""

import math

import pytest
import torch

from model import SenseVoiceEncoderSmall


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

INPUT_SIZE = 20
OUTPUT_SIZE = 64
ATTENTION_HEADS = 4
LINEAR_UNITS = 128
NUM_BLOCKS = 3  # -> encoders0 has 1 layer, encoders has 2
TP_BLOCKS = 2
KERNEL_SIZE = 11  # SenseVoiceSmall's value; FSMN half-width is (11 - 1) // 2 == 5
SEED = 0

# Streaming geometries as (pad_left, stride, pad_right).  Chosen to cover
# pad_right == 0, pad_left == 0, both zero, and both non-zero.
GEOMETRIES = [
    (0, 10, 0),
    (0, 10, 5),
    (5, 10, 0),
    (5, 10, 5),
    (3, 8, 4),
    (0, 6, 2),
]

# Geometries with pad_right >= 1, i.e. the ones that support a lookahead flush
# and ``encoder_chunk_look_back != 0`` (the attention cache is built by dropping
# the last pad_right frames and would be empty otherwise).
LOOKAHEAD_GEOMETRIES = [g for g in GEOMETRIES if g[2] > 0]

# Deviation envelope for a multi-chunk stream against full attention, measured
# with the encoder built below over all 20 combinations of
# GEOMETRIES x look_back {0, -1} x T {40, 37 frames}:
#   worst max  1.9585  (geometry (0,  6, 2), look_back=0,  T=37)
#   best  max  1.2877  (geometry (3,  8, 4), look_back=-1, T=40)
#   worst mean 0.3713  (geometry (0,  6, 2), look_back=0,  T=40)
#   best  mean 0.1089  (geometry (5, 10, 5), look_back=-1, T=40)
# The bounds sit just outside that envelope.  They are deliberately *two-sided*:
# the upper bound catches a regression that makes streaming wilder, and the
# lower bound catches someone silently turning streaming into full attention
# (e.g. by leaking the whole utterance into the window), which would make the
# streaming path a lie rather than a bug.  These are outputs of a *randomly
# initialised* encoder normalised by tp_norm, so an O(1) deviation is the
# expected magnitude -- it does not translate to an O(1) WER change on trained
# weights, it just says the two computations are genuinely different.
STREAMING_MAX_DEVIATION_UPPER = 2.1
STREAMING_MEAN_DEVIATION_UPPER = 0.42
STREAMING_MAX_DEVIATION_LOWER = 0.6
STREAMING_MEAN_DEVIATION_LOWER = 0.05


def build_encoder(
    dtype: torch.dtype = torch.float64,
    chunk_size=None,
    stride=None,
    pad_left=None,
    seed: int = SEED,
) -> SenseVoiceEncoderSmall:
    """A small, deterministically initialised encoder in ``eval()`` mode.

    Dropout is disabled everywhere so that ``train()`` and ``eval()`` differ only
    in the chunk-config draw, and so that repeated calls are reproducible without
    reseeding.
    """
    torch.manual_seed(seed)
    encoder = SenseVoiceEncoderSmall(
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
        chunk_size=chunk_size,
        stride=stride,
        pad_left=pad_left,
    )
    encoder.eval()
    return encoder.to(dtype)


def make_features(
    time: int, batch: int = 1, dtype: torch.dtype = torch.float64, seed: int = 1
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, time, INPUT_SIZE, generator=generator).to(dtype)


def run_forward(encoder, x, lengths=None):
    """``forward`` on a clone -- it scales ``xs_pad`` in place."""
    if lengths is None:
        lengths = [x.size(1)] * x.size(0)
    with torch.no_grad():
        return encoder(x.clone(), torch.tensor(lengths))


def stream(encoder, x, pad_left, stride_, pad_right, look_back=0, chunk_lengths=None):
    """Feed ``x`` through ``forward_chunk`` and return ``(concatenated, per_call_sizes)``.

    ``chunk_lengths`` overrides the default fixed-``stride`` schedule.  The final
    ``tail_chunk`` flush is only issued when ``pad_right > 0``: with no lookahead
    there is nothing withheld, and when ``pad_left + pad_right == 0`` the cached
    overlap is empty, which would hand the FSMN conv1d a zero-length input.
    """
    if chunk_lengths is None:
        chunk_lengths = [stride_] * math.ceil(x.size(1) / stride_)
    cache = encoder.init_chunk_cache(
        pad_left=pad_left,
        stride=stride_,
        pad_right=pad_right,
        encoder_chunk_look_back=look_back,
        batch_size=x.size(0),
        dtype=x.dtype,
    )
    outputs = []
    start = 0
    with torch.no_grad():
        for length in chunk_lengths:
            chunk = x[:, start : start + length, :]
            start += length
            if chunk.size(1) == 0:
                continue
            out, cache = encoder.forward_chunk(chunk.clone(), cache)
            outputs.append(out)
        if pad_right > 0:
            cache["tail_chunk"] = True
            out, cache = encoder.forward_chunk(x[:, :0, :], cache)
            outputs.append(out)
    return torch.cat(outputs, dim=1), [out.size(1) for out in outputs]


def capture_first_layer_input(encoder, x, cache):
    """Return the tensor handed to ``encoders0[0]`` on the first ``forward_chunk``.

    ``register_forward_pre_hook`` cannot be used here: ``forward_chunk`` calls
    ``encoder_layer.forward_chunk(...)`` directly rather than going through
    ``nn.Module.__call__``, so no hook fires.  Wrapping the bound method is the
    equivalent interception point.
    """
    captured = {}
    layer = encoder.encoders0[0]
    original = layer.forward_chunk

    def spy(xs_pad, *args, **kwargs):
        captured.setdefault("x", xs_pad.detach().clone())
        return original(xs_pad, *args, **kwargs)

    layer.forward_chunk = spy
    try:
        with torch.no_grad():
            encoder.forward_chunk(x.clone(), cache)
    finally:
        layer.forward_chunk = original
    return captured["x"]


def upstream_cache(encoder, pad_left, stride_, pad_right, batch=1, dtype=torch.float64):
    """Our cache, with ``feats`` reset to funasr's ``pad_left + pad_right`` zeros.

    Mirrors ``funasr.models.scama.model``/``paraformer_streaming.model``, which
    both build the cache with
    ``torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims))``.
    """
    cache = encoder.init_chunk_cache(
        pad_left=pad_left,
        stride=stride_,
        pad_right=pad_right,
        batch_size=batch,
        dtype=dtype,
    )
    cache["feats"] = torch.zeros((batch, pad_left + pad_right, INPUT_SIZE), dtype=dtype)
    return cache


# ---------------------------------------------------------------------------
# (A) Single-chunk full-coverage equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("time", [17, 33, 37])
def test_one_forward_chunk_covering_the_whole_utterance_equals_forward_exactly(dtype, time):
    """The strongest equivalence claim available, and it is bit-exact.

    With ``pad_left = pad_right = 0`` and ``stride = T`` the streamed window is
    the entire utterance, so ``forward_chunk`` performs literally the same
    computation as ``forward``: same absolute positions (``start_idx`` is 0),
    same unmasked attention (``forward``'s length mask is all ones for a single
    full-length utterance, and ``masked_fill`` with an all-false mask is a
    no-op), same FSMN input.  Measured max absolute difference is exactly 0.0 in
    both float64 and float32 -- not "small", *zero*.  If this ever becomes
    non-zero, ``forward_chunk`` has stopped being the same function and the
    streaming path must be re-derived, not re-toleranced.
    """
    encoder = build_encoder(dtype=dtype)
    x = make_features(time, dtype=dtype)

    reference, _ = run_forward(encoder, x)
    cache = encoder.init_chunk_cache(
        pad_left=0, stride=time, pad_right=0, dtype=dtype
    )
    with torch.no_grad():
        streamed, _ = encoder.forward_chunk(x.clone(), cache)

    assert streamed.shape == reference.shape
    assert (streamed - reference).abs().max().item() == 0.0
    assert torch.equal(streamed, reference)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_full_coverage_equivalence_holds_for_a_multi_utterance_batch(dtype):
    """Equal-length batching changes nothing: no padding means no mask asymmetry.

    Streaming has no length argument, so it can only be trusted when every item
    in the batch is the same length.  This pins that the batch dimension itself
    introduces no drift, isolating "streaming is unsafe for ragged batches" as a
    statement about padding rather than about batching.
    """
    encoder = build_encoder(dtype=dtype)
    time = 33
    x = make_features(time, batch=2, dtype=dtype)

    reference, _ = run_forward(encoder, x)
    cache = encoder.init_chunk_cache(
        pad_left=0, stride=time, pad_right=0, batch_size=2, dtype=dtype
    )
    with torch.no_grad():
        streamed, _ = encoder.forward_chunk(x.clone(), cache)

    assert (streamed - reference).abs().max().item() == 0.0


def test_full_coverage_equivalence_is_unaffected_by_trim():
    """At ``pad_left = pad_right = 0`` trimming is a no-op, so ``trim=False`` also matches.

    This separates the two things ``forward_chunk`` does -- compute, and select
    which frames are final -- and shows group (A) is testing the computation.
    """
    encoder = build_encoder()
    time = 33
    x = make_features(time)

    reference, _ = run_forward(encoder, x)
    cache = encoder.init_chunk_cache(
        pad_left=0, stride=time, pad_right=0, dtype=torch.float64
    )
    with torch.no_grad():
        raw, _ = encoder.forward_chunk(x.clone(), cache, trim=False)

    assert raw.shape == reference.shape
    assert (raw - reference).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# (B) Layer-0 input equivalence -- pins a deliberate deviation from funasr
# ---------------------------------------------------------------------------
#
# MEASURED RESULT, recorded because it is the opposite of the hoped-for one:
# the two initialisations do **not** produce the same layer-0 input whenever
# ``pad_right > 0``.  Upstream's window is exactly ``pad_right`` frames LONGER,
# and those extra frames are all-zero leading frames.  Concretely, for a first
# chunk of ``stride`` frames:
#
#   geometry (pl, st, pr)   ours    upstream   extra   leading    aligned
#                           frames  frames             |max|      max diff
#   (0, 10, 0)              10      10         0       -          0.0
#   (0, 10, 5)              10      15         5       0.0        0.0
#   (5, 10, 0)              15      15         0       -          0.0
#   (5, 10, 5)              15      20         5       0.0        0.0
#   (3,  8, 4)              11      15         4       0.0        0.0
#   (0,  6, 2)               6       8         2       0.0        0.0
#
# So a plain "max abs difference" is not even well defined for pad_right > 0 --
# the shapes differ.  The honest statements, each asserted below, are:
#   * the extra ``pad_right`` frames are exactly 0.0, and
#   * after dropping them the two windows are bit-identical (max diff 0.0), and
#   * those zeros are NOT inert: they participate in self-attention and the FSMN
#     convolution, corrupting the first chunk's output by up to 1.514661 and
#     adding ``pad_right`` phantom output frames.

@pytest.mark.parametrize("pad_left,stride_,pad_right", GEOMETRIES)
def test_upstream_feats_init_makes_the_first_layer0_window_longer_by_pad_right(
    pad_left, stride_, pad_right
):
    """Upstream seeds ``feats`` with ``pad_left + pad_right`` zeros; we seed ``pad_left``.

    Why it matters: ``_add_overlap_chunk`` prepends the cached frames verbatim,
    so the seed length *is* the first window's left context.  Upstream's extra
    ``pad_right`` frames are pure fiction -- they stand for a lookahead that has
    not been withheld yet, because on the very first call nothing has been
    withheld.  Every later call is fine in both variants, since the cache is
    re-cut to ``pad_left + pad_right`` real frames.
    """
    encoder = build_encoder()
    x = make_features(stride_)

    ours = capture_first_layer_input(
        encoder,
        x,
        encoder.init_chunk_cache(
            pad_left=pad_left, stride=stride_, pad_right=pad_right, dtype=torch.float64
        ),
    )
    upstream = capture_first_layer_input(
        encoder, x, upstream_cache(encoder, pad_left, stride_, pad_right)
    )

    assert ours.size(1) == pad_left + stride_
    assert upstream.size(1) == pad_left + pad_right + stride_
    assert upstream.size(1) - ours.size(1) == pad_right


@pytest.mark.parametrize("pad_left,stride_,pad_right", LOOKAHEAD_GEOMETRIES)
def test_the_extra_upstream_layer0_frames_are_exactly_zero(pad_left, stride_, pad_right):
    """The surplus is padding, not data -- which is why it is safe for us to drop it.

    Restricted to ``pad_right >= 1``: at ``pad_right == 0`` the two seeds are the
    same length and there is no surplus slice to inspect.
    """
    encoder = build_encoder()
    x = make_features(stride_)

    upstream = capture_first_layer_input(
        encoder, x, upstream_cache(encoder, pad_left, stride_, pad_right)
    )

    surplus = upstream[:, :pad_right, :]
    assert surplus.numel() > 0
    assert surplus.abs().max().item() == 0.0


@pytest.mark.parametrize("pad_left,stride_,pad_right", GEOMETRIES)
def test_layer0_input_matches_upstream_bit_for_bit_after_dropping_the_extra_zeros(
    pad_left, stride_, pad_right
):
    """Our window is upstream's window minus ``pad_right`` leading zero frames.

    Measured max absolute difference over the aligned frames: 0.0 for every
    geometry above, including ``pad_right = 0`` where the two coincide outright.
    That bounds the deviation precisely: we changed the amount of zero padding
    and nothing else.
    """
    encoder = build_encoder()
    x = make_features(stride_)

    ours = capture_first_layer_input(
        encoder,
        x,
        encoder.init_chunk_cache(
            pad_left=pad_left, stride=stride_, pad_right=pad_right, dtype=torch.float64
        ),
    )
    upstream = capture_first_layer_input(
        encoder, x, upstream_cache(encoder, pad_left, stride_, pad_right)
    )

    aligned = upstream[:, pad_right:, :]
    assert aligned.shape == ours.shape
    assert (aligned - ours).abs().max().item() == 0.0


@pytest.mark.parametrize("pad_left,stride_,pad_right", LOOKAHEAD_GEOMETRIES)
def test_upstream_feats_init_emits_pad_right_phantom_frames_for_the_whole_stream(
    pad_left, stride_, pad_right
):
    """The real damage: upstream's stream is ``pad_right`` frames too long.

    ``forward_chunk`` trims ``[pad_left : len - pad_right]``.  With upstream's
    seed the first call's window is ``pad_right`` longer, so it emits ``stride``
    frames instead of ``stride - pad_right`` -- and the surplus frames are
    encodings of zero padding, not of audio.  The streamed output then no longer
    lines up one-to-one with the input and no longer matches ``forward``'s length,
    which is exactly the contract CTC decoding depends on.
    """
    encoder = build_encoder()
    time = 40
    x = make_features(time)

    ours, our_sizes = stream(encoder, x, pad_left, stride_, pad_right)
    upstream_out, upstream_sizes = stream_with_upstream_seed(
        encoder, x, pad_left, stride_, pad_right
    )

    assert ours.size(1) == time
    assert our_sizes[0] == stride_ - pad_right
    assert upstream_out.size(1) == time + pad_right
    assert upstream_sizes[0] == stride_


@pytest.mark.parametrize(
    "pad_left,stride_,pad_right,expected_max",
    [
        (0, 10, 5, 1.514661),
        (5, 10, 5, 0.370553),
        (0, 8, 4, 1.244551),
    ],
)
def test_upstream_phantom_zero_frames_corrupt_exactly_the_first_emitted_chunk(
    pad_left, stride_, pad_right, expected_max
):
    """The extra zeros are not inert -- they enter attention and the FSMN conv.

    After realigning upstream's stream (dropping its ``pad_right`` phantom
    frames) the remaining frames still differ from ours, but *only* over the
    first call's ``stride - pad_right`` emitted frames: everything from frame
    ``stride`` onward is bit-identical, because by then the cache holds real
    frames in both variants.  The peak deviations are pinned to six decimal
    places below; they are the concrete cost of upstream's seed, and the reason
    ``init_chunk_cache`` deviates from it.
    """
    encoder = build_encoder()
    time = 40
    x = make_features(time)

    ours, _ = stream(encoder, x, pad_left, stride_, pad_right)
    upstream_out, _ = stream_with_upstream_seed(encoder, x, pad_left, stride_, pad_right)
    aligned = upstream_out[:, pad_right:, :]

    per_frame = (aligned - ours).abs().amax(dim=-1)[0]

    assert per_frame[:stride_].max().item() == pytest.approx(expected_max, abs=1e-6)
    assert per_frame[stride_:].max().item() == 0.0
    # The corruption is confined to the frames the first call emitted.
    assert per_frame[stride_ - pad_right :].max().item() == 0.0


def stream_with_upstream_seed(encoder, x, pad_left, stride_, pad_right, look_back=0):
    """``stream`` but with funasr's ``pad_left + pad_right`` zero seed for ``feats``."""
    cache = upstream_cache(
        encoder, pad_left, stride_, pad_right, batch=x.size(0), dtype=x.dtype
    )
    cache["encoder_chunk_look_back"] = look_back
    outputs = []
    with torch.no_grad():
        for start in range(0, x.size(1), stride_):
            out, cache = encoder.forward_chunk(x[:, start : start + stride_, :].clone(), cache)
            outputs.append(out)
        if pad_right > 0:
            cache["tail_chunk"] = True
            out, cache = encoder.forward_chunk(x[:, :0, :], cache)
            outputs.append(out)
    return torch.cat(outputs, dim=1), [out.size(1) for out in outputs]


# ---------------------------------------------------------------------------
# (C) Multi-chunk streaming vs. full attention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad_left,stride_,pad_right", GEOMETRIES)
@pytest.mark.parametrize("time", [40, 37])
def test_streamed_output_has_exactly_as_many_frames_as_full_attention(
    pad_left, stride_, pad_right, time
):
    """The shape contract, which must hold *exactly* even though the values do not.

    CTC decoding, timestamps and the streaming session layer all assume streamed
    frame k corresponds to full-attention frame k.  ``time=37`` makes the last
    real chunk short, exercising the ragged-tail path.
    """
    encoder = build_encoder()
    x = make_features(time)

    reference, olens = run_forward(encoder, x)
    streamed, _ = stream(encoder, x, pad_left, stride_, pad_right)

    assert streamed.shape == reference.shape
    assert streamed.size(1) == time == int(olens[0])


@pytest.mark.parametrize("pad_left,stride_,pad_right", GEOMETRIES)
@pytest.mark.parametrize("time", [40, 37])
def test_per_call_emission_schedule_follows_the_documented_contract(
    pad_left, stride_, pad_right, time
):
    """First call emits ``stride - pad_right``, later calls emit their input length, tail emits ``pad_right``.

    This is what makes "frames line up one-to-one" checkable: the schedule is a
    pure function of the geometry and the utterance length, with no dependence on
    the audio, so a caller can map streamed frame indices back to input frames.
    """
    encoder = build_encoder()
    x = make_features(time)

    _, sizes = stream(encoder, x, pad_left, stride_, pad_right)

    chunk_lengths = [
        min(stride_, time - start) for start in range(0, time, stride_)
    ]
    expected = [chunk_lengths[0] - pad_right] + chunk_lengths[1:]
    if pad_right > 0:
        expected.append(pad_right)

    assert sizes == expected
    assert sum(sizes) == time


@pytest.mark.parametrize("pad_left,stride_,pad_right", GEOMETRIES)
@pytest.mark.parametrize("look_back", [0, -1])
@pytest.mark.parametrize("time", [40, 37])
def test_multi_chunk_streaming_is_close_to_but_not_equal_to_full_attention(
    pad_left, stride_, pad_right, look_back, time
):
    """Streaming is a genuinely different computation, and the tests say so.

    The bounds come from measurement (see ``STREAMING_*`` above), not from what
    would be convenient.  Both directions are asserted: exceeding the upper bound
    means streaming regressed; falling below the lower bound means the streaming
    path stopped being streaming.  See the module docstring for the six
    divergence sources behind the gap.
    """
    if look_back != 0 and pad_right < 1:
        pytest.skip("encoder_chunk_look_back != 0 requires pad_right >= 1")

    encoder = build_encoder()
    x = make_features(time)

    reference, _ = run_forward(encoder, x)
    streamed, _ = stream(encoder, x, pad_left, stride_, pad_right, look_back)

    deviation = (streamed - reference).abs()
    max_dev = deviation.max().item()
    mean_dev = deviation.mean().item()

    assert max_dev < STREAMING_MAX_DEVIATION_UPPER, f"max deviation grew to {max_dev}"
    assert mean_dev < STREAMING_MEAN_DEVIATION_UPPER, f"mean deviation grew to {mean_dev}"
    assert max_dev > STREAMING_MAX_DEVIATION_LOWER, (
        f"max deviation collapsed to {max_dev}: streaming should NOT match full "
        "attention. If the streaming window was legitimately widened, re-measure "
        "and update STREAMING_MAX_DEVIATION_LOWER rather than deleting this check."
    )
    assert mean_dev > STREAMING_MEAN_DEVIATION_LOWER, (
        f"mean deviation collapsed to {mean_dev}: see the note above."
    )


@pytest.mark.parametrize("pad_left,stride_,pad_right", LOOKAHEAD_GEOMETRIES)
def test_unbounded_look_back_reduces_the_mean_deviation_versus_no_look_back(
    pad_left, stride_, pad_right
):
    """More history can only help, and measurably does.

    ``encoder_chunk_look_back=-1`` keeps every past chunk's keys and values, so
    each frame attends to strictly more context than at ``look_back=0`` while the
    right-hand lookahead stays fixed.  Measured mean deviation drops on every
    geometry, e.g. 0.2316 -> 0.2032 at (0, 10, 5) and 0.1264 -> 0.1090 at
    (5, 10, 5).  This is the invariant that survives a reimplementation, unlike
    the absolute numbers.
    """
    encoder = build_encoder()
    time = 40
    x = make_features(time)
    reference, _ = run_forward(encoder, x)

    no_look_back, _ = stream(encoder, x, pad_left, stride_, pad_right, look_back=0)
    all_look_back, _ = stream(encoder, x, pad_left, stride_, pad_right, look_back=-1)

    mean_without = (no_look_back - reference).abs().mean().item()
    mean_with = (all_look_back - reference).abs().mean().item()

    assert mean_with < mean_without


@pytest.mark.parametrize("pad_left,stride_,pad_right", [(0, 10, 5), (5, 10, 5), (0, 10, 0)])
def test_streaming_deviation_is_algorithmic_not_floating_point(pad_left, stride_, pad_right):
    """float32 and float64 report the same deviation to ~1e-7 relative.

    This is what licenses reading the group (C) numbers as real: if the gap were
    rounding noise, doubling the mantissa would shrink it by orders of magnitude.
    It does not move at all, so the gap is the truncated context, and production
    float32 behaves like the float64 measurement.
    """
    time = 40

    encoder64 = build_encoder(dtype=torch.float64)
    x64 = make_features(time, dtype=torch.float64)
    reference64, _ = run_forward(encoder64, x64)
    streamed64, _ = stream(encoder64, x64, pad_left, stride_, pad_right)

    encoder32 = build_encoder(dtype=torch.float32)
    x32 = make_features(time, dtype=torch.float32)
    reference32, _ = run_forward(encoder32, x32)
    streamed32, _ = stream(encoder32, x32, pad_left, stride_, pad_right)

    max64 = (streamed64 - reference64).abs().max().item()
    max32 = (streamed32 - reference32).abs().max().item()
    mean64 = (streamed64 - reference64).abs().mean().item()
    mean32 = (streamed32 - reference32).abs().mean().item()

    # Measured relative agreement is <= 4.2e-7 (max) and <= 6.4e-8 (mean).
    assert abs(max64 - max32) / max64 < 1e-5
    assert abs(mean64 - mean32) / mean64 < 1e-5


# ---------------------------------------------------------------------------
# (D) Boundary invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad_left,stride_,pad_right", GEOMETRIES)
@pytest.mark.parametrize("look_back", [0, -1])
@pytest.mark.parametrize("longer_time", [43, 47, 50])
def test_frames_before_the_tail_do_not_depend_on_where_the_tail_falls(
    pad_left, stride_, pad_right, look_back, longer_time
):
    """Extending the utterance must not retroactively change already-emitted frames.

    This is the property that makes streaming *streaming*: output already handed
    to the caller is final.  A regression here (e.g. a cache that peeks at the
    whole buffer) would only show up as decoded text changing after the fact,
    which is very hard to debug downstream -- hence pinning it directly.

    Feeding a common prefix at a fixed stride, every full-stride call is
    byte-identical between the short and the long run; measured max absolute
    difference is exactly 0.0 for all geometries and both look-back settings.
    """
    if look_back != 0 and pad_right < 1:
        pytest.skip("encoder_chunk_look_back != 0 requires pad_right >= 1")

    encoder = build_encoder()
    base = make_features(80)
    short_time = 40

    short, _ = stream(encoder, base[:, :short_time, :], pad_left, stride_, pad_right, look_back)
    long, _ = stream(encoder, base[:, :longer_time, :], pad_left, stride_, pad_right, look_back)

    # Frames emitted by the calls that are full-stride in *both* runs.
    settled = (short_time // stride_) * stride_ - pad_right

    assert settled > 0
    assert (long[:, :settled, :] - short[:, :settled, :]).abs().max().item() == 0.0


@pytest.mark.parametrize("pad_left,stride_,pad_right", LOOKAHEAD_GEOMETRIES)
def test_the_tail_flush_emits_the_frames_the_lookahead_withheld(pad_left, stride_, pad_right):
    """The ``tail_chunk`` flush is a re-run of the cached overlap, not new work.

    It re-encodes ``cache["feats"]`` -- frames the stream has already seen -- and
    trims off the ``pad_left`` left context, yielding exactly the ``pad_right``
    frames whose lookahead never arrived.  Without it the stream would be
    ``pad_right`` frames short, silently truncating the end of every utterance.
    """
    encoder = build_encoder()
    time = 40
    x = make_features(time)

    streamed, sizes = stream(encoder, x, pad_left, stride_, pad_right)

    assert sizes[-1] == pad_right
    # Skipping the flush would leave the stream exactly pad_right frames short.
    assert sum(sizes[:-1]) == time - pad_right
    assert streamed.size(1) == time
    # The flushed frames really are the last frames of the utterance, not a repeat
    # of frames an earlier call already emitted.
    assert not torch.equal(
        streamed[:, -pad_right:, :], streamed[:, -2 * pad_right : -pad_right, :]
    )


def test_resplitting_the_same_audio_at_different_points_does_change_the_output():
    """Chunk boundaries are part of the computation -- a negative result, pinned.

    One might hope streaming were split-invariant.  It is not, and cannot be:
    each call's self-attention and FSMN see exactly the window
    ``[pad_left + chunk + pad_right]``, so moving a boundary moves every
    receptive field that straddles it.  Measured max absolute difference between
    a uniform stride-10 schedule and a ``[4, 10, 10, 10, 6]`` schedule over the
    same 40 frames is 2.1736.

    The practical consequence, and why this test exists: a caller must not vary
    its chunk length mid-stream and expect stable output.  Feed a constant
    ``stride`` and let only the final partial chunk be short -- which is what
    ``test_frames_before_the_tail_do_not_depend_on_where_the_tail_falls`` shows
    is safe.
    """
    encoder = build_encoder()
    x = make_features(40)

    uniform, _ = stream(encoder, x, 0, 10, 0)
    ragged, _ = stream(encoder, x, 0, 10, 0, chunk_lengths=[4, 10, 10, 10, 6])

    assert uniform.shape == ragged.shape
    assert (uniform - ragged).abs().max().item() > 0.1


# ---------------------------------------------------------------------------
# (E) Training-path sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("training", [False, True])
def test_full_attention_sentinel_forward_equals_a_chunkless_encoder_exactly(training):
    """``chunk_size=(-1,)`` must be a no-op wrapper around the original forward.

    This is the mechanism that mixes full-attention batches into chunk training,
    so any drift here would silently change what the model learns on those steps.
    Measured max absolute difference is 0.0 in both ``train()`` and ``eval()``:
    in eval ``random_choice`` returns index 0, in train it draws from a
    single-entry tuple, and both land on the sentinel.
    """
    chunkless = build_encoder(chunk_size=None)
    sentinel = build_encoder(chunk_size=(-1,))
    if training:
        sentinel.train()

    time = 33
    x = make_features(time)

    reference, ref_lens = run_forward(chunkless, x)
    got, got_lens = run_forward(sentinel, x)

    assert got.shape == reference.shape
    assert (got - reference).abs().max().item() == 0.0
    assert got_lens.tolist() == ref_lens.tolist()


@pytest.mark.parametrize(
    "chunk_size,stride_,pad_left",
    [
        ((15,), (10,), (0,)),
        ((20,), (10,), (5,)),
        ((10,), (10,), (0,)),
        ((16,), (8,), (4,)),
    ],
)
@pytest.mark.parametrize("lengths", [[37], [40, 23, 11], [5]])
def test_chunk_training_forward_returns_the_original_lengths(chunk_size, stride_, pad_left, lengths):
    """``remove_chunk`` restores the time axis, so the CTC loss needs no changes.

    funasr's ``SANMEncoderChunkOpt`` returns the chunk-*expanded* sequence and
    expects the caller to cope.  We fold it back, which is the whole reason CTC
    targets, ``ys_pad_lens`` and the loss call stay untouched between the chunk
    and full-attention paths.  ``[5]`` is shorter than one chunk and ``[40, 23,
    11]`` is ragged -- both are cases where an off-by-one in the fold would
    surface as a length mismatch at loss time rather than here.
    """
    encoder = build_encoder(chunk_size=chunk_size, stride=stride_, pad_left=pad_left)
    x = make_features(max(lengths), batch=len(lengths))

    with torch.no_grad():
        out, olens = encoder(x.clone(), torch.tensor(lengths))

    assert olens.tolist() == lengths
    assert out.shape == (len(lengths), max(lengths), OUTPUT_SIZE)
    assert olens.dtype == torch.int32


@pytest.mark.parametrize("lengths", [[40, 23, 11]])
def test_chunk_training_forward_zeroes_the_padding_of_shorter_utterances(lengths):
    """The fold-back must not leak chunk-expanded values past each utterance's length.

    ``remove_chunk`` masks in place; if it stopped doing so, padding frames would
    carry real activations into the CTC loss for the shorter items in a batch.
    """
    encoder = build_encoder(chunk_size=(15,), stride=(10,), pad_left=(0,))
    x = make_features(max(lengths), batch=len(lengths))

    with torch.no_grad():
        out, _ = encoder(x.clone(), torch.tensor(lengths))

    for i, length in enumerate(lengths):
        if length == max(lengths):
            continue  # the longest item has no padding to check
        assert out[i, length:].abs().max().item() == 0.0
