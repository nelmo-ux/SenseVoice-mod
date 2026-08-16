"""Characterization (pinning) tests for ``funasr.models.scama.chunk_utilis.overlap_chunk``.

SenseVoiceSmall's SCAMA-style chunk training/inference reuses funasr's
``overlap_chunk`` directly rather than vendoring a copy of it.  ``overlap_chunk``
is *not* part of funasr's public, stable API, so these tests pin down the exact
behaviour we depend on.  If funasr is upgraded and these tests fail, the chunk
training code must be re-validated against the new implementation -- do not
"fix" the tests by loosening the assertions.

Everything here is pure numpy/torch shape-and-value checking: no checkpoint is
loaded and no model code from this repository is imported, so the suite stays
fast and stays green while the model side is still being written.
"""

import inspect
import math

import numpy as np
import pytest
import torch

import funasr
from funasr.models.scama.chunk_utilis import overlap_chunk


# SenseVoiceSmall uses an FSMN kernel_size of 11, so the shift is (11 - 1) // 2.
SENSEVOICE_SHFIT_FSMN = 5

# (chunk_size, stride, pad_left, shfit_fsmn) combinations with a non-negative
# pad_right (== chunk_size - stride - pad_left).
VALID_PARAMS = [
    ((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN),
    ((16,), (10,), (3,), SENSEVOICE_SHFIT_FSMN),
    ((10,), (8,), (2,), SENSEVOICE_SHFIT_FSMN),
    ((8,), (4,), (0,), SENSEVOICE_SHFIT_FSMN),
    ((6,), (6,), (0,), 0),  # no overlap, no FSMN shift
]

# Batches of encoder lengths: single utterance, ragged batch, exact multiple of
# the stride, and a batch shorter than one chunk.
LENGTH_SETS = [
    [30],
    [30, 17, 8],
    [40],
    [5],
]


def build(chunk_size, stride, pad_left, shfit_fsmn, encoder_att_look_back_factor=(1,)):
    """Construct an ``overlap_chunk`` the way the SenseVoice chunk encoder does."""
    return overlap_chunk(
        chunk_size=chunk_size,
        stride=stride,
        pad_left=pad_left,
        encoder_att_look_back_factor=encoder_att_look_back_factor,
        shfit_fsmn=shfit_fsmn,
    )


def make_lengths(lengths, dtype=torch.int32):
    return torch.tensor(lengths, dtype=dtype)


def make_features(lengths, dim=3, seed=0):
    """Deterministic, all-nonzero features so zeroing is detectable."""
    generator = torch.Generator().manual_seed(seed)
    batch, time = len(lengths), max(lengths)
    x = torch.rand(batch, time, dim, generator=generator) + 1.0
    return x


def zero_padded_reference(x, lengths):
    """The input with everything at/after each utterance's length zeroed out."""
    reference = x[:, : max(lengths), :].clone()
    for i, length in enumerate(lengths):
        reference[i, length:] = 0.0
    return reference


# ---------------------------------------------------------------------------
# 5. funasr version pinning
# ---------------------------------------------------------------------------


def test_funasr_version_is_pinned_to_the_validated_release():
    assert funasr.__version__ == "1.4.1", (
        f"funasr is {funasr.__version__}, but the SCAMA chunk masks were validated "
        "against 1.4.1. overlap_chunk is not a stable public API: re-verify its "
        "behaviour (round-trip, mask value ranges, chunk_outs tuple order) before "
        "updating this pin."
    )


# ---------------------------------------------------------------------------
# 4. Parameter derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
def test_get_chunk_size_returns_the_five_element_parameter_tuple(
    chunk_size, stride, pad_left, shfit_fsmn
):
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)

    result = oc.get_chunk_size(0)

    assert result == (chunk_size[0], stride[0], pad_left[0], 1, chunk_size[0] + shfit_fsmn)


def test_chunk_size_pad_shift_is_chunk_size_plus_shfit_fsmn():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)

    _, _, _, _, chunk_size_pad_shift = oc.get_chunk_size(0)

    assert chunk_size_pad_shift == 16 + SENSEVOICE_SHFIT_FSMN == 21


def test_get_chunk_size_also_publishes_the_values_as_attributes():
    oc = build((16,), (10,), (3,), SENSEVOICE_SHFIT_FSMN)

    oc.get_chunk_size(0)

    assert oc.chunk_size_cur == 16
    assert oc.stride_cur == 10
    assert oc.pad_left_cur == 3
    assert oc.chunk_size_pad_shift_cur == 21
    assert oc.decoder_att_look_back_factor_cur == 1


def test_scalar_tuples_are_broadcast_to_the_chunk_size_length():
    """pad_left / look-back factors given as 1-tuples are repeated per chunk size."""
    oc = overlap_chunk(
        chunk_size=(4, 8, 16),
        stride=(2, 6, 10),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )

    assert list(oc.pad_left) == [0, 0, 0]
    assert list(oc.encoder_att_look_back_factor) == [1, 1, 1]
    assert list(oc.decoder_att_look_back_factor) == [1, 1, 1]


def test_get_chunk_size_raises_index_error_beyond_the_configured_chunk_sizes():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)

    with pytest.raises(IndexError):
        oc.get_chunk_size(1)


@pytest.mark.parametrize(
    "chunk_size,stride,pad_left",
    [
        ((10,), (8,), (5,)),  # pad_right == -3, pad_left too large
        ((8,), (10,), (0,)),  # pad_right == -2, stride larger than chunk_size
    ],
)
def test_negative_pad_right_raises_value_error_inside_gen_chunk_mask(
    chunk_size, stride, pad_left
):
    """pad_right = chunk_size - stride - pad_left must stay non-negative.

    A negative value is not validated at construction time; it surfaces as a
    numpy "negative dimensions are not allowed" ValueError only once
    ``gen_chunk_mask`` runs, so configuration must be checked by the caller.
    """
    oc = build(chunk_size, stride, pad_left, SENSEVOICE_SHFIT_FSMN)

    with pytest.raises(ValueError):
        oc.gen_chunk_mask(make_lengths([20]), 0)


def test_pad_right_is_chunk_size_minus_stride_minus_pad_left():
    chunk_size, stride, pad_left = 16, 10, 3
    oc = build((chunk_size,), (stride,), (pad_left,), SENSEVOICE_SHFIT_FSMN)
    outs = oc.gen_chunk_mask(make_lengths([30]), 0)
    pad_right = chunk_size - stride - pad_left

    # Each per-chunk block of x_rm_mask is laid out as
    # [shfit_fsmn | pad_left | stride | pad_right] columns.
    x_rm_mask = outs[2]
    block = SENSEVOICE_SHFIT_FSMN + pad_left + stride + pad_right

    assert pad_right == 3
    assert block == chunk_size + SENSEVOICE_SHFIT_FSMN
    # The first block's selected columns start after shfit_fsmn + pad_left.
    first_selected = int(np.flatnonzero(x_rm_mask[0])[0])
    assert first_selected == SENSEVOICE_SHFIT_FSMN + pad_left


# ---------------------------------------------------------------------------
# 3. Dynamic chunk selection
# ---------------------------------------------------------------------------


def test_random_choice_always_returns_zero_for_a_single_chunk_size():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)

    torch.manual_seed(0)
    choices = {oc.random_choice(training=True) for _ in range(50)}

    assert choices == {0}


def test_random_choice_ignores_decoding_ind_while_training_on_a_single_chunk_size():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)

    assert oc.random_choice(training=True, decoding_ind=3) == 0


def test_random_choice_samples_every_chunk_size_during_training():
    """This is the dynamic-chunk-training mechanism: a random config per step."""
    oc = overlap_chunk(
        chunk_size=(4, 8, 16),
        stride=(2, 6, 10),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )

    torch.manual_seed(1234)
    choices = [oc.random_choice(training=True) for _ in range(200)]

    assert set(choices) == {0, 1, 2}
    assert all(0 <= ind < len(oc.chunk_size) for ind in choices)


def test_random_choice_is_deterministic_under_a_fixed_seed():
    oc = overlap_chunk(
        chunk_size=(4, 8, 16),
        stride=(2, 6, 10),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )

    torch.manual_seed(7)
    first = [oc.random_choice(training=True) for _ in range(20)]
    torch.manual_seed(7)
    second = [oc.random_choice(training=True) for _ in range(20)]

    assert first == second


@pytest.mark.parametrize("decoding_ind", [0, 1, 2])
def test_random_choice_returns_decoding_ind_when_not_training(decoding_ind):
    oc = overlap_chunk(
        chunk_size=(4, 8, 16),
        stride=(2, 6, 10),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )

    assert oc.random_choice(training=False, decoding_ind=decoding_ind) == decoding_ind


def test_random_choice_defaults_to_zero_when_not_training_without_decoding_ind():
    oc = overlap_chunk(
        chunk_size=(4, 8, 16),
        stride=(2, 6, 10),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )

    assert oc.random_choice(training=False) == 0


def test_random_choice_does_not_range_check_decoding_ind():
    """An out-of-range decoding_ind is returned as-is and only fails later."""
    oc = overlap_chunk(
        chunk_size=(4, 8),
        stride=(2, 6),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )

    assert oc.random_choice(training=False, decoding_ind=5) == 5
    with pytest.raises(IndexError):
        oc.get_chunk_size(5)


# ---------------------------------------------------------------------------
# chunk_outs tuple layout / getter indices
# ---------------------------------------------------------------------------


def test_chunk_outs_has_eight_entries_in_a_fixed_order():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)

    outs = oc.gen_chunk_mask(make_lengths([30]), 0)

    assert isinstance(outs, tuple)
    assert len(outs) == 8
    assert outs[0] is oc.x_add_mask
    assert outs[1] is oc.x_len_chunk
    assert outs[2] is oc.x_rm_mask
    assert outs[3] is oc.x_len
    assert outs[4] is oc.mask_shfit_chunk
    assert outs[5] is oc.mask_chunk_predictor
    assert outs[6] is oc.mask_att_chunk_encoder
    assert outs[7] is oc.mask_shift_att_chunk_decoder


@pytest.mark.parametrize(
    "method_name,expected_idx",
    [
        ("get_x_add_mask", 0),
        ("get_x_len_chunk", 1),
        ("get_x_rm_mask", 2),
        ("get_x_len", 3),
        ("get_mask_shfit_chunk", 4),
        ("get_mask_chunk_predictor", 5),
        ("get_mask_att_chunk_encoder", 6),
        ("get_mask_shift_att_chunk_decoder", 7),
    ],
)
def test_getter_default_idx_matches_the_chunk_outs_slot(method_name, expected_idx):
    """``get_mask_att_chunk_encoder`` is idx=6, not 5 -- do not rely on ordering by name."""
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)
    signature = inspect.signature(getattr(oc, method_name))

    assert signature.parameters["idx"].default == expected_idx


# ---------------------------------------------------------------------------
# 2. Shapes and invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_x_len_chunk_matches_the_documented_formula(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)
    x_len = np.array(lengths, dtype=np.int32)

    outs = oc.gen_chunk_mask(make_lengths(lengths), 0)

    chunk_num_batch = np.ceil(x_len / stride[0]).astype(np.int32)
    expected = (
        (chunk_num_batch - 1) * (chunk_size[0] + shfit_fsmn)
        + shfit_fsmn
        + pad_left[0]
        + x_len
        - (chunk_num_batch - 1) * stride[0]
    )

    np.testing.assert_array_equal(outs[1], expected)


def test_x_len_chunk_dtype_follows_the_input_length_dtype():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)

    int32_outs = oc.gen_chunk_mask(make_lengths([30], dtype=torch.int32), 0)
    int64_outs = oc.gen_chunk_mask(make_lengths([30], dtype=torch.int64), 0)

    assert int32_outs[1].dtype == np.int32
    assert int64_outs[1].dtype == np.int64


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_mask_shapes_are_consistent_with_the_expanded_length(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)

    outs = oc.gen_chunk_mask(make_lengths(lengths), 0)
    (
        x_add_mask,
        x_len_chunk,
        x_rm_mask,
        x_len,
        mask_shfit_chunk,
        mask_chunk_predictor,
        mask_att_chunk_encoder,
        mask_shift_att_chunk_decoder,
    ) = outs

    x_len_max = max(lengths)
    x_len_chunk_max = int(x_len_chunk.max())

    assert x_add_mask.shape == (x_len_chunk_max, x_len_max + pad_left[0])
    assert x_rm_mask.shape == (x_len_max, x_len_chunk_max)
    assert mask_shfit_chunk.shape == (x_len_chunk_max, 1)
    assert mask_chunk_predictor.shape == (x_len_chunk_max, 1)
    assert mask_att_chunk_encoder.shape == (x_len_chunk_max, x_len_chunk_max)
    assert mask_shift_att_chunk_decoder.shape == (x_len_chunk_max, 1)
    np.testing.assert_array_equal(x_len, np.array(lengths))


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_all_masks_are_multiplicative_zero_one_masks(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    """model.py multiplies these into attention/FSMN masks, so they must be {0, 1}."""
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)

    outs = oc.gen_chunk_mask(make_lengths(lengths), 0)

    for idx in (0, 2, 4, 5, 6, 7):
        values = np.unique(outs[idx])
        assert np.isin(values, [0, 1]).all(), f"chunk_outs[{idx}] holds {values}"


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_x_add_mask_copies_at_most_one_input_frame_per_output_row(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)

    x_add_mask = oc.gen_chunk_mask(make_lengths(lengths), 0)[0]

    assert set(np.unique(x_add_mask.sum(axis=1))) <= {0.0, 1.0}


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_x_rm_mask_selects_exactly_one_expanded_frame_per_original_frame(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)

    x_rm_mask = oc.gen_chunk_mask(make_lengths(lengths), 0)[2]

    np.testing.assert_array_equal(x_rm_mask.sum(axis=1), np.ones(max(lengths)))


def test_mask_shfit_chunk_zeroes_exactly_the_fsmn_shift_rows():
    chunk_size, stride, shfit_fsmn = 6, 4, 2
    oc = build((chunk_size,), (stride,), (0,), shfit_fsmn)

    mask_shfit_chunk = oc.gen_chunk_mask(make_lengths([12]), 0)[4].flatten()

    # Blocks of [shfit_fsmn zeros | chunk_size ones], truncated to x_len_chunk_max.
    expected = np.tile(
        np.concatenate([np.zeros(shfit_fsmn), np.ones(chunk_size)]), 3
    )[: len(mask_shfit_chunk)]
    np.testing.assert_array_equal(mask_shfit_chunk, expected)


def test_mask_chunk_predictor_keeps_only_the_stride_frames_of_each_chunk():
    chunk_size, stride, shfit_fsmn = 6, 4, 2
    oc = build((chunk_size,), (stride,), (0,), shfit_fsmn)

    mask_chunk_predictor = oc.gen_chunk_mask(make_lengths([12]), 0)[5].flatten()

    # [shfit_fsmn + pad_left zeros | stride ones | (chunk_size - stride - pad_left) zeros]
    expected = np.tile(
        np.concatenate([np.zeros(shfit_fsmn), np.ones(stride), np.zeros(chunk_size - stride)]),
        3,
    )[: len(mask_chunk_predictor)]
    np.testing.assert_array_equal(mask_chunk_predictor, expected)


def test_mask_att_chunk_encoder_blanks_the_fsmn_shift_rows():
    """FSMN shift rows attend to nothing at all once the mask is multiplied in."""
    chunk_size, stride, shfit_fsmn = 6, 4, 2
    oc = build((chunk_size,), (stride,), (0,), shfit_fsmn)

    mask_att = oc.gen_chunk_mask(make_lengths([12]), 0)[6]

    assert mask_att[:shfit_fsmn].sum() == 0
    block = chunk_size + shfit_fsmn
    assert mask_att[block : block + shfit_fsmn].sum() == 0


def test_mask_att_chunk_encoder_hides_future_chunks_and_limits_the_look_back():
    """Chunk c may attend to its own chunk plus ``encoder_att_look_back_factor`` back."""
    chunk_size, stride, shfit_fsmn = 6, 4, 2
    block = chunk_size + shfit_fsmn
    oc = build((chunk_size,), (stride,), (0,), shfit_fsmn, encoder_att_look_back_factor=(1,))

    mask_att = oc.gen_chunk_mask(make_lengths([12]), 0)[6]

    # Chunk 0's rows must not see anything belonging to chunk 1 or later.
    assert mask_att[shfit_fsmn:block, block:].sum() == 0
    # Chunk 2's rows must not see chunk 0 (only one chunk of look-back is allowed).
    assert mask_att[2 * block + shfit_fsmn :, :block].sum() == 0
    # But chunk 2's rows do attend within chunk 1 and chunk 2.
    assert mask_att[2 * block + shfit_fsmn :, block:].sum() > 0


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_units", [1, 4])
def test_mask_getters_tile_over_the_batch_and_return_float32(batch_size, num_units):
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)
    outs = oc.gen_chunk_mask(make_lengths([30, 12][:batch_size] or [30]), 0)
    x_len_chunk_max = int(outs[1].max())

    mask_shfit = oc.get_mask_shfit_chunk(outs, "cpu", batch_size=batch_size, num_units=num_units)
    mask_pred = oc.get_mask_chunk_predictor(
        outs, "cpu", batch_size=batch_size, num_units=num_units
    )
    mask_att = oc.get_mask_att_chunk_encoder(outs, "cpu", batch_size=batch_size)
    mask_dec = oc.get_mask_shift_att_chunk_decoder(outs, "cpu", batch_size=batch_size)

    assert mask_shfit.shape == (batch_size, x_len_chunk_max, num_units)
    assert mask_pred.shape == (batch_size, x_len_chunk_max, num_units)
    assert mask_att.shape == (batch_size, x_len_chunk_max, x_len_chunk_max)
    assert mask_dec.shape == (batch_size, 1, x_len_chunk_max)
    for mask in (mask_shfit, mask_pred, mask_att, mask_dec):
        assert mask.dtype == torch.float32
        assert torch.isin(mask, torch.tensor([0.0, 1.0])).all()


def test_mask_getters_honour_an_explicit_dtype():
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)
    outs = oc.gen_chunk_mask(make_lengths([30]), 0)

    mask = oc.get_mask_att_chunk_encoder(outs, "cpu", batch_size=1, dtype=torch.float16)

    assert mask.dtype == torch.float16


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_split_chunk_then_remove_chunk_restores_shape_values_and_lengths(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    """split_chunk -> remove_chunk is an *exact* round trip.

    Verified for every combination above, including pad_left > 0 and ragged
    batches: the restored tensor equals the input with the per-utterance padding
    zeroed, bit for bit.  The only information lost is what was already padding.
    """
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)
    x_len = make_lengths(lengths)
    outs = oc.gen_chunk_mask(x_len, 0)
    x = make_features(lengths)
    reference = zero_padded_reference(x, lengths)

    x_chunk, x_len_chunk = oc.split_chunk(x.clone(), x_len, outs)
    restored, restored_len = oc.remove_chunk(x_chunk, x_len_chunk, outs)

    assert restored.shape == reference.shape
    assert torch.equal(restored, reference)
    np.testing.assert_array_equal(restored_len.numpy(), np.array(lengths))


@pytest.mark.parametrize("chunk_size,stride,pad_left,shfit_fsmn", VALID_PARAMS)
@pytest.mark.parametrize("lengths", LENGTH_SETS)
def test_split_chunk_expands_the_time_axis_to_x_len_chunk(
    chunk_size, stride, pad_left, shfit_fsmn, lengths
):
    oc = build(chunk_size, stride, pad_left, shfit_fsmn)
    x_len = make_lengths(lengths)
    outs = oc.gen_chunk_mask(x_len, 0)
    x = make_features(lengths)

    x_chunk, x_len_chunk = oc.split_chunk(x.clone(), x_len, outs)

    chunk_num = math.ceil(max(lengths) / stride[0])
    # Every chunk after the first re-materialises (chunk_size - stride) overlapping
    # frames plus shfit_fsmn FSMN-shift rows, and the first chunk adds shfit_fsmn
    # rows plus pad_left frames.
    expected_growth = (chunk_num - 1) * (chunk_size[0] - stride[0] + shfit_fsmn) + shfit_fsmn + pad_left[0]

    assert x_chunk.shape == (len(lengths), int(x_len_chunk.max()), x.shape[-1])
    assert int(x_len_chunk.max()) == max(lengths) + expected_growth
    assert x_len_chunk.dtype == x_len.dtype
    np.testing.assert_array_equal(x_len_chunk.numpy(), outs[1])


def test_split_chunk_zeroes_the_expanded_frames_beyond_each_utterance_length():
    lengths = [30, 12]
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)
    x_len = make_lengths(lengths)
    outs = oc.gen_chunk_mask(x_len, 0)

    x_chunk, x_len_chunk = oc.split_chunk(make_features(lengths), x_len, outs)

    for i in range(len(lengths)):
        assert x_chunk[i, int(x_len_chunk[i]) :].abs().sum() == 0


def test_split_chunk_mutates_its_input_tensor_in_place():
    """Gotcha: split_chunk applies the length mask with ``x *= ...`` on a view.

    Callers that still need the original features must pass a clone.
    """
    lengths = [30, 12]
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)
    x_len = make_lengths(lengths)
    outs = oc.gen_chunk_mask(x_len, 0)
    x = make_features(lengths)
    before = x.clone()

    oc.split_chunk(x, x_len, outs)

    assert not torch.equal(x, before)
    assert x[1, 12:].abs().sum() == 0
    assert torch.equal(x[0], before[0])


def test_remove_chunk_mutates_its_input_tensor_in_place():
    lengths = [30, 12]
    oc = build((16,), (10,), (0,), SENSEVOICE_SHFIT_FSMN)
    x_len = make_lengths(lengths)
    outs = oc.gen_chunk_mask(x_len, 0)
    x_chunk, x_len_chunk = oc.split_chunk(make_features(lengths), x_len, outs)
    x_chunk = torch.ones_like(x_chunk)
    before = x_chunk.clone()

    oc.remove_chunk(x_chunk, x_len_chunk, outs)

    assert not torch.equal(x_chunk, before)
    assert x_chunk[1, int(x_len_chunk[1]) :].abs().sum() == 0


def test_round_trip_is_exact_for_a_second_chunk_config_of_a_dynamic_setup():
    """Each index of a multi-chunk-size config round-trips independently."""
    lengths = [30, 17]
    oc = overlap_chunk(
        chunk_size=(8, 16),
        stride=(6, 10),
        pad_left=(0,),
        encoder_att_look_back_factor=(1,),
        shfit_fsmn=SENSEVOICE_SHFIT_FSMN,
    )
    x_len = make_lengths(lengths)
    x = make_features(lengths)
    reference = zero_padded_reference(x, lengths)

    for ind in range(len(oc.chunk_size)):
        outs = oc.gen_chunk_mask(x_len, ind)
        x_chunk, x_len_chunk = oc.split_chunk(x.clone(), x_len, outs)
        restored, _ = oc.remove_chunk(x_chunk, x_len_chunk, outs)

        assert torch.equal(restored, reference), f"round trip failed for ind={ind}"
