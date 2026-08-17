"""Tests for the emotion-slot mask and the rich-loss weight in ``SenseVoiceSmall``.

funasr assembles the training ``text`` tensor as
``lid_ids + emo_ids + event_ids + punc_itn_bottom_ids + target_ids``
(``funasr/datasets/sense_voice_datasets/datasets.py:408``), so ``text[:, :4]`` is
``[language, emotion, event, textnorm]`` and the **emotion slot is index 1**.

``scripts/prepare_vn_data.py`` writes strings into the manifest, so it cannot put
``ignore_id`` (-1) into the emotion slot for utterances with no emotion label; it
writes the ``<|SER|>`` sentinel instead and ``SenseVoiceSmall`` maps that id to
``ignore_id`` just before the rich CE loss.

No checkpoint is downloaded here: every model is a tiny, randomly initialised
``SenseVoiceSmall`` built with a fixed seed.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import funasr
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import force_gatherable

from model import SER_TOKEN_ID, SenseVoiceSmall, _optional_scalar


REPO_ROOT = Path(__file__).resolve().parent.parent
BPE_MODEL = REPO_ROOT / "models" / "SenseVoiceSmall" / "chn_jpn_yue_eng_ko_spectok.bpe.model"

# Real ids from chn_jpn_yue_eng_ko_spectok.bpe.model (see the sentencepiece test below).
SER = 24991  # <|SER|> -- the "no emotion label" sentinel written by the data prep
NEUTRAL = 25004  # <|NEUTRAL|>
HAPPY = 25001  # <|HAPPY|>
SAD = 25002  # <|SAD|>

# An "old-style" batch: every clip carries a real emotion label and no sentinel appears.
# This is what a pre-<|SER|> manifest looks like, and it must keep working untouched.
LABELLED = [NEUTRAL, HAPPY, SAD, NEUTRAL]
ZH = 24884  # <|zh|>, a key of SenseVoiceSmall.lid_int_dict
SPEECH = 24993  # <|Speech|>
WOITN = 25017  # <|woitn|>, a key of SenseVoiceSmall.textnorm_int_dict
VOCAB_SIZE = 25055  # the real vocabulary; the token ids above must be valid indices

IGNORE_ID = -1
INPUT_DIM = 8  # tiny stand-in for the real 80-dim LFR fbank
ENC_DIM = 8
SEED = 1234


def build_model(seed: int = SEED, **model_conf) -> SenseVoiceSmall:
    """A minimal, randomly initialised SenseVoiceSmall (1 block, 8-dim, no dropout)."""
    torch.manual_seed(seed)
    # The shipped models/SenseVoiceSmall/config.yaml sets length_normalized_loss=true,
    # and that is what makes LabelSmoothingLoss normalise by the non-ignored count.
    model_conf.setdefault("length_normalized_loss", True)
    model = SenseVoiceSmall(
        encoder="SenseVoiceEncoderSmall",
        encoder_conf=dict(
            output_size=ENC_DIM,
            attention_heads=2,
            linear_units=16,
            num_blocks=1,
            tp_blocks=1,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            input_layer="pe",
            normalize_before=True,
            kernel_size=11,
            sanm_shfit=0,
            selfattention_layer_type="sanm",
        ),
        input_size=INPUT_DIM,
        vocab_size=VOCAB_SIZE,
        ignore_id=IGNORE_ID,
        blank_id=0,
        sos=1,
        eos=2,
        **model_conf,
    )
    model.eval()
    return model


@dataclass
class Batch:
    """One training batch.  ``encode`` mutates ``speech_lengths`` in place, so every
    accessor hands out fresh tensors."""

    speech: torch.Tensor
    speech_lengths: torch.Tensor
    text: torch.Tensor
    text_lengths: torch.Tensor

    def fresh(self) -> "Batch":
        return Batch(
            self.speech.clone(),
            self.speech_lengths.clone(),
            self.text.clone(),
            self.text_lengths.clone(),
        )


def make_batch(emotions, seed: int = 0, frames: int = 12) -> Batch:
    """A batch whose only per-row difference is the emotion slot."""
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(emotions)
    speech = torch.randn(batch_size, frames, INPUT_DIM, generator=generator)
    speech_lengths = torch.full((batch_size,), frames, dtype=torch.int32)
    transcript = [100, 101, 102]
    text = torch.tensor(
        [[ZH, emo, SPEECH, WOITN, *transcript] for emo in emotions], dtype=torch.int64
    )
    text_lengths = torch.full((batch_size,), text.size(1), dtype=torch.int32)
    return Batch(speech, speech_lengths, text, text_lengths)


def reference_forward(model: SenseVoiceSmall, batch: Batch, seed: int = SEED):
    """The pre-change loss computation, inlined.

    ``encode`` draws ``torch.rand(1)`` per row for language-id dropout, so the seed
    must be reset before every comparable call.
    """
    torch.manual_seed(seed)
    fresh = batch.fresh()
    encoder_out, encoder_out_lens = model.encode(
        fresh.speech, fresh.speech_lengths, fresh.text
    )
    loss_ctc, _ = model._calc_ctc_loss(
        encoder_out[:, 4:, :],
        encoder_out_lens - 4,
        fresh.text[:, 4:],
        fresh.text_lengths - 4,
    )
    decoder_out = model.ctc.ctc_lo(encoder_out[:, :4, :])
    ys_pad = fresh.text[:, :4].contiguous()
    loss_rich = model.criterion_att(decoder_out, ys_pad)
    acc_rich = th_accuracy(
        decoder_out.view(-1, model.vocab_size), ys_pad, ignore_label=model.ignore_id
    )
    return loss_ctc + loss_rich, loss_ctc, loss_rich, acc_rich


def run_forward(model: SenseVoiceSmall, batch: Batch, seed: int = SEED):
    torch.manual_seed(seed)
    fresh = batch.fresh()
    return model(fresh.speech, fresh.speech_lengths, fresh.text, fresh.text_lengths)


# ---------------------------------------------------------------------------
# 1. Backward compatibility
# ---------------------------------------------------------------------------


def test_defaults_leave_the_options_unset():
    model = build_model()

    assert model.emo_mask_token_id is None
    assert model.rich_loss_weight == 1.0


def test_forward_is_numerically_identical_to_the_pre_change_formula():
    """Top priority: an unconfigured model must reproduce the old numbers bit for bit."""
    model = build_model()
    batch = make_batch(LABELLED)

    loss, stats, _ = run_forward(model, batch)
    ref_loss, ref_ctc, ref_rich, ref_acc = reference_forward(model, batch)

    assert torch.equal(loss, ref_loss.detach()[None])
    assert torch.equal(stats["loss"], ref_loss.detach()[None])
    assert torch.equal(stats["loss_ctc"], ref_ctc.detach()[None])
    assert torch.equal(stats["loss_rich"], ref_rich.detach()[None])
    assert float(stats["acc_rich"]) == ref_acc


def test_acc_emo_is_reported_and_finite_for_an_unconfigured_model():
    model = build_model()
    batch = make_batch(LABELLED)

    _, stats, _ = run_forward(model, batch)

    assert "acc_emo" in stats
    assert torch.isfinite(stats["acc_emo"]).all()
    assert 0.0 <= float(stats["acc_emo"]) <= 1.0


def test_an_old_style_all_labelled_batch_never_trips_the_guard():
    """The compatibility guarantee, pinned rather than assumed.

    A pre-<|SER|> manifest gives every clip a real emotion label, so the
    misconfiguration guard below is inert and old configs are untouched.
    """
    model = build_model()

    loss, stats, _ = run_forward(model, make_batch([NEUTRAL] * 4))

    assert torch.isfinite(loss).all()
    assert torch.isfinite(stats["acc_emo"]).all()


# ---------------------------------------------------------------------------
# 1b. Misconfiguration guard: sentinel present but masking not switched on
# ---------------------------------------------------------------------------


def test_sentinel_without_emo_mask_token_id_raises():
    """The blocking failure this guard exists for.

    A manifest built with --emo-labels but a job that forgot EMO_MASK_TOKEN_ID would
    otherwise fit <|SER|> as the majority emotion class: acc_emo climbs towards the
    sentinel's share of the data and looks healthy while the emotion head learns only
    to emit the sentinel.  Nothing else cross-checks the manifest against the flags,
    so fail at step 1 rather than after days of GPU time.
    """
    model = build_model()

    with pytest.raises(RuntimeError) as excinfo:
        run_forward(model, make_batch([NEUTRAL, SER, HAPPY, NEUTRAL]))

    message = str(excinfo.value)
    # The message must name the fix, not just the symptom.
    assert "EMO_MASK_TOKEN_ID=24991" in message
    assert "++model_conf.emo_mask_token_id=24991" in message
    assert "<|SER|>" in message


def test_the_guard_fires_on_a_single_sentinel_row():
    model = build_model()

    with pytest.raises(RuntimeError):
        run_forward(model, make_batch([NEUTRAL, NEUTRAL, NEUTRAL, SER]))


def test_the_guard_only_inspects_the_emotion_slot():
    """id 24991 elsewhere in the sequence is not this failure mode and must not raise."""
    model = build_model()
    batch = make_batch([NEUTRAL, HAPPY])
    batch.text[:, 4] = SER_TOKEN_ID  # inside the transcript, not the emotion slot

    loss, _, _ = run_forward(model, batch)

    assert torch.isfinite(loss).all()


def test_configuring_the_mask_disarms_the_guard():
    """With the option set, the same batch trains normally -- the guard is about the
    *combination* of sentinel-bearing data and absent configuration."""
    model = build_model(emo_mask_token_id=SER)

    loss, stats, _ = run_forward(model, make_batch([NEUTRAL, SER, HAPPY, NEUTRAL]))

    assert torch.isfinite(loss).all()
    assert stats["acc_emo"] is not None


def test_ser_token_id_constant_matches_the_documented_sentinel():
    assert SER_TOKEN_ID == SER == 24991


# ---------------------------------------------------------------------------
# 1c. Startup logging of the resolved configuration
# ---------------------------------------------------------------------------


def test_init_logs_the_resolved_options(caplog):
    """funasr builds model_conf as ``deep_update(model_conf, kwargs.get("model_conf", {}))``
    then ``deep_update(model_conf, kwargs)`` (funasr/auto/auto_model.py:615-618), so a
    stray top-level key silently overrides the model_conf entry.  The model cannot
    prevent that, so it records what it actually resolved.
    """
    with caplog.at_level("INFO"):
        build_model(emo_mask_token_id="24991", rich_loss_weight="0.5")

    messages = [r.getMessage() for r in caplog.records]
    line = next(m for m in messages if "rich-loss config" in m)
    # The *resolved* values, not the strings that were passed in.
    assert "emo_mask_token_id=24991" in line
    assert "rich_loss_weight=0.5" in line
    assert "length_normalized_loss=True" in line


def test_init_logs_the_unset_defaults_too(caplog):
    with caplog.at_level("INFO"):
        build_model()

    line = next(m for m in (r.getMessage() for r in caplog.records) if "rich-loss config" in m)

    assert "emo_mask_token_id=None" in line
    assert "rich_loss_weight=1.0" in line


def test_init_logs_once_not_per_step(caplog):
    model = build_model(emo_mask_token_id=SER)
    caplog.clear()

    with caplog.at_level("INFO"):
        run_forward(model, make_batch([NEUTRAL, SER]))
        run_forward(model, make_batch([NEUTRAL, SER]))

    assert [m for m in (r.getMessage() for r in caplog.records) if "rich-loss config" in m] == []


# ---------------------------------------------------------------------------
# 2. Masking blocks the gradient
# ---------------------------------------------------------------------------


def capture_rich_target(model: SenseVoiceSmall) -> dict:
    """Record the arguments ``forward`` hands to ``_calc_rich_ce_loss``."""
    captured: dict = {}
    original = model._calc_rich_ce_loss

    def spy(encoder_out, ys_pad):
        captured["encoder_out"] = encoder_out.detach().clone()
        captured["ys_pad"] = ys_pad.detach().clone()
        return original(encoder_out, ys_pad)

    model._calc_rich_ce_loss = spy
    return captured


def test_forward_maps_the_sentinel_to_ignore_id_in_the_emotion_slot_only():
    model = build_model(emo_mask_token_id=SER)
    captured = capture_rich_target(model)
    batch = make_batch([SER, NEUTRAL, SER, HAPPY])

    run_forward(model, batch)

    ys_pad = captured["ys_pad"]
    assert torch.equal(ys_pad[:, 1], torch.tensor([IGNORE_ID, NEUTRAL, IGNORE_ID, HAPPY]))
    # Language / event / textnorm slots are untouched.
    assert torch.equal(ys_pad[:, 0], batch.text[:, 0])
    assert torch.equal(ys_pad[:, 2], batch.text[:, 2])
    assert torch.equal(ys_pad[:, 3], batch.text[:, 3])


def test_forward_does_not_mutate_the_incoming_text_tensor():
    """``text`` is shared with the CTC branch, so the mask must run on a copy."""
    model = build_model(emo_mask_token_id=SER)
    batch = make_batch([SER, NEUTRAL])
    text = batch.text.clone()
    before = text.clone()

    torch.manual_seed(SEED)
    model(batch.speech, batch.speech_lengths, text, batch.text_lengths)

    assert torch.equal(text, before)


def test_masked_emotion_slot_receives_exactly_zero_gradient():
    model = build_model(emo_mask_token_id=SER)
    encoder_out = torch.randn(3, 4, ENC_DIM, generator=torch.Generator().manual_seed(0))
    encoder_out.requires_grad_(True)
    ys_pad = torch.tensor(
        [[ZH, IGNORE_ID, SPEECH, WOITN]] * 3, dtype=torch.int64
    )  # every emotion slot masked

    loss_rich, _, _ = model._calc_rich_ce_loss(encoder_out, ys_pad)
    loss_rich.backward()

    assert torch.equal(encoder_out.grad[:, 1, :], torch.zeros(3, ENC_DIM))
    # The unmasked slots do get gradient, so the zero above is not vacuous.
    assert encoder_out.grad[:, 0, :].abs().sum() > 0


def test_loss_rich_is_independent_of_the_logits_under_a_masked_slot():
    model = build_model(emo_mask_token_id=SER)
    encoder_out = torch.randn(3, 4, ENC_DIM, generator=torch.Generator().manual_seed(1))
    perturbed = encoder_out.clone()
    perturbed[:, 1, :] += 100.0
    ys_pad = torch.tensor([[ZH, IGNORE_ID, SPEECH, WOITN]] * 3, dtype=torch.int64)

    loss_a, _, _ = model._calc_rich_ce_loss(encoder_out, ys_pad)
    loss_b, _, _ = model._calc_rich_ce_loss(perturbed, ys_pad)

    assert torch.equal(loss_a, loss_b)


# ---------------------------------------------------------------------------
# 3. The denominator shrinks with the mask
# ---------------------------------------------------------------------------


def compact(model: SenseVoiceSmall, encoder_out: torch.Tensor, ys_pad: torch.Tensor):
    """Squeeze the non-ignored (row, slot) pairs into a single-row batch of logits."""
    decoder_out = model.ctc.ctc_lo(encoder_out)
    keep = ys_pad != IGNORE_ID
    return decoder_out[keep].unsqueeze(0), ys_pad[keep].unsqueeze(0)


def test_masked_positions_leave_the_length_normalised_denominator():
    """With length_normalized_loss=True the loss is a mean over *non-ignored* targets.

    Half the rows carry the sentinel; the resulting loss must equal the loss over a
    batch made only of the surviving targets, which can only hold if the denominator
    is the non-ignored count (14) rather than the padded count (16).
    """
    model = build_model(emo_mask_token_id=SER)
    encoder_out = torch.randn(4, 4, ENC_DIM, generator=torch.Generator().manual_seed(2))
    ys_pad = torch.tensor(
        [
            [ZH, IGNORE_ID, SPEECH, WOITN],
            [ZH, IGNORE_ID, SPEECH, WOITN],
            [ZH, NEUTRAL, SPEECH, WOITN],
            [ZH, HAPPY, SPEECH, WOITN],
        ],
        dtype=torch.int64,
    )
    assert int((ys_pad != IGNORE_ID).sum()) == 14

    loss_rich, _, _ = model._calc_rich_ce_loss(encoder_out, ys_pad)
    kept_out, kept_ys = compact(model, encoder_out, ys_pad)
    expected = model.criterion_att(kept_out, kept_ys)

    assert torch.allclose(loss_rich, expected, rtol=0, atol=1e-6)


def test_without_length_normalisation_the_denominator_is_the_row_count():
    """Gotcha worth pinning: length_normalized_loss=False divides by the batch size.

    The shipped config sets it to true.  If a future config turns it off, masking
    still removes the position from the numerator and from the gradient, but it no
    longer rescales the loss -- masked rows simply contribute less.
    """
    model = build_model(emo_mask_token_id=SER, length_normalized_loss=False)
    encoder_out = torch.randn(4, 4, ENC_DIM, generator=torch.Generator().manual_seed(2))
    ys_pad = torch.tensor([[ZH, IGNORE_ID, SPEECH, WOITN]] * 4, dtype=torch.int64)

    loss_rich, _, _ = model._calc_rich_ce_loss(encoder_out, ys_pad)
    kept_out, kept_ys = compact(model, encoder_out, ys_pad)
    # kept_* is a single row, so its denominator is 1 rather than 4.
    expected = model.criterion_att(kept_out, kept_ys) / 4

    assert torch.allclose(loss_rich, expected, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. rich_loss_weight
# ---------------------------------------------------------------------------


def test_rich_loss_weight_scales_only_the_total_loss():
    model = build_model(rich_loss_weight=0.5)
    batch = make_batch(LABELLED)

    _, stats, _ = run_forward(model, batch)

    assert torch.equal(stats["loss"], stats["loss_ctc"] + 0.5 * stats["loss_rich"])


def test_stats_loss_rich_stays_unweighted():
    weighted = build_model(rich_loss_weight=0.5)
    unweighted = build_model(rich_loss_weight=1.0)
    batch = make_batch(LABELLED)

    _, weighted_stats, _ = run_forward(weighted, batch)
    _, plain_stats, _ = run_forward(unweighted, batch)

    assert torch.equal(weighted_stats["loss_rich"], plain_stats["loss_rich"])
    assert torch.equal(weighted_stats["loss_ctc"], plain_stats["loss_ctc"])
    assert not torch.equal(weighted_stats["loss"], plain_stats["loss"])


def test_weight_of_one_is_bit_identical_to_the_pre_change_total():
    model = build_model(rich_loss_weight=1.0)
    batch = make_batch([NEUTRAL, HAPPY])

    _, stats, _ = run_forward(model, batch)
    ref_loss, _, _, _ = reference_forward(model, batch)

    assert torch.equal(stats["loss"], ref_loss.detach()[None])


# ---------------------------------------------------------------------------
# 5. acc_emo semantics
# ---------------------------------------------------------------------------


def test_acc_emo_scores_the_emotion_slot_alone():
    model = build_model()
    encoder_out = torch.randn(4, 4, ENC_DIM, generator=torch.Generator().manual_seed(3))
    predicted = model.ctc.ctc_lo(encoder_out).argmax(-1)
    # Emotion slot always right, every other slot deliberately wrong.
    ys_pad = torch.full((4, 4), 0, dtype=torch.int64)
    ys_pad[:, 1] = predicted[:, 1]
    for slot in (0, 2, 3):
        ys_pad[:, slot] = (predicted[:, slot] + 1) % VOCAB_SIZE

    _, acc_rich, acc_emo = model._calc_rich_ce_loss(encoder_out, ys_pad)

    assert acc_emo == 1.0
    assert acc_rich == 0.25
    assert acc_emo != acc_rich


def test_acc_emo_is_none_when_every_emotion_slot_is_masked():
    """``None``, never NaN.

    Both trainers strip None-valued stats before logging
    (``stats = {k: v for k, v in stats.items() if v is not None}`` at
    funasr/train_utils/trainer.py:427 and :604, trainer_ds.py:732), so the metric is
    simply absent for that step.  A NaN would instead be forwarded verbatim -- into
    the step log line (trainer.py:741) and as a NaN point in the TensorBoard series
    (trainer.py:757) -- and acc_emo is the only in-training signal that the emotion
    head is learning, so it must not be corruptible by an unlucky batch.
    """
    model = build_model(emo_mask_token_id=SER)
    batch = make_batch([SER, SER, SER])

    _, stats, _ = run_forward(model, batch)

    assert stats["acc_emo"] is None
    # acc_rich still covers the three unmasked slots, so it stays a real number.
    assert torch.isfinite(stats["acc_rich"]).all()


def test_force_gatherable_passes_none_through_untouched():
    """funasr/train_utils/device_funcs.py:60-61 returns None as-is.

    Pinned because the ``is None`` branch sits *after* the float branch: a NaN float
    would be converted to ``tensor([nan])`` and survive into the logs instead.
    """
    assert force_gatherable({"acc_emo": None}, "cpu") == {"acc_emo": None}
    assert torch.isnan(force_gatherable({"acc_emo": float("nan")}, "cpu")["acc_emo"]).all()


def test_the_trainers_drop_none_valued_stats_before_logging():
    """Pin the reporter contract this design depends on.

    If a funasr upgrade stops filtering None, ``trainer.py:741`` would raise on
    ``None.detach()`` instead of silently skipping the metric -- loud, but it would
    need re-validating here.
    """
    stats = {"loss_rich": torch.tensor([1.0]), "acc_emo": None}

    filtered = {k: v for k, v in stats.items() if v is not None}

    assert "acc_emo" not in filtered
    assert set(filtered) == {"loss_rich"}
    trainer_source = (
        Path(funasr.__file__).parent / "train_utils" / "trainer.py"
    ).read_text(encoding="utf-8")
    assert "stats = {k: v for k, v in stats.items() if v is not None}" in trainer_source


def test_an_all_masked_batch_does_not_affect_a_later_batch():
    """A None step must not poison the next step's acc_emo (the NaN failure mode)."""
    model = build_model(emo_mask_token_id=SER)
    normal = make_batch([NEUTRAL, HAPPY, SER], seed=7)

    _, before, _ = run_forward(model, normal)
    _, masked, _ = run_forward(model, make_batch([SER, SER, SER], seed=7))
    _, after, _ = run_forward(model, normal)

    assert masked["acc_emo"] is None
    assert after["acc_emo"] is not None
    assert torch.equal(after["acc_emo"], before["acc_emo"])
    assert torch.isfinite(after["acc_emo"]).all()


def test_all_masked_batch_still_produces_a_finite_loss():
    model = build_model(emo_mask_token_id=SER)
    batch = make_batch([SER, SER, SER])

    loss, stats, _ = run_forward(model, batch)

    assert torch.isfinite(loss).all()
    assert torch.isfinite(stats["loss_rich"]).all()


# ---------------------------------------------------------------------------
# 6. The sentinel is a single token
# ---------------------------------------------------------------------------


def test_ser_sentinel_is_a_single_token_with_id_24991():
    """If <|SER|> ever tokenised to more than one id, the slot-1 mask would be wrong."""
    if not BPE_MODEL.exists():
        pytest.skip(
            f"UNVERIFIED: the binding of token id {SER} to <|SER|> was NOT checked, because "
            f"{BPE_MODEL} is missing (checkpoint not downloaded). model.SER_TOKEN_ID and the "
            f"emotion-slot mask both hardcode {SER}; if the BPE model ever changes, only this "
            f"test would notice, and it did not run."
        )
    spm = pytest.importorskip("sentencepiece")

    processor = spm.SentencePieceProcessor(model_file=str(BPE_MODEL))

    assert processor.encode("<|SER|>") == [SER]
    assert processor.get_piece_size() == VOCAB_SIZE
    # The emotion tokens the sentinel stands in for are single tokens too.
    for token, expected in (("<|NEUTRAL|>", NEUTRAL), ("<|HAPPY|>", HAPPY)):
        assert processor.encode(token) == [expected]


# ---------------------------------------------------------------------------
# 7. Hydra-style string values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("none", None),
        ("None", None),
        ("null", None),
        (SER, SER),
        ("24991", SER),
        (" 24991 ", SER),
    ],
)
def test_optional_scalar_coerces_int_like_config_values(value, expected):
    assert _optional_scalar("emo_mask_token_id", value, int) == expected


@pytest.mark.parametrize("value,expected", [(None, 1.0), ("", 1.0), ("0.5", 0.5), (0.5, 0.5)])
def test_optional_scalar_coerces_float_like_config_values(value, expected):
    assert _optional_scalar("rich_loss_weight", value, float, 1.0) == expected


@pytest.mark.parametrize("value", [True, False])
def test_optional_scalar_rejects_booleans(value):
    """bool subclasses int, so True would otherwise become 1 -- a valid-looking token id."""
    with pytest.raises(ValueError) as excinfo:
        _optional_scalar("emo_mask_token_id", value, int)

    assert "emo_mask_token_id" in str(excinfo.value)


def test_optional_scalar_names_the_option_when_coercion_fails():
    with pytest.raises(ValueError) as excinfo:
        _optional_scalar("emo_mask_token_id", "twenty-four thousand", int)

    message = str(excinfo.value)
    assert "emo_mask_token_id" in message
    assert "twenty-four thousand" in message


def test_a_boolean_option_is_rejected_at_model_construction():
    with pytest.raises(ValueError) as excinfo:
        build_model(emo_mask_token_id=True)

    assert "emo_mask_token_id" in str(excinfo.value)


def test_string_options_behave_like_their_numeric_forms():
    """``++model_conf.emo_mask_token_id=24991`` arrives as a string via hydra."""
    from_strings = build_model(emo_mask_token_id="24991", rich_loss_weight="0.5")
    from_numbers = build_model(emo_mask_token_id=SER, rich_loss_weight=0.5)

    assert from_strings.emo_mask_token_id == SER
    assert isinstance(from_strings.emo_mask_token_id, int)
    assert from_strings.rich_loss_weight == 0.5
    assert isinstance(from_strings.rich_loss_weight, float)

    batch = make_batch([SER, NEUTRAL, HAPPY, SER])
    _, string_stats, _ = run_forward(from_strings, batch)
    _, number_stats, _ = run_forward(from_numbers, batch)

    for key in ("loss", "loss_ctc", "loss_rich", "acc_rich", "acc_emo"):
        assert torch.equal(string_stats[key], number_stats[key]), key
