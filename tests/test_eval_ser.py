"""Unit tests for the SER (speech emotion recognition) evaluation surface.

Two scripts are covered: the ``ser`` block ``scripts/eval_chunk_gap.py`` adds to
its per-epoch report, and ``scripts/eval_ser_jvnv.py``, the external JVNV
benchmark.  Nothing here loads a model, touches the network or decodes audio --
every test drives pure functions over literals and over empty files in
``tmp_path``, so the suite runs in well under a second.

Four groups of assertions here are *pinning* tests and must not be relaxed:

Emotion parsing reads the leading tag block only
    SenseVoice emits its metadata as ``<|ja|><|HAPPY|><|Speech|><|woitn|>``
    before the transcript.  An emotion token that appears *inside* the
    transcript is content, not a prediction.  A naive search over the whole
    string would silently score transcript text as a prediction, and the failure
    would be invisible in aggregate.

Macro-F1 averages reference-present classes, unpredicted ones included at 0.0
    This is the whole reason macro-F1 is here rather than accuracy alone.  The
    round-2 failure is a head that answers one class for everything; averaged
    over *predicted* classes such a model scores near 1.0, and averaged over
    reference classes it scores near 0.  ``test_degenerate_predictor_*`` pins
    the gap.

Degeneracy is reported, not silently scored
    The round-2 manifest carried ``emo_target = "<|NEUTRAL|>"`` on *every* clip.
    Scored naively, a collapsed model gets 100% accuracy on it -- the most
    misleading possible number.  ``summarise_ser`` must return
    ``status="degenerate"`` there, and ``status="na"`` when no clip carries an
    emotion label at all.

The CER path is untouched
    The round-2 CER numbers must stay reproducible byte-for-byte, so the CER
    helpers are pinned against hand-computed values.  If these fail, the SER
    work has changed something it had no business changing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ``eval_chunk_gap`` imports torch transitively; skip rather than fail on a
# torch-less checkout, matching tests/test_streaming_ctc_decode.py.
pytest.importorskip("torch")

HAPPY = "<|HAPPY|>"
SAD = "<|SAD|>"
ANGRY = "<|ANGRY|>"
NEUTRAL = "<|NEUTRAL|>"
MASK = "<|SER|>"


def _load(name: str):
    """Load a script in ``scripts/`` by path.

    ``scripts/`` is not a package, so there is nothing to import.  Same loader
    shape as ``tests/test_detect_label_noise.py``.  The module is registered in
    ``sys.modules`` under its own name so that ``eval_ser_jvnv``'s
    ``from eval_chunk_gap import ...`` binds to the very object these tests hold,
    rather than to a second copy.

    Args:
        name: Module name, matching the file stem.

    Returns:
        The executed module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gap():
    """``scripts/eval_chunk_gap.py``."""
    return _load("eval_chunk_gap")


@pytest.fixture(scope="module")
def jvnv(gap):
    """``scripts/eval_ser_jvnv.py``, sharing ``gap``'s module object."""
    return _load("eval_ser_jvnv")


def make_clip(gap, key: str, emo_target: Optional[str], reference: str = "テスト"):
    """Build an :class:`EvalClip` with no audio behind it.

    Args:
        gap: The ``eval_chunk_gap`` module.
        key: Clip key.
        emo_target: The manifest's emotion label, or ``None``.
        reference: Ground-truth transcript.

    Returns:
        The clip.  Its ``path`` is never opened by anything under test.
    """
    return gap.EvalClip(
        key=key,
        path=Path(f"/audio/{key}.wav"),
        reference=reference,
        language="ja",
        use_itn=False,
        emo_target=emo_target,
    )


def make_decodes(gap, by_model: Dict[str, Dict[str, str]]):
    """Wrap ``{model: {key: full_decode}}`` into ``ClipDecode`` objects.

    Args:
        gap: The ``eval_chunk_gap`` module.
        by_model: Raw full-attention decode text per model and clip.

    Returns:
        ``{model: {key: ClipDecode}}``.
    """
    return {
        name: {
            key: gap.ClipDecode(full=text, chunk=text, chunk_last_partial=text)
            for key, text in by_key.items()
        }
        for name, by_key in by_model.items()
    }


# ------------------------------------------------------- emotion tag parsing


def test_extracts_emotion_from_normal_tag_block(gap):
    text = "<|ja|><|HAPPY|><|Speech|><|woitn|>こんにちは"
    assert gap.extract_emotion_tag(text) == HAPPY


def test_no_tags_at_all_yields_no_prediction(gap):
    assert gap.extract_emotion_tag("こんにちは") is None
    assert gap.extract_emotion_tag("") is None


def test_tags_without_an_emotion_yield_no_prediction(gap):
    """A decode can carry language/event/ITN tags and no emotion at all."""
    assert gap.extract_emotion_tag("<|ja|><|Speech|><|woitn|>こんにちは") is None


def test_tag_order_is_not_assumed(gap):
    """The slot order is a model detail; parsing must not depend on it."""
    assert gap.extract_emotion_tag("<|Speech|><|woitn|><|SAD|><|ja|>本文") == SAD


def test_emotion_token_inside_the_transcript_is_not_a_prediction(gap):
    """The load-bearing one: only the leading block is metadata.

    A search over the whole string would return ``<|HAPPY|>`` here and score
    transcript content as the model's answer.
    """
    text = "<|ja|><|Speech|><|woitn|>彼は<|HAPPY|>と言った"
    assert gap.extract_emotion_tag(text) is None


def test_leading_block_wins_over_a_later_emotion_token(gap):
    text = "<|ja|><|SAD|><|Speech|><|woitn|>本文<|HAPPY|>続き"
    assert gap.extract_emotion_tag(text) == SAD


def test_leading_whitespace_is_tolerated(gap):
    assert gap.extract_emotion_tag("  <|ja|><|ANGRY|><|Speech|>本文") == ANGRY


def test_emo_unknown_is_a_prediction_not_a_blank(gap):
    """Banning it is a documented setting, so it has to be observable."""
    assert gap.extract_emotion_tag("<|ja|><|EMO_UNKNOWN|><|Speech|>本文") == (
        gap.EMO_UNKNOWN_TOKEN
    )


def test_leading_rich_tags_stops_at_the_transcript(gap):
    tags = gap.leading_rich_tags("<|ja|><|HAPPY|>本文<|Speech|>")
    assert tags == ["<|ja|>", "<|HAPPY|>"]


def test_leading_rich_tags_is_empty_when_text_starts_with_content(gap):
    assert gap.leading_rich_tags("本文<|ja|>") == []


# --------------------------------------------------------- metric arithmetic


def test_metrics_match_a_hand_computed_example(gap):
    """Six clips, three classes, one of which is never predicted.

    references  H H H S S A     supports 3 / 2 / 1
    predictions H H S S H H

    correct   = 3 of 6                              -> accuracy 0.5
    HAPPY     : predicted 4, TP 2 -> P 1/2, R 2/3   -> F1 4/7
    SAD       : predicted 2, TP 1 -> P 1/2, R 1/2   -> F1 1/2
    ANGRY     : predicted 0, TP 0 -> P 0,   R 0     -> F1 0
    macro-F1  = (4/7 + 1/2 + 0) / 3
    """
    references = [HAPPY, HAPPY, HAPPY, SAD, SAD, ANGRY]
    predictions = [HAPPY, HAPPY, SAD, SAD, HAPPY, HAPPY]
    metrics = gap.classification_metrics(references, predictions)

    assert metrics["num_scored"] == 6
    assert metrics["num_correct"] == 3
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((4 / 7 + 0.5 + 0.0) / 3)

    happy = metrics["per_class"][HAPPY]
    assert happy["support"] == 3
    assert happy["num_predicted"] == 4
    assert happy["precision"] == pytest.approx(0.5)
    assert happy["recall"] == pytest.approx(2 / 3)
    assert happy["f1"] == pytest.approx(4 / 7)

    sad = metrics["per_class"][SAD]
    assert (sad["precision"], sad["recall"], sad["f1"]) == pytest.approx((0.5, 0.5, 0.5))


def test_class_with_zero_predicted_support_scores_zero_and_enters_the_average(gap):
    """A class the model abandoned must be punished, not skipped."""
    references = [HAPPY, HAPPY, HAPPY, SAD, SAD, ANGRY]
    predictions = [HAPPY, HAPPY, SAD, SAD, HAPPY, HAPPY]
    metrics = gap.classification_metrics(references, predictions)

    angry = metrics["per_class"][ANGRY]
    assert angry["support"] == 1
    assert angry["num_predicted"] == 0
    assert angry["precision"] == 0.0
    assert angry["f1"] == 0.0
    # Three classes in the denominator, not the two that were predicted.
    assert metrics["macro_f1_classes"] == sorted([ANGRY, HAPPY, SAD])
    assert metrics["macro_f1"] == pytest.approx((4 / 7 + 0.5 + 0.0) / 3)


def test_confusion_matrix_shape_and_counts(gap):
    references = [HAPPY, HAPPY, HAPPY, SAD, SAD, ANGRY]
    predictions = [HAPPY, HAPPY, SAD, SAD, HAPPY, HAPPY]
    confusion = gap.classification_metrics(references, predictions)["confusion"]

    # One row per reference class, every row carrying the same columns.
    assert sorted(confusion) == sorted([ANGRY, HAPPY, SAD])
    columns = [sorted(row) for row in confusion.values()]
    assert columns == [sorted([ANGRY, HAPPY, SAD])] * 3

    assert confusion[HAPPY] == {ANGRY: 0, HAPPY: 2, SAD: 1}
    assert confusion[SAD] == {ANGRY: 0, HAPPY: 1, SAD: 1}
    assert confusion[ANGRY] == {ANGRY: 0, HAPPY: 1, SAD: 0}


def test_macro_f1_covers_reference_classes_only(gap):
    """A class the model invents is not added to the macro denominator.

    It still has to be visible, so it appears as a confusion column and in the
    prediction distribution.
    """
    references = [HAPPY, HAPPY, SAD, SAD]
    predictions = [HAPPY, ANGRY, SAD, ANGRY]
    metrics = gap.classification_metrics(references, predictions)

    assert metrics["macro_f1_classes"] == sorted([HAPPY, SAD])
    assert ANGRY not in metrics["per_class"]
    assert metrics["prediction_distribution"][ANGRY] == 2
    assert ANGRY in metrics["confusion"][HAPPY]


def test_degenerate_predictor_scores_far_below_a_balanced_one(gap):
    """The point of macro-F1: accuracy alone hides a collapsed head."""
    references = [HAPPY, HAPPY, SAD, SAD, ANGRY, ANGRY]

    collapsed = gap.classification_metrics(references, [HAPPY] * 6)
    balanced = gap.classification_metrics(references, list(references))

    assert balanced["macro_f1"] == pytest.approx(1.0)
    # Two of three classes score 0, the third 0.5 -> 1/6.
    assert collapsed["macro_f1"] == pytest.approx(0.5 / 3)
    assert collapsed["macro_f1"] < balanced["macro_f1"] / 3
    # And it is diagnosable as a collapse, not merely as a low score.
    assert collapsed["dominant_prediction"] == {
        "label": HAPPY,
        "count": 6,
        "share": pytest.approx(1.0),
    }


def test_unpredicted_clips_are_scored_as_misses_not_dropped(gap):
    """``None`` must lower the accuracy; dropping it would inflate it."""
    references = [HAPPY, SAD, ANGRY, HAPPY]
    metrics = gap.classification_metrics(references, [HAPPY, None, None, HAPPY])

    assert metrics["num_scored"] == 4
    assert metrics["num_pred_none"] == 2
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["prediction_distribution"][gap.NO_PREDICTION] == 2
    assert metrics["confusion"][SAD][gap.NO_PREDICTION] == 1


def test_empty_population_reports_none_rather_than_zero(gap):
    metrics = gap.classification_metrics([], [])
    assert metrics["num_scored"] == 0
    assert metrics["accuracy"] is None
    assert metrics["macro_f1"] is None


def test_misaligned_inputs_raise(gap):
    with pytest.raises(ValueError):
        gap.classification_metrics([HAPPY, SAD], [HAPPY])


# ------------------------------------------------------ population accounting


def test_masked_missing_and_unparseable_clips_are_excluded_and_counted(gap):
    clips = [
        make_clip(gap, "a", HAPPY),
        make_clip(gap, "b", SAD),
        make_clip(gap, "c", MASK),
        make_clip(gap, "d", None),
        make_clip(gap, "e", "<|BOGUS|>"),
        make_clip(gap, "f", ""),
    ]
    decodes = make_decodes(
        gap,
        {
            "base": {
                "a": f"<|ja|>{HAPPY}<|Speech|><|woitn|>あ",
                "b": f"<|ja|>{HAPPY}<|Speech|><|woitn|>い",
                "c": f"<|ja|>{SAD}<|Speech|><|woitn|>う",
                "d": f"<|ja|>{SAD}<|Speech|><|woitn|>え",
                "e": f"<|ja|>{SAD}<|Speech|><|woitn|>お",
                "f": f"<|ja|>{SAD}<|Speech|><|woitn|>か",
            }
        },
    )
    block = gap.summarise_ser(clips, decodes, ban_emo_unk=False)

    assert block["status"] == "ok"
    assert block["population"] == {
        "num_val_clips": 6,
        "num_scored": 2,
        "num_excluded_mask": 1,
        # The empty string counts as missing, not as junk.
        "num_excluded_missing": 2,
        "num_excluded_unparseable": 1,
    }
    # num_scored is stated by the metric block itself, not only by the header.
    assert block["per_model"]["base"]["num_scored"] == 2
    # Scored over exactly the two labelled clips: 'a' right, 'b' wrong.
    assert block["per_model"]["base"]["accuracy"] == pytest.approx(0.5)


def test_ban_emo_unk_setting_is_recorded(gap):
    clips = [make_clip(gap, "a", HAPPY), make_clip(gap, "b", SAD)]
    decodes = make_decodes(gap, {"base": {"a": HAPPY, "b": SAD}})

    assert gap.summarise_ser(clips, decodes, ban_emo_unk=True)["ban_emo_unk"] is True
    assert gap.summarise_ser(clips, decodes, ban_emo_unk=False)["ban_emo_unk"] is False


def test_status_na_when_no_clip_carries_emo_target(gap):
    clips = [make_clip(gap, "a", None), make_clip(gap, "b", None)]
    decodes = make_decodes(gap, {"base": {"a": HAPPY, "b": SAD}})
    block = gap.summarise_ser(clips, decodes, ban_emo_unk=False)

    assert block["status"] == "na"
    assert block["reason"] == "manifest has no emo_target field"
    assert block["population"]["num_scored"] == 0
    # No numbers at all: there are none to report.
    assert "per_model" not in block


def test_status_na_when_every_label_is_masked(gap):
    clips = [make_clip(gap, "a", MASK), make_clip(gap, "b", MASK)]
    decodes = make_decodes(gap, {"base": {"a": HAPPY, "b": SAD}})
    block = gap.summarise_ser(clips, decodes, ban_emo_unk=False)

    assert block["status"] == "na"
    assert "masked" in block["reason"]
    assert block["population"]["num_excluded_mask"] == 2


def test_round2_all_neutral_manifest_is_degenerate_not_perfect(gap):
    """The exact round-2 shape: constant ``<|NEUTRAL|>`` reference.

    A collapsed model scores 1.0 here.  The number is still reported -- but
    under a status that stops it being read as quality, which is the entire
    point of this test.
    """
    clips = [make_clip(gap, key, NEUTRAL) for key in ("a", "b", "c")]
    decodes = make_decodes(
        gap,
        {
            "finetuned": {
                key: f"<|ja|>{NEUTRAL}<|Speech|><|woitn|>本文" for key in ("a", "b", "c")
            }
        },
    )
    block = gap.summarise_ser(clips, decodes, ban_emo_unk=False)

    assert block["status"] == "degenerate"
    assert NEUTRAL in block["reason"]
    assert block["reference_classes"] == [NEUTRAL]
    assert block["population"]["num_scored"] == 3
    # Reported, and flatly misleading without the status beside it.
    assert block["per_model"]["finetuned"]["accuracy"] == pytest.approx(1.0)


def test_ser_block_reports_base_versus_finetuned_delta(gap):
    clips = [make_clip(gap, "a", HAPPY), make_clip(gap, "b", SAD)]
    decodes = make_decodes(
        gap,
        {
            "base": {"a": HAPPY, "b": SAD},
            "finetuned": {"a": NEUTRAL, "b": NEUTRAL},
        },
    )
    block = gap.summarise_ser(clips, decodes, ban_emo_unk=False)

    assert block["per_model"]["base"]["accuracy"] == pytest.approx(1.0)
    assert block["per_model"]["finetuned"]["accuracy"] == pytest.approx(0.0)
    assert block["delta"]["accuracy_finetuned_minus_base"] == pytest.approx(-1.0)


def test_prediction_comes_from_the_full_attention_decode(gap):
    """The chunk decode must not be consulted, even when it disagrees."""
    clips = [make_clip(gap, "a", HAPPY)]
    decodes = {
        "base": {
            "a": gap.ClipDecode(
                full=f"<|ja|>{HAPPY}<|Speech|><|woitn|>本文",
                chunk=f"<|ja|>{SAD}<|Speech|><|woitn|>本文",
                chunk_last_partial="本文",
            )
        }
    }
    block = gap.summarise_ser(clips, decodes, ban_emo_unk=False)
    assert block["per_model"]["base"]["accuracy"] == pytest.approx(1.0)


# ----------------------------------------------------- manifest / CER safety


def test_load_val_clips_reads_emo_target(gap, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in (
                {"key": "a", "source": str(audio), "target": "あ", "emo_target": HAPPY},
                {"key": "b", "source": str(audio), "target": "い", "emo_target": MASK},
                {"key": "c", "source": str(audio), "target": "う"},
            )
        ),
        encoding="utf-8",
    )

    clips, notes = gap.load_val_clips(manifest, None, "ja", False)
    assert notes == []
    assert [clip.emo_target for clip in clips] == [HAPPY, MASK, None]


def test_emo_target_does_not_affect_the_transcript_fields(gap, tmp_path):
    """Adding the field to a manifest must change nothing the CER path reads."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"")
    base_record = {
        "key": "a",
        "source": str(audio),
        "target": "こんにちは、世界。",
        "text_language": "<|ja|>",
        "with_or_wo_itn": "<|woitn|>",
    }
    without = tmp_path / "without.jsonl"
    without.write_text(json.dumps(base_record, ensure_ascii=False) + "\n", encoding="utf-8")
    with_emo = tmp_path / "with.jsonl"
    with_emo.write_text(
        json.dumps({**base_record, "emo_target": HAPPY}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    plain_clips, plain_notes = gap.load_val_clips(without, None, "en", True)
    tagged_clips, tagged_notes = gap.load_val_clips(with_emo, None, "en", True)
    assert plain_notes == tagged_notes == []
    (plain,), (tagged,) = plain_clips, tagged_clips

    assert (plain.key, plain.reference, plain.language, plain.use_itn) == (
        tagged.key,
        tagged.reference,
        tagged.language,
        tagged.use_itn,
    )
    assert plain.emo_target is None and tagged.emo_target == HAPPY


def test_cer_helpers_are_untouched(gap):
    """Hand-computed CER/WER over a fixed pair set.

    The round-2 numbers have to stay reproducible byte-for-byte, so this pins
    the CER path directly rather than through a model run.

    Pair 1: reference ``こんにちは、世界。`` normalises to ``こんにちは世界``
    (7 chars, punctuation stripped) and the hypothesis matches -> 0 edits.
    Pair 2: ``ありがとう`` (5 chars) against ``あリがとう``, one katakana
    substitution NFKC does *not* fold -> 1 edit.
    """
    pairs = [("こんにちは、世界。", "こんにちは世界"), ("ありがとう", "あリがとう")]

    assert gap.normalize_chars("こんにちは、世界。") == "こんにちは世界"
    assert gap.normalize_words("こんにちは、世界。") == ["こんにちは", "世界"]
    assert gap.pair_cer(*pairs[0]) == pytest.approx(0.0)
    assert gap.pair_cer(*pairs[1]) == pytest.approx(0.2)

    metrics = gap.corpus_metrics(pairs)
    assert metrics["num_clips"] == 2
    assert metrics["ref_chars"] == 12
    assert metrics["char_edits"] == 1
    assert metrics["cer"] == pytest.approx(1 / 12)
    assert metrics["mean_cer"] == pytest.approx(0.1)
    assert metrics["ref_words"] == 3
    assert metrics["word_edits"] == 3
    assert metrics["wer"] == pytest.approx(1.0)


def test_rich_tags_including_emotion_are_still_stripped_before_cer(gap):
    """The SER work must not leak tags into the transcript comparison."""
    hypothesis = f"<|ja|>{HAPPY}<|Speech|><|woitn|>こんにちは世界"
    assert gap.pair_cer("こんにちは、世界。", hypothesis) == pytest.approx(0.0)


def test_summarise_val_keeps_cer_metrics_and_adds_ser_alongside(gap):
    clips = [
        make_clip(gap, "a", HAPPY, reference="こんにちは"),
        make_clip(gap, "b", SAD, reference="さようなら"),
    ]
    decodes = make_decodes(
        gap,
        {
            "base": {
                "a": f"<|ja|>{HAPPY}<|Speech|><|woitn|>こんにちは",
                "b": f"<|ja|>{SAD}<|Speech|><|woitn|>さようなら",
            }
        },
    )
    result = gap.summarise_val(clips, decodes, keep_punctuation=False)

    expected = gap.corpus_metrics(
        [
            ("こんにちは", decodes["base"]["a"].full),
            ("さようなら", decodes["base"]["b"].full),
        ]
    )
    assert result["per_model"]["base"]["chunk"] == expected
    assert result["per_model"]["base"]["full"] == expected
    assert result["per_model"]["base"]["chunk_minus_full_cer"] == pytest.approx(0.0)
    # The new block is a sibling, never a component.
    assert result["ser"]["status"] == "ok"
    assert set(result) == {"per_model", "ser"}


# ------------------------------------------------------------ JVNV path parsing


@pytest.mark.parametrize(
    ("emotion", "token"),
    [
        ("anger", "<|ANGRY|>"),
        ("disgust", "<|DISGUSTED|>"),
        ("fear", "<|FEARFUL|>"),
        ("happy", "<|HAPPY|>"),
        ("sad", "<|SAD|>"),
        ("surprise", "<|SURPRISED|>"),
    ],
)
def test_jvnv_layout_maps_every_emotion(jvnv, emotion, token):
    """The real layout: ``<root>/<speaker>/<emotion>/<speaker>_<emotion>_<n>.wav``."""
    root = Path("/data/jvnv_corpus_v1_no_nv")
    path = root / "F1" / emotion / f"F1_{emotion}_1.wav"
    assert jvnv.emotion_from_path(path, root) == token


def test_jvnv_flat_layout_reads_the_filename(jvnv):
    root = Path("/data/jvnv")
    assert jvnv.emotion_from_path(root / "F1_sad_01.wav", root) == SAD


def test_jvnv_ignores_components_above_the_corpus_root(jvnv):
    """A corpus staged under a path containing an emotion word must not poison it."""
    root = Path("/mnt/sad-machine/jvnv")
    path = root / "F1" / "happy" / "F1_happy_3.wav"
    assert jvnv.emotion_from_path(path, root) == HAPPY


def test_jvnv_speaker_is_read_from_the_path(jvnv):
    root = Path("/data/jvnv")
    assert jvnv.speaker_from_path(root / "M2" / "fear" / "M2_fear_1.wav", root) == "M2"
    assert jvnv.speaker_from_path(root / "fear" / "x.wav", root) is None


def test_jvnv_unparseable_path_fails_loudly(jvnv):
    """Skipping it would shrink the population behind the accuracy in silence."""
    root = Path("/data/jvnv")
    with pytest.raises(ValueError, match="cannot determine the emotion"):
        jvnv.emotion_from_path(root / "F1" / "mystery" / "F1_x_1.wav", root)


def test_jvnv_ambiguous_path_fails_loudly(jvnv):
    root = Path("/data/jvnv")
    with pytest.raises(ValueError, match="ambiguous"):
        jvnv.emotion_from_path(root / "F1" / "anger" / "F1_happy_1.wav", root)


def test_jvnv_has_no_neutral_class(jvnv):
    """``<|NEUTRAL|>`` is always wrong here, so it must not be mappable."""
    assert "neutral" not in jvnv.JVNV_EMOTION_TO_TOKEN
    assert NEUTRAL not in jvnv.JVNV_EMOTION_TO_TOKEN.values()


# --------------------------------------------------------------- JVNV corpus


def build_corpus(root: Path, speakers=("F1", "M1"), emotions=("anger", "happy", "sad")):
    """Create an empty-file JVNV tree.

    Args:
        root: Directory to create.
        speakers: Speaker directories.
        emotions: Emotion directories under each speaker.

    Returns:
        The root.
    """
    for speaker in speakers:
        for emotion in emotions:
            directory = root / speaker / emotion
            directory.mkdir(parents=True, exist_ok=True)
            for index in (1, 2):
                (directory / f"{speaker}_{emotion}_{index}.wav").write_bytes(b"")
    return root


def test_discover_clips_walks_the_tree(jvnv, tmp_path):
    clips = jvnv.discover_clips(build_corpus(tmp_path / "jvnv"))

    assert len(clips) == 12
    assert clips[0].key == "F1/anger/F1_anger_1.wav"
    assert {clip.speaker for clip in clips} == {"F1", "M1"}
    assert {clip.emotion for clip in clips} == {ANGRY, HAPPY, SAD}


def test_discover_clips_filters_speakers(jvnv, tmp_path):
    clips = jvnv.discover_clips(build_corpus(tmp_path / "jvnv"), speakers=["m1"])
    assert {clip.speaker for clip in clips} == {"M1"}


def test_discover_clips_limit_keeps_every_class(jvnv, tmp_path):
    """A path-sorted ``[:limit]`` would return one speaker's anger and nothing else."""
    clips = jvnv.discover_clips(build_corpus(tmp_path / "jvnv"), limit=6)

    assert len(clips) == 6
    assert {clip.emotion for clip in clips} == {ANGRY, HAPPY, SAD}
    assert {clip.speaker for clip in clips} == {"F1", "M1"}


def test_discover_clips_rejects_an_unlabelled_clip(jvnv, tmp_path):
    root = build_corpus(tmp_path / "jvnv")
    (root / "F1" / "mystery").mkdir()
    (root / "F1" / "mystery" / "F1_x_1.wav").write_bytes(b"")

    with pytest.raises(ValueError, match="cannot determine the emotion"):
        jvnv.discover_clips(root)


def test_discover_clips_requires_a_staged_corpus(jvnv, tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        jvnv.discover_clips(tmp_path / "absent")


def test_discover_clips_rejects_an_empty_tree(jvnv, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no audio"):
        jvnv.discover_clips(empty)


# ------------------------------------------------------------- JVNV model CLI


def test_checkpoint_spec_parses_label_and_path(jvnv):
    label, path = jvnv.parse_checkpoint_spec("round2-ep3=/outputs/round2_full/model.pt.ep3")
    assert label == "round2-ep3"
    assert path == Path("/outputs/round2_full/model.pt.ep3")


@pytest.mark.parametrize("value", ["/outputs/model.pt.ep3", "label=", "=path", ""])
def test_checkpoint_spec_rejects_anything_but_label_equals_path(jvnv, value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        jvnv.parse_checkpoint_spec(value)


def test_resolve_models_keys_by_label_and_orders_base_first(jvnv, tmp_path):
    import argparse

    model_dir = tmp_path / "SenseVoiceSmall"
    model_dir.mkdir()
    checkpoint = tmp_path / "model.pt.ep3"
    checkpoint.write_bytes(b"")

    specs = jvnv.resolve_models(
        argparse.Namespace(
            base=model_dir,
            model_dir=model_dir,
            checkpoint=[("round2-ep3", checkpoint)],
        )
    )
    assert [spec.label for spec in specs] == ["base", "round2-ep3"]
    assert specs[0].checkpoint is None
    assert specs[1].checkpoint == checkpoint


def test_resolve_models_rejects_duplicate_labels(jvnv, tmp_path):
    import argparse

    model_dir = tmp_path / "SenseVoiceSmall"
    model_dir.mkdir()
    checkpoint = tmp_path / "model.pt.ep3"
    checkpoint.write_bytes(b"")

    with pytest.raises(ValueError, match="duplicate"):
        jvnv.resolve_models(
            argparse.Namespace(
                base=None,
                model_dir=model_dir,
                checkpoint=[("r2", checkpoint), ("r2", checkpoint)],
            )
        )


def test_resolve_models_reports_every_missing_path_at_once(jvnv, tmp_path):
    import argparse

    model_dir = tmp_path / "SenseVoiceSmall"
    model_dir.mkdir()

    with pytest.raises(ValueError) as excinfo:
        jvnv.resolve_models(
            argparse.Namespace(
                base=None,
                model_dir=model_dir,
                checkpoint=[
                    ("r2", tmp_path / "absent2.pt"),
                    ("r3", tmp_path / "absent3.pt"),
                ],
            )
        )
    assert "absent2.pt" in str(excinfo.value) and "absent3.pt" in str(excinfo.value)


def test_resolve_models_requires_at_least_one_model(jvnv, tmp_path):
    import argparse

    with pytest.raises(ValueError, match="nothing to evaluate"):
        jvnv.resolve_models(
            argparse.Namespace(base=None, model_dir=tmp_path, checkpoint=[])
        )


def test_jvnv_and_gap_share_one_metric_implementation(jvnv, gap):
    """Not merely equal results: literally the same function object."""
    assert jvnv.classification_metrics is gap.classification_metrics
    assert jvnv.extract_emotion_tag is gap.extract_emotion_tag
