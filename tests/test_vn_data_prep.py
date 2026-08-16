"""Unit tests for ``scripts/prepare_vn_data.py`` (VisualNovel corpus prep).

The script builds the Japanese fine-tuning corpus from a 1.28 GB gated
HuggingFace dataset.  Nothing here touches that dataset: every test drives the
four pure helpers -- ``normalize_text``, ``compute_source_len``,
``build_record`` and ``split_by_speaker`` -- over literals and over tiny wavs
synthesised into ``tmp_path`` with the stdlib :mod:`wave` module.  The suite
therefore needs no network, no HF token, no ffmpeg and no ML dependency.

Three things here are *pinning* tests rather than free-choice assertions, and
must not be "fixed" by loosening them:

Punctuation convention
    The corpus must use the orthography the pretrained model already emits:
    full-width ``！？。、：；``, ``…`` intact, ``～`` intact.  An earlier revision
    applied blanket NFKC, which decomposed ``…`` into 120,921 ASCII dots across
    this corpus and produced 8,398 ASCII ``?``/7,095 ASCII ``!``/968 ``~`` with
    zero full-width forms -- training targets that contradicted the model's own
    tokenisation.  The punctuation section below locks each of those
    regressions down individually.

``source_len`` scale
    ``scripts/make_smoke_data.py`` and the shipped ``data/*.jsonl`` store 100
    frames per second (10 ms per frame, no LFR division).  The two generators
    write into the same trainer, so they have to agree.  The parametrised
    ground-truth test below reads the real durations out of ``data/smoke/*.wav``
    and checks ``compute_source_len`` reproduces the ``source_len`` already on
    disk.  If it fails, one of the two pipelines has drifted.

Speaker-disjointness
    The split holds out *whole speakers*.  Speaker leakage does not crash
    anything and does not show up in any metric -- it silently turns the
    validation loss into an optimistic lie, because the model is scored on
    voices it memorised during training.  The overlap assertions are the only
    thing standing between the corpus and that failure, so they are deliberately
    exhaustive.
"""

import ast
import importlib.util
import json
import os
import sys
import urllib.error
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "scripts" / "prepare_vn_data.py"


def _load_prepare_vn_data():
    """Import ``scripts/prepare_vn_data.py`` by path.

    ``scripts/`` is not a package, so there is nothing to ``import``.  The
    module must be registered in ``sys.modules`` *before* ``exec_module`` runs:
    it defines ``@dataclass`` classes, and :mod:`dataclasses` resolves field
    annotations via ``sys.modules[cls.__module__]``, which blows up with an
    ``AttributeError`` if the module is not yet registered.
    """
    spec = importlib.util.spec_from_file_location("prepare_vn_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vn = _load_prepare_vn_data() if SCRIPT_PATH.is_file() else None

# Keeps the file collectable while the script is still being written; every
# test is reported as skipped rather than erroring the whole run.
pytestmark = pytest.mark.skipif(vn is None, reason=f"{SCRIPT_PATH} does not exist yet")

SMOKE_TRAIN_JSONL = ROOT / "data" / "smoke_train.jsonl"
TRAIN_EXAMPLE_JSONL = ROOT / "data" / "train_example.jsonl"

# The schema of data/train_example.jsonl, in order.  json.dumps preserves dict
# insertion order, so the generated manifests are byte-comparable with the
# shipped examples field-for-field.
EXPECTED_FIELDS = [
    "key",
    "text_language",
    "emo_target",
    "event_target",
    "with_or_wo_itn",
    "target",
    "source",
    "target_len",
    "source_len",
]


# --- helpers ----------------------------------------------------------------


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wav_duration(path):
    """Duration in seconds, read from the wav header only."""
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def write_wav(path, seconds, sample_rate=16000):
    """Write ``seconds`` of PCM16 silence.  Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * sample_rate))
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def make_record(title, speaker, voice, text="こんにちは", duration_sec=2.0):
    """A record whose ``source`` follows the pipeline's audio/<title>/<speaker>/ layout.

    Production records carry no explicit ``speaker`` field -- ``build_record``
    does not emit one -- so the speaker is recovered from the path.  These
    fixtures reproduce that, exercising the same branch the real corpus takes.
    """
    return vn.build_record(
        f"{title}_{speaker}_{voice}",
        text,
        f"/vn/audio/{title}/{speaker}/{voice}.wav",
        duration_sec,
    )


def corpus(titles=3, speakers_per_title=4, clips_per_speaker=5):
    """A synthetic multi-title corpus whose speaker names are globally unique.

    Exercises the general splitting machinery.  It cannot on its own prove the
    leak guard works, because with unique names the bare-name and
    title-qualified groupings coincide -- see ``franchise_corpus``.
    """
    return [
        make_record(f"title{t}", f"spk{t}_{s}", f"v{c}")
        for t in range(titles)
        for s in range(speakers_per_title)
        for c in range(clips_per_speaker)
    ]


def franchise_corpus(clips_per_speaker=5):
    """A corpus shaped like the real one: the same voice recurs across titles.

    These archives are sequels and fandiscs, so a character name appearing in
    two titles is normally the same voice actor (observed: 七瀬 across Ai Kiss 2
    and its Extra, 東雲/栞 across Criminal Border 2nd and 3rd).  ``nanase`` and
    ``shinonome`` below are those recurring voices; the rest are title-local.

    Every leak assertion must be exercised against *this* shape.  On a corpus of
    globally unique names a title-qualified split key looks disjoint, which is
    exactly how the original leak passed review.
    """
    layout = {
        "titleA": ["nanase", "aoi", "kaede"],
        "titleB": ["nanase", "shinonome", "rin"],
        "titleC": ["shinonome", "yuki", "mei"],
    }
    return [
        make_record(title, speaker, f"v{c}")
        for title, speakers in layout.items()
        for speaker in speakers
        for c in range(clips_per_speaker)
    ]


def speakers_of(records):
    """The set of split keys present -- the *bare* speaker name.

    This MUST be ``speaker_group_key`` and not ``record_speaker``.  The
    title-qualified id is bookkeeping only; using it here makes every leak
    assertion in this file vacuous, because a split that separates
    ``titleA/nanase`` from ``titleB/nanase`` looks disjoint under it while the
    same voice sits on both sides.  ``test_the_leak_guard_detects_a_leaky_split``
    pins that distinction.
    """
    return {vn.speaker_group_key(record) for record in records}


def qualified_speakers_of(records):
    """Title-qualified ids -- bookkeeping only, never a leakage check."""
    return {vn.record_speaker(record) for record in records}


def titles_of(records):
    return {vn.record_title(record) for record in records}


def keys_of(records):
    return sorted(record["key"] for record in records)


def _group_sizes(records):
    """Clip count per split key (bare speaker name)."""
    sizes = {}
    for record in records:
        key = vn.speaker_group_key(record)
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def over_cap_corpus():
    """A corpus where *every* speaker exceeds ``VAL_MAX_CLIPS_PER_SPEAKER``.

    Exercises the surplus-drop path, which a fixture of 5-clip speakers never
    reaches.  Every speaker holds ``2 * cap`` clips -- above the cap, below the
    10x size limit -- so whichever speakers are chosen, half of each one's
    clips are surplus and must be discarded rather than handed back to train.
    Making every speaker over-cap keeps the drop deterministic across seeds; a
    fixture with cheap speakers available lets the splitter avoid the path
    entirely, which is correct behaviour but tests nothing here.
    """
    cap = vn.VAL_MAX_CLIPS_PER_SPEAKER
    return [
        make_record(title, f"spk_{title}_{index}", f"v{c}")
        for title in ("titleA", "titleB")
        for index in range(4)
        for c in range(cap * 2)
    ]


# ===========================================================================
# normalize_text
# ===========================================================================

# --- NFKC ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ＡＢＣ１２３", "ABC123"),  # full-width ASCII folded
        ("１２３ちゃん", "123ちゃん"),
        ("ｱｲｳ", "アイウ"),  # half-width katakana composed
        ("ｶﾞﾝﾊﾞﾚ", "ガンバレ"),  # ...including the combining dakuten
    ],
)
def test_nfkc_normalisation_is_applied(raw, expected):
    assert vn.normalize_text(raw) == expected


def test_nfkc_does_not_reach_japanese_punctuation():
    # Blanket NFKC folds U+FF01/U+FF1F to "!"/"?".  That is wrong for this
    # corpus: see the punctuation-convention section below for the decoded
    # evidence.  The script holds Japanese punctuation out of NFKC's reach and
    # applies the fold only to alphanumerics and spaces.
    assert vn.normalize_text("ラーメン！？") == "ラーメン！？"


def test_ideographic_full_stop_survives():
    assert vn.normalize_text("そうか。") == "そうか。"


# --- punctuation convention (regression-locked) ---
#
# The corpus must match the orthography the pretrained model already emits.
# Blanket NFKC contradicted it, measurably: across this corpus it produced
# 120,921 ASCII dots from "……", 8,398 ASCII "?" and 7,095 ASCII "!" with zero
# full-width forms, plus 968 "~" from "～".  Each assertion below pins one of
# those regressions.  Do not "simplify" them back into a plain NFKC call.


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ああ……", "ああ……"),  # the 120,921-dot regression
        ("ああ…", "ああ…"),
        ("うふふ…！", "うふふ…！"),
    ],
)
def test_ellipsis_is_preserved_and_never_decomposed(raw, expected):
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["ああ……", "ああ…", "うふふ…！", "ＡＢＣ…", "そう……ね"],
)
def test_normalising_japanese_never_introduces_an_ascii_dot(raw):
    # The single highest-volume defect: "…" is one token to the model, but
    # NFKC turns it into three ASCII dots, so every ellipsis became three
    # wrong tokens.
    assert "." not in vn.normalize_text(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ああ...", "ああ…"),  # a 3-dot run is one ellipsis
        ("ああ......", "ああ……"),  # a 6-dot run is two
    ],
)
def test_ascii_dot_runs_collapse_back_to_ellipsis(raw, expected):
    # Text that already went through a lossy NFKC pass is repaired rather than
    # left as ASCII.
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("あ！い", "あ！い"),
        ("あ？い", "あ？い"),
        ("あ！？い", "あ！？い"),
    ],
)
def test_full_width_exclamation_and_question_survive(raw, expected):
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("あ!い", "あ！い"),
        ("あ?い", "あ？い"),
        ("あ!?い", "あ！？い"),
    ],
)
def test_ascii_exclamation_and_question_convert_to_full_width(raw, expected):
    # Convergent, not merely preserving: whichever width the VN engine emitted,
    # the corpus ends up on the model's convention.
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("そう～", "そう～"),  # U+FF5E preserved, not folded to "~"
        ("そう〜", "そう〜"),  # U+301C preserved
        ("あ~い", "あ～い"),  # ASCII tilde converted up
    ],
)
def test_wave_dash_is_preserved_full_width(raw, expected):
    # "～" marks vowel lengthening in VN dialogue, so it is pronounced content
    # rather than decoration.
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize("mark", ["！", "？", "。", "、", "：", "；", "…", "‥", "～", "〜"])
def test_every_protected_mark_survives_in_its_full_width_form(mark):
    result = vn.normalize_text(f"あい{mark}うえ")

    assert result == f"あい{mark}うえ"


@pytest.mark.parametrize("ascii_char", ["!", "?", "~"])
def test_no_bare_ascii_punctuation_survives_in_japanese_text(ascii_char):
    # Note "." is deliberately absent: see the carve-out below.
    result = vn.normalize_text(f"あい{ascii_char}うえ")

    assert ascii_char not in result


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.5秒", "1.5秒"),
        ("Ver1.5です", "Ver1.5です"),
        ("あい.うえ", "あい.うえ"),
    ],
)
def test_a_lone_ascii_dot_is_left_alone(raw, expected):
    # Only *runs* of dots are folded ellipses.  A single dot is a decimal point
    # or an abbreviation, so collapsing it would corrupt "1.5秒" into "1…5秒".
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize("raw,expected", [("ああ..", "ああ…"), ("ああ...", "ああ…")])
def test_a_run_of_two_or_more_dots_is_a_folded_ellipsis(raw, expected):
    assert vn.normalize_text(raw) == expected


def test_the_useful_half_of_nfkc_is_still_applied():
    # Protecting punctuation must not cost the alphanumeric folding: that part
    # of NFKC is genuinely wanted.
    assert vn.normalize_text("ＡＢＣ１２３！？…") == "ABC123！？…"


# --- engine escape artifacts ---


def test_engine_backslash_quote_artifact_is_stripped():
    # Observed verbatim in the corpus: the VN engine leaks its own escape
    # sequence into the transcript text.  The escape goes; the full-width
    # punctuation stays.
    assert vn.normalize_text(r'に\"ぇっ！？') == "にぇっ！？"


def test_backslash_is_deleted_anywhere_in_the_line():
    assert vn.normalize_text(r"あ\い") == "あい"


def test_bare_double_quotes_are_deleted():
    assert vn.normalize_text('"あ"') == "あ"


def test_a_line_of_only_escape_residue_becomes_empty():
    assert vn.normalize_text(r"\"") == ""


# --- wrapping quotes and brackets ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("「こんにちは」", "こんにちは"),
        ("『やあ』", "やあ"),
        ("（ふふ）", "ふふ"),
        ("「ラーメン！？」", "ラーメン！？"),
        ("「ああ……」", "ああ……"),
    ],
)
def test_whole_line_wrapping_pairs_are_removed(raw, expected):
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("あ「い」う", "あ「い」う"),
        ("あ『い』う", "あ『い』う"),
        ("Ａ「い」Ｂ", "A「い」B"),
    ],
)
def test_the_same_pairs_survive_mid_sentence(raw, expected):
    # Only a pair wrapping the *whole* line is engine chrome; mid-sentence the
    # very same characters are quoted speech inside the line and are content.
    assert vn.normalize_text(raw) == expected


def test_two_quoted_spans_are_not_mangled():
    # The naive "starts with 「 and ends with 」" rule would eat the outer
    # characters and produce A」と「B.  The closer must be the line's *first*
    # occurrence for the pair to count as wrapping.
    assert vn.normalize_text("「A」と「B」") == "「A」と「B」"


def test_an_empty_quoted_line_becomes_empty():
    assert vn.normalize_text("「」") == ""


# --- whitespace ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ああ　　ん", "ああ ん"),  # ideographic spaces, NFKC'd then collapsed
        ("あ\tい", "あ い"),
        ("「ああ」\n「いい」", "「ああ」 「いい」"),
        ("あ    い", "あ い"),
    ],
)
def test_repeated_whitespace_is_collapsed_to_one_space(raw, expected):
    assert vn.normalize_text(raw) == expected


def test_leading_and_trailing_whitespace_is_stripped():
    assert vn.normalize_text("　あ　") == "あ"


# --- preservation of load-bearing Japanese characters ---


@pytest.mark.parametrize(
    "text",
    [
        "ラーメン",  # long-vowel mark
        "そーゆー",  # long-vowel mark after hiragana
        "ヴァーチャル",
        "こんにちは",  # hiragana
        "アイウエオ",  # katakana
        "日本語",  # kanji
        "ーーー",  # a drawn-out vocalisation is a legitimate transcript
    ],
)
def test_load_bearing_characters_pass_through_untouched(text):
    # These are pronounced content: dropping them would desynchronise the
    # transcript from the audio and teach the model to omit them.
    assert vn.normalize_text(text) == text


@pytest.mark.parametrize("mark", ["ー", "！", "？", "。", "、"])
def test_sentence_marks_are_never_deleted_from_a_line(mark):
    result = vn.normalize_text(f"あい{mark}うえ")

    # NFKC may fold the *form* (！->!), but the mark must still be present.
    assert len(result) == len("あいうえ") + 1
    assert result.startswith("あい")
    assert result.endswith("うえ")


# --- idempotence ---


@pytest.mark.parametrize(
    "raw",
    [
        r'に\"ぇっ！？',
        "「こんにちは」",
        "『やあ』",
        "（ふふ）",
        "「A」と「B」",
        "ＡＢＣ１２３",
        "ｶﾞﾝﾊﾞﾚ",
        "ああ　　ん",
        "ラーメン！？",
        "そうか。",
        "うふふ…！",
        # The convergent rewrites are the real idempotence risk: a second pass
        # must not re-collapse an ellipsis or re-widen an already-full-width
        # mark into something else.
        "ああ……",
        "ああ...",
        "ああ......",
        "そう～",
        "あ~い",
        "あ!?い",
        "ＡＢＣ…",
        "",
        "   ",
        "！？。",
        "あ「い」う",
    ],
)
def test_normalisation_is_idempotent(raw):
    # The pipeline may re-normalise text that was already written out once, so
    # a second pass must be a no-op rather than eating another bracket layer.
    once = vn.normalize_text(raw)

    assert vn.normalize_text(once) == once


# --- degenerate input ---


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " ",
        "   ",
        "　",
        "\n\t ",
        "！？。",  # punctuation only
        "、、、",
        "…",
        "-----",
        "!!!",
        "\\",
        '""',
    ],
)
def test_contentless_input_normalises_to_empty_string(raw):
    # Empty is the caller's documented signal to drop the clip: CTC cannot
    # train on a zero-length target and the trainer crashes rather than
    # skipping such a sample.
    assert vn.normalize_text(raw) == ""


def test_normalize_text_always_returns_a_string():
    for raw in ["", "   ", "あ", "「」", r"\"", "！？"]:
        assert isinstance(vn.normalize_text(raw), str)


# ===========================================================================
# compute_source_len
# ===========================================================================


def test_smoke_dataset_is_present_for_the_ground_truth_tests():
    # Guards the parametrisation below: if data/ were regenerated away, the
    # cross-pipeline check would silently collapse to zero cases.
    assert SMOKE_TRAIN_JSONL.is_file()
    assert read_jsonl(SMOKE_TRAIN_JSONL)


@pytest.mark.skipif(not SMOKE_TRAIN_JSONL.is_file(), reason="data/smoke_train.jsonl missing")
@pytest.mark.parametrize("record", read_jsonl(SMOKE_TRAIN_JSONL) if SMOKE_TRAIN_JSONL.is_file() else [])
def test_matches_the_source_len_already_written_by_make_smoke_data(record):
    """Cross-pipeline ground truth: both generators feed the same trainer.

    ``make_smoke_data.py`` computes ``int(samples / rate * 1000 / 10)``.  This
    reads the wav it actually produced and checks ``compute_source_len`` agrees
    with the number sitting in the manifest.
    """
    wav = Path(record["source"])
    if not wav.is_file():
        pytest.skip(f"generated smoke wav missing: {wav}")

    assert vn.compute_source_len(wav_duration(wav)) == record["source_len"]


def test_the_documented_scale_is_one_hundred_frames_per_second():
    # A 1.0 s smoke clip has source_len 100 on disk.  This is the whole
    # convention in one assertion.
    assert vn.compute_source_len(1.0) == 100


@pytest.mark.parametrize(
    "duration,expected",
    [
        (0.5, 50),  # MIN_SECONDS
        (1.0, 100),
        (1.273, 127),
        (2.0, 200),
        (4.0, 400),
        (20.0, 2000),  # MAX_SECONDS
    ],
)
def test_frame_counts_at_the_documented_durations(duration, expected):
    assert vn.compute_source_len(duration) == expected


def test_min_and_max_seconds_constants_map_to_sane_frame_counts():
    assert vn.compute_source_len(vn.MIN_SECONDS) == 50
    assert vn.compute_source_len(vn.MAX_SECONDS) == 2000


def test_frame_ms_constant_agrees_with_make_smoke_data():
    assert vn.FRAME_MS == 10


def test_result_is_a_plain_int():
    result = vn.compute_source_len(1.5)

    assert isinstance(result, int)
    assert not isinstance(result, bool)


@pytest.mark.parametrize("duration", [0.0, -0.001, -1.0, -100.0])
def test_non_positive_durations_give_zero_rather_than_raising(duration):
    assert vn.compute_source_len(duration) == 0


def test_monotonic_non_decreasing_in_duration():
    durations = [0.0, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0]

    lengths = [vn.compute_source_len(d) for d in durations]

    assert lengths == sorted(lengths)


def test_strictly_increasing_across_the_working_range():
    lengths = [vn.compute_source_len(d / 10) for d in range(5, 201)]

    assert all(b > a for a, b in zip(lengths, lengths[1:]))


def test_sub_frame_durations_truncate_toward_zero():
    # int() truncation, not rounding: a partial frame is not a frame.
    assert vn.compute_source_len(0.019) == 1
    assert vn.compute_source_len(0.009) == 0


def test_lfr_n_divides_the_frame_count_when_requested():
    # Exposed for callers wanting post-LFR counts; must not be the default,
    # because the on-disk manifests are calibrated at the pre-LFR scale.
    assert vn.compute_source_len(1.0, lfr_n=6) == 100 // 6
    assert vn.compute_source_len(1.0, lfr_n=1) == 100


# ===========================================================================
# build_record
# ===========================================================================


@pytest.fixture
def sample_wav(tmp_path):
    return write_wav(tmp_path / "audio" / "title" / "spk" / "v0.wav", 2.0)


def test_emits_exactly_the_expected_fields(sample_wav):
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert set(record) == set(EXPECTED_FIELDS)


def test_field_order_matches_the_shipped_example(sample_wav):
    # json.dumps follows insertion order, so this keeps generated manifests
    # field-for-field comparable with data/train_example.jsonl.
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert list(record) == EXPECTED_FIELDS


def test_schema_matches_data_train_example_jsonl(sample_wav):
    shipped = read_jsonl(TRAIN_EXAMPLE_JSONL)[0]

    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert list(record) == list(shipped)


@pytest.mark.parametrize(
    "field,expected",
    [
        ("text_language", "<|ja|>"),
        ("emo_target", "<|NEUTRAL|>"),
        ("event_target", "<|Speech|>"),
        ("with_or_wo_itn", "<|woitn|>"),
    ],
)
def test_fixed_tag_values(field, expected, sample_wav):
    # Japanese speech, no ITN, no emotion labels in this corpus.
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert record[field] == expected


def test_tag_constants_match_the_emitted_tags():
    assert vn.TEXT_LANGUAGE == "<|ja|>"
    assert vn.EMO_TARGET == "<|NEUTRAL|>"
    assert vn.EVENT_TARGET == "<|Speech|>"
    assert vn.WITH_OR_WO_ITN == "<|woitn|>"


def test_key_and_target_are_passed_through(sample_wav):
    record = vn.build_record("mykey", "にぇっ！？", sample_wav, 2.0)

    assert record["key"] == "mykey"
    assert record["target"] == "にぇっ！？"


# --- source path ---


def test_source_is_absolute(sample_wav):
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert Path(record["source"]).is_absolute()


def test_relative_input_paths_are_made_absolute():
    # The trainer resolves ``source`` relative to nothing, so a relative path
    # in the manifest is an unopenable file at training time.
    record = vn.build_record("k", "こんにちは", "data/vn/audio/t/s/v.wav", 2.0)

    assert Path(record["source"]).is_absolute()
    assert record["source"].endswith("data/vn/audio/t/s/v.wav")


def test_source_is_a_string_not_a_path_object(sample_wav):
    # Path is not JSON-serialisable; the manifest writer would die on it.
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert isinstance(record["source"], str)


def test_accepts_both_str_and_path_inputs(sample_wav):
    from_path = vn.build_record("k", "こんにちは", sample_wav, 2.0)
    from_str = vn.build_record("k", "こんにちは", str(sample_wav), 2.0)

    assert from_path["source"] == from_str["source"]


def test_source_points_at_the_wav_that_exists(sample_wav):
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert Path(record["source"]).is_file()


# --- lengths ---


def test_target_len_counts_characters_for_japanese(sample_wav):
    # Pinned choice: make_smoke_data.py uses len(text.split()) because its
    # transcripts are English.  Japanese is unsegmented, so whitespace
    # splitting would report 1 for every line and destroy the length signal
    # the token-batch sampler relies on.  Characters are counted instead.
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert record["target_len"] == 5


def test_target_len_excludes_spaces(sample_wav):
    record = vn.build_record("k", "あ い う", sample_wav, 2.0)

    assert record["target_len"] == 3


@pytest.mark.parametrize(
    "text,expected",
    [
        ("こんにちは", 5),
        ("にぇっ！？", 5),
        ("日本語", 3),
        ("ラーメン", 4),
        ("あ", 1),
        ("", 0),
    ],
)
def test_target_len_over_representative_transcripts(text, expected, sample_wav):
    record = vn.build_record("k", text, sample_wav, 2.0)

    assert record["target_len"] == expected


def test_japanese_target_len_is_not_the_whitespace_word_count(sample_wav):
    # The regression this guards: a Japanese line has no spaces, so a
    # word-count implementation returns 1 for every clip in the corpus.
    record = vn.build_record("k", "これは日本語の文です", sample_wav, 2.0)

    assert record["target_len"] > 1


def test_source_len_delegates_to_compute_source_len(sample_wav):
    record = vn.build_record("k", "こんにちは", sample_wav, 3.5)

    assert record["source_len"] == vn.compute_source_len(3.5)


def test_source_len_matches_the_real_wav_duration(sample_wav):
    duration = wav_duration(sample_wav)

    record = vn.build_record("k", "こんにちは", sample_wav, duration)

    assert record["source_len"] == 200


def test_lengths_are_plain_ints(sample_wav):
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert isinstance(record["target_len"], int)
    assert isinstance(record["source_len"], int)


# --- overrides ---


def test_tags_can_be_overridden_by_keyword(sample_wav):
    record = vn.build_record(
        "k", "hello", sample_wav, 2.0, text_language="<|en|>", with_or_wo_itn="<|withitn|>"
    )

    assert record["text_language"] == "<|en|>"
    assert record["with_or_wo_itn"] == "<|withitn|>"


# --- serialisation ---


def test_round_trips_through_json(sample_wav):
    record = vn.build_record("k", "にぇっ！？", sample_wav, 2.0)

    assert json.loads(json.dumps(record, ensure_ascii=False)) == record


def test_json_dumps_keeps_japanese_unescaped(sample_wav):
    # The manifest writer uses ensure_ascii=False, matching make_smoke_data.py.
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert "こんにちは" in json.dumps(record, ensure_ascii=False)


def test_every_value_is_a_json_primitive(sample_wav):
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert all(isinstance(value, (str, int)) for value in record.values())


def test_serialises_as_one_line(sample_wav):
    # jsonl: a record containing a raw newline would corrupt the manifest.
    record = vn.build_record("k", "こんにちは", sample_wav, 2.0)

    assert "\n" not in json.dumps(record, ensure_ascii=False)


# ===========================================================================
# split_by_speaker
# ===========================================================================

# --- speaker identity ---


def test_speaker_is_recovered_from_the_wav_path():
    record = make_record("mytitle", "myspeaker", "v0")

    assert vn.record_speaker(record) == "mytitle/myspeaker"


def test_speaker_id_is_qualified_with_the_title():
    # The same speaker name recurs across titles; unqualified ids would merge
    # two different voices into one held-out "speaker".
    a = make_record("titleA", "sameName", "v0")
    b = make_record("titleB", "sameName", "v0")

    assert vn.record_speaker(a) != vn.record_speaker(b)


def test_an_explicit_speaker_field_wins_over_the_path():
    record = make_record("t", "s", "v0")
    record["speaker"] = "explicit"

    assert vn.record_speaker(record) == "explicit"


# --- the leakage guarantee ---


def test_train_and_val_share_no_speaker():
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert speakers_of(train) & speakers_of(val) == set()


@pytest.mark.parametrize("seed", range(10))
def test_no_speaker_leaks_at_any_seed(seed):
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=seed)

    assert speakers_of(train) & speakers_of(val) == set()


@pytest.mark.parametrize("val_frac", [0.05, 0.1, 0.2, 0.3, 0.5])
def test_no_speaker_leaks_at_any_val_fraction(val_frac):
    records = corpus()

    train, val = vn.split_by_speaker(records, val_frac, seed=0)

    assert speakers_of(train) & speakers_of(val) == set()


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 2),
        (3, 4, 5),
        (5, 2, 1),
        (2, 10, 3),
        (10, 1, 7),
    ],
)
def test_no_speaker_leaks_across_corpus_shapes(shape):
    titles, speakers_per_title, clips = shape
    records = corpus(titles, speakers_per_title, clips)

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert speakers_of(train) & speakers_of(val) == set()


def test_no_speaker_leaks_on_a_franchise_corpus():
    # The shape that actually matters: recurring voices across titles.
    records = franchise_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert speakers_of(train) & speakers_of(val) == set()


@pytest.mark.parametrize("seed", range(10))
def test_no_speaker_leaks_on_a_franchise_corpus_at_any_seed(seed):
    records = franchise_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=seed)

    assert speakers_of(train) & speakers_of(val) == set()


def test_a_recurring_voice_lands_entirely_on_one_side():
    # nanase is in titleA and titleB.  Both title-qualified halves must go the
    # same way, because they are one voice actor.
    records = franchise_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    in_train = any(vn.speaker_group_key(r) == "nanase" for r in train)
    in_val = any(vn.speaker_group_key(r) == "nanase" for r in val)
    assert in_train != in_val


def test_the_leak_guard_detects_a_leaky_split():
    """The guard must be *seen* to fail; a green regression test proves nothing.

    This reproduces the exact defect review found: a split holding out only the
    title-qualified group ``titleB/nanase`` leaves the *same voice* in train via
    ``titleA/nanase``.  Under the title-qualified id that split looks perfectly
    disjoint -- which is why ``speakers_of`` must use ``speaker_group_key``.
    """
    records = franchise_corpus()
    leaky_val = [r for r in records if vn.record_speaker(r) == "titleB/nanase"]
    leaky_train = [r for r in records if vn.record_speaker(r) != "titleB/nanase"]

    # Both halves are non-empty, so this is a real split, not a degenerate one.
    assert leaky_val and leaky_train

    # The vacuous check: title-qualified ids call this clean.
    assert qualified_speakers_of(leaky_train) & qualified_speakers_of(leaky_val) == set()

    # The real check: the voice is on both sides, and the guard says so.
    assert speakers_of(leaky_train) & speakers_of(leaky_val) == {"nanase"}


def test_a_held_out_speaker_leaves_no_clip_behind_in_train():
    # Partial hold-out is the subtle form of leakage: the speaker is "in val"
    # but some of their clips still trained the model.  Note val may contain
    # *fewer* clips than the speaker owns (the per-speaker cap drops the
    # surplus); what must never happen is a held-out voice appearing in train.
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    held_out = speakers_of(val)
    assert not [r for r in train if vn.speaker_group_key(r) in held_out]


def test_val_receives_at_most_the_cap_from_each_speaker():
    records = corpus()

    _, val = vn.split_by_speaker(records, 0.2, seed=0)

    counts = {}
    for record in val:
        key = vn.speaker_group_key(record)
        counts[key] = counts.get(key, 0) + 1
    assert all(count <= vn.VAL_MAX_CLIPS_PER_SPEAKER for count in counts.values())


# --- partition ---


def test_no_record_lands_in_both_splits():
    # The half of the old "exactly one split" claim that is still absolute.
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert set(keys_of(train)) & set(keys_of(val)) == set()


def test_no_record_is_invented():
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert set(keys_of(train)) | set(keys_of(val)) <= set(keys_of(records))


def test_output_never_exceeds_the_input():
    # NOT equality: surplus clips of a capped val speaker are dropped outright.
    # Returning them to train would put that voice on both sides, which is the
    # leak the split exists to prevent -- so the total legitimately shrinks.
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert len(train) + len(val) <= len(records)


def test_nothing_is_dropped_when_no_speaker_exceeds_the_cap():
    # Under the cap the split is a true partition; this pins that the shrinkage
    # above is caused by capping and nothing else.
    records = corpus(clips_per_speaker=3)
    assert all(
        count <= vn.VAL_MAX_CLIPS_PER_SPEAKER
        for count in _group_sizes(records).values()
    )

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert sorted(keys_of(train) + keys_of(val)) == keys_of(records)


# --- surplus dropping (only reachable when a speaker exceeds the cap) ---


@pytest.mark.parametrize("seed", range(6))
def test_surplus_is_actually_dropped_when_speakers_exceed_the_cap(seed):
    records = over_cap_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=seed)

    # Strictly fewer: this is the assertion a fixture under the cap cannot make.
    assert len(train) + len(val) < len(records)


def test_dropped_clips_are_exactly_the_surplus_of_the_val_speakers():
    records = over_cap_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    sizes = _group_sizes(records)
    held_out = speakers_of(val)
    expected_dropped = sum(
        sizes[name] - vn.VAL_MAX_CLIPS_PER_SPEAKER for name in held_out
    )
    assert len(records) - len(train) - len(val) == expected_dropped


def test_each_val_speaker_contributes_exactly_the_cap():
    records = over_cap_corpus()

    _, val = vn.split_by_speaker(records, 0.2, seed=0)

    counts = _group_sizes(val)
    assert counts
    assert all(count == vn.VAL_MAX_CLIPS_PER_SPEAKER for count in counts.values())


def test_dropped_surplus_is_not_handed_back_to_train():
    """The whole point of dropping: the surplus must not become training data.

    Returning a capped speaker's leftover clips to train would put that voice on
    both sides -- a leak wearing the disguise of "wasting no data".
    """
    records = over_cap_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    held_out = speakers_of(val)
    assert held_out
    assert not [r for r in train if vn.speaker_group_key(r) in held_out]


@pytest.mark.parametrize("seed", range(6))
def test_no_speaker_leaks_even_when_surplus_is_dropped(seed):
    records = over_cap_corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=seed)

    assert speakers_of(train) & speakers_of(val) == set()


def test_no_record_is_duplicated():
    records = corpus()

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    combined = keys_of(train) + keys_of(val)
    assert len(combined) == len(set(combined))


def test_the_input_list_is_not_mutated():
    records = corpus()
    before = keys_of(records)

    vn.split_by_speaker(records, 0.2, seed=0)

    assert keys_of(records) == before


def test_returns_two_plain_lists():
    train, val = vn.split_by_speaker(corpus(), 0.2, seed=0)

    assert isinstance(train, list)
    assert isinstance(val, list)


# --- determinism ---


def test_same_seed_gives_the_same_split():
    records = corpus()

    first = vn.split_by_speaker(records, 0.2, seed=7)
    second = vn.split_by_speaker(records, 0.2, seed=7)

    assert keys_of(first[0]) == keys_of(second[0])
    assert keys_of(first[1]) == keys_of(second[1])


def test_the_split_does_not_depend_on_input_order():
    # Speakers are sorted before shuffling, so a reordered manifest must not
    # silently produce a different validation set for the same seed.
    records = corpus()
    reversed_records = list(reversed(records))

    _, val_a = vn.split_by_speaker(records, 0.2, seed=3)
    _, val_b = vn.split_by_speaker(reversed_records, 0.2, seed=3)

    assert speakers_of(val_a) == speakers_of(val_b)


def test_different_seeds_give_different_splits():
    records = corpus()

    chosen = {
        tuple(sorted(speakers_of(vn.split_by_speaker(records, 0.2, seed=seed)[1])))
        for seed in range(8)
    }

    assert len(chosen) > 1


def test_global_random_state_is_not_disturbed():
    # The script seeds a private random.Random; using the module-level RNG
    # would make unrelated callers' sequences depend on split_by_speaker.
    import random

    random.seed(1234)
    expected = random.random()

    random.seed(1234)
    vn.split_by_speaker(corpus(), 0.2, seed=0)

    assert random.random() == expected


# --- val fraction ---


@pytest.mark.parametrize("val_frac", [0.1, 0.2, 0.3])
def test_val_fraction_is_approximately_honoured(val_frac):
    records = corpus(titles=4, speakers_per_title=5, clips_per_speaker=5)

    _, val = vn.split_by_speaker(records, val_frac, seed=0)

    # Whole speakers are held out, so the fraction is granular, not exact; the
    # implementation caps a split at 1.5x the target.
    assert 0 < len(val) <= max(1, round(len(records) * val_frac)) * 1.5


def test_val_is_never_empty_for_a_multi_speaker_corpus():
    records = corpus()

    _, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert val


def test_train_keeps_the_bulk_of_a_small_val_fraction():
    records = corpus(titles=4, speakers_per_title=5, clips_per_speaker=5)

    train, val = vn.split_by_speaker(records, 0.05, seed=0)

    assert len(train) > len(val)


def test_one_prolific_speaker_cannot_swallow_the_val_budget():
    # A speaker with far more clips than the target must be skipped, otherwise
    # validation collapses onto a single voice.
    records = corpus(titles=3, speakers_per_title=4, clips_per_speaker=2)
    records += [make_record("title0", "hog", f"v{i}") for i in range(200)]

    _, val = vn.split_by_speaker(records, 0.05, seed=0)

    assert "title0/hog" not in speakers_of(val)


@pytest.mark.parametrize("val_frac", [0.0, 1.0, -0.1, 1.5, 2.0])
def test_out_of_range_val_fraction_raises(val_frac):
    with pytest.raises(ValueError):
        vn.split_by_speaker(corpus(), val_frac, seed=0)


def test_default_val_fraction_constant_is_small():
    assert 0.0 < vn.VAL_FRACTION < 0.1


# --- title coverage ---


def test_val_spans_more_than_one_title():
    # A validation set drawn from a single visual novel measures adaptation to
    # that title's recording chain, not generalisation across the corpus.
    records = corpus(titles=4, speakers_per_title=4, clips_per_speaker=5)

    _, val = vn.split_by_speaker(records, 0.05, seed=0)

    assert len(titles_of(val)) >= 2


@pytest.mark.parametrize("seed", range(6))
def test_val_spans_multiple_titles_at_any_seed(seed):
    records = corpus(titles=4, speakers_per_title=4, clips_per_speaker=5)

    _, val = vn.split_by_speaker(records, 0.05, seed=seed)

    assert len(titles_of(val)) >= 2


def test_single_title_corpus_does_not_force_a_second_title():
    records = corpus(titles=1, speakers_per_title=6, clips_per_speaker=4)

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert titles_of(val) == {"title0"}
    assert speakers_of(train) & speakers_of(val) == set()


# --- degenerate corpora ---


def test_empty_input_returns_two_empty_lists():
    assert vn.split_by_speaker([], 0.2, seed=0) == ([], [])


def test_a_single_speaker_keeps_everything_trainable():
    # Holding out the only speaker would empty train and make the run
    # impossible; the corpus is returned intact with no validation instead.
    records = [make_record("t", "only", f"v{i}") for i in range(5)]

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert len(train) == 5
    assert val == []


def test_a_single_record_does_not_crash():
    train, val = vn.split_by_speaker([make_record("t", "s", "v0")], 0.2, seed=0)

    assert len(train) == 1
    assert val == []


def test_two_speakers_one_clip_each():
    records = [make_record("t", "a", "v0"), make_record("t", "b", "v0")]

    train, val = vn.split_by_speaker(records, 0.5, seed=0)

    assert len(train) + len(val) == 2
    assert speakers_of(train) & speakers_of(val) == set()


def test_one_record_per_speaker_never_empties_train():
    records = [make_record(f"title{i % 3}", f"spk{i}", "v0") for i in range(30)]

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert train
    assert val
    assert speakers_of(train) & speakers_of(val) == set()


@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 13])
def test_train_is_never_empty_for_any_small_corpus(count):
    # An empty train set is an unrunnable fine-tune; the split must degrade to
    # "everything trains" rather than produce one.
    records = [make_record(f"title{i % 2}", f"spk{i}", "v0") for i in range(count)]

    train, _ = vn.split_by_speaker(records, 0.3, seed=0)

    assert train


def test_one_name_across_two_titles_is_one_voice_and_stays_on_one_side():
    """A recurring name is a single voice actor, not two speakers.

    The previous name for this test ("...is split as two speakers") described
    the *defect*: treating (title, speaker) as the split key produced a formally
    disjoint split that still leaked the voice.  With bare-name grouping there
    is exactly one group here, so the whole corpus stays trainable rather than
    being cut down the middle of one voice.
    """
    records = [make_record("titleA", "same", f"v{i}") for i in range(4)]
    records += [make_record("titleB", "same", f"v{i}") for i in range(4)]

    train, val = vn.split_by_speaker(records, 0.4, seed=0)

    assert _group_sizes(records) == {"same": 8}
    assert len(train) == 8
    assert val == []
    assert speakers_of(train) & speakers_of(val) == set()


# ===========================================================================
# End-to-end over a synthesised corpus (no network, no real dataset)
# ===========================================================================


@pytest.fixture
def fake_corpus(tmp_path):
    """A miniature on-disk corpus: index.json plus real (tiny) wavs.

    Mirrors the layout the script produces -- ``audio/<title>/<speaker>/<voice>.wav``
    with an ``index.json`` of ``{"Speaker", "Voice", "Text"}`` entries -- at a
    scale where the whole thing is a few kilobytes.
    """
    entries = []
    for title in ("titleA", "titleB", "titleC"):
        for speaker_index in range(3):
            speaker = f"spk{speaker_index}"
            for voice_index in range(4):
                voice = f"v{voice_index}"
                seconds = 1.0 + 0.5 * voice_index
                wav = write_wav(
                    tmp_path / "audio" / title / speaker / f"{voice}.wav", seconds
                )
                entries.append(
                    {
                        "Speaker": speaker,
                        "Voice": voice,
                        "Text": f"「これは{title}の\\\"せりふ{voice_index}です……!?」",
                        "title": title,
                        "wav": wav,
                        "seconds": seconds,
                    }
                )

    index_json = tmp_path / "index.json"
    index_json.write_text(
        json.dumps(
            [
                {"Speaker": e["Speaker"], "Voice": e["Voice"], "Text": e["Text"]}
                for e in entries
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return entries, index_json


def build_records(entries):
    records = []
    for entry in entries:
        text = vn.normalize_text(entry["Text"])
        if not text:
            continue
        records.append(
            vn.build_record(
                f"{entry['title']}_{entry['Speaker']}_{entry['Voice']}",
                text,
                entry["wav"],
                entry["seconds"],
            )
        )
    return records


def test_index_json_parses_into_the_documented_entry_shape(fake_corpus):
    _, index_json = fake_corpus

    entries = json.loads(index_json.read_text(encoding="utf-8"))

    assert isinstance(entries, list)
    assert all({"Speaker", "Voice", "Text"} <= set(entry) for entry in entries)


def test_synthesised_corpus_produces_one_record_per_clip(fake_corpus):
    entries, _ = fake_corpus

    records = build_records(entries)

    assert len(records) == len(entries) == 36


def test_synthesised_records_all_have_non_empty_targets(fake_corpus):
    entries, _ = fake_corpus

    records = build_records(entries)

    # A zero-length target crashes the trainer, so the pipeline must never
    # emit one.
    assert all(record["target"] for record in records)
    assert all(record["target_len"] > 0 for record in records)


def test_synthesised_records_strip_the_wrapping_and_escape_artifacts(fake_corpus):
    entries, _ = fake_corpus

    records = build_records(entries)

    for record in records:
        assert not record["target"].startswith("「")
        assert "\\" not in record["target"]
        assert '"' not in record["target"]


def test_no_target_in_the_whole_corpus_carries_ascii_punctuation(fake_corpus):
    entries, _ = fake_corpus

    records = build_records(entries)

    # The corpus-wide form of the regression: it is not enough for individual
    # calls to be right, no manifest line may reach the trainer with ASCII
    # punctuation in Japanese text.
    for record in records:
        for ascii_char in (".", "!", "?", "~"):
            assert ascii_char not in record["target"]


def test_corpus_targets_keep_the_full_width_convention(fake_corpus):
    entries, _ = fake_corpus

    records = build_records(entries)

    assert all(record["target"].endswith("……！？") for record in records)


def test_synthesised_source_lens_match_the_wav_headers(fake_corpus):
    entries, _ = fake_corpus

    records = build_records(entries)

    for record in records:
        expected = vn.compute_source_len(wav_duration(Path(record["source"])))
        assert record["source_len"] == expected


def test_whole_manifest_round_trips_through_jsonl(fake_corpus, tmp_path):
    entries, _ = fake_corpus
    records = build_records(entries)
    manifest = tmp_path / "train.jsonl"

    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )

    assert read_jsonl(manifest) == records


def test_end_to_end_split_is_speaker_disjoint_and_multi_title(fake_corpus):
    entries, _ = fake_corpus
    records = build_records(entries)

    train, val = vn.split_by_speaker(records, 0.2, seed=0)

    assert speakers_of(train) & speakers_of(val) == set()
    assert set(keys_of(train)) & set(keys_of(val)) == set()
    assert set(keys_of(train)) | set(keys_of(val)) <= set(keys_of(records))
    # Not equality: surplus from a capped val speaker is dropped, not returned
    # to train.  This fixture happens to sit under the cap, so nothing is
    # dropped here -- the drop path itself is covered by over_cap_corpus().
    assert len(train) + len(val) <= len(records)
    assert len(titles_of(val)) >= 2


def test_end_to_end_records_match_the_shipped_schema(fake_corpus):
    entries, _ = fake_corpus
    shipped = read_jsonl(TRAIN_EXAMPLE_JSONL)[0]

    records = build_records(entries)

    assert all(list(record) == list(shipped) for record in records)


# ===========================================================================
# Offline guarantees
# ===========================================================================


def test_importing_the_script_performs_no_network_call():
    # The module is imported at collection time by every test in this file; if
    # import had side effects the suite could not run offline.
    assert vn.HF_BASE_URL.startswith("https://huggingface.co/")


def looks_like_an_embedded_secret(value):
    """Heuristic for a high-entropy credential sitting in source as a literal.

    Deliberately structural: it must flag a re-hardcoded password *without this
    file knowing what the password is*, because writing the value here to test
    for it would commit the very secret the check exists to keep out of git.

    Mixed case plus digits, no separators, 16-64 chars.  Pure-hex blobs are
    exempt -- the dataset's repo id is a 38-char hex string and is not a
    credential.
    """
    if not (16 <= len(value) <= 64) or not value.isalnum():
        return False
    if not (
        any(c.islower() for c in value)
        and any(c.isupper() for c in value)
        and any(c.isdigit() for c in value)
    ):
        return False
    try:
        int(value, 16)  # an id or digest, not a password
    except ValueError:
        return True
    return False


def string_literals(path):
    """Every string constant in ``path``, docstrings included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


@pytest.mark.parametrize("path", [SCRIPT_PATH, Path(__file__)], ids=["script", "tests"])
def test_no_high_entropy_secret_is_embedded_as_a_literal(path):
    """The 7z password must not be hardcoded anywhere -- including in here.

    An earlier revision had it in the script; the first version of this
    tripwire then re-committed it into the test file, which is no better, a
    secret in ``tests/`` is as permanently in git history as one in
    ``scripts/``.  So this checks both files and never names the value.
    """
    offenders = [v for v in string_literals(path) if looks_like_an_embedded_secret(v)]

    # Report shape, never the value: a failure message is CI-log output, and
    # echoing the credential there would undo the point of the test.  The
    # assertion is made against the *redacted* list rather than the raw one
    # because pytest's assertion rewriting prints whatever expression it is
    # given -- asserting on `offenders` would leak the secret into the log even
    # with a custom message attached.
    redacted = [f"{len(v)} chars starting {v[:2]}..." for v in offenders]
    assert redacted == [], (
        f"{path.name} embeds {len(offenders)} credential-shaped literal(s).  "
        f"Read the secret from ${vn.ARCHIVE_PASSWORD_ENV} or "
        f"{vn.ARCHIVE_PASSWORD_FILE.name} instead of hardcoding it."
    )


def test_the_secret_detector_actually_detects_a_secret():
    # Guards the tripwire above from rotting into a tautology: a heuristic that
    # silently stopped matching would leave it green forever.
    assert looks_like_an_embedded_secret("aB3" + "x7Qm2ptL9wRs4v")
    assert not looks_like_an_embedded_secret("56697375616C4E6F76656C5F44617461736574")
    assert not looks_like_an_embedded_secret("VN_ARCHIVE_PASSWORD")
    assert not looks_like_an_embedded_secret("short1A")


def test_the_configured_password_does_not_appear_in_the_source():
    """Exact-value check against whatever this machine is actually configured with.

    Complements the structural test: that one catches any credential-shaped
    literal, this one catches *the* password even if it were reformatted into a
    shape the heuristic misses.  Skipped when no password is configured, so a
    fresh checkout with no secret available stays green.
    """
    configured = os.environ.get(vn.ARCHIVE_PASSWORD_ENV, "").strip()
    if not configured and vn.ARCHIVE_PASSWORD_FILE.is_file():
        configured = vn.ARCHIVE_PASSWORD_FILE.read_text(encoding="utf-8").strip()
    if not configured:
        pytest.skip(
            f"no archive password configured (${vn.ARCHIVE_PASSWORD_ENV} unset "
            "and no password file); structural check still applies"
        )

    for path in (SCRIPT_PATH, Path(__file__)):
        assert configured not in path.read_text(encoding="utf-8")


def test_the_password_env_constant_is_stable():
    assert vn.ARCHIVE_PASSWORD_ENV == "VN_ARCHIVE_PASSWORD"


def test_the_archive_password_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(vn.ARCHIVE_PASSWORD_ENV, "  s3cret  ")

    assert vn.read_archive_password() == "s3cret"


def test_a_missing_archive_password_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv(vn.ARCHIVE_PASSWORD_ENV, raising=False)
    monkeypatch.setattr(vn, "ARCHIVE_PASSWORD_FILE", tmp_path / "absent")

    # SystemExit rather than a silent empty password, which would surface much
    # later as a corrupt-archive error.
    with pytest.raises(SystemExit):
        vn.read_archive_password()


def test_the_archive_password_is_never_logged(monkeypatch, capsys):
    monkeypatch.setenv(vn.ARCHIVE_PASSWORD_ENV, "sentinel-password-value")

    vn.log(f"extracting with password from ${vn.ARCHIVE_PASSWORD_ENV}")

    assert "sentinel-password-value" not in capsys.readouterr().out


def test_the_hf_token_is_scrubbed_from_download_transport_errors(
    monkeypatch, tmp_path, capsys
):
    """A transport error that echoes the request must not leak the HF token.

    ``download_archive`` is the one place that attaches ``Authorization:
    Bearer``, and it puts the exception text into two messages: the retry log
    line and the final ``SystemExit``.  Today's ``URLError.__str__`` does not
    embed request headers, so the error is simulated rather than provoked --
    the point is that the *scrubbing* holds if a dependency ever starts
    formatting exceptions that way.  Both sites are exercised: ``attempts=2``
    gives exactly one retry log before the raise.
    """
    token = "hf_sentinel-token-value"

    def explode(request, timeout=None):
        raise urllib.error.URLError(
            f"connection reset while sending Authorization: Bearer {token}"
        )

    monkeypatch.setattr(vn.urllib.request, "urlopen", explode)
    monkeypatch.setattr(vn.time, "sleep", lambda seconds: None)

    with pytest.raises(SystemExit) as excinfo:
        vn.download_archive(
            "GalGame/Studio_Title.7z",
            tmp_path / "Studio_Title.7z",
            token,
            attempts=2,
        )

    raised = str(excinfo.value)
    logged = capsys.readouterr().out
    assert token not in raised
    assert token not in logged
    # Not just absent -- the surrounding detail must still be there, so the
    # scrub is proven to be a substitution rather than a dropped message.
    assert "***" in raised
    assert "***" in logged
    assert "attempt 1 failed" in logged


# ---------------------------------------------------------------------------
# --archives basename collision
#
# Every stage keys an archive by its basename: the download writes
# <out-dir>/archives/<name>, extraction unpacks <out-dir>/raw/<stem> and the
# manifest records <stem> as the title.  Two repo paths sharing a basename
# therefore collapse onto one local file, and download_archive returns early
# for a file that already exists -- so the second archive silently resolves to
# the first one's contents and the manifest fingerprint records a lie.
# DEFAULT_ARCHIVES cannot reach this; --list-archives spans the whole repo, so
# paths pasted off it can.
# ---------------------------------------------------------------------------


def test_unique_basenames_are_accepted_unchanged():
    # The load-bearing case: a resume of an already-downloaded corpus must pass
    # the guard untouched, so it can go straight to extraction.
    sixteen = [f"GalGame/Studio {i}_Title {i}.7z" for i in range(16)]

    assert vn.check_archive_basenames(sixteen) is None


def test_the_shipped_default_archives_pass_the_guard():
    assert vn.check_archive_basenames(vn.DEFAULT_ARCHIVES) is None


def test_same_basename_in_two_directories_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        vn.check_archive_basenames(
            ["GalGame/Ai Kiss 2.7z", "Voice/Ai Kiss 2.7z", "GalGame/Other.7z"]
        )

    message = str(excinfo.value)
    # The colliding remote paths must be named: the whole failure mode is that
    # you cannot tell from the local layout which archive you actually got.
    assert "GalGame/Ai Kiss 2.7z" in message
    assert "Voice/Ai Kiss 2.7z" in message
    assert "GalGame/Other.7z" not in message


def test_a_repeated_path_is_refused():
    with pytest.raises(SystemExit):
        vn.check_archive_basenames(["GalGame/A.7z", "GalGame/A.7z"])


def test_the_collision_guard_runs_before_any_download(monkeypatch):
    # Fail-fast is the point: a collision must cost nothing, not surface after
    # hundreds of GB have been fetched.
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("download attempted despite a basename collision")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit):
        vn.main(["--archives", "GalGame/A.7z", "Other/A.7z"])


# ---------------------------------------------------------------------------
# --list-format
# ---------------------------------------------------------------------------


def test_list_format_without_list_archives_is_an_error(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("pipeline started despite an inapplicable flag")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(["--list-format", "plain"])

    assert "--list-format" in str(excinfo.value)


def test_list_format_defaults_to_none_so_it_stays_out_of_the_manifest():
    # parse_args defaults it to None (not "text") purely so the combination
    # above is detectable.  Nothing else may observe the difference.
    assert vn.parse_args([]).list_format is None
    assert vn.DEFAULT_LIST_FORMAT == "text"


def test_the_listing_table_columns_line_up(capsys):
    vn.print_archive_listing([("GalGame/a.7z", 1), ("X/b.7z", 45678901234)], "text")

    lines = capsys.readouterr().out.splitlines()
    header, rows = lines[1], lines[2:-1]
    # The header's leading "# " is part of its first column, so a 3-wide 'idx'
    # header aligns with a 5-wide index field.  Each column must end where the
    # header's label ends.
    for column in ("idx", "bytes", "size", "cumulative"):
        end = header.index(column) + len(column)
        for row in rows:
            # The row's value for this column ends exactly at the header
            # label's right edge, and a separator follows.
            assert row[:end] == row[:end].rstrip(), (column, header, row)
            assert row[end : end + 1] == " ", (column, header, row)


# ---------------------------------------------------------------------------
# Extraction resume: completeness must be *verified*, never assumed
#
# The bug this section pins: the resume check used to read "the output
# directory exists" as "this archive is fully extracted".  An extraction killed
# partway -- OOM, timeout, evicted node, full disk -- leaves a directory that is
# indistinguishable from a finished one, so the next run skipped it and the
# pipeline built a silently truncated corpus.  The only downstream symptom is a
# raised ``dropped_missing_audio`` in the filter summary, which reads as a
# dataset quirk rather than as a truncated extraction.
#
# Everything below drives real (tiny) password-protected .7z archives built in
# tmp_path.  The production archives are ~28 GiB each and gated, so they are
# never touched; py7zr is imported lazily in the script and skipped here the
# same way when it is not installed.
# ---------------------------------------------------------------------------

try:  # py7zr is an extraction-only extra: commented out in requirements.txt
    import py7zr
except ImportError:  # pragma: no cover - depends on the machine, not the code
    py7zr = None

# Skipped per test rather than with a module-level importorskip: that would skip
# this *whole* file, and everything above it is deliberately dependency-free.
requires_py7zr = pytest.mark.skipif(py7zr is None, reason="py7zr is not installed")

# Not a real credential: the archives below are built by this test, and the
# value is deliberately non-alnum so the embedded-secret detector above stays
# meaningful.
TEST_ARCHIVE_PASSWORD = "vn-test-password"


def build_test_archive(
    tmp_path,
    name="Studio_Test.7z",
    clips=3,
    password=TEST_ARCHIVE_PASSWORD,
    nested=False,
):
    """Build a small encrypted .7z with an ``index.json`` and ``clips`` oggs.

    ``nested`` mirrors the other real archive layout, where everything sits one
    directory down and ``find_index_json`` has to descend a level.

    Returns ``(archive_path, file_count)`` where ``file_count`` is the number of
    non-directory entries -- what a complete extraction must put on disk.
    """
    source = tmp_path / "source" / Path(name).stem
    (source / "spk").mkdir(parents=True, exist_ok=True)
    entries = [
        {"Speaker": "spk", "Voice": f"v{i:03d}", "Text": f"せりふ{i}"} for i in range(clips)
    ]
    (source / "index.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    for entry in entries:
        (source / "spk" / f"{entry['Voice']}.ogg").write_bytes(b"OggS-not-really-audio")

    archive = tmp_path / "archives" / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, "w", password=password) as handle:
        if nested:
            handle.writeall(source, arcname=source.name)
        else:
            # Written entry by entry rather than with writeall(arcname=".")
            # because py7zr refuses to extract a "." directory entry.
            for path in sorted(source.rglob("*")):
                handle.write(path, arcname=path.relative_to(source).as_posix())
    return archive, 1 + clips


@pytest.fixture
def extracted(tmp_path):
    """A freshly extracted archive: ``(archive, raw_dir, target, file_count)``."""
    archive, file_count = build_test_archive(tmp_path)
    raw_dir = tmp_path / "raw"
    root = vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)
    target = raw_dir / archive.stem
    assert root == target
    return archive, raw_dir, target, file_count


def forbid_re_extraction(monkeypatch):
    """Make any re-extraction attempt fail the test loudly instead of running."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("re-extracted an extraction that was already complete")

    monkeypatch.setattr(vn, "discard_partial_extraction", explode)


# --- the marker itself ------------------------------------------------------


@requires_py7zr
def test_a_successful_extraction_writes_a_completion_marker(extracted):
    archive, _raw_dir, target, file_count = extracted

    marker = vn.read_extract_marker(target)
    assert marker is not None, "no marker written: every later run has to re-verify by count"
    assert marker["archive"] == archive.name
    assert marker["archive_bytes"] == archive.stat().st_size
    assert marker["file_count"] == file_count
    assert marker["completed_at"]


@requires_py7zr
def test_the_marker_never_records_the_password(extracted):
    _archive, _raw_dir, target, _file_count = extracted

    text = (target / vn.EXTRACT_MARKER_NAME).read_text(encoding="utf-8")
    assert TEST_ARCHIVE_PASSWORD not in text


@requires_py7zr
def test_the_marker_is_a_dotfile_so_it_cannot_reach_the_manifest(extracted):
    _archive, _raw_dir, target, file_count = extracted

    # Two independent guards.  The name is a dotfile, and the only place the
    # raw tree is walked (find_index_json) descends into directories only, so
    # the marker can never be resolved as an index or counted as dataset audio.
    assert vn.EXTRACT_MARKER_NAME.startswith(".")
    assert vn.find_index_json(target) == target / "index.json"
    assert vn.count_extracted_files(target) == file_count


# --- 1. a truncated extraction is detected ----------------------------------


@requires_py7zr
def test_a_truncated_extraction_is_re_extracted_not_skipped(extracted, capsys):
    """The core regression: a half-populated directory must not pass for done."""
    archive, raw_dir, target, file_count = extracted
    # Simulate a kill mid-extraction: no marker was ever written, and one clip
    # never made it to disk.
    (target / vn.EXTRACT_MARKER_NAME).unlink()
    (target / "spk" / "v002.ogg").unlink()
    assert vn.count_extracted_files(target) == file_count - 1

    capsys.readouterr()
    root = vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)
    out = capsys.readouterr().out

    assert root == target
    assert (target / "spk" / "v002.ogg").is_file(), "the missing clip was not restored"
    assert vn.count_extracted_files(target) == file_count
    assert vn.read_extract_marker(target)["file_count"] == file_count
    # The reasoning has to be legible in the log: silence is what let the
    # truncated corpus through in the first place.
    assert "incomplete extraction" in out
    assert f"archive holds {file_count} files, {file_count - 1} on disk" in out


@requires_py7zr
def test_a_truncated_extraction_missing_its_index_is_also_re_extracted(extracted):
    archive, raw_dir, target, file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).unlink()
    (target / "index.json").unlink()
    (target / "spk" / "v000.ogg").unlink()

    root = vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)

    assert vn.find_index_json(root) == target / "index.json"
    assert vn.count_extracted_files(target) == file_count


@requires_py7zr
def test_re_extraction_does_not_merge_with_stale_orphans(extracted):
    """py7zr writes in place, so leftovers would survive and be indexed."""
    archive, raw_dir, target, file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).unlink()
    # Two clips short and one file that is not in the archive at all, so the
    # directory is unambiguously incomplete (a one-for-one swap would balance
    # the count -- see the marker tests for the check that catches that).
    (target / "spk" / "v001.ogg").unlink()
    (target / "spk" / "v002.ogg").unlink()
    orphan = target / "spk" / "from_a_previous_revision.ogg"
    orphan.write_bytes(b"stale")

    vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)

    assert not orphan.exists(), "a stale file survived the re-extraction"
    assert vn.count_extracted_files(target) == file_count


# --- 2. a marked, complete extraction is skipped ----------------------------


@requires_py7zr
def test_a_complete_extraction_with_a_valid_marker_is_skipped(extracted, monkeypatch, capsys):
    archive, raw_dir, target, file_count = extracted
    forbid_re_extraction(monkeypatch)
    stamp = (target / "index.json").stat().st_mtime_ns

    capsys.readouterr()
    # Empty password on purpose: a marker-verified archive must be skippable
    # with no secret available at all.
    root = vn.extract_archive(archive, raw_dir, "")
    out = capsys.readouterr().out

    assert root == target
    assert (target / "index.json").stat().st_mtime_ns == stamp, "files were rewritten"
    assert "already extracted, skipping" in out
    assert f"marker verified: {file_count} files" in out


@requires_py7zr
def test_a_marker_verified_archive_needs_no_password(extracted):
    archive, raw_dir, _target, _file_count = extracted

    assert vn.extraction_password_need(archive, raw_dir) == "skip"


# --- 3. a complete extraction with no marker (the pre-existing corpora) -----


@requires_py7zr
def test_an_unmarked_but_complete_extraction_is_verified_by_count_and_backfilled(
    extracted, monkeypatch, capsys
):
    """The 53.8 h and ~187 h corpora on disk predate markers.

    Re-extracting them on the next run would be a costly surprise, so an
    unmarked directory whose file count matches the archive's entry count is
    accepted -- and the marker is written retroactively so the next run is
    cheap.
    """
    archive, raw_dir, target, file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).unlink()
    forbid_re_extraction(monkeypatch)

    capsys.readouterr()
    root = vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)
    out = capsys.readouterr().out

    assert root == target
    assert "verified by count" in out
    assert f"{file_count} files == {file_count} archive entries" in out

    marker = vn.read_extract_marker(target)
    assert marker is not None, "the marker was not backfilled"
    assert marker["file_count"] == file_count
    assert marker["archive_bytes"] == archive.stat().st_size


@requires_py7zr
def test_the_backfilled_marker_makes_the_next_run_free(extracted, monkeypatch, capsys):
    archive, raw_dir, target, _file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).unlink()
    vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)

    forbid_re_extraction(monkeypatch)
    capsys.readouterr()
    vn.extract_archive(archive, raw_dir, "")

    assert "marker verified" in capsys.readouterr().out


@requires_py7zr
def test_an_unmarked_extraction_wants_the_password_only_to_verify(extracted):
    archive, raw_dir, target, _file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).unlink()

    # "verify", not "extract": missing the secret must not abort a run over an
    # already-extracted corpus.
    assert vn.extraction_password_need(archive, raw_dir) == "verify"


@requires_py7zr
def test_without_a_password_an_unmarked_extraction_is_skipped_but_flagged(
    extracted, monkeypatch, capsys
):
    archive, raw_dir, target, _file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).unlink()
    forbid_re_extraction(monkeypatch)

    capsys.readouterr()
    root = vn.extract_archive(archive, raw_dir, "")
    out = capsys.readouterr().out

    assert root == target
    assert "NOT verified" in out
    # Nothing was verified, so nothing may be certified: writing a marker here
    # would launder an unchecked directory into a trusted one forever.
    assert vn.read_extract_marker(target) is None


# --- 4. a marker that disagrees with disk -----------------------------------


@requires_py7zr
@pytest.mark.parametrize(
    "field,value",
    [
        ("file_count", 999),
        ("archive_bytes", 1),
        ("archive", "Some_Other_Title.7z"),
    ],
)
def test_a_marker_that_disagrees_with_disk_triggers_re_extraction(
    extracted, capsys, field, value
):
    archive, raw_dir, target, file_count = extracted
    marker_path = target / vn.EXTRACT_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    # Something is also actually missing, to prove the re-extraction repairs it.
    (target / "spk" / "v000.ogg").unlink()

    capsys.readouterr()
    vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)
    out = capsys.readouterr().out

    assert "marker disagrees with disk" in out
    assert (target / "spk" / "v000.ogg").is_file()
    assert vn.read_extract_marker(target)["file_count"] == file_count


@requires_py7zr
def test_a_disagreeing_marker_makes_the_password_mandatory(extracted):
    archive, raw_dir, target, _file_count = extracted
    marker_path = target / vn.EXTRACT_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["file_count"] += 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert vn.extraction_password_need(archive, raw_dir) == "extract"


@requires_py7zr
def test_a_corrupt_marker_is_treated_as_absent(extracted):
    _archive, _raw_dir, target, _file_count = extracted
    (target / vn.EXTRACT_MARKER_NAME).write_text("{not json", encoding="utf-8")

    assert vn.read_extract_marker(target) is None


# --- the wipe is scoped ------------------------------------------------------


@requires_py7zr
def test_the_stale_wipe_refuses_anything_but_one_archives_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "Title" / "spk").mkdir(parents=True)

    # The whole raw tree, and anything below the per-archive level, is off
    # limits: only <raw>/<stem> may ever be deleted.
    with pytest.raises(SystemExit):
        vn.discard_partial_extraction(raw_dir, raw_dir)
    with pytest.raises(SystemExit):
        vn.discard_partial_extraction(raw_dir / "Title" / "spk", raw_dir)
    assert (raw_dir / "Title" / "spk").is_dir()


@requires_py7zr
def test_the_stale_wipe_removes_only_the_named_archive(tmp_path):
    raw_dir = tmp_path / "raw"
    doomed = raw_dir / "Title_A"
    keeper = raw_dir / "Title_B"
    doomed.mkdir(parents=True)
    keeper.mkdir(parents=True)
    (doomed / "partial.ogg").write_bytes(b"x")
    (keeper / "index.json").write_text("[]", encoding="utf-8")

    vn.discard_partial_extraction(doomed, raw_dir)

    assert not doomed.exists()
    assert (keeper / "index.json").is_file()


# --- entry counting ---------------------------------------------------------


@requires_py7zr
def test_the_archive_entry_count_ignores_directories(tmp_path):
    archive, file_count = build_test_archive(tmp_path, clips=5)

    assert vn.count_archive_entries(archive, TEST_ARCHIVE_PASSWORD) == file_count


@requires_py7zr
def test_the_archive_entry_count_is_none_without_a_password(tmp_path):
    archive, _file_count = build_test_archive(tmp_path)

    # None means "cannot verify" and must never be read as zero, which would
    # condemn every complete extraction to a re-run.
    assert vn.count_archive_entries(archive, "") is None


@requires_py7zr
def test_the_archive_entry_count_is_none_when_the_archive_cannot_be_read(tmp_path):
    broken = tmp_path / "archives" / "Broken.7z"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a 7z container at all")

    assert vn.count_archive_entries(broken, TEST_ARCHIVE_PASSWORD) is None


@requires_py7zr
def test_an_unextracted_archive_needs_the_password(tmp_path):
    archive, _file_count = build_test_archive(tmp_path)

    assert vn.extraction_password_need(archive, tmp_path / "raw") == "extract"


@requires_py7zr
def test_the_resume_check_works_on_the_nested_archive_layout(tmp_path, capsys):
    """Some archives put everything one directory down; the marker still sits
    at the raw dir's root, so the count must not be thrown off by the level."""
    archive, file_count = build_test_archive(tmp_path, clips=4, nested=True)
    raw_dir = tmp_path / "raw"
    target = raw_dir / archive.stem

    root = vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)
    assert root == target / archive.stem
    assert vn.find_index_json(target) == root / "index.json"
    assert vn.read_extract_marker(target)["file_count"] == file_count

    (root / "spk" / "v000.ogg").unlink()
    (target / vn.EXTRACT_MARKER_NAME).unlink()

    capsys.readouterr()
    vn.extract_archive(archive, raw_dir, TEST_ARCHIVE_PASSWORD)

    assert "incomplete extraction" in capsys.readouterr().out
    assert (root / "spk" / "v000.ogg").is_file()
