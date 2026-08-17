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
import shutil
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


# --- the stray "n" of a line break that lost its backslash ---
#
# The game script's line breaks were escaped as "\n"; somewhere upstream the
# backslash was lost and the "n" stayed, wedged between two Japanese characters.
# 89 occurrences in a 4,000-clip sample -- ~12,000 clips of the 549,404-clip
# manifest -- and invisible in the transcripts alone: they read as ordinary
# Japanese until the clips are ranked by CER against an in-domain teacher ASR
# model.  The transcript is otherwise correct, so this is repaired rather than
# dropped, and the repair must be narrow enough to leave every legitimate "n"
# alone (the counter-examples below are the real test).


@pytest.mark.parametrize(
    "raw,expected",
    [
        # All three verbatim from the corpus.
        (
            "だってだってぇ、お風呂上がりはこの飲むプリンで一n杯きゅーって行くのが最高で",
            "だってだってぇ、お風呂上がりはこの飲むプリンで一杯きゅーって行くのが最高で",
        ),
        (
            "あ、それって、あたしがあnんまりにも色気がありすぎちゃったから、"
            "にーちゃんのn我慢が限界になって",
            "あ、それって、あたしがあんまりにも色気がありすぎちゃったから、"
            "にーちゃんの我慢が限界になって",
        ),
        (
            "はいっ！わたくしとしては、教師と生徒の愛ある調n教する話は",
            "はいっ！わたくしとしては、教師と生徒の愛ある調教する話は",
        ),
    ],
)
def test_a_lone_n_between_japanese_characters_is_deleted(raw, expected):
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("一n杯", "一杯"),  # kanji on both sides
        ("あnん", "あん"),  # hiragana on both sides
        ("ンnン", "ンン"),  # katakana on both sides
        ("ーnー", "ーー"),  # the long-vowel mark counts as Japanese
        ("そのn人", "その人"),  # kana then kanji
    ],
)
def test_the_repair_needs_a_kana_or_kanji_on_each_side(raw, expected):
    assert vn.normalize_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        # A Latin word embedded in Japanese: its "n" has a Latin neighbour.
        "今日はfineな気分",
        "ワンnight",
        "nightは夜",
        # A romanised name -- the reason the rule may not simply delete "n"
        # between any two characters.
        "Nanaseと呼んで",
        "田中nakaさん",
        # Doubled "n": each one has the other beside it, so neither is isolated.
        "あnnまり",
        "一nn杯",
        # A digit neighbour is not Japanese either.
        "第2n回",
        "n2は変数",
        # Uppercase is left alone: the lost escape is lowercase "\n", while an
        # uppercase N between Japanese characters is a real placeholder or
        # abbreviation ("第N回", "N響").
        "第N回",
        # Nothing to sit between.
        "n",
        "んn",
    ],
)
def test_a_legitimate_n_is_never_touched(text):
    # normalize_text may still fold width or punctuation elsewhere in the line;
    # what must hold is that the "n" count does not change.
    assert vn.normalize_text(text).count("n") == text.count("n")


def test_an_escape_that_still_has_its_backslash_is_also_repaired():
    # The backslash deletion runs first and would otherwise *create* exactly the
    # defect above: "\n" -> "n" wedged inside the word.
    assert vn.normalize_text("に\\nぇ") == "にぇ"


def test_a_real_line_break_is_still_whitespace_not_a_stray_n():
    # An actual newline character is collapsed by the whitespace rule; only the
    # leftover literal "n" is deleted.
    assert vn.normalize_text("あ\nい") == "あ い"


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
        # The stray-"n" repair: deleting the "n" joins its two neighbours, and a
        # second pass must not then find a new "n" to delete.
        "一n杯",
        "あnnまり",
        "今日はfineな気分",
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
# --pin-val-keys: a val set that survives a corpus rebuild
#
# split_by_speaker computes a fresh split over whatever corpus it is handed, so
# a bigger corpus yields a *different* val set and round 2's metrics cannot be
# compared with round 1's.  Worse, round 1's val clips end up in round 2's
# training set -- measured once at 547 of 772 clips (70.9%), which is why a plan
# to re-score an old val had to be abandoned: it would have been scoring on
# training data.  test_growing_the_corpus_moves_round_one_val_clips_into_train
# reproduces that failure, and everything after it pins the fix.
#
# The two properties that matter:
#   1. val is EXACTLY the pinned clips -- not a superset, so the two rounds are
#      measured on the same data;
#   2. speaker-disjointness still holds -- a pinned clip's speaker is kept out
#      of train entirely, exactly as the unpinned split does it.
# ===========================================================================


def round_corpus(titles, speakers=5, clips=6):
    """A corpus of title-local speakers, sized so each title fills a val quota."""
    return [
        make_record(title, f"spk_{title}_{index}", f"v{clip}")
        for title in titles
        for index in range(speakers)
        for clip in range(clips)
    ]


def pin_split(records, keys):
    return vn.split_by_pinned_keys(records, keys)


# --- the failure this flag exists to prevent -------------------------------


def test_growing_the_corpus_moves_round_one_val_clips_into_train():
    """The hazard, seen failing.  A green fix test alone would prove nothing.

    Round 2 adds archives that sort *before* the round-1 titles, which shifts
    the per-title candidate shuffle and therefore which speakers are held out.
    Every round-1 val clip then lands in round-2 train.  The 100% here is this
    fixture's number, not a law -- the real corpus measured 70.9%.
    """
    round1 = round_corpus(["t0", "t1"])
    _, val1 = vn.split_by_speaker(round1, 0.2, seed=0)
    round2 = round1 + round_corpus(["a0", "a1"])

    train2, _ = vn.split_by_speaker(round2, 0.2, seed=0)

    pinned = {r["key"] for r in val1}
    assert pinned
    assert pinned <= {r["key"] for r in train2}


def test_pinning_keeps_every_round_one_val_clip_out_of_round_two_train():
    # Same corpora, same seed, with the pin: the exact clips round 1 was scored
    # on are held out of round 2's training set.
    round1 = round_corpus(["t0", "t1"])
    _, val1 = vn.split_by_speaker(round1, 0.2, seed=0)
    round2 = round1 + round_corpus(["a0", "a1"])

    train2, val2, _ = pin_split(round2, keys_of(val1))

    assert keys_of(val2) == keys_of(val1)
    assert set(keys_of(train2)) & set(keys_of(val1)) == set()


def test_the_pinned_val_is_identical_however_the_corpus_grows():
    # The whole point: one pin file, several corpora, one val set.  If this
    # holds, two rounds' numbers are measured on the same clips.
    round1 = round_corpus(["t0", "t1"])
    _, val1 = vn.split_by_speaker(round1, 0.2, seed=0)
    keys = keys_of(val1)

    for extra in (["a0"], ["a0", "a1"], ["m0", "m1", "z9"]):
        _, val2, _ = pin_split(round1 + round_corpus(extra), keys)

        assert keys_of(val2) == keys


# --- pinned clips reach val ------------------------------------------------


def test_every_pinned_clip_that_exists_lands_in_val():
    records = corpus()
    keys = [records[0]["key"], records[7]["key"], records[-1]["key"]]

    _, val, _ = pin_split(records, keys)

    assert set(keys) <= set(keys_of(val))


def test_val_is_exactly_the_pinned_set_and_never_a_superset():
    # No top-up toward --val-frac: a val that grows with the corpus is the
    # thing being fixed, so "pinned plus extra speakers" is not offered.
    records = corpus()
    keys = keys_of(records[:3])

    _, val, _ = pin_split(records, keys)

    assert keys_of(val) == sorted(keys)


def test_a_pinned_clip_never_appears_in_train():
    records = franchise_corpus()
    keys = keys_of([r for r in records if vn.speaker_group_key(r) == "aoi"])

    train, val, _ = pin_split(records, keys)

    assert set(keys_of(train)) & set(keys) == set()
    assert keys_of(val) == sorted(keys)


def test_the_per_speaker_cap_does_not_trim_the_pinned_set():
    # The unpinned split caps a speaker at VAL_MAX_CLIPS_PER_SPEAKER (40) to
    # stop one voice owning val.  A pin file is an explicit instruction and is
    # honoured whole -- trimming it would silently change what round 2 is
    # scored on, which is exactly the failure being fixed.
    records = over_cap_corpus()
    one_speaker = [r for r in records if vn.speaker_group_key(r) == "spk_titleA_0"]
    assert len(one_speaker) > vn.VAL_MAX_CLIPS_PER_SPEAKER

    _, val, _ = pin_split(records, keys_of(one_speaker))

    assert len(val) == len(one_speaker)


def test_the_val_fraction_does_not_bound_the_pinned_set():
    # Half the corpus pinned is far past the 2.5% default; the pin wins.
    records = corpus(titles=2, speakers_per_title=4, clips_per_speaker=5)
    half = [r for r in records if vn.speaker_group_key(r).endswith(("_0", "_1"))]

    _, val, _ = pin_split(records, keys_of(half))

    assert len(val) == len(half) == len(records) // 2


# --- speaker-disjointness, which pinning must not quietly break ------------


def test_a_pinned_speaker_is_absent_from_train():
    records = corpus()
    keys = [records[0]["key"]]

    train, val, _ = pin_split(records, keys)

    assert speakers_of(train) & speakers_of(val) == set()


@pytest.mark.parametrize("index", [0, 5, 11, 23, 44])
def test_no_speaker_leaks_whichever_single_clip_is_pinned(index):
    records = corpus()

    train, val, _ = pin_split(records, [records[index]["key"]])

    assert speakers_of(train) & speakers_of(val) == set()


def test_pinning_one_clip_of_a_recurring_voice_holds_out_every_title():
    # The franchise case: pinning a titleA clip of nanase must also keep
    # nanase's titleB clips out of train, because it is one voice actor.
    records = franchise_corpus()
    one = next(
        r
        for r in records
        if vn.speaker_group_key(r) == "nanase" and vn.record_title(r) == "titleA"
    )

    train, _, _ = pin_split(records, [one["key"]])

    assert not [r for r in train if vn.speaker_group_key(r) == "nanase"]


def test_the_non_pinned_clips_of_a_pinned_speaker_are_dropped_from_both_sides():
    # They cannot go to train (that is the leak) and they cannot go to val
    # (that would make val bigger than the pinned set), so they are dropped --
    # the same trade the unpinned split makes with surplus clips.
    records = corpus(titles=2, speakers_per_title=3, clips_per_speaker=5)
    speaker_clips = [r for r in records if vn.speaker_group_key(r) == "spk0_0"]
    pinned, rest = speaker_clips[:2], speaker_clips[2:]

    train, val, report = pin_split(records, keys_of(pinned))

    assert keys_of(val) == keys_of(pinned)
    assert set(keys_of(train)) & set(keys_of(rest)) == set()
    assert report.train_clips_dropped == len(rest) == 3
    assert len(train) + len(val) + report.train_clips_dropped == len(records)


# --- reporting -------------------------------------------------------------


def test_missing_pinned_keys_are_reported_not_silently_ignored():
    # A stale or typo'd pin file that quietly pins less than it names would
    # produce a val set that looks healthy and is not the intended one.
    records = corpus()
    keys = [records[0]["key"], "titleZ_ghost_v9", records[1]["key"], "typo"]

    _, val, report = pin_split(records, keys)

    assert report.requested == 4
    assert report.found == 2
    assert report.missing == ["titleZ_ghost_v9", "typo"]
    assert len(val) == 2


def test_the_report_counts_reconcile_with_the_split():
    records = franchise_corpus()
    keys = keys_of(records[:6]) + ["absent"]

    train, val, report = pin_split(records, keys)

    assert report.requested == report.found + len(report.missing)
    assert report.found == len(val)
    assert report.speakers == len(speakers_of(val))
    assert report.train_clips_dropped == len(records) - len(train) - len(val)


def test_duplicate_pinned_keys_are_counted_and_pinned_once():
    records = corpus()
    key = records[0]["key"]

    _, val, report = pin_split(records, [key, key, key])

    assert report.requested == 1
    assert keys_of(val) == [key]


def test_the_pin_report_reaches_the_manifest_with_the_full_missing_list():
    records = corpus()
    _, _, report = pin_split(records, [records[0]["key"], "gone_a", "gone_b"])
    report.keys_file = "/pins/round1.txt"

    payload = report.as_dict()

    assert payload["keys_file"] == "/pins/round1.txt"
    assert payload["keys_requested"] == 3
    assert payload["keys_found"] == 1
    assert payload["keys_missing"] == 2
    # The full list, never a sample: this is the record of which round-1 clips
    # the new corpus could not reproduce, and a truncated one cannot be diffed
    # against the next rebuild.
    assert payload["missing_keys"] == ["gone_a", "gone_b"]
    assert json.loads(json.dumps(payload)) == payload


def test_the_log_states_requested_found_missing_and_the_dropped_clips(capsys):
    report = vn.PinnedValReport(
        requested=772,
        found=770,
        missing=["stale_a", "stale_b"],
        speakers=23,
        train_clips_dropped=1_234,
        keys_file="/pins/round1.txt",
    )

    vn.log_pinned_val_report(report)

    out = capsys.readouterr().out
    assert "772 keys requested" in out
    assert "770 found in corpus" in out
    assert "2 missing" in out
    assert "23 pinned speaker(s) held out of train" in out
    assert "1234 further clip(s)" in out
    assert "stale_a" in out and "stale_b" in out
    # The decision, restated where the operator will see it.
    assert "val is EXACTLY the pinned set" in out


def test_a_long_missing_list_is_truncated_in_the_log_but_still_counted(capsys):
    missing = [f"gone_{i}" for i in range(vn.PIN_MISSING_KEYS_LOGGED + 5)]
    report = vn.PinnedValReport(requested=100, found=75, missing=missing)

    vn.log_pinned_val_report(report)

    out = capsys.readouterr().out
    # Truncation keeps a genuinely different corpus from flooding the log; the
    # count and manifest.json still carry every key.
    assert f"gone_{vn.PIN_MISSING_KEYS_LOGGED - 1}" in out
    assert f"gone_{vn.PIN_MISSING_KEYS_LOGGED}" not in out
    assert "... and 5 more missing key(s)" in out
    assert f"{len(missing)} missing" in out


# --- loud failures ---------------------------------------------------------


def test_a_pin_file_matching_nothing_is_fatal():
    # The silent-failure case: falling back to an ordinary split here would
    # write a perfectly healthy-looking val set that is not the pinned one.
    records = corpus()

    with pytest.raises(SystemExit) as excinfo:
        pin_split(records, ["not_a_key", "also_not_a_key"])

    message = str(excinfo.value)
    assert "--pin-val-keys" in message
    assert "2 pinned keys" in message


def test_pinning_every_speaker_is_refused_rather_than_emptying_train():
    records = corpus(titles=2, speakers_per_title=2, clips_per_speaker=3)

    with pytest.raises(SystemExit) as excinfo:
        pin_split(records, keys_of(records))

    assert "train empty" in str(excinfo.value)


def test_an_empty_key_list_is_refused():
    with pytest.raises(SystemExit):
        pin_split(corpus(), [])


# --- the pin file ----------------------------------------------------------


def test_the_pin_file_is_one_key_per_line(tmp_path):
    path = tmp_path / "pins.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert vn.read_pin_val_keys(path) == ["alpha", "beta", "gamma"]


def test_the_pin_file_ignores_blank_lines_comments_and_stray_whitespace(tmp_path):
    path = tmp_path / "pins.txt"
    path.write_text(
        "# round 1 val, 2026-08-14\n\n  alpha  \n\n# a note\nbeta\n\n",
        encoding="utf-8",
    )

    # Comments matter: the file records which run it came from, and no
    # production key can start with '#' -- keys are built from slugify output.
    assert vn.read_pin_val_keys(path) == ["alpha", "beta"]


def test_the_pin_file_collapses_duplicates_keeping_the_first_position(tmp_path):
    path = tmp_path / "pins.txt"
    path.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

    assert vn.read_pin_val_keys(path) == ["alpha", "beta"]


def test_a_missing_pin_file_fails_loudly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        vn.read_pin_val_keys(tmp_path / "nope.txt")

    assert "--pin-val-keys" in str(excinfo.value)


def test_a_pin_file_with_no_keys_fails_loudly(tmp_path):
    path = tmp_path / "pins.txt"
    path.write_text("# only a comment\n\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        vn.read_pin_val_keys(path)

    assert "no keys" in str(excinfo.value)


def test_a_real_val_jsonl_round_trips_through_the_pin_file(tmp_path):
    # How the flag is actually used: keys are cut out of a written val.jsonl.
    records = corpus()
    _, val = vn.split_by_speaker(records, 0.2, seed=0)
    val_path = tmp_path / "val.jsonl"
    vn.write_jsonl(val, val_path)
    pins = tmp_path / "pins.txt"
    pins.write_text(
        "".join(f"{r['key']}\n" for r in read_jsonl(val_path)), encoding="utf-8"
    )

    _, pinned, report = pin_split(records, vn.read_pin_val_keys(pins))

    assert keys_of(pinned) == keys_of(val)
    assert report.missing == []


# --- the CLI ---------------------------------------------------------------


def test_pin_val_keys_defaults_to_none_so_the_flag_is_opt_in():
    assert vn.parse_args([]).pin_val_keys is None


def test_pin_val_keys_is_parsed_as_a_path():
    assert vn.parse_args(["--pin-val-keys", "pins.txt"]).pin_val_keys == Path("pins.txt")


def test_the_cli_surface_is_exactly_these_options():
    """The CLI surface, pinned deliberately.

    Listed exhaustively so that adding an option is a conscious edit here: this
    script's flags are recorded in manifest.json's ``config`` and quoted in the
    training docs, and a silently grown CLI makes an old command line's meaning
    ambiguous.  Six options have been added under this rule so far --
    ``--pin-val-keys``, ``--manifest-only``, ``--drop-kana-only-titles``, its
    ``--kana-only-title-threshold``, ``--emo-labels`` and its
    ``--allow-sparse-emo-labels``.
    """
    assert sorted(vars(vn.parse_args([]))) == [
        "allow_sparse_emo_labels",
        "archives",
        "download_workers",
        "drop_kana_only_titles",
        "emo_labels",
        "kana_only_title_threshold",
        "limit_hours",
        "list_archives",
        "list_format",
        "manifest_only",
        "max_seconds",
        "min_seconds",
        "out_dir",
        "pin_val_keys",
        "seed",
        "skip_download",
        "val_cover_all_titles",
        "val_frac",
        "val_max_clips_per_speaker",
        "workers",
    ]


def test_the_pin_file_is_read_before_any_download(monkeypatch):
    # Fail-fast, like the basename guard: a typo'd path must cost nothing, not
    # surface after a 1000-hour corpus has been fetched and converted.
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("download attempted despite an unreadable pin file")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(["--pin-val-keys", "/nonexistent/pins.txt"])

    assert "--pin-val-keys" in str(excinfo.value)


# --- the unpinned path must be exactly what it was -------------------------


def test_the_unpinned_split_is_unchanged_by_the_pinning_code():
    """Golden result for the default path, so a rebuild reproduces round 1.

    The existing corpus was built by this splitter and its train/val files are
    fingerprinted; nothing added for --pin-val-keys may perturb the unpinned
    split.  The keys below were produced by the splitter as it stood before the
    flag existed.
    """
    train, val = vn.split_by_speaker(franchise_corpus(), 0.2, seed=0)

    assert keys_of(val) == [
        "titleA_aoi_v0",
        "titleA_aoi_v1",
        "titleA_aoi_v2",
        "titleA_aoi_v3",
        "titleA_aoi_v4",
        "titleB_rin_v0",
        "titleB_rin_v1",
        "titleB_rin_v2",
        "titleB_rin_v3",
        "titleB_rin_v4",
    ]
    assert len(train) == 35


def test_the_unpinned_manifest_carries_no_pin_section():
    # An unpinned run's manifest.json must be byte-for-byte what it was before
    # the flag existed, so the two corpora's manifests stay comparable.
    records = corpus()
    train, val = vn.split_by_speaker(records, 0.2, seed=0)
    args = vn.parse_args([])
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))

    manifest = vn.build_manifest(args, {}, stats, train, val, 0)

    assert "val_pin" not in manifest
    assert "pin_val_keys" not in manifest["config"]


def test_a_pinned_run_records_the_pin_in_the_manifest():
    records = corpus()
    train, val, report = pin_split(records, keys_of(records[:2]))
    report.keys_file = "/pins/round1.txt"
    args = vn.parse_args([])
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))

    manifest = vn.build_manifest(args, {}, stats, train, val, 0, pin_report=report)

    assert manifest["val_pin"]["keys_file"] == "/pins/round1.txt"
    assert manifest["val_pin"]["keys_found"] == 2


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


# ===========================================================================
# --manifest-only: rebuilding the split after the audio sources are pruned
# ===========================================================================
# A ~1,000 h corpus does not fit on the cluster alongside its own intermediates,
# so it is built in batches and each batch's .7z and extracted .ogg files are
# deleted once converted, leaving raw/<stem>/index.json as the only survivor of
# the raw tree.  --manifest-only produces the final whole-corpus split from that
# state: index.json plus the converted wavs, no archives, no credentials.
#
# Reading a wav header needs soundfile, and a *full* run additionally decodes
# with librosa.  Both are skipped rather than imported eagerly, so the rest of
# this file keeps its promise of running with no audio dependency at all.
# ---------------------------------------------------------------------------

requires_soundfile = pytest.mark.skipif(
    importlib.util.find_spec("soundfile") is None, reason="soundfile is not installed"
)
requires_audio_stack = pytest.mark.skipif(
    importlib.util.find_spec("soundfile") is None
    or importlib.util.find_spec("librosa") is None,
    reason="soundfile and librosa are needed to run a real conversion",
)

MANIFEST_ONLY_ARCHIVES = ["GalGame/Studio_A.7z", "GalGame/Studio_B.7z"]


def forbid_acquisition(monkeypatch, *, extraction=True):
    """Make every credential read and every archive touch fail the test loudly.

    --manifest-only claims to need no HF token, no archive password and no
    archive.  The claim is only worth anything if breaking it is an error rather
    than a fallback, so the whole acquisition half is booby-trapped.
    """

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("--manifest-only touched the acquisition path")

    for name in ("read_hf_token", "read_archive_password", "find_archive_password",
                 "download_all", "list_repo_archives"):
        monkeypatch.setattr(vn, name, explode)
    if extraction:
        monkeypatch.setattr(vn, "extract_archive", explode)
        monkeypatch.setattr(vn, "extraction_password_need", explode)


def write_index(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )
    return root / "index.json"


def build_pruned_corpus(out_dir, clips_per_speaker=4, speakers=3):
    """The exact on-disk state a pruned batch leaves behind.

    ``raw/<stem>/index.json`` and nothing else in the raw tree; the converted
    wavs under ``audio/``; no archives directory at all.  Entries use the
    ``FilePath`` shape, so the (absent) sources would have been ``.ogg`` beside
    the index -- what pruning deletes.
    """
    for archive in MANIFEST_ONLY_ARCHIVES:
        stem = Path(archive).stem
        title = vn.slugify(stem, "unknown_title")
        entries = []
        for speaker_index in range(speakers):
            speaker = f"spk{speaker_index}"
            for voice_index in range(clips_per_speaker):
                voice = f"v{voice_index}"
                write_wav(
                    out_dir / "audio" / title / speaker / f"{voice}.wav",
                    1.0 + 0.25 * voice_index,
                )
                entries.append(
                    {
                        "Speaker": speaker,
                        "FilePath": f"{speaker}\\{voice}.ogg",
                        "Text": f"「{stem}の\\\"せりふ{voice_index}です……!?」",
                    }
                )
        write_index(out_dir / "raw" / stem, entries)
    return out_dir


def manifest_only_argv(out_dir, *extra):
    return [
        "--out-dir", str(out_dir),
        "--archives", *MANIFEST_ONLY_ARCHIVES,
        "--manifest-only", *extra,
    ]


# --- a real full build, for the equivalence tests ---------------------------
# The sources are addressed through the ``FilePath`` entry shape, which carries
# its own extension, so they can be the same stdlib-written wavs used
# everywhere else in this file: the real decoder runs over them without the
# fixture needing an ogg encoder.  Pruning then deletes them exactly as the
# cluster does.


def build_full_corpus(out_dir, include_out_of_range=False, speakers=3, clips=4):
    for archive in MANIFEST_ONLY_ARCHIVES:
        stem = Path(archive).stem
        root = out_dir / "raw" / stem
        entries = []
        for speaker_index in range(speakers):
            speaker = f"spk{speaker_index}"
            for voice_index in range(clips):
                voice = f"v{voice_index}"
                seconds = 1.0 + 0.25 * voice_index
                if include_out_of_range:
                    seconds = {0: 0.2, 1: 21.0}.get(voice_index, seconds)
                write_wav(root / speaker / f"{voice}.wav", seconds)
                entries.append(
                    {
                        "Speaker": speaker,
                        "FilePath": f"{speaker}\\{voice}.wav",
                        "Text": f"「{stem}の\\\"せりふ{voice_index}です……!?」",
                    }
                )
        # One entry the archive never shipped: dropped_missing_audio on both
        # sides, so the reclassification arithmetic has a non-zero baseline.
        entries.append(
            {"Speaker": "spk0", "FilePath": "spk0\\absent.wav", "Text": "ない"}
        )
        write_index(root, entries)
    return out_dir


def stub_extraction(monkeypatch):
    """Skip download and extraction only; planning and conversion stay real."""
    monkeypatch.setattr(vn, "read_hf_token", lambda: "")
    monkeypatch.setattr(vn, "extraction_password_need", lambda archive, raw_dir: "skip")
    monkeypatch.setattr(
        vn, "extract_archive", lambda archive, raw_dir, password: raw_dir / archive.stem
    )

    def download_all(archives, archive_dir, token, workers):
        archive_dir.mkdir(parents=True, exist_ok=True)
        placed = {}
        for remote in archives:
            local = archive_dir / Path(remote).name
            local.touch()
            placed[remote] = local
        return placed

    monkeypatch.setattr(vn, "download_all", download_all)
    # The conversion pool spawns fresh interpreters, which re-import the module
    # by name; this test file loads it from a path, so the directory holding it
    # has to be importable in the children too.
    monkeypatch.syspath_prepend(str(SCRIPT_PATH.parent))


def full_argv(out_dir, *extra):
    return [
        "--out-dir", str(out_dir),
        "--archives", *MANIFEST_ONLY_ARCHIVES,
        "--workers", "1", *extra,
    ]


def prune_corpus(out_dir):
    """Exactly what each cluster batch does once it has converted: drop the
    archives and every extracted source, keep ``raw/<stem>/index.json``."""
    shutil.rmtree(out_dir / "archives", ignore_errors=True)
    for path in sorted((out_dir / "raw").rglob("*"), reverse=True):
        if path.is_file() and path.name != "index.json":
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


# --- the CLI option ---------------------------------------------------------


def test_manifest_only_is_off_by_default():
    assert vn.parse_args([]).manifest_only is False


def test_manifest_only_is_a_flag():
    assert vn.parse_args(["--manifest-only"]).manifest_only is True


# --- locating the pruned inputs --------------------------------------------


def test_pruned_indexes_are_found_from_the_archive_names(tmp_path):
    raw_dir = tmp_path / "raw"
    for archive in MANIFEST_ONLY_ARCHIVES:
        write_index(raw_dir / Path(archive).stem, [])

    found = vn.locate_pruned_indexes(MANIFEST_ONLY_ARCHIVES, raw_dir)

    assert found == [
        ("Studio_A", raw_dir / "Studio_A" / "index.json"),
        ("Studio_B", raw_dir / "Studio_B" / "index.json"),
    ]


def test_pruned_indexes_are_found_in_the_nested_archive_layout(tmp_path):
    # The other real layout: everything one directory down.  Pruning keeps the
    # index where the archive put it, so the lookup has to descend too.
    raw_dir = tmp_path / "raw"
    write_index(raw_dir / "Studio_A" / "Studio_A", [])

    found = vn.locate_pruned_indexes(["GalGame/Studio_A.7z"], raw_dir)

    assert found == [("Studio_A", raw_dir / "Studio_A" / "Studio_A" / "index.json")]


def test_a_missing_index_json_is_fatal_rather_than_skipped(tmp_path):
    """Silently dropping an archive would emit a quietly smaller corpus.

    That is the same failure the extraction markers exist to prevent, and here
    it is worse: by this point the archive is deleted, so the omission cannot be
    reconstructed later from anything on disk.
    """
    raw_dir = tmp_path / "raw"
    write_index(raw_dir / "Studio_A", [])

    with pytest.raises(SystemExit) as excinfo:
        vn.locate_pruned_indexes(MANIFEST_ONLY_ARCHIVES, raw_dir)

    message = str(excinfo.value)
    assert "Studio_B" in message
    assert "1 of 2 archive(s)" in message


def test_the_missing_index_list_is_truncated_but_counted(tmp_path):
    archives = [f"GalGame/S{i:02d}.7z" for i in range(25)]

    with pytest.raises(SystemExit) as excinfo:
        vn.locate_pruned_indexes(archives, tmp_path / "raw")

    message = str(excinfo.value)
    assert f"+{25 - vn.MANIFEST_ONLY_MISSING_LOGGED} more" in message
    assert "25 of 25" in message


# --- planning against a pruned raw tree ------------------------------------


def test_planning_keeps_clips_whose_source_ogg_has_been_deleted(tmp_path):
    """The crux of the mode: the sources are gone by construction.

    Without this, every entry would be dropped as missing audio and the mode
    would rebuild an empty corpus from a perfectly good one.
    """
    index = write_index(
        tmp_path / "raw" / "Studio_A",
        [{"Speaker": "spk", "FilePath": "spk\\v0.ogg", "Text": "せりふ"}],
    )
    stats = vn.FilterStats()

    tasks = vn.plan_archive(
        index, "Studio_A", tmp_path / "audio", stats,
        vn.ArchiveStats(archive="Studio_A", title="Studio_A"), manifest_only=True,
    )

    assert [task.key for task in tasks] == ["Studio_A__spk__v0"]
    assert stats.missing_audio == 0


def test_planning_without_the_flag_still_requires_the_source(tmp_path):
    # The default path is unchanged: a missing ogg is still a missing clip.
    index = write_index(
        tmp_path / "raw" / "Studio_A",
        [{"Speaker": "spk", "FilePath": "spk\\v0.ogg", "Text": "せりふ"}],
    )
    stats = vn.FilterStats()

    tasks = vn.plan_archive(
        index, "Studio_A", tmp_path / "audio", stats,
        vn.ArchiveStats(archive="Studio_A", title="Studio_A"),
    )

    assert tasks == []
    assert stats.missing_audio == 1


def test_planning_with_the_flag_still_drops_empty_text(tmp_path):
    # Only the audio check is relaxed; every text filter still applies.
    index = write_index(
        tmp_path / "raw" / "Studio_A",
        [{"Speaker": "spk", "FilePath": "spk\\v0.ogg", "Text": "　"}],
    )
    stats = vn.FilterStats()

    tasks = vn.plan_archive(
        index, "Studio_A", tmp_path / "audio", stats,
        vn.ArchiveStats(archive="Studio_A", title="Studio_A"), manifest_only=True,
    )

    assert tasks == []
    assert stats.empty_text == 1


# --- probing an already-converted wav --------------------------------------


@requires_soundfile
def test_probing_reports_a_converted_wav_and_its_duration(tmp_path):
    wav = write_wav(tmp_path / "v0.wav", 2.0)

    dst, duration, status, detail = vn.probe_converted(str(wav), 0.5, 20.0)

    assert (dst, status, detail) == (str(wav), "ok", "")
    assert duration == pytest.approx(2.0)


@requires_soundfile
def test_probing_a_wav_that_was_never_converted_reports_missing(tmp_path):
    dst, duration, status, _ = vn.probe_converted(str(tmp_path / "gone.wav"), 0.5, 20.0)

    assert status == "missing"
    assert duration == 0.0


@requires_soundfile
@pytest.mark.parametrize(
    "seconds,expected", [(0.25, "too_short"), (21.0, "too_long")]
)
def test_probing_applies_the_same_duration_bounds_as_conversion(
    tmp_path, seconds, expected
):
    wav = write_wav(tmp_path / "v0.wav", seconds)

    assert vn.probe_converted(str(wav), 0.5, 20.0)[2] == expected


@requires_soundfile
def test_probing_an_unreadable_wav_is_an_error_not_a_silent_drop(tmp_path):
    # A conversion killed mid-write leaves a file that is not a wav; counting it
    # as merely "missing" would hide a truncated batch.
    broken = tmp_path / "v0.wav"
    broken.write_bytes(b"not a wav at all")

    assert vn.probe_converted(str(broken), 0.5, 20.0)[2] == "error"


# --- the mode end to end ----------------------------------------------------


@requires_soundfile
def test_manifest_only_rebuilds_from_a_pruned_corpus(tmp_path, monkeypatch, capsys):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)

    assert vn.main(manifest_only_argv(tmp_path)) == 0

    train = read_jsonl(tmp_path / "train.jsonl")
    val = read_jsonl(tmp_path / "val.jsonl")
    assert len(train) + len(val) == 24
    assert not (tmp_path / "archives").exists()


@requires_soundfile
def test_manifest_only_announces_the_mode_in_the_log(tmp_path, monkeypatch, capsys):
    """A rebuilt manifest is indistinguishable from a full build's on disk, so
    the run that produced it has to say which it was."""
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)

    vn.main(manifest_only_argv(tmp_path))

    out = capsys.readouterr().out
    assert "mode    : MANIFEST-ONLY" in out
    assert "no download, no extraction, no conversion" in out
    # The progress lines must not claim to be converting anything either.
    assert "convert:" not in out
    assert "scan   :" in out


@requires_soundfile
def test_manifest_only_records_the_mode_in_the_manifest(tmp_path, monkeypatch):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)

    vn.main(manifest_only_argv(tmp_path))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "--manifest-only" in manifest["manifest_only"]["note"]


@requires_soundfile
def test_manifest_only_needs_no_hf_token_or_archive_password(
    tmp_path, monkeypatch, capsys
):
    """The mode does no work that requires a credential, so it must demand none.

    Failing for a missing secret while doing nothing that needs it would strand
    the final split on a machine that legitimately has neither.
    """
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                 vn.ARCHIVE_PASSWORD_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vn, "HF_TOKEN_FILE", tmp_path / "absent-token")
    monkeypatch.setattr(vn, "ARCHIVE_PASSWORD_FILE", tmp_path / "absent-password")

    assert vn.main(manifest_only_argv(tmp_path)) == 0


@requires_soundfile
def test_an_index_entry_without_a_converted_wav_counts_as_missing_audio(
    tmp_path, monkeypatch
):
    build_pruned_corpus(tmp_path)
    # One clip that was in the index but never produced a wav -- the shape of an
    # upstream-missing ogg, and of a clip the full run dropped for duration.
    (tmp_path / "audio" / "Studio_A" / "spk0" / "v0.wav").unlink()
    forbid_acquisition(monkeypatch)

    vn.main(manifest_only_argv(tmp_path))

    stats = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert stats["filter_stats"]["dropped_missing_audio"] == 1
    assert stats["filter_stats"]["kept"] == 23
    assert stats["totals"]["clips"] + stats["totals"]["val_surplus_clips_dropped"] == 23


@requires_soundfile
def test_manifest_only_composes_with_the_split_options(tmp_path, monkeypatch):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)

    vn.main(manifest_only_argv(tmp_path, "--val-frac", "0.3", "--seed", "7"))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["val_fraction"] == 0.3
    assert manifest["config"]["seed"] == 7
    # The speaker-disjointness guarantee is not weakened by the mode.
    train_speakers = {r["key"].rsplit("__", 1)[0] for r in
                      read_jsonl(tmp_path / "train.jsonl")}
    val_speakers = {r["key"].rsplit("__", 1)[0] for r in
                    read_jsonl(tmp_path / "val.jsonl")}
    assert train_speakers.isdisjoint(val_speakers)


@requires_soundfile
def test_manifest_only_composes_with_pinned_val_keys(tmp_path, monkeypatch):
    """The 58-archive final split is exactly this: both flags, one invocation."""
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(tmp_path))
    pinned = [record["key"] for record in read_jsonl(tmp_path / "val.jsonl")]
    pins = tmp_path / "pins.txt"
    pins.write_text("".join(f"{key}\n" for key in pinned), encoding="utf-8")

    vn.main(manifest_only_argv(tmp_path, "--pin-val-keys", str(pins)))

    assert [r["key"] for r in read_jsonl(tmp_path / "val.jsonl")] == sorted(pinned)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["val_pin"]["keys_found"] == len(pinned)
    assert manifest["val_pin"]["keys_missing"] == 0
    assert "manifest_only" in manifest


# --- refusing to write nothing ----------------------------------------------


def test_manifest_only_fails_when_nothing_has_been_converted(tmp_path, monkeypatch):
    """Pointed at an unbuilt corpus it must not overwrite good outputs with an
    empty manifest -- and it must say so before doing any work."""
    build_pruned_corpus(tmp_path)
    shutil.rmtree(tmp_path / "audio")
    forbid_acquisition(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(manifest_only_argv(tmp_path))

    message = str(excinfo.value)
    assert "--manifest-only" in message
    assert "no converted wav" in message


def test_manifest_only_fails_when_the_audio_tree_holds_no_wav(tmp_path, monkeypatch):
    build_pruned_corpus(tmp_path)
    for wav in (tmp_path / "audio").rglob("*.wav"):
        wav.unlink()
    forbid_acquisition(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(manifest_only_argv(tmp_path))

    assert "no converted wav" in str(excinfo.value)


def test_the_empty_corpus_check_runs_before_any_output_is_written(
    tmp_path, monkeypatch
):
    build_pruned_corpus(tmp_path)
    shutil.rmtree(tmp_path / "audio")
    (tmp_path / "train.jsonl").write_text("PREVIOUS RUN\n", encoding="utf-8")
    forbid_acquisition(monkeypatch)

    with pytest.raises(SystemExit):
        vn.main(manifest_only_argv(tmp_path))

    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == "PREVIOUS RUN\n"


@requires_soundfile
def test_manifest_only_fails_when_no_index_entry_has_audio(tmp_path, monkeypatch):
    """A different failure from "nothing converted": there *is* audio, but none
    of it belongs to the archives named."""
    build_pruned_corpus(tmp_path)
    write_index(
        tmp_path / "raw" / "Studio_A",
        [{"Speaker": "ghost", "FilePath": "ghost\\v9.ogg", "Text": "だれもいない"}],
    )
    write_index(
        tmp_path / "raw" / "Studio_B",
        [{"Speaker": "ghost", "FilePath": "ghost\\v9.ogg", "Text": "だれもいない"}],
    )
    forbid_acquisition(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(manifest_only_argv(tmp_path))

    assert "would be empty" in str(excinfo.value)


# --- equivalence with a full build ------------------------------------------


@requires_audio_stack
def test_manifest_only_reproduces_the_manifest_a_full_run_produced(
    tmp_path, monkeypatch
):
    """The guarantee the mode exists to provide, end to end.

    A full run (real planning, real conversion, real split) is followed by the
    cluster's pruning -- archives deleted, every extracted source deleted,
    ``index.json`` kept -- and then a manifest-only rebuild.  The two
    ``train.jsonl`` / ``val.jsonl`` pairs must be byte-identical: if they are
    not, the final 1,000 h split is not the corpus that was converted.
    """
    full = build_full_corpus(tmp_path / "full")
    stub_extraction(monkeypatch)
    assert vn.main(full_argv(full)) == 0
    before = {
        name: (full / name).read_bytes() for name in ("train.jsonl", "val.jsonl")
    }

    prune_corpus(full)
    for name in before:
        (full / name).unlink()
    forbid_acquisition(monkeypatch)

    assert vn.main(manifest_only_argv(full)) == 0

    after = {
        name: (full / name).read_bytes() for name in ("train.jsonl", "val.jsonl")
    }
    assert after == before


@requires_audio_stack
def test_manifest_only_reproduces_a_pinned_split_after_pruning(tmp_path, monkeypatch):
    full = build_full_corpus(tmp_path / "full")
    stub_extraction(monkeypatch)
    vn.main(full_argv(full))
    pins = full / "pins.txt"
    pins.write_text(
        "".join(f"{r['key']}\n" for r in read_jsonl(full / "val.jsonl")),
        encoding="utf-8",
    )
    vn.main(full_argv(full, "--pin-val-keys", str(pins)))
    before = {
        name: (full / name).read_bytes() for name in ("train.jsonl", "val.jsonl")
    }

    prune_corpus(full)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(full, "--pin-val-keys", str(pins)))

    after = {
        name: (full / name).read_bytes() for name in ("train.jsonl", "val.jsonl")
    }
    assert after == before


@requires_audio_stack
def test_the_kept_clip_count_survives_pruning_though_the_reasons_shift(
    tmp_path, monkeypatch
):
    """Duration drops are reclassified, the corpus is not.

    A clip the full run dropped for being too short never produced a wav, and
    its source is gone, so manifest-only can only call it missing audio.  That
    moves counts between filter_stats reasons -- and must move nothing between
    kept and dropped.
    """
    full = build_full_corpus(tmp_path / "full", include_out_of_range=True)
    stub_extraction(monkeypatch)
    vn.main(full_argv(full))
    before = json.loads((full / "manifest.json").read_text(encoding="utf-8"))

    prune_corpus(full)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(full))
    after = json.loads((full / "manifest.json").read_text(encoding="utf-8"))

    assert before["filter_stats"]["dropped_too_short"] > 0
    assert before["filter_stats"]["dropped_too_long"] > 0
    assert after["filter_stats"]["dropped_too_short"] == 0
    assert after["filter_stats"]["dropped_too_long"] == 0
    assert after["filter_stats"]["dropped_missing_audio"] == (
        before["filter_stats"]["dropped_missing_audio"]
        + before["filter_stats"]["dropped_too_short"]
        + before["filter_stats"]["dropped_too_long"]
    )
    assert after["filter_stats"]["kept"] == before["filter_stats"]["kept"]
    assert after["totals"]["clips"] == before["totals"]["clips"]


# ---------------------------------------------------------------------------
# Resolving an index entry to the audio actually on disk
#
# The bug this section pins: an index entry was resolved to
# ``<Speaker>/<Voice>.ogg`` and nowhere else.  That layout is a property of
# *most* archives, not of the dataset, and assuming it silently discarded
# 54,420 clips (~90 h) of a 1,000 h build.  Silently, because every one of them
# landed in ``dropped_missing_audio``, which reads as an upstream dataset gap
# rather than as a bug here -- the audio was fully present in all four affected
# titles (.ogg count == index entry count).
#
# Three distinct real causes, one synthetic tree each below:
#   1. CIRCUS D.C. III P.P.  -- Voice already carries ".ogg" (16,214 clips);
#   2. NanaWind Haruoto      -- the directory is not the index's Speaker
#                               (23,478 clips);
#   3. Libido Soft Hinekure  -- Voice names no file under the speaker's
#                               directory at all (8,228 clips).
#
# The two properties that must NOT move while fixing that are pinned here too:
# a title that already resolved must resolve to exactly the same file (the
# existing corpora are fingerprinted), and the manifest's speaker must keep
# coming from the index, because cause 2 is precisely a directory name that
# disagrees with it and the speaker-disjoint split is keyed on that label.
# ---------------------------------------------------------------------------


def plan_tree(tmp_path, entries, files, *, title="Studio_A", manifest_only=False):
    """Plan one synthetic title.  ``files`` are paths relative to the index."""
    root = tmp_path / "raw" / title
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Planning only tests existence; conversion is a separate stage.
        path.touch()
    index = write_index(root, entries)
    stats = vn.FilterStats()
    tasks = vn.plan_archive(
        index,
        title,
        tmp_path / "audio",
        stats,
        vn.ArchiveStats(archive=title, title=title),
        manifest_only=manifest_only,
    )
    return tasks, stats, root


# --- strip_audio_suffix -----------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("DC3PP_0518_FREE_AOI_AOI01.ogg", "DC3PP_0518_FREE_AOI_AOI01"),
        ("clip.OGG", "clip"),  # compared case-folded
        ("clip.wav", "clip"),
        ("clip.mp3", "clip"),
        ("clip.flac", "clip"),
        ("clip.m4a", "clip"),
        ("clip", "clip"),
        # Only ONE extension comes off, so a doubled name loses one layer only.
        ("clip.ogg.ogg", "clip.ogg"),
        # Not extensions.  Path.stem would wrongly cut both of these, which is
        # why the helper matches against AUDIO_SUFFIXES instead.
        ("Mr.N", "Mr.N"),
        ("2.5", "2.5"),
        ("", ""),
    ],
)
def test_strip_audio_suffix_removes_one_audio_extension_only(name, expected):
    assert vn.strip_audio_suffix(name) == expected


# --- the layout that already worked must not move ---------------------------


def test_the_documented_layout_resolves_to_exactly_the_same_file(tmp_path):
    """The reproducibility guarantee: <Speaker>/<Voice>.ogg still wins outright.

    The existing corpora are fingerprinted, so a title that resolved 100% before
    must produce byte-identical output.  The decoy is the sharp edge: the same
    stem exists elsewhere in the title, and must lose to the literal path.
    """
    entries = [{"Speaker": "spk", "Voice": "v0", "Text": "せりふ"}]
    tasks, stats, root = plan_tree(
        tmp_path, entries, ["spk/v0.ogg", "elsewhere/v0.ogg"]
    )

    assert [task.src for task in tasks] == [str(root / "spk" / "v0.ogg")]
    assert stats.resolved_exact == 1
    assert (stats.resolved_extension_stripped, stats.resolved_stem_elsewhere) == (0, 0)
    assert stats.missing_audio == 0


def test_the_filepath_shape_also_resolves_at_its_literal_path(tmp_path):
    # Criminal Border's shape: a Windows-style relative path with the extension.
    entries = [{"Speaker": "琴子", "FilePath": "琴子\\ep2_kot0002.ogg", "Text": "せりふ"}]
    tasks, stats, root = plan_tree(tmp_path, entries, ["琴子/ep2_kot0002.ogg"])

    assert [task.src for task in tasks] == [str(root / "琴子" / "ep2_kot0002.ogg")]
    assert stats.resolved_exact == 1


# --- cause 1: Voice already carries the extension ---------------------------


def test_a_voice_that_already_carries_its_extension_resolves(tmp_path):
    """CIRCUS D.C. III P.P., 16,214 clips.

    ``Voice`` is ``...AOI01.ogg``, so appending ".ogg" looked for "....ogg.ogg".
    """
    entries = [
        {
            "Speaker": "AOI",
            "Voice": "DC3PP_0518_FREE_AOI_AOI01.ogg",
            "Text": "せりふ",
        }
    ]
    tasks, stats, root = plan_tree(
        tmp_path, entries, ["AOI/DC3PP_0518_FREE_AOI_AOI01.ogg"]
    )

    assert [task.src for task in tasks] == [
        str(root / "AOI" / "DC3PP_0518_FREE_AOI_AOI01.ogg")
    ]
    assert stats.missing_audio == 0
    assert stats.resolved_extension_stripped == 1


def test_the_doubled_extension_never_reaches_the_converted_wav_name(tmp_path):
    """The wav must be ``<voice>.wav``, not ``<voice>_ogg.wav``.

    ``Voice`` names the destination as well as the source, so an extension left
    on it would be slugified into the wav name and into the manifest key -- and
    --manifest-only, which recomputes both from the index alone, would then look
    for a file the full run never wrote.
    """
    entries = [{"Speaker": "AOI", "Voice": "clip01.ogg", "Text": "せりふ"}]
    tasks, _stats, _root = plan_tree(tmp_path, entries, ["AOI/clip01.ogg"])

    assert Path(tasks[0].dst).name == "clip01.wav"
    assert tasks[0].key == "Studio_A__AOI__clip01"


# --- cause 2: the directory is not the index's Speaker ----------------------


def test_audio_in_a_directory_other_than_the_speaker_resolves(tmp_path):
    """NanaWind Haruoto Alice Gram, 23,478 clips.

    The index says 女子生徒Ｂ; the file sits under 男子生徒ａ.  The stem is the
    only reliable link between the two.
    """
    entries = [{"Speaker": "女子生徒Ｂ", "Voice": "com6_mob01_6", "Text": "せりふ"}]
    tasks, stats, root = plan_tree(
        tmp_path, entries, ["男子生徒ａ/com6_mob01_6.ogg"]
    )

    assert [task.src for task in tasks] == [
        str(root / "男子生徒ａ" / "com6_mob01_6.ogg")
    ]
    assert stats.missing_audio == 0
    assert stats.resolved_stem_elsewhere == 1


def test_the_speaker_label_comes_from_the_index_not_the_directory(tmp_path):
    """The split depends on this, so it must not follow the file.

    Taking the speaker from the directory would relabel all 23,478 of cause 2's
    clips as 男子生徒ａ, silently re-partitioning the speaker-disjoint split --
    a change that shows up in no metric until val is measuring a voice train has
    already seen.
    """
    entries = [{"Speaker": "女子生徒Ｂ", "Voice": "com6_mob01_6", "Text": "せりふ"}]
    tasks, _stats, _root = plan_tree(
        tmp_path, entries, ["男子生徒ａ/com6_mob01_6.ogg"]
    )

    assert tasks[0].speaker == f"Studio_A/{vn.slugify('女子生徒Ｂ')}"
    assert "男子生徒" not in tasks[0].speaker
    assert "男子生徒" not in tasks[0].dst
    assert "男子生徒" not in tasks[0].key
    # And the split key derived from the record is the index's speaker.
    record = vn.build_record(tasks[0].key, tasks[0].text, tasks[0].dst, 1.0)
    record["speaker"] = tasks[0].speaker
    assert vn.speaker_group_key(record) == vn.slugify("女子生徒Ｂ")


# --- cause 3: no file of that name under any spelling tried -----------------


def test_a_stem_that_differs_only_in_case_resolves(tmp_path):
    """Case-insensitivity is the mechanism, tested independently of the diagnosis.

    Libido Soft (8,228 clips) is the one cause I could not settle against the
    real archives -- they are not on this machine, and its index's ``Voice``
    (``Mizuki001``) shares no stem with the sample of disk names I was given
    (``朔夜・京子/kyoko0305.ogg``).  So this pins what the resolver does about
    case, and ``filter_stats`` is what will say whether that was enough on the
    real title; the test below pins that anything genuinely absent still drops.
    """
    entries = [{"Speaker": "Mizuki", "Voice": "Mizuki001", "Text": "せりふ"}]
    # A different directory, so the case-insensitive filesystem this runs on
    # cannot resolve it at the literal path and mask the stem lookup.
    tasks, stats, root = plan_tree(tmp_path, entries, ["朔夜・京子/mizuki001.ogg"])

    assert [task.src for task in tasks] == [str(root / "朔夜・京子" / "mizuki001.ogg")]
    assert stats.resolved_stem_elsewhere == 1


def test_a_voice_naming_no_file_anywhere_is_still_dropped(tmp_path):
    """The resolver must not invent a match.

    A wrong clip is worse than a missing one: it trains CTC against a transcript
    that is certainly not what the audio says.
    """
    entries = [{"Speaker": "Mizuki", "Voice": "Mizuki001", "Text": "せりふ"}]
    tasks, stats, _root = plan_tree(tmp_path, entries, ["朔夜・京子/kyoko0305.ogg"])

    assert tasks == []
    assert stats.missing_audio == 1
    assert (stats.resolved_exact, stats.resolved_stem_elsewhere) == (0, 0)


def test_an_entry_naming_no_audio_at_all_is_dropped(tmp_path):
    entries = [{"Speaker": "spk", "Text": "せりふ"}]
    tasks, stats, _root = plan_tree(tmp_path, entries, ["spk/v0.ogg"])

    assert tasks == []
    assert stats.missing_audio == 1


# --- stem collisions --------------------------------------------------------


def test_a_stem_collision_prefers_the_file_under_the_index_speaker(tmp_path):
    """Two files, one stem: the entry's own speaker directory decides."""
    entries = [{"Speaker": "spk", "Voice": "v0.ogg", "Text": "せりふ"}]
    tasks, stats, root = plan_tree(
        tmp_path, entries, ["aaa_first_by_sort/v0.ogg", "spk/v0.ogg"]
    )

    assert [task.src for task in tasks] == [str(root / "spk" / "v0.ogg")]
    assert stats.stem_collisions == 1


def test_a_stem_collision_with_no_speaker_match_is_deterministic(tmp_path):
    """Sorted path order, so two machines build the same corpus.

    Filesystem walk order is not stable across machines or across a re-extract,
    so an arbitrary pick would make the corpus unreproducible -- exactly the
    property the fingerprints rely on.
    """
    entries = [{"Speaker": "spk", "Voice": "v0", "Text": "せりふ"}]
    files = ["zzz/v0.ogg", "aaa/v0.ogg", "mmm/v0.ogg"]
    tasks, stats, root = plan_tree(tmp_path, entries, files)

    assert [task.src for task in tasks] == [str(root / "aaa" / "v0.ogg")]
    assert stats.stem_collisions == 1
    # Same tree, planned again: same pick.
    again, _stats, _root = plan_tree(
        tmp_path / "again", entries, list(reversed(files))
    )
    assert Path(again[0].src).parent.name == "aaa"


# --- cost: one walk per title, never per entry ------------------------------


def test_the_on_disk_index_is_built_once_per_title(tmp_path, monkeypatch):
    """Not once per entry: a title holds ~25,000 files and there are 58 of them.

    Per-entry scanning would be quadratic -- ~625 million stat calls for one
    title -- which is the difference between a walk that costs seconds and a
    corpus build that never finishes.
    """
    builds = []
    original = vn.TitleAudioIndex._build
    monkeypatch.setattr(
        vn.TitleAudioIndex,
        "_build",
        lambda self: builds.append(self.root) or original(self),
    )
    # 50 entries, none of which resolves at its literal path.
    entries = [
        {"Speaker": "spk", "Voice": f"v{i}", "Text": "せりふ"} for i in range(50)
    ]
    tasks, stats, _root = plan_tree(
        tmp_path, entries, [f"other/v{i}.ogg" for i in range(50)]
    )

    assert len(tasks) == 50
    assert stats.resolved_stem_elsewhere == 50
    assert len(builds) == 1


def test_a_title_that_resolves_exactly_never_walks_its_tree(tmp_path, monkeypatch):
    # The lazy build is what keeps the fix free for the titles that were already
    # fine -- 54 of the 58 archives.
    builds = []
    monkeypatch.setattr(
        vn.TitleAudioIndex, "_build", lambda self: builds.append(self.root) or {}
    )
    entries = [
        {"Speaker": "spk", "Voice": f"v{i}", "Text": "せりふ"} for i in range(20)
    ]
    tasks, stats, _root = plan_tree(
        tmp_path, entries, [f"spk/v{i}.ogg" for i in range(20)]
    )

    assert len(tasks) == 20
    assert stats.resolved_exact == 20
    assert builds == []


# --- the counters reach the summary -----------------------------------------


def test_the_resolution_counters_are_reported_in_the_filter_summary():
    """One collapsed number is what hid the loss; four named ones is the fix."""
    summary = vn.FilterStats().as_dict()

    for name in (
        "resolved_audio_exact_path",
        "resolved_audio_extension_stripped",
        "resolved_audio_stem_match_elsewhere",
        "audio_stem_collisions",
        "dropped_missing_audio",
    ):
        assert name in summary


def test_every_entry_with_text_is_accounted_for_by_exactly_one_counter(tmp_path):
    """No clip may vanish between "resolved" and "dropped"."""
    entries = [
        {"Speaker": "spk", "Voice": "v0", "Text": "せりふ"},           # exact
        {"Speaker": "spk", "Voice": "v1.ogg", "Text": "せりふ"},       # extension
        {"Speaker": "spk", "Voice": "v2", "Text": "せりふ"},           # elsewhere
        {"Speaker": "spk", "Voice": "gone", "Text": "せりふ"},         # missing
        {"Speaker": "spk", "Voice": "v0", "Text": "……"},              # empty text
    ]
    _tasks, stats, _root = plan_tree(
        tmp_path, entries, ["spk/v0.ogg", "spk/v1.ogg", "other/v2.ogg"]
    )

    assert (stats.resolved_exact, stats.resolved_extension_stripped) == (1, 1)
    assert (stats.resolved_stem_elsewhere, stats.missing_audio) == (1, 1)
    assert stats.empty_text == 1
    assert stats.index_entries == 5


# --- --manifest-only is unaffected ------------------------------------------


def test_manifest_only_still_plans_without_touching_the_audio_tree(tmp_path):
    """The oggs are gone by construction there, so nothing is resolved on disk.

    The wav is what that mode checks, later, in probe_converted.
    """
    entries = [{"Speaker": "spk", "Voice": "v0.ogg", "Text": "せりふ"}]
    tasks, stats, _root = plan_tree(tmp_path, entries, [], manifest_only=True)

    assert [task.key for task in tasks] == ["Studio_A__spk__v0"]
    assert stats.missing_audio == 0
    assert (stats.resolved_exact, stats.resolved_stem_elsewhere) == (0, 0)


# ===========================================================================
# Kana-only titles (--drop-kana-only-titles)
# ===========================================================================
#
# Some titles transcribe their dialogue in hiragana where the audio plainly
# contains kanji -- the same clip reads "いちおうはでんとうとかくしきのあるめい
# もんがくえんで…" in the corpus and "一応は伝統と格式のある名門学園で…" to an
# in-domain teacher ASR model.  Neither is a transcription *error*, which is why
# no per-clip text filter can see it; what it does is teach the model to emit
# kana where kanji is expected.  Measured over 4,000 clips: 285 of the 2,872
# transcripts longer than 10 characters (9.9%) hold no kanji at all, at a median
# normalised CER of 0.417 against 0.103 for the rest -- two populations, not a
# tail.  JADE_Love_Destination is 100% kanji-free among its substantial lines.
#
# Two properties are pinned here and must not be loosened:
#   * the unit of the decision is the TITLE.  A short kana-only line ("うん",
#     "そうなんだ") is ordinary Japanese, and a per-clip rule would strip the
#     corpus of backchannels and interjections;
#   * the filter is OPT-IN.  A corpus built without the flag must be exactly
#     what it was, down to the manifest's filter_stats keys.

# A substantial line (> KANA_ONLY_MIN_CHARS characters) with no kanji in it.
KANA_ONLY_LINE = "いちおうはでんとうとかくしきのあるめいもんがくえんで"
# The same utterance as the rest of the corpus writes it.
KANJI_LINE = "一応は伝統と格式のある名門学園で"
# Backchannels: kana-only, but far too short to say anything about a title.
BACKCHANNEL_LINES = ["うん", "そうなんだ", "ふふっ"]


def plan_kana_title(tmp_path, texts, *, title="Kana_Title", **kwargs):
    """Plan a synthetic title whose transcripts are exactly ``texts``."""
    root = tmp_path / "raw" / title
    entries = []
    for index, text in enumerate(texts):
        source = root / "spk" / f"v{index}.ogg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        entries.append({"Speaker": "spk", "Voice": f"v{index}", "Text": text})
    index_path = write_index(root, entries)
    stats = vn.FilterStats()
    tasks = vn.plan_archive(
        index_path,
        title,
        tmp_path / "audio",
        stats,
        vn.ArchiveStats(archive=title, title=title),
        **kwargs,
    )
    return tasks, stats


# --- has_kanji --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("一応は伝統と格式のある名門学園で", True),
        ("いちおうはでんとうとかくしきのある", False),
        ("カタカナダケノセリフデス", False),
        ("うん", False),
        ("", False),
        ("ABC123", False),  # Latin is not kanji
        ("ーーー", False),  # a drawn-out vocalisation
        ("々", False),  # the repeat mark alone is not a kanji occurrence
        ("あ亜", True),
    ],
)
def test_has_kanji_recognises_only_kanji(text, expected):
    assert vn.has_kanji(text) is expected


# --- kana_only_fraction -----------------------------------------------------


def test_a_title_written_entirely_in_kana_measures_one():
    assert vn.kana_only_fraction([KANA_ONLY_LINE] * 5) == 1.0


def test_a_title_written_with_kanji_measures_zero():
    assert vn.kana_only_fraction([KANJI_LINE] * 5) == 0.0


def test_backchannels_do_not_count_as_evidence():
    """The reason the measurement has a length floor at all.

    Every title is full of "うん"; counting them would put an ordinary title's
    kanji-free fraction near a kana-only title's and destroy the separation the
    9.9% / median-CER measurement found.
    """
    assert vn.kana_only_fraction(BACKCHANNEL_LINES + [KANJI_LINE] * 2) == 0.0


def test_a_title_with_no_substantial_transcript_is_unmeasurable():
    # None, not 0.0 and not 1.0: the fraction is undefined over an empty
    # denominator, and a title too small to measure must not be dropped on a
    # measurement that was never taken.
    assert vn.kana_only_fraction(BACKCHANNEL_LINES) is None
    assert vn.kana_only_fraction([]) is None


@pytest.mark.parametrize("length,counted", [(10, False), (11, True)])
def test_the_length_floor_is_strictly_greater_than_ten(length, counted):
    # 10 is KANA_ONLY_MIN_CHARS, and the measurement that produced the 0.8
    # default was taken over transcripts *longer than* 10 characters (2,872 of
    # them).  A line of exactly 10 is below the floor, so it cannot vote.
    assert vn.KANA_ONLY_MIN_CHARS == 10
    line = "あ" * length

    assert vn.kana_only_fraction([line]) == (1.0 if counted else None)


def test_the_fraction_is_over_substantial_lines_only():
    # 2 of 4 substantial lines are kanji-free; the backchannels are invisible.
    texts = [KANA_ONLY_LINE, KANA_ONLY_LINE, KANJI_LINE, KANJI_LINE]

    assert vn.kana_only_fraction(texts + BACKCHANNEL_LINES) == 0.5


# --- the title-level filter -------------------------------------------------


def test_a_kana_only_title_is_dropped_whole(tmp_path):
    tasks, stats = plan_kana_title(
        tmp_path,
        [KANA_ONLY_LINE] * 4 + BACKCHANNEL_LINES,
        drop_kana_only_titles=True,
    )

    assert tasks == []
    assert stats.kana_only_titles == ["Kana_Title"]
    # Every clip of the title, backchannels included: the title goes, not the
    # lines that triggered the verdict.
    assert stats.kana_only_title_clips == 7


def test_an_ordinary_title_survives_the_filter(tmp_path):
    tasks, stats = plan_kana_title(
        tmp_path,
        [KANJI_LINE] * 4 + BACKCHANNEL_LINES,
        drop_kana_only_titles=True,
    )

    assert len(tasks) == 7
    assert stats.kana_only_titles == []
    assert stats.kana_only_title_clips == 0


def test_a_title_of_only_backchannels_is_kept(tmp_path):
    """Nothing measurable, so nothing to act on.

    A title whose every line is a short interjection is unusual, but dropping it
    on an undefined fraction would be dropping it on no evidence.
    """
    tasks, stats = plan_kana_title(
        tmp_path, BACKCHANNEL_LINES, drop_kana_only_titles=True
    )

    assert len(tasks) == 3
    assert stats.kana_only_titles == []


def test_individual_kana_only_clips_are_never_dropped(tmp_path):
    """The rule the module docstring is emphatic about.

    A kana-only line inside a normal title is ordinary Japanese -- unvoiced
    kanji, a backchannel, an interjection -- and dropping those would bias the
    corpus against exactly the utterances an ASR model finds hardest.
    """
    tasks, stats = plan_kana_title(
        tmp_path,
        [KANJI_LINE] * 4 + [KANA_ONLY_LINE] + BACKCHANNEL_LINES,
        drop_kana_only_titles=True,
    )

    assert sorted(task.text for task in tasks) == sorted(
        [KANJI_LINE] * 4 + [KANA_ONLY_LINE] + BACKCHANNEL_LINES
    )
    assert stats.kana_only_title_clips == 0


@pytest.mark.parametrize(
    "kana_lines,kanji_lines,dropped",
    [
        (4, 1, False),  # exactly 0.8 -- at the threshold, not above it
        (5, 1, True),   # 0.833
        (5, 0, True),   # 1.0, i.e. JADE_Love_Destination
        (1, 4, False),  # 0.2, an ordinary title
    ],
)
def test_the_threshold_is_an_exclusive_bound(
    tmp_path, kana_lines, kanji_lines, dropped
):
    # 0.8 is the default because the affected titles measure ~1.0 while ordinary
    # ones sit far below (9.9% corpus-wide), so the bound is deliberately far
    # from both populations.  Exclusive, so a title sitting exactly on it stays:
    # the drop must be justified by evidence past the bound, not at it.
    assert vn.KANA_ONLY_TITLE_THRESHOLD == 0.8
    texts = [KANA_ONLY_LINE] * kana_lines + [KANJI_LINE] * kanji_lines
    tasks, stats = plan_kana_title(tmp_path, texts, drop_kana_only_titles=True)

    assert (tasks == []) is dropped
    assert (stats.kana_only_titles == ["Kana_Title"]) is dropped


def test_the_threshold_is_configurable(tmp_path):
    texts = [KANA_ONLY_LINE, KANJI_LINE, KANJI_LINE, KANJI_LINE]  # 0.25

    kept, _ = plan_kana_title(tmp_path, texts, drop_kana_only_titles=True)
    dropped, stats = plan_kana_title(
        tmp_path / "strict",
        texts,
        drop_kana_only_titles=True,
        kana_only_threshold=0.2,
    )

    assert len(kept) == 4
    assert dropped == []
    assert stats.kana_only_titles == ["Kana_Title"]


def test_the_excluded_title_is_named_in_the_log(tmp_path, capsys):
    # An excluded title is otherwise indistinguishable in the output from one
    # that was never passed to --archives, which is the failure mode the
    # resolution counters were added for.
    plan_kana_title(tmp_path, [KANA_ONLY_LINE] * 3, drop_kana_only_titles=True)

    log = capsys.readouterr().out
    assert "Kana_Title" in log
    assert "no kanji" in log


# --- opt-in: the default path is untouched ----------------------------------


def test_without_the_flag_a_kana_only_title_is_kept_in_full(tmp_path):
    """The no-op guarantee, stated against the worst case.

    A training run is scored against corpora built before this filter existed,
    so the default must reproduce them: even a 100% kana-only title keeps every
    clip unless the flag is passed.
    """
    tasks, stats = plan_kana_title(tmp_path, [KANA_ONLY_LINE] * 5)

    assert len(tasks) == 5
    assert stats.kana_only_titles == []
    assert stats.kana_only_title_clips == 0
    assert stats.kana_only_filter_applied is False


def test_the_filter_summary_gains_no_keys_when_the_filter_is_off(tmp_path):
    # manifest.json's filter_stats must be byte-for-byte what it was, so the
    # counters may not appear as zeroes on a default run.
    _tasks, stats = plan_kana_title(tmp_path, [KANA_ONLY_LINE] * 5)

    summary = stats.as_dict()

    assert "dropped_kana_only_title_clips" not in summary
    assert "excluded_kana_only_titles" not in summary


def test_the_filter_summary_reports_the_exclusions_when_it_is_on(tmp_path):
    _tasks, stats = plan_kana_title(
        tmp_path, [KANA_ONLY_LINE] * 5, drop_kana_only_titles=True
    )

    summary = stats.as_dict()

    assert summary["excluded_kana_only_titles"] == ["Kana_Title"]
    assert summary["dropped_kana_only_title_clips"] == 5
    # Reported alongside the existing counters, not instead of them.
    assert summary["index_entries"] == 5
    assert summary["kept"] == 0


def test_an_engaged_filter_that_excludes_nothing_still_says_so(tmp_path):
    # An empty list is a measurement; an absent key is "not measured".  The two
    # must not be confused when reading a manifest.
    _tasks, stats = plan_kana_title(
        tmp_path, [KANJI_LINE] * 5, drop_kana_only_titles=True
    )

    summary = stats.as_dict()

    assert summary["excluded_kana_only_titles"] == []
    assert summary["dropped_kana_only_title_clips"] == 0


# --- the CLI ----------------------------------------------------------------


def test_drop_kana_only_titles_defaults_to_off():
    assert vn.parse_args([]).drop_kana_only_titles is False
    assert vn.parse_args(["--drop-kana-only-titles"]).drop_kana_only_titles is True


def test_the_threshold_defaults_to_none_so_it_stays_out_of_the_manifest():
    # Same trick as --list-format: None (not 0.8) is what makes "passed without
    # --drop-kana-only-titles" detectable instead of silently ineffective.
    assert vn.parse_args([]).kana_only_title_threshold is None
    parsed = vn.parse_args(["--kana-only-title-threshold", "0.5"])
    assert parsed.kana_only_title_threshold == 0.5


def test_the_threshold_without_the_flag_is_an_error(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("pipeline started despite an inapplicable flag")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(["--kana-only-title-threshold", "0.5"])

    assert "--kana-only-title-threshold" in str(excinfo.value)


@pytest.mark.parametrize("value", ["0", "0.0", "1.5", "-0.2"])
def test_a_threshold_outside_zero_to_one_is_rejected(monkeypatch, value):
    # A fraction of clips can only live in (0, 1].  0 would drop every title
    # that holds a single kanji-free line; above 1 the filter can never fire and
    # the run would silently be an unfiltered one.
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("pipeline started despite an invalid threshold")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(["--drop-kana-only-titles", "--kana-only-title-threshold", value])

    assert "(0, 1]" in str(excinfo.value)


# --- the manifest -----------------------------------------------------------


def test_a_default_run_records_no_kana_filter_in_the_manifest():
    records = corpus()
    train, val = vn.split_by_speaker(records, 0.2, seed=0)
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))

    manifest = vn.build_manifest(vn.parse_args([]), {}, stats, train, val, 0)

    assert "drop_kana_only_titles" not in manifest["config"]
    assert "kana_only_title_threshold" not in manifest["config"]


def test_a_filtered_run_records_the_threshold_it_used():
    # The title list in filter_stats is only reproducible against the threshold
    # it was taken at, so the two travel together.
    records = corpus()
    train, val = vn.split_by_speaker(records, 0.2, seed=0)
    args = vn.parse_args(
        ["--drop-kana-only-titles", "--kana-only-title-threshold", "0.6"]
    )
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))
    stats.kana_only_filter_applied = True
    stats.kana_only_titles.append("JADE_Love_Destination")

    manifest = vn.build_manifest(args, {}, stats, train, val, 0)

    assert manifest["config"]["drop_kana_only_titles"] is True
    assert manifest["config"]["kana_only_title_threshold"] == 0.6
    assert manifest["config"]["kana_only_min_chars"] == vn.KANA_ONLY_MIN_CHARS
    assert manifest["filter_stats"]["excluded_kana_only_titles"] == [
        "JADE_Love_Destination"
    ]


def test_the_default_threshold_reaches_the_manifest_when_unset():
    records = corpus()
    train, val = vn.split_by_speaker(records, 0.2, seed=0)
    args = vn.parse_args(["--drop-kana-only-titles"])
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))

    manifest = vn.build_manifest(args, {}, stats, train, val, 0)

    assert manifest["config"]["kana_only_title_threshold"] == 0.8


# ---------------------------------------------------------------------------
# Per-clip emotion targets (--emo-labels)
#
# The bug this section pins: every record's ``emo_target`` was the hardcoded
# ``<|NEUTRAL|>``, so rounds 1 and 2 trained the emotion head against a
# constant and collapsed it -- the head learned the constant and nothing else.
# Round 3 joins real per-clip pseudo-labels in from a file the labeller writes.
#
# Three properties are pinned here and must not be loosened:
#   * a clip the label file does NOT mention gets ``<|SER|>``, never
#     ``<|NEUTRAL|>``.  Stamping neutral on an unmeasured clip is the same
#     mistake as before, only quieter; the sentinel is what model.py rewrites
#     to ignore_id, so the clip trains CTC and skips the emotion head entirely;
#   * the flag is OPT-IN and the default path is byte-identical.  The round-2
#     manifests are reproducible only while a run without --emo-labels stamps
#     ``<|NEUTRAL|>`` on everything, exactly as it always did;
#   * every rejection is loud.  A label file is a training target for each clip
#     it touches, and this repo has already lost a round to a data-prep bug
#     that failed silently.
# ---------------------------------------------------------------------------

# Two clips of the pruned fixture corpus, spelled the way plan_archive builds a
# key: "<title>__<speaker_slug>__<voice_slug>".
EMO_KEY_A = "Studio_A__spk0__v0"
EMO_KEY_B = "Studio_A__spk0__v1"


def write_emo_labels(path, entries):
    """Write a --emo-labels JSONL file from ``[{...}, ...]`` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return path


def emo_labels_for(path, mapping):
    """The common case: a plain ``{key: emo_target}`` file."""
    return write_emo_labels(
        path, [{"key": key, "emo_target": target} for key, target in mapping.items()]
    )


def emo_records(mapping):
    """Records whose keys are exactly ``mapping``'s, for the applier tests."""
    return [
        vn.build_record(key, "こんにちは", f"/vn/audio/t/s/{index}.wav", 2.0)
        for index, key in enumerate(mapping)
    ]


def all_emo_targets(path):
    return {record["emo_target"] for record in read_jsonl(path)}


# --- the tokens themselves --------------------------------------------------


def test_the_mask_sentinel_is_ser_and_not_neutral():
    """The whole point of the flag: an unlabelled clip is masked, not asserted
    neutral.  model.py maps this exact token (id 24991) to ignore_id."""
    assert vn.EMO_MASK_TARGET == "<|SER|>"
    assert vn.EMO_MASK_TARGET != vn.EMO_TARGET


def test_the_seven_emotion_tokens_are_exactly_sensevoices():
    assert set(vn.EMO_LABEL_TARGETS) == {
        "<|HAPPY|>",
        "<|SAD|>",
        "<|ANGRY|>",
        "<|NEUTRAL|>",
        "<|FEARFUL|>",
        "<|DISGUSTED|>",
        "<|SURPRISED|>",
    }


def test_emo_unknown_is_not_an_acceptable_target():
    # <|EMO_UNKNOWN|> is the "no prediction" token; webui.py maps it to the
    # empty string at inference.  As a *target* it teaches the model to predict
    # "unknown", which is worse than masking the clip.
    assert vn.EMO_UNKNOWN_TARGET not in vn.EMO_ALL_TARGETS


def test_the_eight_possible_targets_are_the_seven_plus_the_mask():
    assert set(vn.EMO_ALL_TARGETS) == set(vn.EMO_LABEL_TARGETS) | {vn.EMO_MASK_TARGET}
    assert len(vn.EMO_ALL_TARGETS) == 8


def test_the_default_emotion_target_is_still_neutral():
    # Pins round-2 reproducibility at the constant itself: build_record's
    # default is what a run without --emo-labels writes on every clip.
    assert vn.EMO_TARGET == "<|NEUTRAL|>"


# --- reading the label file -------------------------------------------------


def test_the_label_file_is_one_json_object_per_line(tmp_path):
    path = emo_labels_for(
        tmp_path / "emo.jsonl", {EMO_KEY_A: "<|HAPPY|>", EMO_KEY_B: "<|SAD|>"}
    )

    labels = vn.read_emo_labels(path)

    assert labels.targets == {EMO_KEY_A: "<|HAPPY|>", EMO_KEY_B: "<|SAD|>"}


@pytest.mark.parametrize("target", vn.EMO_ALL_TARGETS if vn else [])
def test_every_one_of_the_eight_targets_is_accepted(tmp_path, target):
    path = emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: target})

    assert vn.read_emo_labels(path).targets == {EMO_KEY_A: target}


def test_unknown_fields_are_ignored_rather_than_rejected(tmp_path):
    """The labeller records its own provenance and this script must not have to
    track its schema -- only 'key' and 'emo_target' are contractual."""
    path = write_emo_labels(
        tmp_path / "emo.jsonl",
        [
            {
                "key": EMO_KEY_A,
                "emo_target": "<|ANGRY|>",
                "decision": "audio_conf",
                "audio_label": "angry",
                "audio_score": 0.87,
                "text_label": "neutral",
                "something_invented_next_quarter": {"nested": [1, 2]},
            },
            # merge_emo_labels.MergedRow.to_json emits null for a labeller that
            # had nothing to say about the clip, so nulls are the ordinary wire
            # shape and must not be mistaken for a malformed row.
            {
                "key": EMO_KEY_B,
                "emo_target": "<|SER|>",
                "decision": "missing_masked",
                "audio_label": None,
                "audio_score": None,
                "text_label": None,
            },
        ],
    )

    assert vn.read_emo_labels(path).targets == {
        EMO_KEY_A: "<|ANGRY|>",
        EMO_KEY_B: "<|SER|>",
    }


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "emo.jsonl"
    path.write_text(
        f'\n{{"key": "{EMO_KEY_A}", "emo_target": "<|SAD|>"}}\n\n  \n',
        encoding="utf-8",
    )

    assert vn.read_emo_labels(path).targets == {EMO_KEY_A: "<|SAD|>"}


def test_the_file_digest_is_the_sha256_of_its_bytes(tmp_path):
    """The labeller is re-run and overwrites the same path, so the manifest has
    to say which revision of that path the corpus was built from."""
    import hashlib

    path = emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|HAPPY|>"})

    labels = vn.read_emo_labels(path)

    assert labels.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert labels.path == str(path)


# --- loud failures ----------------------------------------------------------


def test_a_missing_label_file_fails_loudly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(tmp_path / "absent.jsonl")

    assert "--emo-labels" in str(excinfo.value)
    assert "cannot read" in str(excinfo.value)


def test_a_malformed_line_names_its_line_number(tmp_path):
    path = tmp_path / "emo.jsonl"
    path.write_text(
        f'{{"key": "{EMO_KEY_A}", "emo_target": "<|SAD|>"}}\n'
        f'{{"key": "{EMO_KEY_B}", "emo_target": ...\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    message = str(excinfo.value)
    assert "malformed JSON line" in message
    # The line number, not just the file: a labeller writing 500k lines is only
    # debuggable if the bad one is named.
    assert f"{path}:2" in message


def test_a_duplicate_key_is_fatal(tmp_path):
    path = write_emo_labels(
        tmp_path / "emo.jsonl",
        [
            {"key": EMO_KEY_A, "emo_target": "<|HAPPY|>"},
            {"key": EMO_KEY_A, "emo_target": "<|SAD|>"},
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    message = str(excinfo.value)
    assert "duplicate key" in message
    assert EMO_KEY_A in message


def test_a_target_outside_the_eight_tokens_is_fatal(tmp_path):
    path = emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|EXCITED|>"})

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    message = str(excinfo.value)
    # Both the offending key and its value, so the bad line is findable.
    assert EMO_KEY_A in message
    assert "<|EXCITED|>" in message


@pytest.mark.parametrize("target", ["NEUTRAL", "<|neutral|>", "happy", ""])
def test_a_near_miss_spelling_is_not_quietly_accepted(tmp_path, target):
    # A token that is close but not exact would be tokenised as ordinary text
    # and train the emotion slot on garbage, so no fuzzy matching is applied.
    path = emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: target})

    with pytest.raises(SystemExit):
        vn.read_emo_labels(path)


def test_emo_unknown_is_rejected_by_name(tmp_path):
    """Rejected with its own explanation, not just as "not in the list".

    It is the one wrong value a well-meaning labeller will actually produce --
    it is a real SenseVoice token and the obvious thing to emit for a clip the
    classifier could not call -- so the error has to say what to do instead.
    """
    path = emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|EMO_UNKNOWN|>"})

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    message = str(excinfo.value)
    assert "<|EMO_UNKNOWN|>" in message
    assert "never be a training target" in message
    assert vn.EMO_MASK_TARGET in message


@pytest.mark.parametrize(
    "entry",
    [
        {"emo_target": "<|SAD|>"},
        {"key": EMO_KEY_A},
        {},
    ],
    ids=["no-key", "no-target", "empty"],
)
def test_a_line_missing_a_required_field_is_fatal(tmp_path, entry):
    path = write_emo_labels(tmp_path / "emo.jsonl", [entry])

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    assert "missing required field" in str(excinfo.value)


@pytest.mark.parametrize(
    "entry",
    [{"key": 7, "emo_target": "<|SAD|>"}, {"key": EMO_KEY_A, "emo_target": 3}],
    ids=["key", "emo_target"],
)
def test_a_non_string_field_is_fatal(tmp_path, entry):
    path = write_emo_labels(tmp_path / "emo.jsonl", [entry])

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    assert "must be a string" in str(excinfo.value)


def test_a_json_line_that_is_not_an_object_is_fatal(tmp_path):
    path = tmp_path / "emo.jsonl"
    path.write_text('["Studio_A__spk0__v0", "<|SAD|>"]\n', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    assert "expected a JSON object per line" in str(excinfo.value)


def test_a_label_file_with_no_labels_fails_loudly(tmp_path):
    path = tmp_path / "emo.jsonl"
    path.write_text("\n\n   \n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        vn.read_emo_labels(path)

    assert "holds no labels" in str(excinfo.value)


# --- stamping the records ---------------------------------------------------


def test_a_listed_clip_takes_the_files_target(tmp_path):
    mapping = {EMO_KEY_A: "<|HAPPY|>", EMO_KEY_B: "<|ANGRY|>"}
    records = emo_records(mapping)
    labels = vn.read_emo_labels(emo_labels_for(tmp_path / "emo.jsonl", mapping))

    vn.apply_emo_labels(records, labels)

    assert [r["emo_target"] for r in records] == ["<|HAPPY|>", "<|ANGRY|>"]


def test_an_unlisted_clip_is_masked_and_never_called_neutral(tmp_path):
    """The regression this whole flag exists to prevent, at its narrowest."""
    records = emo_records({EMO_KEY_A: "", EMO_KEY_B: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|SAD|>"})
    )

    vn.apply_emo_labels(records, labels)

    assert records[0]["emo_target"] == "<|SAD|>"
    assert records[1]["emo_target"] == vn.EMO_MASK_TARGET
    assert records[1]["emo_target"] != "<|NEUTRAL|>"


def test_an_explicitly_masked_row_matches_an_omitted_one(tmp_path):
    """The labeller emits <|SER|> outright for a clip it could not call (its
    'missing_masked' / 'disagree_masked' decisions), so the two spellings of
    "no label" have to produce the same record."""
    records = emo_records({EMO_KEY_A: "", EMO_KEY_B: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: vn.EMO_MASK_TARGET})
    )

    # allow_sparse: this file labels nothing at all, which the coverage floor
    # would otherwise refuse.  What is under test here is the stamping.
    vn.apply_emo_labels(records, labels, allow_sparse=True)

    assert records[0]["emo_target"] == records[1]["emo_target"] == vn.EMO_MASK_TARGET


def test_an_explicit_mask_still_counts_as_a_matched_key(tmp_path):
    # It matched a clip; it just carries no emotion.  labelled_clips, not
    # keys_matched, is what says how much the emotion head sees.
    mapping = {EMO_KEY_A: vn.EMO_MASK_TARGET, EMO_KEY_B: "<|SAD|>"}
    labels = vn.read_emo_labels(emo_labels_for(tmp_path / "emo.jsonl", mapping))
    records = emo_records(mapping)
    vn.apply_emo_labels(records, labels)

    stats = vn.build_emo_label_stats(labels, records, [])

    assert stats.keys_matched == 2
    assert stats.labelled_clips == 1


def test_the_matched_keys_are_returned(tmp_path):
    records = emo_records({EMO_KEY_A: "", EMO_KEY_B: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(
            tmp_path / "emo.jsonl",
            {EMO_KEY_A: "<|SAD|>", "Studio_Z__ghost__v9": "<|HAPPY|>"},
        )
    )

    assert vn.apply_emo_labels(records, labels) == {EMO_KEY_A}


def test_stamping_keeps_the_field_in_its_schema_position(tmp_path):
    # json.dumps preserves insertion order and the shipped examples are compared
    # field-for-field, so the stamp must overwrite emo_target in place rather
    # than re-adding it at the end.
    records = emo_records({EMO_KEY_A: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|SURPRISED|>"})
    )

    vn.apply_emo_labels(records, labels)

    assert list(records[0]) == EXPECTED_FIELDS


def test_a_label_file_matching_nothing_is_fatal(tmp_path):
    """Zero overlap means the labels were built against a different corpus.

    Masking every clip and carrying on would produce a run that looks healthy,
    trains no emotion head at all, and is only distinguishable from a good one
    by reading the manifest -- exactly the silent failure mode this file's
    other sections were written after.
    """
    records = emo_records({EMO_KEY_A: "", EMO_KEY_B: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {"Other_Corpus__spk__v0": "<|SAD|>"})
    )

    with pytest.raises(SystemExit) as excinfo:
        vn.apply_emo_labels(records, labels)

    message = str(excinfo.value)
    assert "--emo-labels" in message
    assert "different corpus" in message


def test_a_single_matching_clip_is_enough_to_proceed(tmp_path):
    # The guard is against a *wholly* mismatched file; a corpus that grew since
    # the labels were written is ordinary and must still build.
    records = emo_records({EMO_KEY_A: "", EMO_KEY_B: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(
            tmp_path / "emo.jsonl", {EMO_KEY_A: "<|SAD|>", "absent__spk__v0": "<|SAD|>"}
        )
    )

    assert vn.apply_emo_labels(records, labels) == {EMO_KEY_A}


# --- the statistics block ---------------------------------------------------


def emo_stats_over(tmp_path, train_targets, val_targets):
    """An EmoLabelStats over synthetic splits carrying exactly these targets."""
    mapping = {}
    train, val = [], []
    for index, target in enumerate(train_targets + val_targets):
        key = f"T__spk{index}__v{index}"
        record = vn.build_record(key, "あ", f"/vn/audio/T/spk{index}/v.wav", 2.0)
        record["emo_target"] = target
        if target != vn.EMO_MASK_TARGET:
            mapping[key] = target
        (train if index < len(train_targets) else val).append(record)
    labels = vn.read_emo_labels(emo_labels_for(tmp_path / "emo.jsonl", mapping))
    return vn.build_emo_label_stats(labels, train, val)


def test_the_stats_count_every_target_in_each_split(tmp_path):
    stats = emo_stats_over(
        tmp_path,
        ["<|HAPPY|>", "<|HAPPY|>", "<|SAD|>", vn.EMO_MASK_TARGET],
        ["<|ANGRY|>", vn.EMO_MASK_TARGET],
    )

    block = stats.as_dict()
    assert block["train"]["counts"]["<|HAPPY|>"] == 2
    assert block["train"]["counts"]["<|SAD|>"] == 1
    assert block["train"]["counts"][vn.EMO_MASK_TARGET] == 1
    assert block["val"]["counts"]["<|ANGRY|>"] == 1
    assert block["val"]["counts"][vn.EMO_MASK_TARGET] == 1
    assert block["train"]["clips"] == 4
    assert block["val"]["clips"] == 2


def test_all_eight_targets_are_listed_even_at_zero(tmp_path):
    # A block whose keys depend on what happened to occur cannot be diffed
    # between two rounds, which is the one thing this block is written for.
    stats = emo_stats_over(tmp_path, ["<|HAPPY|>"], ["<|HAPPY|>"])

    block = stats.as_dict()
    for split in ("train", "val"):
        assert list(block[split]["counts"]) == list(vn.EMO_ALL_TARGETS)
        assert list(block[split]["fractions"]) == list(vn.EMO_ALL_TARGETS)
    assert block["train"]["counts"]["<|FEARFUL|>"] == 0


def test_the_fractions_are_over_that_splits_clips(tmp_path):
    stats = emo_stats_over(
        tmp_path, ["<|HAPPY|>", "<|SAD|>", "<|SAD|>", "<|SAD|>"], ["<|ANGRY|>"]
    )

    block = stats.as_dict()
    assert block["train"]["fractions"]["<|SAD|>"] == 0.75
    assert block["train"]["fractions"]["<|HAPPY|>"] == 0.25
    assert block["val"]["fractions"]["<|ANGRY|>"] == 1.0
    assert sum(block["train"]["fractions"].values()) == pytest.approx(1.0)


def test_the_masked_clips_are_excluded_from_labelled_clips(tmp_path):
    stats = emo_stats_over(
        tmp_path, ["<|HAPPY|>", vn.EMO_MASK_TARGET, vn.EMO_MASK_TARGET], ["<|SAD|>"]
    )

    block = stats.as_dict()
    assert block["train"]["labelled_clips"] == 1
    assert block["val"]["labelled_clips"] == 1
    assert stats.labelled_clips == 2


def test_keys_that_matched_nothing_are_counted(tmp_path):
    mapping = {EMO_KEY_A: "<|SAD|>", "gone__spk__v0": "<|HAPPY|>"}
    labels = vn.read_emo_labels(emo_labels_for(tmp_path / "emo.jsonl", mapping))
    records = emo_records({EMO_KEY_A: ""})
    vn.apply_emo_labels(records, labels)

    stats = vn.build_emo_label_stats(labels, records, [])

    block = stats.as_dict()
    assert block["keys_in_file"] == 2
    assert block["keys_matched"] == 1
    assert block["keys_matched_nothing"] == 1


def test_the_stats_record_the_file_and_its_digest(tmp_path):
    path = emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|SAD|>"})
    labels = vn.read_emo_labels(path)
    records = emo_records({EMO_KEY_A: ""})
    vn.apply_emo_labels(records, labels)

    block = vn.build_emo_label_stats(labels, records, []).as_dict()

    assert block["labels_file"] == str(path)
    assert block["labels_sha256"] == labels.sha256
    assert block["mask_target"] == vn.EMO_MASK_TARGET


# --- the degenerate-distribution warning ------------------------------------


@pytest.mark.parametrize(
    "happy,others,degenerate",
    [
        # The bound is EMO_DEGENERATE_FRACTION (0.9) and it is EXCLUSIVE: 90/100
        # is the documented ceiling of an acceptable skew, 91/100 is over it.
        (90, 10, False),
        (91, 9, True),
        (100, 0, True),
        (50, 50, False),
    ],
)
def test_the_warning_bound_is_exclusive(tmp_path, happy, others, degenerate):
    stats = emo_stats_over(
        tmp_path, ["<|HAPPY|>"] * happy + ["<|SAD|>"] * others, ["<|SAD|>"] * 0
    )

    dominant = stats.dominant_label()

    assert (dominant is not None) is degenerate
    if degenerate:
        assert dominant[0] == "<|HAPPY|>"


def test_the_mask_cannot_hide_a_constant_emotion(tmp_path):
    """Measured over the labelled clips only.

    100 masked clips beside 10 targets that are all <|HAPPY|> is 9% of the
    corpus but 100% of what the emotion head sees, which is the collapse.
    """
    stats = emo_stats_over(
        tmp_path, ["<|HAPPY|>"] * 10 + [vn.EMO_MASK_TARGET] * 100, []
    )

    assert stats.dominant_label() == ("<|HAPPY|>", 1.0)


def test_a_wholly_masked_corpus_reports_no_dominant_label(tmp_path):
    # Nothing to be degenerate about, and the fraction would divide by zero.
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|SAD|>"})
    )
    masked = emo_records({EMO_KEY_B: ""})
    vn.apply_emo_labels(masked + emo_records({EMO_KEY_A: ""}), labels)

    stats = vn.build_emo_label_stats(labels, masked, [])

    assert stats.labelled_clips == 0
    assert stats.dominant_label() is None


def test_the_warning_names_the_collapse_it_is_guarding_against(capsys, tmp_path):
    stats = emo_stats_over(tmp_path, ["<|HAPPY|>"] * 95 + ["<|SAD|>"] * 5, [])

    vn.log_emo_label_report(stats)

    out = capsys.readouterr().out
    assert "WARNING: degenerate emotion distribution" in out
    assert "<|HAPPY|> covers 95.0%" in out
    assert "collapsed the emotion head" in out


def test_no_warning_for_a_healthy_distribution(capsys, tmp_path):
    stats = emo_stats_over(tmp_path, ["<|HAPPY|>"] * 5 + ["<|SAD|>"] * 5, [])

    vn.log_emo_label_report(stats)

    out = capsys.readouterr().out
    assert "degenerate" not in out
    assert "<|HAPPY|> 5 (50.0%)" in out


# --- the CLI ----------------------------------------------------------------


def test_emo_labels_defaults_to_none_so_the_flag_is_opt_in():
    assert vn.parse_args([]).emo_labels is None


def test_emo_labels_is_parsed_as_a_path():
    assert vn.parse_args(["--emo-labels", "e.jsonl"]).emo_labels == Path("e.jsonl")


def test_the_label_file_is_read_before_any_download(monkeypatch):
    # Fail-fast, like the pin file and the basename guard: a typo'd path must
    # cost nothing, not surface after the corpus has been fetched and converted.
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("download attempted despite an unreadable label file")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(["--emo-labels", "/nonexistent/emo.jsonl"])

    assert "--emo-labels" in str(excinfo.value)


# --- the manifest -----------------------------------------------------------


def test_a_default_run_records_no_emotion_stats_in_the_manifest():
    # An unlabelled run's manifest.json must be byte-for-byte what it was before
    # the flag existed, so round 2's manifests stay reproducible.
    records = corpus()
    train, val = vn.split_by_speaker(records, 0.2, seed=0)
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))

    manifest = vn.build_manifest(vn.parse_args([]), {}, stats, train, val, 0)

    assert "emo_label_stats" not in manifest


def test_a_labelled_run_records_the_stats_in_the_manifest(tmp_path):
    records = corpus()
    train, val = vn.split_by_speaker(records, 0.2, seed=0)
    labels = vn.read_emo_labels(
        emo_labels_for(
            tmp_path / "emo.jsonl", {record["key"]: "<|SAD|>" for record in train[:3]}
        )
    )
    vn.apply_emo_labels(records, labels)
    emo_stats = vn.build_emo_label_stats(labels, train, val)
    stats = vn.FilterStats(index_entries=len(records), kept=len(records))

    manifest = vn.build_manifest(
        vn.parse_args([]), {}, stats, train, val, 0, emo_stats=emo_stats
    )

    block = manifest["emo_label_stats"]
    assert block["keys_in_file"] == 3
    assert block["keys_matched"] == 3
    assert block["train"]["counts"]["<|SAD|>"] == 3
    assert block["val"]["counts"][vn.EMO_MASK_TARGET] == len(val)


# --- end to end over a synthesised corpus -----------------------------------


@requires_soundfile
def test_without_the_flag_every_clip_is_still_neutral(tmp_path, monkeypatch):
    """The round-2 reproducibility pin, taken through main().

    The existing corpus was built with a hardcoded <|NEUTRAL|> on every record.
    Nothing added for --emo-labels may perturb that path.
    """
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)

    assert vn.main(manifest_only_argv(tmp_path)) == 0

    assert all_emo_targets(tmp_path / "train.jsonl") == {"<|NEUTRAL|>"}
    assert all_emo_targets(tmp_path / "val.jsonl") <= {"<|NEUTRAL|>"}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "emo_label_stats" not in manifest


@requires_soundfile
def test_labelled_clips_take_their_target_and_the_rest_are_masked(
    tmp_path, monkeypatch
):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    labelled = {
        "Studio_A__spk0__v0": "<|HAPPY|>",
        "Studio_A__spk0__v1": "<|SAD|>",
        "Studio_B__spk2__v3": "<|ANGRY|>",
    }
    path = emo_labels_for(tmp_path / "emo.jsonl", labelled)

    assert vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path))) == 0

    written = read_jsonl(tmp_path / "train.jsonl") + read_jsonl(tmp_path / "val.jsonl")
    by_key = {record["key"]: record["emo_target"] for record in written}
    for key, target in labelled.items():
        assert by_key[key] == target
    unlabelled = set(by_key) - set(labelled)
    assert unlabelled  # otherwise the mask assertion below is vacuous
    assert {by_key[key] for key in unlabelled} == {"<|SER|>"}


@requires_soundfile
def test_the_manifest_stats_reconcile_with_the_written_files(tmp_path, monkeypatch):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    path = emo_labels_for(
        tmp_path / "emo.jsonl",
        {"Studio_A__spk0__v0": "<|HAPPY|>", "Studio_B__spk1__v2": "<|SAD|>"},
    )

    vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["emo_label_stats"]
    assert block["labels_file"] == str(path)
    assert block["keys_in_file"] == 2
    assert block["train"]["clips"] == manifest["totals"]["train_clips"]
    assert block["val"]["clips"] == manifest["totals"]["val_clips"]
    # Every clip is accounted for by exactly one of the eight targets.
    for split in ("train", "val"):
        assert sum(block[split]["counts"].values()) == block[split]["clips"]
    assert block["keys_matched"] + block["keys_matched_nothing"] == 2


@requires_soundfile
def test_the_run_prints_the_emotion_summary(tmp_path, monkeypatch, capsys):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    # 6 of the fixture's 24 clips: 25%, comfortably over the coverage floor.
    path = emo_labels_for(
        tmp_path / "emo.jsonl",
        {f"Studio_A__spk0__v{index}": "<|HAPPY|>" for index in range(4)}
        | {f"Studio_B__spk1__v{index}": "<|SAD|>" for index in range(2)},
    )

    vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    out = capsys.readouterr().out
    assert "emo     :" in out
    assert "6 keys" in out
    assert "<|SER|>" in out
    # The number an operator needs before submitting a training job, on its own
    # line: keys_matched counts explicitly-masked rows too, labelled_clips does
    # not.
    assert "labelled 6 of 24 clips (25.0%)" in out


@requires_soundfile
def test_a_degenerate_corpus_is_called_out_at_the_end_of_the_run(
    tmp_path, monkeypatch, capsys
):
    """22 of 24 clips labelled <|HAPPY|> is 91.7% -- over the 90% bound."""
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(tmp_path))
    keys = [
        record["key"]
        for record in read_jsonl(tmp_path / "train.jsonl")
        + read_jsonl(tmp_path / "val.jsonl")
    ]
    mapping = {key: "<|HAPPY|>" for key in keys[:22]}
    mapping.update({key: "<|SAD|>" for key in keys[22:24]})
    path = emo_labels_for(tmp_path / "emo.jsonl", mapping)
    capsys.readouterr()

    vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    assert "WARNING: degenerate emotion distribution" in capsys.readouterr().out


@requires_soundfile
def test_a_balanced_corpus_raises_no_warning(tmp_path, monkeypatch, capsys):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(tmp_path))
    keys = [
        record["key"]
        for record in read_jsonl(tmp_path / "train.jsonl")
        + read_jsonl(tmp_path / "val.jsonl")
    ]
    mapping = {key: "<|HAPPY|>" for key in keys[:12]}
    mapping.update({key: "<|SAD|>" for key in keys[12:24]})
    path = emo_labels_for(tmp_path / "emo.jsonl", mapping)
    capsys.readouterr()

    vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    assert "degenerate" not in capsys.readouterr().out


@requires_soundfile
def test_a_label_file_for_another_corpus_stops_the_run(tmp_path, monkeypatch):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    path = emo_labels_for(
        tmp_path / "emo.jsonl", {"Some_Other_Title__spk0__v0": "<|HAPPY|>"}
    )

    with pytest.raises(SystemExit) as excinfo:
        vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    assert "different corpus" in str(excinfo.value)


# --- interaction with --pin-val-keys ----------------------------------------


@requires_soundfile
def test_emotion_labels_compose_with_a_pinned_val_split(tmp_path, monkeypatch):
    """Round 3 is exactly this: both flags, one invocation.

    The two features touch different fields -- pinning decides *which* clips are
    val, labelling decides what each clip's emo_target is -- and neither may
    disturb the other.
    """
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(tmp_path))
    pinned = [record["key"] for record in read_jsonl(tmp_path / "val.jsonl")]
    pins = tmp_path / "pins.txt"
    pins.write_text("".join(f"{key}\n" for key in pinned), encoding="utf-8")
    keys = [record["key"] for record in read_jsonl(tmp_path / "train.jsonl")] + pinned
    path = emo_labels_for(
        tmp_path / "emo.jsonl",
        {key: ("<|HAPPY|>" if index % 2 else "<|SAD|>") for index, key in
         enumerate(keys)},
    )

    vn.main(
        manifest_only_argv(
            tmp_path, "--pin-val-keys", str(pins), "--emo-labels", str(path)
        )
    )

    val = read_jsonl(tmp_path / "val.jsonl")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["emo_label_stats"]
    # The pin is untouched by the labelling: val is still exactly the pinned set.
    assert [record["key"] for record in val] == sorted(pinned)
    # And the per-split stats add up to the corpus the manifest describes.
    assert block["train"]["clips"] + block["val"]["clips"] == manifest["totals"]["clips"]
    assert block["train"]["clips"] == manifest["totals"]["train_clips"]
    assert block["val"]["clips"] == manifest["totals"]["val_clips"]
    assert block["val"]["labelled_clips"] == len(pinned)
    assert manifest["val_pin"]["keys_found"] == len(pinned)


@requires_soundfile
def test_a_pinned_val_clip_keeps_its_own_label(tmp_path, monkeypatch):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    vn.main(manifest_only_argv(tmp_path))
    pinned = [record["key"] for record in read_jsonl(tmp_path / "val.jsonl")]
    pins = tmp_path / "pins.txt"
    pins.write_text("".join(f"{key}\n" for key in pinned), encoding="utf-8")
    # Only the first pinned clip is labelled; the rest of val must be masked.
    # One clip of 24 is under the coverage floor, so this is also the end-to-end
    # exercise of the deliberate override.
    path = emo_labels_for(tmp_path / "emo.jsonl", {pinned[0]: "<|FEARFUL|>"})

    vn.main(
        manifest_only_argv(
            tmp_path,
            "--pin-val-keys",
            str(pins),
            "--emo-labels",
            str(path),
            "--allow-sparse-emo-labels",
        )
    )

    val = {r["key"]: r["emo_target"] for r in read_jsonl(tmp_path / "val.jsonl")}
    assert val[pinned[0]] == "<|FEARFUL|>"
    assert set(val.values()) - {"<|FEARFUL|>"} <= {"<|SER|>"}


# --- the coverage floor -----------------------------------------------------
#
# Zero overlap is a typo and is caught above.  This is the other half, and the
# likelier accident: a label file that matches *almost* nothing -- a labelling
# job interrupted a few thousand clips in, or a merge of audio and text labels
# taken over different --sample/--seed slices.  Every unmatched clip is masked,
# the run completes, the manifest reports it accurately, and the emotion head
# trains on essentially nothing.  The cost of noticing late is a finished
# two-day training run.


def emo_corpus(count, labelled, tmp_path, target="<|HAPPY|>"):
    """``count`` records of which the first ``labelled`` carry a real emotion."""
    mapping = {f"T__spk{index}__v{index}": "" for index in range(count)}
    records = emo_records(mapping)
    labels = vn.read_emo_labels(
        emo_labels_for(
            tmp_path / "emo.jsonl",
            {key: target for key in list(mapping)[:labelled]},
        )
    )
    return records, labels


def test_the_coverage_floor_is_five_percent():
    # Well beneath the training plan's own gate ("usable labels >= 15% of
    # train"), so this is a floor against a broken file, not a second opinion
    # on that gate.  Raising it to argue about label quality would be a
    # different decision, taken elsewhere.
    assert vn.EMO_MIN_LABELLED_FRACTION == 0.05


@pytest.mark.parametrize(
    "labelled,accepted",
    [
        # The bound is EMO_MIN_LABELLED_FRACTION (0.05) over the kept corpus and
        # it is INCLUSIVE: 5/100 is exactly the floor and passes, 4/100 is below
        # it and does not.
        (5, True),
        (4, False),
        (1, False),
        (100, True),
    ],
)
def test_the_floor_is_an_inclusive_bound(tmp_path, labelled, accepted):
    records, labels = emo_corpus(100, labelled, tmp_path)

    if accepted:
        assert vn.apply_emo_labels(records, labels)
    else:
        with pytest.raises(SystemExit):
            vn.apply_emo_labels(records, labels)


def test_a_nearly_empty_label_file_is_fatal(tmp_path):
    """The finding this section was added for: 0.04% coverage used to pass."""
    records, labels = emo_corpus(5000, 2, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        vn.apply_emo_labels(records, labels)

    message = str(excinfo.value)
    assert "2 of this corpus's 5000 clips" in message
    assert "0.04%" in message


def test_the_floor_error_names_the_likely_causes(tmp_path):
    """A fraction alone does not tell an operator what to go and look at."""
    records, labels = emo_corpus(1000, 1, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        vn.apply_emo_labels(records, labels)

    message = str(excinfo.value)
    assert "interrupted" in message
    assert "--sample/--seed" in message
    assert "--allow-sparse-emo-labels" in message


def test_a_wholly_masked_file_is_caught_by_the_floor(tmp_path):
    """Matching every key while labelling nothing clears the zero-overlap check.

    The floor is what stops it: coverage is measured in clips that came out
    carrying an emotion, not in keys that found a home.
    """
    mapping = {f"T__spk{i}__v{i}": vn.EMO_MASK_TARGET for i in range(20)}
    records = emo_records(mapping)
    labels = vn.read_emo_labels(emo_labels_for(tmp_path / "emo.jsonl", mapping))

    with pytest.raises(SystemExit) as excinfo:
        vn.apply_emo_labels(records, labels)

    assert "0 of this corpus's 20 clips (0.00%)" in str(excinfo.value)


def test_the_override_permits_a_deliberate_pilot(tmp_path):
    # A pilot labelling run over a slice of a large corpus is legitimate and
    # must be expressible without editing the source.
    records, labels = emo_corpus(5000, 2, tmp_path)

    assert len(vn.apply_emo_labels(records, labels, allow_sparse=True)) == 2
    assert records[0]["emo_target"] == "<|HAPPY|>"
    assert records[99]["emo_target"] == vn.EMO_MASK_TARGET


def test_the_override_does_not_excuse_a_mismatched_corpus(tmp_path):
    """Sparse is a judgement call; zero overlap is still a broken input."""
    records = emo_records({EMO_KEY_A: "", EMO_KEY_B: ""})
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {"Other__spk__v0": "<|SAD|>"})
    )

    with pytest.raises(SystemExit) as excinfo:
        vn.apply_emo_labels(records, labels, allow_sparse=True)

    assert "different corpus" in str(excinfo.value)


# --- the override on the CLI and in the manifest ----------------------------


def test_allow_sparse_emo_labels_defaults_to_off():
    assert vn.parse_args([]).allow_sparse_emo_labels is False


def test_allow_sparse_emo_labels_is_a_flag():
    args = vn.parse_args(["--emo-labels", "e.jsonl", "--allow-sparse-emo-labels"])
    assert args.allow_sparse_emo_labels is True


def test_the_override_without_the_labels_flag_is_an_error(monkeypatch):
    # Rejected rather than ignored, like --list-format and
    # --kana-only-title-threshold: an override that silently did nothing would
    # read as a corpus admitted under a relaxed floor when no floor was ever
    # consulted.
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the run proceeded past argument validation")

    monkeypatch.setattr(vn, "download_all", explode)
    monkeypatch.setattr(vn, "read_hf_token", explode)

    with pytest.raises(SystemExit) as excinfo:
        vn.main(["--allow-sparse-emo-labels"])

    assert "--allow-sparse-emo-labels only applies to --emo-labels" in str(
        excinfo.value
    )


def test_the_manifest_records_the_floor_and_whether_it_was_overridden(tmp_path):
    records, labels = emo_corpus(100, 2, tmp_path)
    vn.apply_emo_labels(records, labels, allow_sparse=True)

    block = vn.build_emo_label_stats(
        labels, records, [], sparse_override=True
    ).as_dict()

    # A corpus built under the override is otherwise indistinguishable from one
    # that cleared the floor honestly, and "why did round 3 learn no emotion"
    # is answered here or not at all.
    assert block["sparse_override"] is True
    assert block["min_labelled_fraction"] == vn.EMO_MIN_LABELLED_FRACTION
    assert block["labelled_clips"] == 2
    assert block["labelled_fraction"] == 0.02


def test_an_ordinary_run_records_no_override(tmp_path):
    records, labels = emo_corpus(20, 10, tmp_path)
    vn.apply_emo_labels(records, labels)

    block = vn.build_emo_label_stats(labels, records, []).as_dict()

    assert block["sparse_override"] is False
    assert block["labelled_fraction"] == 0.5


# --- the summary line an operator reads -------------------------------------


def test_the_summary_states_labelled_of_total(capsys, tmp_path):
    stats = emo_stats_over(
        tmp_path, ["<|HAPPY|>", "<|SAD|>"] + [vn.EMO_MASK_TARGET] * 6, ["<|SAD|>"]
    )

    vn.log_emo_label_report(stats)

    out = capsys.readouterr().out
    assert "labelled 3 of 9 clips (33.3%)" in out


def test_the_summary_says_when_the_floor_was_overridden(capsys, tmp_path):
    stats = emo_stats_over(tmp_path, ["<|HAPPY|>"] + [vn.EMO_MASK_TARGET] * 99, [])
    stats.sparse_override = True

    vn.log_emo_label_report(stats)

    assert "--allow-sparse-emo-labels was passed" in capsys.readouterr().out


def test_an_ordinary_run_does_not_mention_the_override(capsys, tmp_path):
    stats = emo_stats_over(tmp_path, ["<|HAPPY|>", "<|SAD|>"], [])

    vn.log_emo_label_report(stats)

    assert "allow-sparse" not in capsys.readouterr().out


# --- defence in depth: the tally trusts nothing -----------------------------


def test_a_target_outside_the_eight_is_refused_by_the_tally(tmp_path):
    """Unreachable while apply_emo_labels is the only writer -- and raised anyway.

    ``_split_block`` sums the eight known targets, so an unknown one would drop
    out of ``clips`` silently and hand the operator an under-count that still
    adds up on its face.  That is the exact shape of failure this block exists
    to prevent, and "unreachable" has a poor record in this script: the
    54,420-clip loss above was also an assumption that held for most inputs.
    """
    labels = vn.read_emo_labels(
        emo_labels_for(tmp_path / "emo.jsonl", {EMO_KEY_A: "<|SAD|>"})
    )
    records = emo_records({EMO_KEY_A: ""})
    vn.apply_emo_labels(records, labels)
    records[0]["emo_target"] = "<|EMO_UNKNOWN|>"

    with pytest.raises(SystemExit) as excinfo:
        vn.build_emo_label_stats(labels, records, [])

    message = str(excinfo.value)
    assert EMO_KEY_A in message
    assert "<|EMO_UNKNOWN|>" in message


# --- end to end -------------------------------------------------------------


@requires_soundfile
def test_a_sparse_label_file_stops_the_run(tmp_path, monkeypatch):
    """1 of 24 clips is 4.2%, under the floor -- the run must not complete."""
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    path = emo_labels_for(tmp_path / "emo.jsonl", {"Studio_A__spk0__v0": "<|HAPPY|>"})

    with pytest.raises(SystemExit) as excinfo:
        vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    assert "1 of this corpus's 24 clips (4.17%)" in str(excinfo.value)


@requires_soundfile
def test_the_sparse_run_writes_no_manifest(tmp_path, monkeypatch):
    # It must refuse before overwriting a good manifest, exactly as the empty
    # corpus check does.
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    path = emo_labels_for(tmp_path / "emo.jsonl", {"Studio_A__spk0__v0": "<|HAPPY|>"})

    with pytest.raises(SystemExit):
        vn.main(manifest_only_argv(tmp_path, "--emo-labels", str(path)))

    assert not (tmp_path / "manifest.json").exists()


@requires_soundfile
def test_the_override_lets_the_sparse_run_through_and_says_so(
    tmp_path, monkeypatch, capsys
):
    build_pruned_corpus(tmp_path)
    forbid_acquisition(monkeypatch)
    path = emo_labels_for(tmp_path / "emo.jsonl", {"Studio_A__spk0__v0": "<|HAPPY|>"})

    assert (
        vn.main(
            manifest_only_argv(
                tmp_path, "--emo-labels", str(path), "--allow-sparse-emo-labels"
            )
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "labelled 1 of 24 clips (4.2%)" in out
    assert "--allow-sparse-emo-labels was passed" in out
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["emo_label_stats"]["sparse_override"] is True


# --- --help must actually render --------------------------------------------


def test_the_help_text_renders():
    """argparse %-formats every help string to expand %(default)s, so a literal
    per-cent sign in one raises TypeError at --help time and nowhere else.

    Caught in review only because --help was run by hand: parse_args() never
    formats help, so the CLI-surface test above passes over the bug entirely.
    A percentage in a help string is natural to write here -- the thresholds
    are all fractions -- so this renders the whole parser once.
    """
    with pytest.raises(SystemExit) as excinfo:
        vn.parse_args(["--help"])

    assert excinfo.value.code == 0


def test_the_help_text_states_the_coverage_floor(capsys):
    with pytest.raises(SystemExit):
        vn.parse_args(["--help"])

    # Rendered as literal per-cent signs, not swallowed as format specs.  Asserted
    # on the unwrapped source rather than the output, because argparse rewraps
    # the paragraph to the terminal width and splits "95%" from "of the corpus".
    out = " ".join(capsys.readouterr().out.split())

    assert "fewer than 5% of clips carry a real emotion" in out
    assert "leaves more than 95% of the corpus unlabelled" in out
