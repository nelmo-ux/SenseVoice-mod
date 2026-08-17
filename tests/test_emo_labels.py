"""Unit tests for the round-3 emotion pseudo-labelling pipeline.

Three scripts are under test -- ``scripts/label_emotions_audio.py``,
``scripts/label_emotions_text.py`` and ``scripts/merge_emo_labels.py`` -- and
none of them is loaded here with a model attached.  Everything below drives the
pure surface: the label vocabularies, the two mapping tables, the consensus
decision rules, the NEUTRAL cap and the response parser.  No download, no
network, no GPU.

What these tests are actually guarding
--------------------------------------

Rounds 1 and 2 wrote ``emo_target="<|NEUTRAL|>"`` for all ~550k clips, and the
emotion head learned the constant -- a silent failure, because the loss looked
fine.  Round 3's replacement has the same silent-failure shape: every way it can
go wrong produces a plausible-looking label file.

* A mapping table that quietly drops a class deletes an emotion from the corpus
  and nothing errors, so :func:`test_audio_mapping_is_total` and its siblings
  pin totality *and* surjectivity in both directions.
* ``<|EMO_UNKNOWN|>`` as a target would be a second high-frequency catch-all --
  the same collapse in new clothes -- so it is asserted absent, not merely
  unused.
* If ``<|SER|>`` were not a single token the model's ``ignore_id`` mapping would
  land on the wrong position and mask the wrong thing, so its id is pinned
  against the real sentencepiece model.
* A neutral cap that trims to ``cap * current_total`` leaves the result *above*
  the cap; :func:`test_neutral_cap_hits_the_cap_exactly` is what distinguishes
  the two formulas.
* A tolerant response parser that reached for a substring match would turn "I
  think this is happy or maybe sad" into a confident ``happy``, which is exactly
  the noise the two-labeller design exists to exclude.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
BPE_MODEL = ROOT / "models" / "SenseVoiceSmall" / "chn_jpn_yue_eng_ko_spectok.bpe.model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str):
    """Load one of the scripts by path.

    ``scripts/`` is not a package, so there is nothing to ``import``.  Same
    loader shape as ``tests/test_detect_label_noise.py``.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audio():
    return _load("label_emotions_audio")


@pytest.fixture(scope="module")
def text():
    return _load("label_emotions_text")


@pytest.fixture(scope="module")
def merge():
    return _load("merge_emo_labels")


# --------------------------------------------------------------------- helpers


def audio_record(label: str, score: float = 0.9, duration: float = 1.0) -> dict:
    """One row as ``label_emotions_audio.py`` writes it."""
    return {"label": label, "score": score, "probs": {label: score}, "duration": duration}


def text_record(label: str, score: float | None = 0.8) -> dict:
    """One row as ``label_emotions_text.py`` writes it."""
    return {"label": label, "score": score, "raw": label}


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


# ------------------------------------------------- mapping totality/surjectivity


def test_audio_mapping_is_total(audio, merge):
    """Every emotion2vec class the labeller can emit has a mapping entry.

    A missing entry does not crash the merge in any obvious place -- it raises
    on one clip, mid-sweep, after hours of GPU time. Pinning the two lists
    against each other catches a checkpoint that renamed a class at import time
    instead.
    """
    assert set(audio.AUDIO_CLASSES) == set(merge.AUDIO_LABEL_TO_TOKEN)
    assert len(audio.AUDIO_CLASSES) == 9


def test_text_mapping_is_total(text, merge):
    """Every class the prompt offers has a mapping entry, and vice versa."""
    assert set(text.TEXT_CLASSES) == set(merge.TEXT_LABEL_TO_TOKEN)
    assert len(text.TEXT_CLASSES) == 10


def test_mapping_images_are_exactly_the_seven_emotion_tokens(merge):
    """Surjectivity in both directions: nothing unreachable, nothing extra.

    The "nothing unreachable" half is the one that matters. A token that no
    class maps to is an emotion the head can never be trained on, which looks
    identical in the loss curve to an emotion the corpus simply lacks.
    """
    non_mask = {
        token
        for table in (merge.AUDIO_LABEL_TO_TOKEN, merge.TEXT_LABEL_TO_TOKEN)
        for token in table.values()
        if token != merge.MASK_TOKEN
    }
    assert non_mask == set(merge.EMOTION_TOKENS)
    assert len(merge.EMOTION_TOKENS) == 7
    # Each table alone must also reach all seven; agreement between two tables
    # with different images could never produce the missing ones.
    for table in (merge.AUDIO_LABEL_TO_TOKEN, merge.TEXT_LABEL_TO_TOKEN):
        images = {token for token in table.values() if token != merge.MASK_TOKEN}
        assert images == set(merge.EMOTION_TOKENS)


def test_emo_unknown_is_never_an_image(merge):
    """``<|EMO_UNKNOWN|>`` must not be emitted as a target.

    Inference bans it (``ban_emo_unk``), so training it spends capacity on a
    token the decoder can never produce. Worse, it would be a second
    high-frequency catch-all class alongside neutral -- the same shape as the
    collapse round 3 exists to undo. Uncertainty goes to the mask, which costs
    the emotion head nothing.
    """
    all_tokens = set(merge.AUDIO_LABEL_TO_TOKEN.values()) | set(
        merge.TEXT_LABEL_TO_TOKEN.values()
    )
    assert "<|EMO_UNKNOWN|>" not in all_tokens
    assert "<|EMO_UNKNOWN|>" not in merge.EMOTION_TOKENS


def test_mask_is_reachable_from_both_sides(merge):
    """Both labellers must be able to abstain, or the filter is one-sided."""
    assert merge.MASK_TOKEN in merge.AUDIO_LABEL_TO_TOKEN.values()
    assert merge.MASK_TOKEN in merge.TEXT_LABEL_TO_TOKEN.values()
    assert merge.AUDIO_LABEL_TO_TOKEN["unknown"] == merge.MASK_TOKEN
    assert merge.AUDIO_LABEL_TO_TOKEN["other"] == merge.MASK_TOKEN
    assert merge.TEXT_LABEL_TO_TOKEN["embarrassed"] == merge.MASK_TOKEN
    assert merge.TEXT_LABEL_TO_TOKEN["sexual"] == merge.MASK_TOKEN


# ---------------------------------------------------------------- the sentinel


def test_mask_sentinel_is_a_single_token(merge):
    """``<|SER|>`` must encode to exactly ``[24991]``.

    The model turns this one position into ``ignore_id``. If the string
    tokenised into two pieces the emotion slot would be off by one and the mask
    would land on a neighbouring position -- corrupting the sequence rather than
    excusing the clip from the emotion loss, and doing so on precisely the
    ~50%+ of clips that get masked.
    """
    sentencepiece = pytest.importorskip("sentencepiece")
    if not BPE_MODEL.is_file():
        pytest.skip(f"sentencepiece model not staged at {BPE_MODEL}")
    processor = sentencepiece.SentencePieceProcessor()
    processor.load(str(BPE_MODEL))
    assert processor.encode(merge.MASK_TOKEN) == [24991]


def test_every_emotion_token_is_a_single_token(merge):
    """The seven targets are single tokens too, for the same reason."""
    sentencepiece = pytest.importorskip("sentencepiece")
    if not BPE_MODEL.is_file():
        pytest.skip(f"sentencepiece model not staged at {BPE_MODEL}")
    processor = sentencepiece.SentencePieceProcessor()
    processor.load(str(BPE_MODEL))
    expected = {
        "<|HAPPY|>": 25001,
        "<|SAD|>": 25002,
        "<|ANGRY|>": 25003,
        "<|NEUTRAL|>": 25004,
        "<|FEARFUL|>": 25005,
        "<|DISGUSTED|>": 25006,
        "<|SURPRISED|>": 25007,
    }
    assert set(expected) == set(merge.EMOTION_TOKENS)
    for token, token_id in expected.items():
        assert processor.encode(token) == [token_id], token


# ------------------------------------------------------------- decision rules


def test_decision_agree_adopts_the_shared_token(merge):
    row = merge.decide("k", audio_record("happy"), text_record("happy"))
    assert row.emo_target == "<|HAPPY|>"
    assert row.decision == "agree"


def test_decision_agree_on_neutral_is_still_agreement(merge):
    """Neutral is a real label here; the cap, not the decision, bounds it."""
    row = merge.decide("k", audio_record("neutral"), text_record("neutral"))
    assert row.emo_target == "<|NEUTRAL|>"
    assert row.decision == "agree"


def test_decision_disagree_masks(merge):
    """Two labellers, two tokens, no principled tiebreak -- so neither wins."""
    row = merge.decide("k", audio_record("sad"), text_record("angry"))
    assert row.emo_target == merge.MASK_TOKEN
    assert row.decision == "disagree_masked"


@pytest.mark.parametrize("aux", ["embarrassed", "sexual"])
def test_decision_aux_text_class_masks(merge, aux):
    """The auxiliary buckets keep the clip and drop only its emotion target.

    They exist so flustered and sexual lines are not forced into one of the
    seven; masking them is the whole point, and the clip stays in the corpus
    training the ASR branch.
    """
    row = merge.decide("k", audio_record("happy"), text_record(aux))
    assert row.emo_target == merge.MASK_TOKEN
    assert row.decision == "aux_masked"


@pytest.mark.parametrize(
    ("audio_label", "text_label"),
    [("other", "happy"), ("unknown", "happy"), ("happy", "other")],
)
def test_decision_other_masks(merge, audio_label, text_label):
    """``other``/``unknown`` are abstentions, from either side."""
    row = merge.decide("k", audio_record(audio_label), text_record(text_label))
    assert row.emo_target == merge.MASK_TOKEN
    assert row.decision == "other_masked"


def test_decision_aux_wins_the_attribution_when_both_sides_abstain(merge):
    """Audio ``other`` + text ``sexual`` is reported as ``aux_masked``.

    A deliberate choice, pinned because it is arbitrary-looking: the target is
    MASK either way, so this only moves a count between two report lines. It
    goes to the aux line because "the text labeller recognised a category we do
    not train" is a specific fact about the corpus that the pilot can act on,
    whereas "a labeller had no opinion" is not.
    """
    row = merge.decide("k", audio_record("other"), text_record("sexual"))
    assert row.emo_target == merge.MASK_TOKEN
    assert row.decision == "aux_masked"


@pytest.mark.parametrize(
    ("audio_rec", "text_rec"),
    [
        (None, "happy"),
        ("happy", None),
        (None, None),
    ],
)
def test_decision_missing_side_masks(merge, audio_rec, text_rec):
    """One labeller alone is never enough -- the design is the intersection."""
    row = merge.decide(
        "k",
        audio_record(audio_rec) if audio_rec else None,
        text_record(text_rec) if text_rec else None,
    )
    assert row.emo_target == merge.MASK_TOKEN
    assert row.decision == "missing_masked"


def test_decision_unparsed_text_label_masks(merge):
    """A ``label: null`` row (parse failure) reaches ``decide`` as missing."""
    row = merge.decide("k", audio_record("happy"), {"label": None, "score": None})
    assert row.decision == "missing_masked"


def test_unknown_class_raises_rather_than_defaulting(merge):
    """A class outside the table is a broken labeller, not a mask.

    Defaulting to the mask would silently delete every clip of a renamed class
    from the emotion signal and report it as ordinary abstention.
    """
    with pytest.raises(KeyError, match="unknown audio class"):
        merge.decide("k", audio_record("elated"), text_record("happy"))
    with pytest.raises(KeyError, match="unknown text class"):
        merge.decide("k", audio_record("happy"), text_record("elated"))


def test_decide_records_the_raw_labels_for_review(merge):
    """The report has to show what each side actually said."""
    row = merge.decide("k", audio_record("sad", score=0.42), text_record("angry"))
    payload = row.to_json()
    assert payload["audio_label"] == "sad"
    assert payload["audio_score"] == pytest.approx(0.42)
    assert payload["text_label"] == "angry"
    assert payload["key"] == "k"


# ------------------------------------- no confidence override (retired 2026-08-14)


@pytest.mark.parametrize(
    ("audio_label", "text_label", "score"),
    [
        pytest.param("happy", "neutral", 1.0, id="text-neutral-audio-certain"),
        pytest.param("surprised", "neutral", 1.0, id="the-domain-shifted-class"),
        pytest.param("happy", "sad", 0.99, id="two-different-emotions"),
        pytest.param("neutral", "happy", 0.99, id="audio-neutral-text-emotional"),
    ],
)
def test_no_audio_confidence_overrides_a_disagreement(merge, audio_label, text_label, score):
    """Confidence buys nothing. Every disagreement masks, at any score.

    ``--audio-conf-fallback`` used to adopt the audio label on the
    text-neutral/audio-confident pattern. The 5,000-clip pilot of 2026-08-14
    retired it: emotion2vec+ large is domain-shifted on this corpus (surprised
    32.1%, happy 24.0%, neutral 1.64%) and returns 1.000 on ordinary
    conversational lines, with ~85% of clips above 0.7. A threshold selects
    nearly everything and the selection is full of confident mislabels, so the
    override imported the shift into precisely the clips the text labeller had
    called neutral.

    Parametrized over the exact cases the flag used to change, including
    ``surprised`` -- the class the shift concentrates in -- so that reinstating
    any confidence rule fails here rather than passing quietly.
    """
    row = merge.decide(
        "k", audio_record(audio_label, score=score), text_record(text_label)
    )
    assert row.emo_target == merge.MASK_TOKEN
    assert row.decision == "disagree_masked"


def test_decide_takes_no_confidence_threshold(merge):
    """The parameter is gone, not defaulted to None.

    A retired knob left in the signature is a knob someone re-enables without
    re-reading why it was retired.
    """
    with pytest.raises(TypeError):
        merge.decide("k", audio_record("happy"), text_record("neutral"), 0.7)
    assert "audio_conf" not in merge.DECISIONS


def test_masked_disagreements_keep_the_audio_evidence(merge):
    """``audio_label`` and ``audio_score`` survive on masked rows.

    Deliberately preserved through the fallback's removal. A second hypothesis
    is still open -- audio ``neutral`` is rare but may be *correct* on short flat
    utterances, so the labeller could be trustworthy in that one direction while
    useless in the other. It is unmeasured, so no rule is built on it, but the
    evidence needed to settle it has to survive in the merge output or the
    question can only be answered by relabelling the corpus.
    """
    row = merge.decide("k", audio_record("surprised", score=1.0), text_record("neutral"))
    payload = row.to_json()
    assert payload["emo_target"] == merge.MASK_TOKEN
    assert payload["audio_label"] == "surprised"
    assert payload["audio_score"] == 1.0
    assert payload["text_label"] == "neutral"


# ---------------------------------------------------------------- neutral cap


def _capped_rows(merge, neutral_count: int, other_count: int, cap: float):
    """Build a synthetic decided set with ascending neutral confidences."""
    rows = [
        merge.MergedRow(
            key=f"n{i:03d}",
            emo_target=merge.NEUTRAL_TOKEN,
            decision="agree",
            audio_label="neutral",
            audio_score=(i + 1) / (neutral_count + 1),
            text_label="neutral",
        )
        for i in range(neutral_count)
    ]
    rows += [
        merge.MergedRow(
            key=f"h{i:03d}",
            emo_target="<|HAPPY|>",
            decision="agree",
            audio_label="happy",
            audio_score=0.9,
            text_label="happy",
        )
        for i in range(other_count)
    ]
    demoted = merge.apply_neutral_cap(rows, cap)
    return rows, demoted


def test_neutral_cap_hits_the_cap_exactly(merge):
    """90% neutral, cap 0.5 -> exactly 50% of the surviving supervision.

    This is the test that separates the right formula from the tempting one.
    Trimming to ``cap * current_total`` would keep 50 of the 100 rows and leave
    the result at 50/60 = 83% neutral, still far over the cap, because demoting
    a neutral shrinks the supervised set as well. The fixed point is
    ``k <= floor(cap * others / (1 - cap))`` -- here 10.
    """
    rows, demoted = _capped_rows(merge, neutral_count=90, other_count=10, cap=0.5)
    kept = [r for r in rows if r.emo_target == merge.NEUTRAL_TOKEN]
    supervised = [r for r in rows if r.emo_target != merge.MASK_TOKEN]
    assert demoted == 80
    assert len(kept) == 10
    assert len(kept) / len(supervised) == pytest.approx(0.5)


def test_neutral_cap_demotes_the_least_confident_first(merge):
    """The kept neutrals are the top-scoring ones; the demoted are the bottom.

    Least-confident-first because a low-confidence neutral is the most likely to
    be a real emotion the acoustic labeller missed -- so it is the cheapest
    supervision to give up.
    """
    rows, _ = _capped_rows(merge, neutral_count=90, other_count=10, cap=0.5)
    kept = sorted(r.key for r in rows if r.emo_target == merge.NEUTRAL_TOKEN)
    # Scores ascend with the index, so the ten survivors are the last ten.
    assert kept == [f"n{i:03d}" for i in range(80, 90)]
    demoted = [r for r in rows if r.decision == "neutral_capped"]
    assert all(r.emo_target == merge.MASK_TOKEN for r in demoted)
    assert max(r.audio_score for r in demoted) < min(
        r.audio_score for r in rows if r.emo_target == merge.NEUTRAL_TOKEN
    )


def test_neutral_cap_leaves_non_neutral_labels_alone(merge):
    """The cap is a bound on neutral, never a global downsample."""
    rows, _ = _capped_rows(merge, neutral_count=90, other_count=10, cap=0.5)
    assert sum(1 for r in rows if r.emo_target == "<|HAPPY|>") == 10


def test_neutral_cap_is_a_noop_when_under_the_cap(merge):
    rows, demoted = _capped_rows(merge, neutral_count=5, other_count=50, cap=0.5)
    assert demoted == 0
    assert all(r.decision == "agree" for r in rows)


def test_neutral_cap_of_one_disables_it(merge):
    rows, demoted = _capped_rows(merge, neutral_count=90, other_count=10, cap=1.0)
    assert demoted == 0
    assert sum(1 for r in rows if r.emo_target == merge.NEUTRAL_TOKEN) == 90


def test_neutral_cap_scores_missing_sort_first(merge):
    """No confidence is not high confidence; unscored neutrals go first."""
    rows = [
        merge.MergedRow("a", merge.NEUTRAL_TOKEN, "agree", "neutral", 0.2, "neutral"),
        merge.MergedRow("b", merge.NEUTRAL_TOKEN, "agree", "neutral", None, "neutral"),
        merge.MergedRow("c", "<|HAPPY|>", "agree", "happy", 0.9, "happy"),
    ]
    assert merge.apply_neutral_cap(rows, 0.5) == 1
    assert rows[1].decision == "neutral_capped"
    assert rows[0].decision == "agree"


def test_neutral_cap_is_deterministic_under_shuffling(merge):
    """Two runs over the same inputs, shuffled, must be byte-identical.

    The whole label file is regenerated on every pilot iteration and diffed
    against the last; a nondeterministic cap would make every diff full of
    noise and hide the change that was actually being evaluated.
    """
    audio_labels = {
        f"n{i:03d}": audio_record("neutral", score=(i + 1) / 91) for i in range(90)
    }
    audio_labels.update({f"h{i:03d}": audio_record("happy", score=0.9) for i in range(10)})
    text_labels = {key: text_record(rec["label"]) for key, rec in audio_labels.items()}

    first = merge.merge(
        merge.LabelFile.from_labels(audio_labels),
        merge.LabelFile.from_labels(text_labels),
        neutral_cap=0.5,
    )

    rng = random.Random(1234)
    shuffled_keys = list(audio_labels)
    rng.shuffle(shuffled_keys)
    shuffled_audio = {key: audio_labels[key] for key in shuffled_keys}
    rng.shuffle(shuffled_keys)
    shuffled_text = {key: text_labels[key] for key in shuffled_keys}
    second = merge.merge(
        merge.LabelFile.from_labels(shuffled_audio),
        merge.LabelFile.from_labels(shuffled_text),
        neutral_cap=0.5,
    )

    assert [r.to_json() for r in first] == [r.to_json() for r in second]
    assert [r.key for r in first] == sorted(r.key for r in first)


# -------------------------------------------------------------------- the join


def test_merge_joins_on_key_and_covers_the_union(merge):
    """Every key in either input appears exactly once in the output."""
    audio_labels = merge.LabelFile.from_labels(
        {"a": audio_record("happy"), "b": audio_record("sad")}
    )
    text_labels = merge.LabelFile.from_labels(
        {"a": text_record("happy"), "c": text_record("angry")}
    )
    rows = merge.merge(audio_labels, text_labels, neutral_cap=1.0)
    by_key = {r.key: r for r in rows}
    assert sorted(by_key) == ["a", "b", "c"]
    assert by_key["a"].emo_target == "<|HAPPY|>"
    assert by_key["b"].decision == "missing_masked"  # no text side
    assert by_key["c"].decision == "missing_masked"  # no audio side


def test_load_label_file_last_duplicate_wins(merge, tmp_path):
    """Append-mode resume can leave a duplicated key; the later row is the redo."""
    path = write_jsonl(
        tmp_path / "audio.jsonl",
        [
            {"key": "a", **audio_record("happy")},
            {"key": "a", **audio_record("sad")},
        ],
    )
    loaded = merge.load_label_file(path)
    assert loaded.labels["a"]["label"] == "sad"
    assert loaded.keys == {"a"}  # a duplicate is one clip, not two


def test_load_label_file_drops_null_labels_but_remembers_the_key(merge, tmp_path):
    """A ``label: null`` row is an abstention, not a label -- and not an absence.

    Both halves are load-bearing. Keeping it out of ``labels`` is what masks the
    clip. Keeping it *in* ``keys`` is what lets the report say "the text labeller
    ran here and produced nothing usable" instead of "these two labellers were
    run over different subsets" -- two faults with completely different fixes
    that were previously indistinguishable in the output.
    """
    path = write_jsonl(
        tmp_path / "text.jsonl",
        [{"key": "a", "label": None, "score": None, "raw": "???"}],
    )
    loaded = merge.load_label_file(path)
    assert loaded.labels == {}
    assert loaded.keys == {"a"}


def test_text_scores_are_diagnostic_only(merge, tmp_path):
    """A text file with no confidences at all must merge identically.

    vLLM's logprobs payload has changed shape across releases, so
    ``_first_token_probability`` can legitimately yield ``None`` for every row.
    That must degrade a diagnostic and nothing else: only ``audio_score`` feeds
    a decision (the cap ordering and the confidence fallback). Pinned because
    the two scores sit in adjacent fields and a future reader could easily wire
    the wrong one into a rule.
    """
    audio_path = write_jsonl(
        tmp_path / "audio.jsonl", [{"key": k, **audio_record("happy")} for k in "abc"]
    )
    scored = write_jsonl(
        tmp_path / "t1.jsonl", [{"key": k, **text_record("happy", score=0.9)} for k in "abc"]
    )
    unscored = write_jsonl(
        tmp_path / "t2.jsonl", [{"key": k, **text_record("happy", score=None)} for k in "abc"]
    )
    audio = merge.load_label_file(audio_path)
    with_scores = merge.merge(audio, merge.load_label_file(scored))
    without = merge.merge(audio, merge.load_label_file(unscored))
    assert [r.emo_target for r in with_scores] == [r.emo_target for r in without]
    assert [r.decision for r in with_scores] == [r.decision for r in without]


def test_load_label_file_keeps_only_label_and_score(merge, tmp_path):
    """The bulky diagnostic fields are dropped on load.

    ``probs`` (9 floats) and ``raw`` are for a human reading the labeller's own
    output; nothing in the merge reads them. At 550k clips across two files,
    carrying them costs roughly a gigabyte of resident memory to be ignored.
    """
    path = write_jsonl(tmp_path / "audio.jsonl", [{"key": "a", **audio_record("happy")}])
    assert merge.load_label_file(path).labels["a"] == {"label": "happy", "score": 0.9}


def test_end_to_end_writes_valid_jsonl_and_stats(merge, tmp_path):
    """Full CLI run: the label file round-trips through ``json.loads``."""
    audio_path = write_jsonl(
        tmp_path / "audio.jsonl",
        [
            {"key": "t__s__1", **audio_record("happy")},
            {"key": "t__s__2", **audio_record("sad")},
            {"key": "t__s__3", **audio_record("neutral", score=0.3)},
            {"key": "t__s__4", **audio_record("other")},
        ],
    )
    text_path = write_jsonl(
        tmp_path / "text.jsonl",
        [
            {"key": "t__s__1", **text_record("happy")},
            {"key": "t__s__2", **text_record("angry")},
            {"key": "t__s__3", **text_record("neutral")},
            {"key": "t__s__4", **text_record("sexual")},
            {"key": "t__s__5", **text_record("happy")},
        ],
    )
    out = tmp_path / "nested" / "emo_labels.jsonl"
    stats_out = tmp_path / "stats.json"

    code = merge.main(
        [
            "--audio", str(audio_path),
            "--text", str(text_path),
            "--out", str(out),
            "--stats-out", str(stats_out),
            "--neutral-cap", "1.0",
            # This fixture is deliberately 80% overlapping so it can exercise
            # the one-sided ``missing_masked`` branch. The overlap guard itself
            # is tested separately below; disabling it here keeps this test
            # about the merge semantics.
            "--min-overlap", "0",
        ]
    )
    assert code == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["key"] for r in rows] == [f"t__s__{i}" for i in range(1, 6)]
    by_key = {r["key"]: r for r in rows}
    assert by_key["t__s__1"]["emo_target"] == "<|HAPPY|>"
    assert by_key["t__s__2"]["decision"] == "disagree_masked"
    assert by_key["t__s__3"]["emo_target"] == "<|NEUTRAL|>"
    assert by_key["t__s__4"]["decision"] == "aux_masked"
    assert by_key["t__s__5"]["decision"] == "missing_masked"
    assert all(r["emo_target"] in (*merge.EMOTION_TOKENS, merge.MASK_TOKEN) for r in rows)

    stats = json.loads(stats_out.read_text(encoding="utf-8"))["stats"]
    assert stats["total_keys"] == 5
    assert stats["num_usable"] == 2
    assert stats["decisions"]["agree"] == 2
    # 3 comparable pairs (1, 2, 3); 2 agreed.
    assert stats["agreement_rate"] == pytest.approx(2 / 3)
    assert stats["disagreement_confusion"]["<|SAD|>"]["<|ANGRY|>"] == 1


def test_stats_report_the_cap_and_the_fallback_state(merge, tmp_path):
    """The pilot reads these fields to decide whether to enable the fallback."""
    audio_path = write_jsonl(
        tmp_path / "audio.jsonl",
        [{"key": f"n{i:02d}", **audio_record("neutral", score=(i + 1) / 100)} for i in range(90)]
        + [{"key": f"h{i:02d}", **audio_record("happy")} for i in range(10)],
    )
    text_path = write_jsonl(
        tmp_path / "text.jsonl",
        [{"key": f"n{i:02d}", **text_record("neutral")} for i in range(90)]
        + [{"key": f"h{i:02d}", **text_record("happy")} for i in range(10)],
    )
    out = tmp_path / "emo.jsonl"
    assert (
        merge.main(
            [
                "--audio", str(audio_path),
                "--text", str(text_path),
                "--out", str(out),
                "--neutral-cap", "0.5",
            ]
        )
        == 0
    )
    stats = json.loads((tmp_path / "emo.jsonl.stats.json").read_text(encoding="utf-8"))["stats"]
    assert stats["neutral_cap"] == {
        "cap": 0.5,
        "enabled": True,
        "neutral_before": 90,
        "neutral_after": 10,
        "demoted": 80,
    }
    assert stats["label_distribution"]["counts"]["<|NEUTRAL|>"] == 10
    assert stats["num_usable"] == 20


# --------------------------------------------------- the join overlap guard (B3)


def _overlap_files(tmp_path, audio_keys, text_keys):
    """Write two label files with the given key sets, all agreeing on happy."""
    audio_path = write_jsonl(
        tmp_path / "audio.jsonl", [{"key": k, **audio_record("happy")} for k in audio_keys]
    )
    text_path = write_jsonl(
        tmp_path / "text.jsonl", [{"key": k, **text_record("happy")} for k in text_keys]
    )
    return audio_path, text_path


def test_overlap_fraction_is_jaccard_over_processed_keys(merge):
    """|A n T| / |A u T|, over what each labeller *processed*."""
    audio = merge.LabelFile(labels={}, keys={"a", "b", "c"})
    text = merge.LabelFile(labels={}, keys={"b", "c", "d"})
    assert merge.overlap_fraction(audio, text) == pytest.approx(0.5)  # 2 shared / 4 union
    assert merge.overlap_fraction(audio, audio) == 1.0
    assert merge.overlap_fraction(
        merge.LabelFile({}, set()), merge.LabelFile({}, set())
    ) == 0.0


def test_overlap_counts_unparsed_clips_as_processed(merge, tmp_path):
    """A labeller that ran everywhere and parsed nothing is not a subset mismatch.

    Those two faults need different fixes -- rerun with matching --sample versus
    fix the prompt -- so the overlap metric must not conflate them. The unparsed
    case is caught by the text labeller's own guard instead.
    """
    audio_path = write_jsonl(
        tmp_path / "audio.jsonl", [{"key": k, **audio_record("happy")} for k in "abcd"]
    )
    text_path = write_jsonl(
        tmp_path / "text.jsonl",
        [{"key": k, "label": None, "score": None, "raw": "?"} for k in "abcd"],
    )
    audio = merge.load_label_file(audio_path)
    text = merge.load_label_file(text_path)
    assert text.labels == {}
    assert merge.overlap_fraction(audio, text) == 1.0


def test_disjoint_key_sets_exit_nonzero(merge, tmp_path):
    """Two labellers over different subsets must fail, not merge cleanly.

    This is the fault the guard exists for: without it the join succeeds, every
    clip lands in ``missing_masked``, and the run reports ``usable labels: 0
    (0.0%)`` at exit 0 -- a complete-looking file that supervises nothing.
    """
    audio_path, text_path = _overlap_files(tmp_path, ["a", "b", "c"], ["x", "y", "z"])
    out = tmp_path / "emo.jsonl"
    argv = ["--audio", str(audio_path), "--text", str(text_path), "--out", str(out)]
    assert merge.main(argv) == 1
    assert not out.exists()


def test_partial_overlap_below_the_floor_exits_nonzero(merge, tmp_path, capsys):
    """Partial mismatch is the dangerous case -- the statistics look plausible.

    A total mismatch is obvious in any summary. A 60% overlap produces an
    agreement rate, a label distribution and a confusion matrix that all read as
    ordinary, with 40% of the corpus quietly unsupervised.
    """
    audio_path, text_path = _overlap_files(
        tmp_path, [f"k{i}" for i in range(8)], [f"k{i}" for i in range(4, 12)]
    )
    out = tmp_path / "emo.jsonl"
    argv = ["--audio", str(audio_path), "--text", str(text_path), "--out", str(out)]
    assert merge.main(argv) == 1
    message = capsys.readouterr().err
    assert "--sample" in message and "--seed" in message  # names the likely cause
    assert not out.exists()


def test_overlap_above_the_floor_passes(merge, tmp_path):
    """A small pre-emption tail must not fail the merge.

    The floor is 0.95, not 1.0, precisely so a labeller killed a few clips short
    of the other still merges.
    """
    keys = [f"k{i:03d}" for i in range(100)]
    audio_path, text_path = _overlap_files(tmp_path, keys, keys[:98])
    out = tmp_path / "emo.jsonl"
    argv = ["--audio", str(audio_path), "--text", str(text_path), "--out", str(out)]
    assert merge.main(argv) == 0


def test_min_overlap_zero_disables_the_guard(merge, tmp_path):
    audio_path, text_path = _overlap_files(tmp_path, ["a"], ["b"])
    out = tmp_path / "emo.jsonl"
    assert (
        merge.main(
            [
                "--audio", str(audio_path), "--text", str(text_path),
                "--out", str(out), "--min-overlap", "0",
            ]
        )
        == 0
    )


def test_stats_report_the_join_shape(merge, tmp_path):
    """``num_audio_only`` / ``num_text_only`` / ``overlap_fraction`` are correct."""
    audio_path, text_path = _overlap_files(tmp_path, ["a", "b", "c"], ["b", "c", "d", "e"])
    out = tmp_path / "emo.jsonl"
    assert (
        merge.main(
            [
                "--audio", str(audio_path), "--text", str(text_path),
                "--out", str(out), "--min-overlap", "0",
            ]
        )
        == 0
    )
    stats = json.loads((tmp_path / "emo.jsonl.stats.json").read_text(encoding="utf-8"))["stats"]
    assert stats["num_audio_keys"] == 3
    assert stats["num_text_keys"] == 4
    assert stats["num_audio_only"] == 1  # "a"
    assert stats["num_text_only"] == 2  # "d", "e"
    assert stats["overlap_fraction"] == pytest.approx(2 / 5)


def test_missing_breakdown_separates_absent_from_unusable(merge, tmp_path):
    """``missing_masked`` alone cannot say which fault occurred.

    "never given to that labeller" is an operator error in how the jobs were
    launched; "labeller ran and produced nothing" is a model or prompt problem.
    Reporting one number for both sends the reader to the wrong place.
    """
    audio_path = write_jsonl(
        tmp_path / "audio.jsonl", [{"key": k, **audio_record("happy")} for k in ["a", "b", "c"]]
    )
    text_path = write_jsonl(
        tmp_path / "text.jsonl",
        [
            {"key": "a", **text_record("happy")},
            {"key": "b", "label": None, "score": None, "raw": "???"},  # ran, unusable
            {"key": "d", **text_record("happy")},  # text-only
        ],
    )
    out = tmp_path / "emo.jsonl"
    assert (
        merge.main(
            [
                "--audio", str(audio_path), "--text", str(text_path),
                "--out", str(out), "--min-overlap", "0",
            ]
        )
        == 0
    )
    stats = json.loads((tmp_path / "emo.jsonl.stats.json").read_text(encoding="utf-8"))["stats"]
    assert stats["missing_breakdown"] == {
        "audio_only": 1,  # "c": audio ran, text never saw it
        "text_only": 1,  # "d": text ran, audio never saw it
        "audio_unusable": 0,
        "text_unusable": 1,  # "b": both ran, text did not parse
    }


# ------------------------------------------------------------- cap validation


def test_the_retired_fallback_flag_is_rejected(merge, tmp_path, capsys):
    """``--audio-conf-fallback`` must not silently no-op if someone still passes it.

    Retired flags in old scripts and job files are the realistic way a removed
    behaviour comes back: argparse rejecting the flag tells the operator to go
    read why it is gone, whereas accepting and ignoring it would let a job that
    *thinks* it enabled the override run as if it had.
    """
    audio_path, text_path = _overlap_files(tmp_path, ["a"], ["a"])
    out = tmp_path / "emo.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        merge.main(
            [
                "--audio", str(audio_path), "--text", str(text_path),
                "--out", str(out), "--audio-conf-fallback", "0.7",
            ]
        )
    assert excinfo.value.code == 2  # argparse: unrecognised argument
    assert not out.exists()


def test_stats_no_longer_report_a_fallback(merge, tmp_path):
    """The stats block for the retired flag is gone, not left reading disabled.

    A permanently-``false`` field reads as a switch waiting to be flipped.
    """
    audio_path, text_path = _overlap_files(tmp_path, ["a"], ["a"])
    out = tmp_path / "emo.jsonl"
    assert (
        merge.main(["--audio", str(audio_path), "--text", str(text_path), "--out", str(out)])
        == 0
    )
    stats = json.loads((tmp_path / "emo.jsonl.stats.json").read_text(encoding="utf-8"))["stats"]
    assert "audio_conf_fallback" not in stats
    # ...but the matrix that will inform the replacement rule stays.
    assert "disagreement_confusion" in stats


def test_neutral_cap_warns_when_it_would_delete_all_supervision(merge, capsys):
    """All-neutral input plus a cap yields a file with nothing in it.

    The arithmetic is right -- with no non-neutral labels to be a majority of,
    the cap allows zero neutrals -- but the outcome deserves a warning rather
    than appearing as an ordinary cap effect in the decision counts.
    """
    rows = [
        merge.MergedRow(f"n{i}", merge.NEUTRAL_TOKEN, "agree", "neutral", 0.5, "neutral")
        for i in range(5)
    ]
    assert merge.apply_neutral_cap(rows, 0.5, warn=sys.stderr) == 5
    assert "no emotion supervision" in capsys.readouterr().err
    assert all(row.emo_target == merge.MASK_TOKEN for row in rows)


def test_neutral_cap_key_tiebreak_is_load_bearing(merge):
    """Equal scores must resolve by key, not by input order.

    ``apply_neutral_cap`` is public and may be handed rows in any order. When
    every neutral shares a confidence -- which happens whenever the audio
    labeller is saturated, and in any file where scores are absent -- Python's
    stable sort would otherwise preserve arrival order, making the demoted set
    depend on how the caller happened to build the list. Blanking ``row.key`` in
    ``_neutral_sort_key`` fails only this test.
    """
    rows = [
        merge.MergedRow(key, merge.NEUTRAL_TOKEN, "agree", "neutral", 0.5, "neutral")
        for key in ["n_zulu", "n_alpha", "n_mike", "n_bravo"]
    ]
    rows.append(merge.MergedRow("h", "<|HAPPY|>", "agree", "happy", 0.9, "happy"))
    assert merge.apply_neutral_cap(rows, 0.5) == 3

    survivor = [r.key for r in rows if r.emo_target == merge.NEUTRAL_TOKEN]
    assert survivor == ["n_zulu"]  # last by key, not last by input order
    demoted = sorted(r.key for r in rows if r.decision == "neutral_capped")
    assert demoted == ["n_alpha", "n_bravo", "n_mike"]


# ------------------------------------------------------------- response parsing


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param("happy", "happy", id="bare"),
        pytest.param(" Happy. ", "happy", id="whitespace-case-and-full-stop"),
        pytest.param("**happy**", "happy", id="markdown-bold"),
        pytest.param("「happy」", "happy", id="japanese-corner-brackets"),
        pytest.param('"sad"', "sad", id="ascii-quotes"),
        pytest.param("`neutral`", "neutral", id="code-span"),
        pytest.param("SEXUAL", "sexual", id="uppercase-aux-class"),
        pytest.param("喜び", "happy", id="japanese-gloss"),
        pytest.param("性的", "sexual", id="japanese-gloss-aux"),
        pytest.param("その他", "other", id="japanese-gloss-other"),
        pytest.param("angry\n", "angry", id="trailing-newline"),
    ],
)
def test_parse_response_accepts_decorated_single_word_answers(text, response, expected):
    """Instruct models decorate one-word answers; that is not a refusal."""
    assert text.parse_response(response) == expected


def test_parse_response_accepts_an_answer_followed_by_reasoning(text):
    """Only the first line is read, so a model that explains itself still counts.

    Deliberate, and the boundary is the line break. Answering "happy" and then
    justifying it is a formatting deviation from the "one word" instruction, not
    a hedge -- the model committed, on the line where the answer goes. A hedge
    *within* the first line is still rejected, which is what keeps this from
    being a substring match by another name.
    """
    assert text.parse_response("happy\nこの台詞は嬉しそうなので") == "happy"
    assert text.parse_response("sad\n\nReasoning: the speaker is crying.") == "sad"
    assert text.parse_response("I think happy\nbut maybe sad") is None


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("I think this is happy or maybe sad", id="hedged-prose"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace-only"),
        pytest.param("elated", id="not-a-class"),
        pytest.param("happy, sad", id="two-classes"),
        pytest.param("感情", id="japanese-non-class"),
    ],
)
def test_parse_response_never_guesses(text, response):
    """``None`` on anything that is not exactly one known class.

    A substring match would read "happy" out of the hedged sentence and record
    it as a confident label. That sentence is the model declining to commit;
    treating it as a decision injects exactly the noise the two-labeller
    intersection exists to keep out. ``None`` propagates to the merge as a
    missing label and masks the clip -- the cheap failure.
    """
    assert text.parse_response(response) is None


# ------------------------------------------------------------------ the prompt


def test_build_prompt_lists_every_class_it_asks_for(text):
    """Prompt glossary and mapping table must not drift apart.

    A class the prompt never mentions is one the model will never emit, so its
    mapping entry is dead and that emotion is quietly absent from the corpus.
    The reverse -- a class in the prompt with no mapping entry -- crashes the
    merge mid-sweep.
    """
    messages = text.build_prompt("こんにちは")
    rendered = "\n".join(message["content"] for message in messages)
    for name in text.TEXT_CLASSES:
        assert name in rendered, name
        assert text.CLASS_GLOSS_JA[name] in rendered, name


def test_build_prompt_is_a_chat_conversation_carrying_the_transcript(text):
    messages = text.build_prompt("嘘だろ！")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "嘘だろ！" in messages[-1]["content"]


def test_build_prompt_routes_uncertainty_to_other_not_neutral(text):
    """Uncertainty must not land in the class that already collapsed once.

    An unsure ``other`` becomes a mask and costs only coverage; an unsure
    ``neutral`` survives the agreement filter and rebuilds the neutral majority
    the whole round is trying to break.
    """
    system = text.build_prompt("あ")[0]["content"]
    assert "neutral ではなく other" in system


def test_prompt_never_offers_emo_unknown(text, merge):
    """The prompt's classes are the mapped ones and nothing else."""
    assert "EMO_UNKNOWN" not in "\n".join(m["content"] for m in text.build_prompt("あ"))
    assert set(text.TEXT_CLASSES) == set(merge.TEXT_LABEL_TO_TOKEN)


# --------------------------------------------------- audio label parsing + CLI


#: The **exact** contents of ``emotion2vec_plus_large/tokens.txt``, one entry per
#: line, at revision ``b9c9fc7fce7dd80c0a59d9d1d1265021e31cb2e8``.
#:
#: sha256 ``866121e470057b847d7a50e9923509141fb2924392f53385a186482a1ec0fb7f``
#: over ``"\n".join(TOKENS_TXT_LINES)`` -- 119 bytes, **no trailing newline**.
#: :func:`test_tokens_fixture_matches_the_staged_artefact` recomputes that digest
#: on every run, so this fixture cannot drift from the artefact silently: edit
#: any character here and the test fails.
#:
#: This list is transcribed from the real file, not reconstructed from the shape
#: the other eight lines suggest. That distinction is the entire point. The
#: previous version of this test asserted the ninth label was ``"<unk>/unknown"``
#: -- a form the model has never emitted at any revision -- and passed, because
#: it fed ``parse_class_label`` a string invented for the occasion and checked
#: the invention against itself. Against the staged model the sweep died on clip
#: 1 of batch 1.
TOKENS_TXT_LINES = (
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",  # bare: no slash, no English half -- the defect this pins
)

TOKENS_TXT_SHA256 = "866121e470057b847d7a50e9923509141fb2924392f53385a186482a1ec0fb7f"

#: Optional: point this at a staged ``tokens.txt`` to check the fixture against
#: the real file rather than against its recorded digest.
TOKENS_TXT_ENV = "SENSEVOICE_EMOTION2VEC_TOKENS"


def test_tokens_fixture_matches_the_staged_artefact():
    """The pinned lines must hash to the digest measured on the staged model.

    Without this the fixture is just another assumption, indistinguishable from
    the one it replaced. With it, the nine lines below are checkable evidence:
    the digest was measured on the artefact, and any edit to the fixture breaks
    the hash.
    """
    content = "\n".join(TOKENS_TXT_LINES)
    assert len(content.encode("utf-8")) == 119
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == TOKENS_TXT_SHA256


def test_tokens_fixture_matches_a_staged_file_when_one_is_available():
    """Verify against the artefact itself when the environment points at it.

    Skipped on a workstation with no staged model. On the cluster this is what
    turns the fixture from a recorded claim into a checked one, and it is the
    test to run first after staging a different revision.
    """
    staged = os.environ.get(TOKENS_TXT_ENV)
    if not staged:
        pytest.skip(f"set {TOKENS_TXT_ENV} to a staged tokens.txt to check against it")
    path = Path(staged)
    if not path.is_file():
        pytest.skip(f"{TOKENS_TXT_ENV}={staged} is not a file")
    actual = path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    assert tuple(actual) == TOKENS_TXT_LINES


def test_parse_class_label_covers_every_line_of_the_real_tokens_file(audio):
    """All nine real labels map onto the nine classes, one-to-one.

    A test about the model's actual output rather than about what we hoped it
    would be. Surjectivity is asserted as well as totality: if two lines
    collapsed onto one class, some class would be unreachable and that emotion
    would be silently absent from the corpus.
    """
    mapped = [audio.parse_class_label(line) for line in TOKENS_TXT_LINES]
    assert len(mapped) == 9
    assert set(mapped) == set(audio.AUDIO_CLASSES)
    assert len(set(mapped)) == len(mapped)  # no two lines share a class


def test_bare_unk_maps_to_unknown(audio):
    """The ninth line has no English half; splitting on "/" yields "<unk>".

    This single line is what killed the first cluster run. Every revision
    checked -- c43c13ee, v2.0.5 and master head -- emits it bare, so there is no
    revision where the split-on-slash rule alone suffices.
    """
    assert audio.parse_class_label("<unk>") == "unknown"
    assert "<unk>" in audio._CLASS_ALIASES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("生气/angry", "angry", id="bilingual"),
        pytest.param("开心/happy", "happy", id="bilingual-happy"),
        pytest.param("其他/other", "other", id="other-class"),
        pytest.param("neutral", "neutral", id="english-only"),
        pytest.param(" 难过/Sad ", "sad", id="whitespace-and-case"),
        pytest.param(" <UNK> ", "unknown", id="bare-unk-whitespace-and-case"),
        # Not observed at any revision, but harmless to accept: if a future
        # checkpoint regularises the ninth line into a pair, the split rule
        # already handles it and this stays green.
        pytest.param("<unk>/unknown", "unknown", id="hypothetical-paired-unk"),
    ],
)
def test_parse_class_label_reads_the_english_half(audio, raw, expected):
    """Only the English half is trusted -- the Chinese half has changed before."""
    assert audio.parse_class_label(raw) == expected


def test_parse_class_label_rejects_the_v205_placeholder_slots(audio):
    """``v2.0.5``'s ``tokens.txt`` has four ``unuse_N`` slots where classes go.

    Pinned as a rejection so that staging that tag fails fast and loudly rather
    than mid-sweep. The module docstring tells operators to stage 767b2e00 or
    later; this is the check that backs the instruction up.
    """
    for slot in ["unuse_0", "unuse_1", "unuse_2", "unuse_3"]:
        with pytest.raises(ValueError, match="unrecognised emotion2vec class"):
            audio.parse_class_label(slot)


def test_parse_class_label_rejects_an_unknown_class(audio):
    """Loud failure beats bucketing into ``other``.

    Bucketing would map a renamed class to the mask and silently remove that
    emotion from ~550k clips of supervision, with nothing in the logs.
    """
    with pytest.raises(ValueError, match="unrecognised emotion2vec class"):
        audio.parse_class_label("愉快/elated")


def test_audio_classes_cover_all_nine(audio):
    """emotion2vec+ large emits nine classes, ``other`` and ``unknown`` included."""
    assert set(audio.AUDIO_CLASSES) == {
        "angry", "disgusted", "fearful", "happy", "neutral",
        "other", "sad", "surprised", "unknown",
    }


def test_resume_skips_keys_already_written(audio, tmp_path):
    """Resume is what makes a multi-hour pre-emptible job survivable."""
    path = write_jsonl(
        tmp_path / "out.jsonl",
        [{"key": "a", **audio_record("happy")}, {"key": "b", **audio_record("sad")}],
    )
    assert audio.read_done_keys(path) == {"a", "b"}
    assert audio.read_done_keys(tmp_path / "absent.jsonl") == set()


def test_resume_tolerates_a_truncated_final_line(audio, tmp_path):
    """A killed job's last line is usually half-written; its clip is redone."""
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"key": "a", **audio_record("happy")}) + '\n{"key": "b", "lab',
        encoding="utf-8",
    )
    assert audio.read_done_keys(path) == {"a"}


def test_sample_is_seeded_and_reproducible(audio, text):
    """Both labellers must draw the same pilot subset from the same seed.

    A pilot is only interpretable if the two labellers saw the same clips;
    otherwise the "disagreement" rate is mostly non-overlap.
    """
    audio_clips = [audio.ManifestClip(key=f"k{i:03d}", source=f"/a/{i}.wav") for i in range(100)]
    text_clips = [text.ManifestClip(key=f"k{i:03d}", target="あ") for i in range(100)]
    audio_keys = [c.key for c in audio.select_clips(audio_clips, sample=10, seed=7)]
    text_keys = [c.key for c in text.select_clips(text_clips, sample=10, seed=7)]
    assert audio_keys == text_keys
    assert len(audio_keys) == 10
    assert audio_keys == [c.key for c in audio.select_clips(audio_clips, sample=10, seed=7)]


def test_limit_and_sample_are_mutually_exclusive(audio):
    """They answer different questions; composing them silently would hide which."""
    clips = [audio.ManifestClip(key=f"k{i}", source=f"/a/{i}.wav") for i in range(10)]
    with pytest.raises(ValueError, match="mutually exclusive"):
        audio.select_clips(clips, limit=3, sample=3)


def test_limit_streams_without_reading_the_whole_manifest(audio):
    """``data/vn/train.jsonl`` is large; a pilot must not parse all of it."""

    def endless():
        index = 0
        while True:
            yield audio.ManifestClip(key=f"k{index}", source=f"/a/{index}.wav")
            index += 1

    assert len(audio.select_clips(endless(), limit=4)) == 4


#: Modules whose presence after a bare import means the laziness was lost.
#: ``torch``/``torchaudio`` cost a multi-second CUDA init, ``funasr`` and
#: ``vllm`` are absent from this venv entirely, and ``soundfile`` /
#: ``transformers`` / ``modelscope`` ride in behind them.
HEAVY_MODULES = (
    "torch",
    "torchaudio",
    "funasr",
    "vllm",
    "transformers",
    "soundfile",
    "modelscope",
)

#: Imports the script by path in a *fresh* interpreter and reports which heavy
#: modules its import graph pulled in.  Printed as JSON on the last stdout line
#: so the assertion can name the offenders.
_IMPORT_PROBE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location({name!r}, {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules[{name!r}] = module
spec.loader.exec_module(module)
print(json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)))
"""


# ------------------------------------------- labeller main() with a stub backend
#
# These are the tests whose absence let B1 and B2 through review: every unit
# below the CLI passed, while the CLI itself turned a total systemic failure
# into exit 0 and an empty output file.


class StubBackend:
    """A labeller backend with scripted per-batch behaviour.

    ``script`` is consulted per batch: ``"ok"`` labels normally, anything else is
    raised as that exception instance. Exhausting it repeats the last entry, so
    ``["boom"]`` means "fail forever".
    """

    def __init__(self, script, emit="neutral", **kwargs):
        self.script = list(script)
        self.emit = emit  # not ``label``: that would shadow the ``label`` method
        self.calls = 0
        self.kwargs = kwargs

    def _step(self):
        entry = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if entry != "ok":
            raise entry


class StubAudioBackend(StubBackend):
    def label(self, paths):
        self._step()
        return [{"label": self.emit, "score": 0.8, "probs": {self.emit: 0.8}} for _ in paths]


class StubTextBackend(StubBackend):
    def label(self, transcripts):
        self._step()
        return [{"label": self.emit, "score": 0.8, "raw": str(self.emit)} for _ in transcripts]


@pytest.fixture
def manifest(tmp_path):
    """A 40-clip manifest, enough for several batches."""
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {"key": f"T__s__{i:03d}", "source": f"/audio/{i}.wav", "target": "こんにちは"},
                ensure_ascii=False,
            )
            + "\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )
    return path


def _install(monkeypatch, module, attr, backend):
    monkeypatch.setattr(module, attr, lambda **kwargs: backend)
    return backend


@pytest.mark.parametrize(
    ("name", "attr", "backend_cls"),
    [
        ("audio", "Emotion2vecLabeller", StubAudioBackend),
        ("text", "VllmLabeller", StubTextBackend),
    ],
)
def test_every_batch_failing_exits_nonzero(
    request, monkeypatch, manifest, tmp_path, name, attr, backend_cls
):
    """A systemic fault must not walk the corpus and exit clean.

    The realistic causes are not ``RuntimeError``: a wrong audio mount raises
    ``FileNotFoundError``, a missing chat template ``ValueError``, a corrupt
    staged model ``OSError``. All of them recur on every batch, so the old
    catch-and-continue produced exit 0 with zero rows written -- the same shape
    as the incident where a run trained on 16.5 hours instead of 813.8 and
    reported success.
    """
    module = request.getfixturevalue(name)
    _install(monkeypatch, module, attr, backend_cls([FileNotFoundError("no such mount")]))
    out = tmp_path / "labels.jsonl"
    code = module.main(
        ["--manifest", str(manifest), "--out", str(out), "--batch-size", "10"]
    )
    assert code == 1
    assert out.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("name", "attr", "backend_cls"),
    [
        ("audio", "Emotion2vecLabeller", StubAudioBackend),
        ("text", "VllmLabeller", StubTextBackend),
    ],
)
def test_consecutive_failures_abort_before_the_corpus_is_exhausted(
    request, monkeypatch, manifest, tmp_path, name, attr, backend_cls
):
    """The abort fires on the run of failures, not merely at the end.

    With ~550k clips the difference is hours of GPU time and a queue slot.
    """
    module = request.getfixturevalue(name)
    backend = _install(monkeypatch, module, attr, backend_cls([OSError("corrupt model")]))
    out = tmp_path / "labels.jsonl"
    code = module.main(
        [
            "--manifest", str(manifest), "--out", str(out),
            "--batch-size", "1", "--max-consecutive-failures", "3",
        ]
    )
    assert code == 1
    assert backend.calls == 3  # stopped at the third, not after all 40


@pytest.mark.parametrize(
    ("name", "attr", "backend_cls"),
    [
        ("audio", "Emotion2vecLabeller", StubAudioBackend),
        ("text", "VllmLabeller", StubTextBackend),
    ],
)
def test_isolated_batch_failures_do_not_abort_the_run(
    request, monkeypatch, manifest, tmp_path, name, attr, backend_cls
):
    """The counter resets on success -- one bad clip must not lose the sweep.

    This is the property the abort must not break: scattered failures are data
    (a truncated wav, one unreadable file), and only a *run* of them is
    systemic.
    """
    module = request.getfixturevalue(name)
    script = [ValueError("bad clip"), "ok"] * 20
    _install(monkeypatch, module, attr, backend_cls(script))
    out = tmp_path / "labels.jsonl"
    code = module.main(
        [
            "--manifest", str(manifest), "--out", str(out),
            "--batch-size", "1", "--max-consecutive-failures", "2",
        ]
    )
    assert code == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 20


@pytest.mark.parametrize(
    ("name", "attr", "backend_cls"),
    [
        ("audio", "Emotion2vecLabeller", StubAudioBackend),
        ("text", "VllmLabeller", StubTextBackend),
    ],
)
def test_successful_run_reports_rows_written(
    request, monkeypatch, manifest, tmp_path, capsys, name, attr, backend_cls
):
    """"How much did this produce" must never require inferring from exit code."""
    module = request.getfixturevalue(name)
    _install(monkeypatch, module, attr, backend_cls(["ok"]))
    out = tmp_path / "labels.jsonl"
    assert module.main(["--manifest", str(manifest), "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert "rows written    : 40/40" in err
    assert "batches skipped : 0" in err


def test_all_responses_unparsed_exits_nonzero(monkeypatch, text, manifest, tmp_path):
    """100% unparsed is a broken prompt, and nothing else reports it.

    The merge masks every clip correctly, so no *wrong* label is produced and no
    downstream step complains -- the emotion head simply receives no supervision
    from the whole corpus, which is the round-1/2 collapse reproduced exactly.
    """
    _install(monkeypatch, text, "VllmLabeller", StubTextBackend(["ok"], emit=None))
    out = tmp_path / "labels.jsonl"
    code = text.main(["--manifest", str(manifest), "--out", str(out)])
    assert code == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40  # the evidence is written; the exit code is the signal
    assert all(row["label"] is None for row in rows)


def test_unparsed_below_the_threshold_still_succeeds(monkeypatch, text, manifest, tmp_path):
    """A minority of unparsed responses is normal and must not fail the run."""
    script = ["ok"] * 40
    backend = StubTextBackend(script, emit="happy")
    calls = {"n": 0}
    original = backend.label

    def alternating(transcripts):
        records = original(transcripts)
        for record in records:
            calls["n"] += 1
            if calls["n"] % 5 == 0:  # 20% unparsed
                record["label"] = None
        return records

    backend.label = alternating
    _install(monkeypatch, text, "VllmLabeller", backend)
    out = tmp_path / "labels.jsonl"
    assert text.main(["--manifest", str(manifest), "--out", str(out)]) == 0


def test_max_unparsed_frac_is_configurable(monkeypatch, text, manifest, tmp_path):
    """A stricter threshold catches a partially broken prompt."""
    _install(monkeypatch, text, "VllmLabeller", StubTextBackend(["ok"], emit=None))
    out = tmp_path / "labels.jsonl"
    assert (
        text.main(
            [
                "--manifest", str(manifest), "--out", str(out),
                "--max-unparsed-frac", "1.0",
            ]
        )
        == 0
    )


def test_hub_id_model_dir_warns(audio, monkeypatch, manifest, tmp_path, capsys):
    """A hub id on a compute node fails late, after the queue time is spent."""
    assert audio.looks_like_hub_id("iic/emotion2vec_plus_large")
    assert not audio.looks_like_hub_id(str(tmp_path))
    assert not audio.looks_like_hub_id("emotion2vec")

    _install(monkeypatch, audio, "Emotion2vecLabeller", StubAudioBackend(["ok"]))
    out = tmp_path / "labels.jsonl"
    assert audio.main(["--manifest", str(manifest), "--out", str(out)]) == 0
    assert "looks like a hub id" in capsys.readouterr().err


@pytest.mark.parametrize(
    "script", ["label_emotions_audio", "label_emotions_text", "merge_emo_labels"]
)
def test_importing_the_module_pulls_in_no_model_stack(script, tmp_path):
    """funasr, vllm and torch must all be imported lazily.

    They are absent from this venv and heavyweight on the cluster. A
    module-level import would make ``--help``, and every test above, unrunnable
    here -- and would cost a multi-second CUDA init on a merge that touches no
    GPU at all.

    The import happens in a **fresh subprocess**, which is the whole point.
    Asserting against this interpreter's ``sys.modules`` would measure the
    *process*, not the module: ``tests/test_rich_loss_mask.py`` imports torch to
    build a model, so under a combined run this test reported on whichever file
    pytest collected first and passed or failed by collection order rather than
    by anything about the code. A subprocess makes the assertion a statement
    about the module's own import graph, which is the property worth pinning.

    Run from an unrelated cwd for the same reason as the ``--help`` tests: these
    scripts run wherever the scheduler drops them.
    """
    probe = _IMPORT_PROBE.format(
        name=script, path=str(SCRIPTS / f"{script}.py"), heavy=list(HEAVY_MODULES)
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    landed = json.loads(result.stdout.strip().splitlines()[-1])
    assert landed == [], (
        f"{script}.py imports the model stack at module level: {', '.join(landed)}. "
        "Move the import inside the backend function that needs it."
    )


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("label_emotions_audio.py", ["--manifest", "--out", "--model-dir", "--device",
                                     "--batch-size", "--limit", "--sample", "--seed",
                                     "--max-consecutive-failures"]),
        ("label_emotions_text.py", ["--manifest", "--out", "--model", "--batch-size",
                                    "--limit", "--sample", "--seed",
                                    "--max-consecutive-failures", "--max-unparsed-frac"]),
        ("merge_emo_labels.py", ["--audio", "--text", "--out", "--stats-out",
                                 "--neutral-cap", "--sample", "--min-overlap"]),
    ],
)
def test_help_runs_without_the_model_stack(script, expected, tmp_path):
    """``--help`` on a machine with neither funasr nor vllm, from any cwd.

    These run inside a container on a cluster node where cwd is whatever the
    scheduler picked, so the run is made from an unrelated directory.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    for flag in expected:
        assert flag in result.stdout, flag


def test_missing_label_file_exits_nonzero_without_writing(merge, tmp_path):
    """Bad input fails before anything is created."""
    out = tmp_path / "emo.jsonl"
    code = merge.main(
        [
            "--audio", str(tmp_path / "nope.jsonl"),
            "--text", str(tmp_path / "nope2.jsonl"),
            "--out", str(out),
        ]
    )
    assert code == 1
    assert not out.exists()
