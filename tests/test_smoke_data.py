"""Unit tests for ``scripts/make_smoke_data.py`` (CPU smoke corpus generator).

The script slices one wav into a handful of dummy clips so that a smoke run can
prove the training plumbing works before a real run spends about two days of GPU
time.  Nothing here touches the repository's own wav or the generated
``data/smoke*`` artefacts: every test writes into ``tmp_path`` over a source wav
synthesised with the stdlib :mod:`wave` module, so the suite needs no fixtures on
disk and cannot clobber a manifest another test reads as ground truth.

Two groups of assertions here are *pinning* tests rather than free-choice ones.

The default is a constant ``<|NEUTRAL|>``
    That constant is what every existing smoke run and every checked-in
    expectation was built on, so ``--emo-mix`` had to be strictly opt-in.  The
    default-path tests below exist to catch a future change that makes variation
    the default and silently shifts output nobody was watching.

The mix is what makes the emotion-slot mask testable
    A manifest may carry the ``<|SER|>`` sentinel for a clip with no reliable
    emotion label, and ``model.py`` maps that token to ``ignore_id`` so the slot
    drops out of the rich cross-entropy loss while the clip still trains CTC.
    A constant emotion target saturates ``acc_emo`` immediately -- that is the
    round-1/2 defect the sentinel exists to fix -- so an all-``<|NEUTRAL|>``
    manifest cannot prove the masking works.  The distribution, legality and
    determinism tests below are the gate on the manifest the smoke run reads.
"""

import array
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "scripts" / "make_smoke_data.py"


def _load_make_smoke_data():
    """Import ``scripts/make_smoke_data.py`` by path -- ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("make_smoke_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_make_smoke_data() if SCRIPT_PATH.is_file() else None

pytestmark = pytest.mark.skipif(
    smoke is None, reason=f"{SCRIPT_PATH} does not exist yet"
)

# The schema of data/train_example.jsonl, in order.  json.dumps preserves dict
# insertion order, so the generated manifests stay byte-comparable with the
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

# The token this script wrote for every record before --emo-mix existed.
NEUTRAL = "<|NEUTRAL|>"
SER = "<|SER|>"
EMO_UNKNOWN = "<|EMO_UNKNOWN|>"

# The source wav's sample rate.  16 kHz is what runtime/llama.cpp/tests/sample.wav
# uses and what the model consumes; the clip lengths in the manifest are derived
# from it, so a synthetic source has to match.
SAMPLE_RATE = 16000


# --- helpers ----------------------------------------------------------------


def write_source_wav(path, seconds=2.0, rate=SAMPLE_RATE):
    """A short mono PCM16 wav, written with the stdlib only.

    Content is irrelevant -- the script slices and wraps whatever it is given --
    but it must be non-silent so nothing downstream mistakes it for a bad read.
    """
    count = int(seconds * rate)
    samples = array.array(
        "h", (int(8000 * np.sin(2 * np.pi * 440 * i / rate)) for i in range(count))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


@pytest.fixture(scope="module")
def source_wav(tmp_path_factory):
    return write_source_wav(tmp_path_factory.mktemp("source") / "sample.wav")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_smoke(out_dir, source_wav, *extra_args):
    """Run the generator into ``out_dir`` and return its two manifests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"
    argv = [
        "--source-wav", str(source_wav),
        "--wav-dir", str(out_dir / "wav"),
        "--train-jsonl", str(train_jsonl),
        "--val-jsonl", str(val_jsonl),
        *extra_args,
    ]
    assert smoke.main(argv) == 0
    return read_jsonl(train_jsonl), read_jsonl(val_jsonl)


def targets(records):
    return [record["emo_target"] for record in records]


def masked_share(values):
    return values.count(SER) / len(values)


def longest_masked_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value == SER else 0
        best = max(best, current)
    return best


# --- the constants the manifest is written from -----------------------------


def test_the_default_emotion_target_is_the_neutral_token():
    """Pinned: every record carried this literal before --emo-mix existed."""
    assert smoke.EMO_TARGET == NEUTRAL


def test_the_mask_sentinel_is_ser():
    """Pinned: model.py rewrites exactly this token to ignore_id.

    ``<|SER|>`` is a single token (id 24991 in
    chn_jpn_yue_eng_ko_spectok.bpe.model), which is what lets the emotion slot be
    swapped for ignore_id without shifting any other position in the text
    tensor.  scripts/prepare_vn_data.py writes the same sentinel; the two
    generators feed one trainer and must agree.
    """
    assert smoke.EMO_MASK_TARGET == SER
    assert smoke.EMO_MASK_TARGET in smoke.EMO_ALL_TARGETS


def test_the_seven_emotion_tokens_are_the_labellable_set():
    assert smoke.EMO_LABEL_TARGETS == (
        "<|HAPPY|>",
        "<|SAD|>",
        "<|ANGRY|>",
        NEUTRAL,
        "<|FEARFUL|>",
        "<|DISGUSTED|>",
        "<|SURPRISED|>",
    )
    assert smoke.EMO_ALL_TARGETS == smoke.EMO_LABEL_TARGETS + (SER,)


def test_emo_unknown_is_not_a_legal_target():
    """<|EMO_UNKNOWN|> is the "no prediction" token, never a training target.

    A clip carrying it as a target teaches the model to predict "unknown"; a clip
    with no usable label belongs behind the mask sentinel instead.
    """
    assert EMO_UNKNOWN not in smoke.EMO_ALL_TARGETS


def test_the_default_mask_rate_matches_what_round_3_expects():
    # Round 3's production corpus is roughly 80-85% masked.  The interesting
    # failure modes -- an entirely masked batch, and a rich loss whose
    # denominator has shrunk a long way -- only appear at a realistic rate, so
    # the smoke default deliberately sits inside that band rather than at a
    # convenient round number like 0.5.
    assert 0.80 <= smoke.DEFAULT_EMO_MASK_RATE <= 0.85


# --- emotion_targets: distribution, legality, determinism -------------------


@pytest.mark.parametrize("count", [12, 50, 400])
@pytest.mark.parametrize("rate", [0.5, 0.8, 0.82, 0.85])
def test_the_sentinel_share_matches_the_requested_rate(count, rate):
    values = smoke.emotion_targets(count, rate, smoke.emotion_rng(0))

    assert len(values) == count
    # The masked count is a whole number of records, so the achievable share is
    # quantised to 1/count; anything wider than that is a real drift in the
    # assignment rather than rounding.
    assert abs(masked_share(values) - rate) <= 1.0 / count


@pytest.mark.parametrize("count", [1, 2, 4, 12, 33])
@pytest.mark.parametrize("rate", [0.0, 0.3, 0.82, 1.0])
@pytest.mark.parametrize("seed", [0, 7, 1234])
def test_every_emitted_target_is_one_of_the_eight_legal_tokens(count, rate, seed):
    values = smoke.emotion_targets(count, rate, smoke.emotion_rng(seed))

    assert set(values) <= set(smoke.EMO_ALL_TARGETS)
    assert EMO_UNKNOWN not in values


@pytest.mark.parametrize("count", [2, 4, 12])
def test_a_small_split_still_gets_both_a_sentinel_and_a_real_label(count):
    """At 82% and four val clips, rounding alone could wipe out one or the other.

    A split with no sentinel does not exercise the mask at all; a split with no
    real label leaves the rich loss with an empty denominator.  Both look like a
    healthy run in the log, so the quota is clamped to keep one of each.
    """
    values = smoke.emotion_targets(count, smoke.DEFAULT_EMO_MASK_RATE, smoke.emotion_rng(0))

    assert SER in values
    assert set(values) - {SER}


def test_a_rate_of_zero_or_one_is_honoured_exactly():
    """The clamp is for rounding artefacts, not for overriding an explicit ask.

    ``--emo-mask-rate 1.0`` is how the "every clip in the batch is masked" crash
    case gets forced deliberately, so it must not be quietly softened to n-1.
    """
    assert smoke.emotion_targets(10, 0.0, smoke.emotion_rng(0)).count(SER) == 0
    assert smoke.emotion_targets(10, 1.0, smoke.emotion_rng(0)).count(SER) == 10


def test_the_real_labels_are_varied_rather_than_a_single_token():
    # A constant target -- of any token, not just <|NEUTRAL|> -- saturates
    # acc_emo, which is the defect this flag exists to expose.  With a low mask
    # rate there is room for several distinct emotions and there must be some.
    values = smoke.emotion_targets(20, 0.2, smoke.emotion_rng(0))
    labelled = [value for value in values if value != SER]

    assert len(set(labelled)) > 1


def test_the_same_seed_reproduces_the_same_assignment():
    first = smoke.emotion_targets(12, 0.82, smoke.emotion_rng(3))
    second = smoke.emotion_targets(12, 0.82, smoke.emotion_rng(3))

    assert first == second


def test_a_different_seed_changes_the_arrangement():
    """Guards against the shuffle silently becoming a no-op (a fixed pattern)."""
    seeds = [0, 1, 2, 3, 4, 5]
    arrangements = {
        tuple(smoke.emotion_targets(12, 0.5, smoke.emotion_rng(seed))) for seed in seeds
    }

    assert len(arrangements) > 1


def test_the_emotion_stream_does_not_collide_with_the_offset_stream():
    """The emotion draw is derived from --seed but must not repeat its numbers.

    Reusing ``default_rng(seed)`` directly would tie the label pattern to the
    clip offsets, so the same seed would always mask the same clip lengths.
    """
    seed = 0
    offsets = np.random.default_rng(seed).permutation(12)
    emotions = smoke.emotion_rng(seed).permutation(12)

    assert not np.array_equal(offsets, emotions)


# --- the summary line -------------------------------------------------------


def test_the_summary_counts_what_the_records_actually_contain():
    records = [{"emo_target": t} for t in [SER, SER, "<|HAPPY|>", SER, "<|SAD|>"]]

    summary = smoke.emotion_counts(records)

    assert "3/5 masked (60.0%)" in summary
    assert "<|HAPPY|>x1" in summary
    assert "<|SAD|>x1" in summary


def test_the_summary_names_an_illegal_token_rather_than_hiding_it():
    """An unexpected token must show up in the log, not be filtered away.

    The whole point of the line is that a run cannot succeed at the wrong thing
    unnoticed, so a target outside the eight legal ones has to be visible.
    """
    records = [{"emo_target": t} for t in [SER, EMO_UNKNOWN]]

    assert EMO_UNKNOWN in smoke.emotion_counts(records)


def test_the_summary_line_reports_the_generated_manifests(tmp_path, source_wav, capsys):
    train, val = run_smoke(tmp_path / "run", source_wav, "--emo-mix")
    line = [
        text
        for text in capsys.readouterr().out.splitlines()
        if text.startswith("emotion :")
    ]

    assert len(line) == 1
    expected_train = smoke.emotion_counts(train)
    expected_val = smoke.emotion_counts(val)
    assert line[0] == f"emotion : train {expected_train} | val {expected_val}"
    # And the counts in it are the manifests' own, not the requested rate.
    assert f"{targets(train).count(SER)}/{len(train)} masked" in line[0]
    assert f"{targets(val).count(SER)}/{len(val)} masked" in line[0]


def test_the_default_run_reports_a_fully_unmasked_corpus(tmp_path, source_wav, capsys):
    run_smoke(tmp_path / "run", source_wav)
    out = capsys.readouterr().out

    assert "emotion : train 0/12 masked (0.0%)" in out
    assert f"{NEUTRAL}x12" in out


# --- the default path, pinned -----------------------------------------------


def test_the_default_writes_a_constant_neutral_target(tmp_path, source_wav):
    """Regression guard: --emo-mix must stay opt-in.

    Existing smoke runs, the checked-in data/smoke_*.jsonl and
    finetune_chunk_slurm.sh's constant-label exemption were all built on this.
    """
    train, val = run_smoke(tmp_path / "run", source_wav)

    assert targets(train) == [NEUTRAL] * len(train)
    assert targets(val) == [NEUTRAL] * len(val)
    assert SER not in targets(train) + targets(val)


def test_the_default_record_schema_is_unchanged(tmp_path, source_wav):
    train, _ = run_smoke(tmp_path / "run", source_wav)

    for record in train:
        assert list(record) == EXPECTED_FIELDS


def test_write_split_defaults_to_neutral_without_targets(tmp_path):
    """The default lives in one place and survives a caller that passes nothing."""
    audio = np.zeros(SAMPLE_RATE, dtype="float32")
    records = smoke.write_split(
        audio, SAMPLE_RATE, tmp_path, tmp_path / "m.jsonl",
        "smoke", 3, 1.0, 2.0, np.random.default_rng(0),
    )

    assert targets(records) == [NEUTRAL] * 3


def test_write_split_rejects_a_target_list_of_the_wrong_length(tmp_path):
    audio = np.zeros(SAMPLE_RATE, dtype="float32")

    with pytest.raises(ValueError, match="2 entries for 3 clips"):
        smoke.write_split(
            audio, SAMPLE_RATE, tmp_path, tmp_path / "m.jsonl",
            "smoke", 3, 1.0, 2.0, np.random.default_rng(0),
            emo_targets=[SER, SER],
        )


# --- the --emo-mix path, end to end -----------------------------------------


def test_emo_mix_varies_the_targets(tmp_path, source_wav):
    train, _ = run_smoke(tmp_path / "run", source_wav, "--emo-mix")

    assert len(set(targets(train))) > 1


def test_both_splits_receive_sentinels(tmp_path, source_wav):
    """Neither split may absorb all the masking.

    A train split with no sentinel proves nothing about the mask; a val split
    with no sentinel means validation never exercises it either, and the smoke
    run passes while the machinery under test was never touched.
    """
    train, val = run_smoke(tmp_path / "run", source_wav, "--emo-mix")

    assert SER in targets(train)
    assert SER in targets(val)


def test_both_splits_also_keep_a_real_emotion(tmp_path, source_wav):
    train, val = run_smoke(tmp_path / "run", source_wav, "--emo-mix")

    assert set(targets(train)) - {SER}
    assert set(targets(val)) - {SER}


@pytest.mark.parametrize("rate", ["0.5", "0.8", "0.85"])
def test_the_requested_rate_reaches_the_manifest(tmp_path, source_wav, rate):
    train, _ = run_smoke(
        tmp_path / "run", source_wav, "--emo-mix", "--emo-mask-rate", rate,
        "--num-train", "60",
    )

    # 60 train clips, so one record is worth 1/60 of the share; see the quota
    # note in test_the_sentinel_share_matches_the_requested_rate.
    assert abs(masked_share(targets(train)) - float(rate)) <= 1.0 / len(train)


def test_no_emo_unknown_ever_reaches_a_manifest(tmp_path, source_wav):
    train, val = run_smoke(
        tmp_path / "run", source_wav, "--emo-mix", "--emo-mask-rate", "0.4"
    )

    for record in train + val:
        assert record["emo_target"] in smoke.EMO_ALL_TARGETS
        assert record["emo_target"] != EMO_UNKNOWN


def test_a_whole_batch_can_come_out_masked_at_the_default_rate(tmp_path, source_wav):
    """The all-masked batch is the edge case that has to not crash.

    The smoke run batches by token count with sort_size=4, which packs 2-4
    neighbouring clips per batch (finetune_chunk.sh), so a run of four
    consecutive masked records means at least one batch can be entirely masked.
    At 82% over 12 clips that happens on its own -- this asserts the arrangement
    is left free to produce it, not that it is manufactured.
    """
    train, _ = run_smoke(tmp_path / "run", source_wav, "--emo-mix")

    assert longest_masked_run(targets(train)) >= 4


def test_the_same_seed_reproduces_identical_manifests(tmp_path, source_wav):
    first = run_smoke(tmp_path / "a", source_wav, "--emo-mix", "--seed", "5")
    second = run_smoke(tmp_path / "b", source_wav, "--emo-mix", "--seed", "5")

    assert [targets(split) for split in first] == [targets(split) for split in second]


def test_a_different_seed_changes_the_manifest(tmp_path, source_wav):
    first, _ = run_smoke(tmp_path / "a", source_wav, "--emo-mix", "--seed", "0",
                         "--emo-mask-rate", "0.5", "--num-train", "24")
    second, _ = run_smoke(tmp_path / "b", source_wav, "--emo-mix", "--seed", "1",
                          "--emo-mask-rate", "0.5", "--num-train", "24")

    assert targets(first) != targets(second)


def test_emo_mix_leaves_the_generated_audio_untouched(tmp_path, source_wav):
    """The flag changes labels only.

    The emotion stream is derived from --seed separately from the clip offsets
    precisely so that a smoke failure under --emo-mix can be attributed to the
    labels rather than to differently sliced audio.
    """
    run_smoke(tmp_path / "plain", source_wav)
    run_smoke(tmp_path / "mixed", source_wav, "--emo-mix")

    plain = sorted((tmp_path / "plain" / "wav").glob("*.wav"))
    mixed = sorted((tmp_path / "mixed" / "wav").glob("*.wav"))
    assert [path.name for path in plain] == [path.name for path in mixed]
    for left, right in zip(plain, mixed):
        assert left.read_bytes() == right.read_bytes()


# --- argument validation ----------------------------------------------------


def test_a_mask_rate_without_the_flag_is_fatal(tmp_path, source_wav):
    """Silently ignoring it would put a rate in the log that nobody applied."""
    with pytest.raises(SystemExit, match="0.5"):
        run_smoke(tmp_path / "run", source_wav, "--emo-mask-rate", "0.5")


@pytest.mark.parametrize("rate", ["-0.1", "1.5"])
def test_a_mask_rate_outside_zero_to_one_is_fatal(tmp_path, source_wav, rate):
    with pytest.raises(SystemExit, match="between 0 and 1"):
        run_smoke(tmp_path / "run", source_wav, "--emo-mix", "--emo-mask-rate", rate)


def test_the_flags_are_documented_in_the_help_text(capsys):
    """The help has to name the sentinel; nobody reads model.py first."""
    with pytest.raises(SystemExit):
        smoke.parse_args(["--help"])
    help_text = capsys.readouterr().out

    assert "--emo-mix" in help_text
    assert "--emo-mask-rate" in help_text
    assert SER in help_text
    # The default rate is not an argparse default (it is applied in main, so that
    # --emo-mask-rate without --emo-mix can be rejected), so it has to be spelled
    # out in the help rather than filled in by %(default)s.
    assert str(smoke.DEFAULT_EMO_MASK_RATE) in help_text
