"""Unit tests for ``scripts/detect_label_noise.py`` (ground-truth noise scan).

The script ranks manifest entries by how far a teacher ASR model's decode is
from the stored transcript, so a human can find *mislabelled ground truth* in
the 1,000 h VisualNovel corpus.  Nothing here loads a teacher: every test drives
the pure surface -- the normalisation projection, the title filter, the ordering
and the summary -- with a stub transcriber, so the suite needs no faster-whisper,
no CTranslate2, no GPU and no audio.

The load-bearing test is the orthography block.  The corpus and the teacher use
different punctuation conventions, and a raw CER between them was measured to
carry a **6.18 CER-point floor from orthography alone** -- 39% of the student's
entire 15.84% CER budget.  Because ``……`` / ``！`` / ``♪`` density is highest on
emotive and NSFW lines, an unnormalised comparison would preferentially flag
exactly those and call a whole content category "noisy".  The whole script is
only meaningful if convention-only differences score **exactly zero**, which is
what :func:`test_orthography_only_difference_scores_zero` pins, pair by pair.

The complement matters just as much and is easy to lose: a projection that
scored everything zero would pass that test.  The tests below it assert that
real wording differences, and the phonemic characters the projection must *not*
touch, still register.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "detect_label_noise.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def detect():
    """Load the script by path.

    ``scripts/`` is not a package, so there is nothing to ``import``.  Same
    loader shape as ``tests/test_extract_weights.py``.
    """
    spec = importlib.util.spec_from_file_location("detect_label_noise", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["detect_label_noise"] = module
    spec.loader.exec_module(module)
    return module


class StubTeacher:
    """Returns canned text per source path; records what it was asked for."""

    def __init__(self, by_source: dict[str, str] | None = None, default: str = "") -> None:
        self._by_source = by_source or {}
        self.default = default
        self.seen: list[str] = []

    def transcribe(self, path: Path) -> str:
        self.seen.append(str(path))
        text = self._by_source.get(str(path), self.default)
        if isinstance(text, Exception):
            raise text
        return text


def make_clip(detect, key: str, target: str, source: str | None = None):
    """Build a :class:`ManifestClip` with a source derived from the key."""
    return detect.ManifestClip(key=key, source=source or f"/audio/{key}.wav", target=target)


def write_manifest(path: Path, records: list[dict]) -> Path:
    """Write a ``data/vn``-shaped manifest jsonl."""
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------- the normalisation core

#: ``(ground truth as our corpus writes it, teacher output)`` pairs that differ
#: *only* in orthographic convention.  Every character class here was counted in
#: ``data/vn/*.jsonl`` against the teacher's measured emission: full-width ``！``
#: 6,534 vs 0, ``？`` 7,980 vs 0, ``。`` 11,158 vs 0, ``―`` 3,502 vs 0, ``～`` 914
#: vs 0, ``♪`` 304 vs 0, and doubled ``……`` vs the teacher's single ``…``.
ORTHOGRAPHY_ONLY_PAIRS = [
    pytest.param("嘘だろ！", "嘘だろ!", id="fullwidth-exclamation"),
    pytest.param("本当に？", "本当に?", id="fullwidth-question"),
    pytest.param("そうだね。", "そうだね", id="ideographic-full-stop"),
    pytest.param("えっ……そんな", "えっ…そんな", id="doubled-vs-single-ellipsis"),
    pytest.param("あ…………うん", "あ…うん", id="long-ellipsis-run"),
    pytest.param("――みんな気になるのさ", "みんな気になるのさ", id="horizontal-bar"),
    pytest.param("やあ～元気？", "やあ元気?", id="wave-dash"),
    pytest.param("デートしよ♪", "デートしよ", id="music-note"),
    pytest.param("行こう 　よ", "行こうよ", id="ascii-and-ideographic-space"),
    pytest.param(
        "――えっ、なに……！？ 行こう♪",
        "えっ、なに…!? 行こう",
        id="all-classes-combined",
    ),
]


@pytest.mark.parametrize(("ground_truth", "teacher"), ORTHOGRAPHY_ONLY_PAIRS)
def test_orthography_only_difference_scores_zero(detect, ground_truth, teacher):
    """Convention-only differences must contribute *exactly* zero CER.

    Not "small": zero.  Anything above zero here is a per-clip offset that
    scales with punctuation density, and punctuation density in this corpus is
    correlated with emotive and NSFW content -- so a nonzero floor becomes a
    content-correlated bias in which clips get ranked as mislabelled.
    """
    assert detect.normalized_cer(ground_truth, teacher) == 0.0
    assert detect.normalize_for_comparison(ground_truth) == detect.normalize_for_comparison(
        teacher
    )


def test_orthography_projection_is_symmetric(detect):
    """The projection cannot depend on which side a string arrived from.

    Ground truth and teacher output go through the same function; if it were
    asymmetric the report's CER would depend on argument order.
    """
    for ground_truth, teacher in [(p.values[0], p.values[1]) for p in ORTHOGRAPHY_ONLY_PAIRS]:
        assert detect.normalized_cer(teacher, ground_truth) == 0.0


@pytest.mark.parametrize(
    ("ground_truth", "teacher", "reason"),
    [
        pytest.param("詳しい話を聞こう", "詳しい話を聞いた", "verb", id="different-verb"),
        pytest.param("蓮くん、デートしよ♪", "律くん、デートしよ", "name", id="wrong-name"),
        pytest.param("おはよう", "こんばんは", "whole line", id="whole-line-wrong"),
        pytest.param("はい", "", "empty decode", id="teacher-said-nothing"),
    ],
)
def test_genuine_wording_difference_scores_nonzero(detect, ground_truth, teacher, reason):
    """Real content differences must survive the projection.

    Guards the failure mode the zero-CER test cannot catch: a projection that
    stripped too much -- kana, kanji, the prolonged sound mark -- would pass
    every orthography case by erasing the text entirely.
    """
    assert detect.normalized_cer(ground_truth, teacher) > 0.0, reason


@pytest.mark.parametrize(
    ("ground_truth", "teacher"),
    [
        # U+30FC prolonged sound mark: 5,994 occurrences in data/vn/*.jsonl and
        # phonemic -- it is vowel length, not decoration.  It looks like the
        # U+2015 bar the projection *does* strip, so this is the trap.
        pytest.param("ラーメン", "ラメン", id="prolonged-sound-mark-U30FC"),
        # U+3007, 51 occurrences, used to censor names -- content, not notation.
        pytest.param("〇〇さん", "さん", id="ideographic-zero-U3007"),
    ],
)
def test_projection_preserves_phonemic_characters(detect, ground_truth, teacher):
    """Characters that carry sound or content must not be stripped."""
    assert detect.normalized_cer(ground_truth, teacher) > 0.0


def test_empty_reference_contract(detect):
    """A purely decorative line is not evidence of a bad label.

    ``♪`` alone normalises to empty on both sides; that must score 0.0, not
    1.0, or every decorative-only line would head the report.  When only the
    reference empties out, the teacher's text is wholly insertion -- 1.0.
    """
    assert detect.normalized_cer("♪", "♪") == 0.0
    assert detect.normalized_cer("……", "") == 0.0
    assert detect.normalized_cer("♪", "こんにちは") == 1.0


def test_normalized_cer_is_edits_over_reference_length(detect):
    """CER is edits / reference chars, and is not clamped at 1.0.

    Unclamped on purpose: a hallucinated hypothesis far longer than its
    reference should outrank a merely-wrong one, since that ordering is the
    report's only product.
    """
    # "はい" normalises to 2 chars; a 1-char substitution over 2 is 0.5.
    assert detect.normalized_cer("はい", "はあ") == pytest.approx(0.5)
    assert detect.normalized_cer("はい", "はいはいはいはい") > 1.0


def test_projection_strips_sensevoice_rich_tags(detect):
    """Rich tags are model metadata, never transcription.

    Reused from ``eval_chunk_gap.normalize_chars``; asserted here because the
    manifest's sibling fields carry this markup and a teacher swap could
    reintroduce it into the hypothesis.
    """
    assert detect.normalize_for_comparison("<|ja|><|NEUTRAL|><|Speech|><|woitn|>おはよう") == (
        detect.normalize_for_comparison("おはよう")
    )


def test_projection_reuses_eval_chunk_gap(detect):
    """The normaliser must be built on the frozen scorer, not a second copy.

    ``scripts/eval_chunk_gap.py`` defines the CER numbers this project reports.
    A private reimplementation here would drift from it silently, which is the
    exact failure this assertion exists to prevent: the projection must equal
    ``normalize_chars`` on any text free of decorative symbols.
    """
    assert detect._EVAL.__name__ == "eval_chunk_gap"
    for text in ["おはよう。", "えっ……！？", "ラーメン食べたい", "<|ja|>やあ～"]:
        assert detect.normalize_for_comparison(text) == detect._EVAL.normalize_chars(text)


# ------------------------------------------------------------ manifest + filter


def test_load_manifest_reads_key_source_target(detect, tmp_path):
    """The manifest reader takes only the three fields it scores on."""
    path = write_manifest(
        tmp_path / "m.jsonl",
        [
            {"key": "A__spk__1", "source": "/a/1.wav", "target": "おはよう", "target_len": 4},
            {"key": "B__spk__2", "source": "/b/2.wav", "target": "こんばんは"},
        ],
    )
    clips = list(detect.load_manifest(path))
    assert [c.key for c in clips] == ["A__spk__1", "B__spk__2"]
    assert clips[0].source == "/a/1.wav"
    assert clips[0].target == "おはよう"


def test_load_manifest_rejects_incomplete_record(detect, tmp_path):
    """A record missing a field raises rather than being skipped.

    Silently dropping records would understate how much of the corpus was
    actually audited, which is indistinguishable in the report from "these
    clips were clean".
    """
    path = write_manifest(tmp_path / "m.jsonl", [{"key": "A", "source": "/a.wav"}])
    with pytest.raises(ValueError, match="target"):
        list(detect.load_manifest(path))


def test_load_title_prefixes_skips_comments_and_blanks(detect, tmp_path):
    """The include file may be annotated with why a title is on the list."""
    path = tmp_path / "titles.txt"
    path.write_text(
        "# uncontaminated: not in the teacher's training set\nGIGA_Ai_Kiss_2\n\n"
        "Purple_software_Criminal_Border_2nd_offence\n",
        encoding="utf-8",
    )
    assert detect.load_title_prefixes(path) == [
        "GIGA_Ai_Kiss_2",
        "Purple_software_Criminal_Border_2nd_offence",
    ]


def test_load_title_prefixes_rejects_empty_file(detect, tmp_path):
    """An empty include list must fail loudly.

    Otherwise it selects nothing and writes a clean empty report, which reads
    as "no label noise found" rather than "nothing was examined".
    """
    path = tmp_path / "titles.txt"
    path.write_text("# only a comment\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no title prefixes"):
        detect.load_title_prefixes(path)


def test_title_filter_keeps_only_listed_prefixes(detect):
    """The filter is a key prefix match -- titles lead the manifest key."""
    clips = [
        make_clip(detect, "GIGA_Ai_Kiss_2__spk__1", "あ"),
        make_clip(detect, "Purple_software_CB__spk__2", "い"),
        make_clip(detect, "GIGA_Ai_Kiss_2__other__3", "う"),
    ]
    selected = detect.select_clips(clips, prefixes=["GIGA_Ai_Kiss_2"])
    assert [c.key for c in selected] == ["GIGA_Ai_Kiss_2__spk__1", "GIGA_Ai_Kiss_2__other__3"]


def test_title_filter_none_keeps_everything(detect):
    """No ``--include-titles`` means no filtering (the CLI warns separately)."""
    clips = [make_clip(detect, "A__1", "あ"), make_clip(detect, "B__2", "い")]
    assert len(detect.select_clips(clips, prefixes=None)) == 2


def test_limit_applies_after_the_title_filter(detect):
    """Order is load-bearing: filter first, then sample.

    The manifest is grouped by title, so limiting first would usually consume
    the budget on excluded titles and select nothing.
    """
    clips = [make_clip(detect, f"X__{i}", "あ") for i in range(3)]
    clips += [make_clip(detect, f"KEEP__{i}", "い") for i in range(3)]
    selected = detect.select_clips(clips, prefixes=["KEEP"], limit=2)
    assert [c.key for c in selected] == ["KEEP__0", "KEEP__1"]


def test_select_clips_consumes_lazily(detect):
    """``--limit`` must not read the whole manifest.

    ``data/vn/train.jsonl`` is large; a sampling run should stop early.
    """

    def endless():
        index = 0
        while True:
            yield make_clip(detect, f"K__{index}", "あ")
            index += 1

    assert len(detect.select_clips(endless(), prefixes=None, limit=3)) == 3


# ------------------------------------------------------------ scoring + ordering


def test_score_clip_keeps_both_raw_and_normalized(detect):
    """A reviewer judges the label from the raw strings, not the projection."""
    clip = make_clip(detect, "A__1", "そうだね。")
    row = detect.score_clip(clip, StubTeacher(default="そうだね"))
    payload = row.to_json()
    assert payload["ground_truth"] == "そうだね。"
    assert payload["teacher_text"] == "そうだね"
    assert payload["ground_truth_normalized"] == payload["teacher_normalized"] == "そうだね"
    assert payload["cer"] == 0.0
    assert payload["error"] is None


def test_score_clip_records_decode_failure_without_raising(detect):
    """One unreadable clip must not lose a multi-hour sweep."""
    clip = make_clip(detect, "A__1", "おはよう")
    teacher = StubTeacher({str(Path("/audio/A__1.wav")): OSError("no such file")})
    row = detect.score_clip(clip, teacher)
    assert row.cer is None
    assert "OSError" in row.error


def test_report_is_sorted_worst_first(detect):
    """Descending CER: reviewer attention runs out before the report does."""
    clips = [
        make_clip(detect, "mid", "おはようございます"),
        make_clip(detect, "clean", "そうだね。"),
        make_clip(detect, "worst", "おはよう"),
    ]
    teacher = StubTeacher(
        {
            "/audio/mid.wav": "おはようこざいます",  # 1 edit / 9
            "/audio/clean.wav": "そうだね",  # orthography only -> 0.0
            "/audio/worst.wav": "まったく違う文章",  # wholly different
        }
    )
    rows = detect.sort_rows([detect.score_clip(c, teacher) for c in clips])
    assert [r.clip.key for r in rows] == ["worst", "mid", "clean"]
    assert [r.cer for r in rows] == sorted((r.cer for r in rows), reverse=True)


def test_report_ordering_breaks_ties_by_key(detect):
    """Ties resolve by key so two runs produce byte-identical reports."""
    clips = [make_clip(detect, key, "おはよう") for key in ["zebra", "alpha", "mike"]]
    rows = detect.sort_rows([detect.score_clip(c, StubTeacher(default="こんばんは")) for c in clips])
    assert [r.clip.key for r in rows] == ["alpha", "mike", "zebra"]


def test_failed_decodes_sort_last(detect):
    """Failures have no CER and need different follow-up than a bad label."""
    good = detect.score_clip(make_clip(detect, "good", "おはよう"), StubTeacher(default="違う"))
    bad = detect.score_clip(
        make_clip(detect, "aaa_failed", "おはよう"),
        StubTeacher({"/audio/aaa_failed.wav": OSError("boom")}),
    )
    rows = detect.sort_rows([bad, good])
    assert [r.clip.key for r in rows] == ["good", "aaa_failed"]


# ------------------------------------------------------------------- summary


def _rows_with_cers(detect, cers, ref_chars=10):
    """Build report rows carrying given CERs, bypassing the teacher."""
    rows = []
    for index, cer in enumerate(cers):
        row = detect.ReportRow(clip=make_clip(detect, f"k{index:03d}", "あ" * ref_chars))
        row.ground_truth_normalized = "あ" * ref_chars
        row.cer = cer
        rows.append(row)
    return rows


def test_summary_counts_and_mean(detect):
    """Counts and the unweighted per-clip mean."""
    summary = detect.summarise(_rows_with_cers(detect, [0.0, 0.5, 1.0]))
    assert summary["num_clips"] == 3
    assert summary["num_scored"] == 3
    assert summary["num_failed"] == 0
    assert summary["mean_cer"] == pytest.approx(0.5)


def test_summary_deciles_are_linear_interpolated(detect):
    """Eleven deciles p0..p100, numpy's ``linear`` method.

    Values 0.0..1.0 in steps of 0.1 make each decile land on a sample, so the
    expected numbers below are the interpolation contract, not a fitted result.
    """
    values = [i / 10 for i in range(11)]
    summary = detect.summarise(_rows_with_cers(detect, values))
    deciles = summary["deciles"]
    assert list(deciles) == [f"p{d * 10}" for d in range(11)]
    assert deciles["p0"] == pytest.approx(0.0)
    assert deciles["p50"] == pytest.approx(0.5)
    assert deciles["p100"] == pytest.approx(1.0)


def test_summary_deciles_interpolate_between_samples(detect):
    """With 2 samples, p50 sits midway -- pins interpolation over nearest-rank."""
    summary = detect.summarise(_rows_with_cers(detect, [0.0, 1.0]))
    assert summary["deciles"]["p50"] == pytest.approx(0.5)
    assert summary["deciles"]["p10"] == pytest.approx(0.1)


def test_summary_threshold_counts_are_inclusive(detect):
    """``counts_above`` counts clips at or above each threshold.

    Inclusive so a clip scoring exactly a threshold is never invisible in
    every bucket. Thresholds bracket the interesting range; they are reporting
    buckets, not a decision rule.
    """
    summary = detect.summarise(_rows_with_cers(detect, [0.05, 0.1, 0.25, 0.6, 1.0, 1.5]))
    counts = summary["counts_above"]
    assert counts["0.1"] == 5  # 0.1, 0.25, 0.6, 1.0, 1.5
    assert counts["0.3"] == 3  # 0.6, 1.0, 1.5
    assert counts["1"] == 2  # 1.0, 1.5 -- unclamped CER is counted as-is
    assert summary["cer_thresholds"] == list(detect.CER_THRESHOLDS)


def test_summary_corpus_cer_weights_by_reference_length(detect):
    """``corpus_cer`` weights by length; ``mean_cer`` does not.

    Both are reported because they answer different questions: mean_cer treats
    each clip as one candidate label (the reading for this report), corpus_cer
    is comparable with the numbers ``eval_chunk_gap.py`` prints.
    """
    short = detect.ReportRow(clip=make_clip(detect, "short", "あ"))
    short.ground_truth_normalized = "あ"
    short.cer = 1.0
    long = detect.ReportRow(clip=make_clip(detect, "long", "あ" * 9))
    long.ground_truth_normalized = "あ" * 9
    long.cer = 0.0

    summary = detect.summarise([short, long])
    assert summary["mean_cer"] == pytest.approx(0.5)
    assert summary["corpus_cer"] == pytest.approx(0.1)  # 1 edit over 10 chars


def test_summary_excludes_failed_decodes_from_statistics(detect):
    """Failures are counted, never folded in as a sentinel CER."""
    rows = _rows_with_cers(detect, [0.0, 1.0])
    failed = detect.ReportRow(clip=make_clip(detect, "zz", "あ"), error="OSError: boom")
    summary = detect.summarise(rows + [failed])
    assert summary["num_clips"] == 3
    assert summary["num_scored"] == 2
    assert summary["num_failed"] == 1
    assert summary["mean_cer"] == pytest.approx(0.5)


def test_summary_of_nothing_is_not_a_crash(detect):
    """An all-failed run still produces a readable summary."""
    summary = detect.summarise([])
    assert summary["num_clips"] == 0
    assert summary["mean_cer"] is None
    assert summary["deciles"] is None


# --------------------------------------------------------- teacher settings/CLI


def test_transcribe_options_pin_the_non_negotiable_settings(detect):
    """Each of these four has a measured cause; none is a tuning choice.

    ``word_timestamps=False``: CTranslate2 segfaults in ``align()`` on this
    model's 2-layer decoder (Whisper-WebUI issue #609) -- a crash, not a slow
    path.  ``initial_prompt=None`` and ``condition_on_previous_text=False``:
    the model card documents that prompt conditioning induces hallucination, so
    both routes into the decoder prompt are closed.  ``no_repeat_ngram_size=5``:
    the card documents repetition loops on emotive/NSFW content, which would
    otherwise flood the top of a CER-ranked report.  ``language="ja"``: forced,
    never autodetected.
    """
    options = detect.teacher_transcribe_options(detect.TeacherConfig())
    assert options["word_timestamps"] is False
    assert options["initial_prompt"] is None
    assert options["condition_on_previous_text"] is False
    assert options["no_repeat_ngram_size"] == 5
    assert options["language"] == "ja"


def test_default_teacher_is_the_ct2_anime_whisper(detect):
    """Defaults name the in-domain teacher; both stay overridable."""
    config = detect.TeacherConfig()
    assert config.model_id == "quantumcookie/anime-whisper-ct2-fp16"
    assert config.backend == "faster-whisper"
    assert detect.TeacherConfig(model_id="other", backend="faster-whisper").model_id == "other"


def test_unknown_backend_is_rejected(detect):
    """Backends are a registry, so a typo fails at startup, not after N hours."""
    with pytest.raises(ValueError, match="unknown backend"):
        detect.build_teacher(detect.TeacherConfig(backend="nope"))


def test_importing_the_module_does_not_import_faster_whisper(detect):
    """The backend import must be lazy.

    faster-whisper is absent from this venv and probably cannot be installed
    on it, so a module-level import would make the CLI and every test above
    unrunnable here.
    """
    assert "faster_whisper" not in sys.modules
    detect.build_teacher(detect.TeacherConfig())  # construction alone must not import
    assert "faster_whisper" not in sys.modules


def test_help_runs_without_faster_whisper():
    """``--help`` on a machine with no teacher installed."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "--include-titles" in result.stdout
    assert "--limit" in result.stdout
    assert "--device" in result.stdout


def test_help_runs_from_an_unrelated_working_directory(tmp_path):
    """The script must not assume a working directory.

    It runs inside a container on a GPU cluster node, where cwd is whatever the
    scheduler picked; its own imports resolve from its file location.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_missing_manifest_exits_nonzero_without_writing(detect, tmp_path):
    """Bad input fails before anything is created."""
    out = tmp_path / "report.jsonl"
    code = detect.main(
        ["--manifest", str(tmp_path / "nope.jsonl"), "--out", str(out)]
    )
    assert code == 1
    assert not out.exists()


def test_empty_selection_exits_nonzero(detect, tmp_path):
    """Selecting nothing is an error, not an empty clean report."""
    manifest = write_manifest(
        tmp_path / "m.jsonl", [{"key": "A__1", "source": "/a.wav", "target": "あ"}]
    )
    titles = tmp_path / "t.txt"
    titles.write_text("NOT_PRESENT\n", encoding="utf-8")
    out = tmp_path / "report.jsonl"
    code = detect.main(
        [
            "--manifest", str(manifest),
            "--out", str(out),
            "--include-titles", str(titles),
        ]
    )
    assert code == 1
    assert not out.exists()


def test_end_to_end_writes_ranked_report_and_summary(detect, tmp_path, monkeypatch):
    """Full run with a stubbed teacher: ranked jsonl plus summary sidecar."""
    manifest = write_manifest(
        tmp_path / "m.jsonl",
        [
            {"key": "KEEP__a", "source": "/audio/a.wav", "target": "そうだね。"},
            {"key": "KEEP__b", "source": "/audio/b.wav", "target": "おはよう"},
            {"key": "SKIP__c", "source": "/audio/c.wav", "target": "無視される"},
        ],
    )
    titles = tmp_path / "t.txt"
    titles.write_text("KEEP\n", encoding="utf-8")
    out = tmp_path / "nested" / "report.jsonl"

    teacher = StubTeacher({"/audio/a.wav": "そうだね", "/audio/b.wav": "まるで違う言葉"})
    monkeypatch.setitem(detect.BACKENDS, "faster-whisper", lambda config: teacher)

    code = detect.main(
        [
            "--manifest", str(manifest),
            "--out", str(out),
            "--include-titles", str(titles),
        ]
    )
    assert code == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["key"] for r in rows] == ["KEEP__b", "KEEP__a"]
    assert rows[1]["cer"] == 0.0  # orthography-only difference
    assert rows[0]["cer"] > 0.0
    assert teacher.seen == ["/audio/a.wav", "/audio/b.wav"]  # SKIP__c never decoded

    summary = json.loads((out.parent / "report.jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["num_clips"] == 2
    assert summary["transcribe_options"]["no_repeat_ngram_size"] == 5
    assert summary["include_titles"] == str(titles)


def test_run_does_not_modify_manifest_or_audio(detect, tmp_path, monkeypatch):
    """Read-only with respect to the corpus.

    The script exists to inform a deletion decision, never to make one.
    """
    manifest = write_manifest(
        tmp_path / "m.jsonl", [{"key": "A__1", "source": "/audio/a.wav", "target": "おはよう"}]
    )
    before = manifest.read_bytes()
    before_mtime = manifest.stat().st_mtime_ns

    monkeypatch.setitem(
        detect.BACKENDS, "faster-whisper", lambda config: StubTeacher(default="違う")
    )
    out = tmp_path / "report.jsonl"
    assert detect.main(["--manifest", str(manifest), "--out", str(out)]) == 0

    assert manifest.read_bytes() == before
    assert manifest.stat().st_mtime_ns == before_mtime
    assert {p.name for p in tmp_path.iterdir()} == {
        "m.jsonl",
        "report.jsonl",
        "report.jsonl.summary.json",
    }
