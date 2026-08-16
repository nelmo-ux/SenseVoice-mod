#!/usr/bin/env python3
"""Build the VisualNovel fine-tuning corpus for SenseVoice chunk-mask training.

Pipeline
--------
1. Download the selected password-protected ``.7z`` archives from the gated
   HuggingFace dataset ``OOPPEENN/56697375616C4E6F76656C5F44617461736574``
   (4 parallel connections, HTTP range resume).
2. Extract them in-process with ``py7zr`` into ``<out-dir>/raw/``.  The archive
   password comes from ``$VN_ARCHIVE_PASSWORD`` (or
   ``~/.cache/sensevoice/vn_archive_password``) and is deliberately never passed
   to a ``7z`` subprocess, whose argv ``ps`` would expose machine-wide.
3. Read each archive's ``index.json`` -- a JSON *list* of
   ``{"Speaker", "Voice", "Text"}`` where the audio lives at
   ``<Speaker>/<Voice>.ogg`` (48 kHz OGG Vorbis, mixed mono/stereo).
4. Transcode every kept clip to 16 kHz mono PCM16 wav under
   ``<out-dir>/audio/<title>/<speaker>/<voice>.wav`` (multiprocessing).
5. Emit ``train.jsonl`` / ``val.jsonl`` in the exact schema of
   ``data/train_example.jsonl`` plus a ``manifest.json`` with counts, hours,
   speaker tallies and per-reason filter statistics.

Every stage is idempotent: an archive that is already downloaded is not
re-fetched, an archive that is already extracted is not re-extracted, and a wav
that already exists is not re-encoded (its duration is read from the header).
An interrupted download resumes from the partial ``.part`` file.

Filtering and normalisation policy (fixed here, deliberately conservative)
-------------------------------------------------------------------------
Dropped clips:

* duration < ``--min-seconds`` (0.5 s)  -- too short to carry a transcript;
* duration > ``--max-seconds`` (20.0 s) -- the MPS memory ceiling for batching;
* the ``.ogg`` referenced by ``index.json`` is missing on disk;
* the decoder raises (corrupt/truncated ogg);
* ``Text`` is empty, whitespace-only, or punctuation-only after normalisation
  (CTC cannot train on a zero-length target and the trainer crashes rather than
  skipping such a sample -- see ``scripts/make_smoke_data.py``);
* the index lists the same audio file twice with *different* transcripts (two
  observed).  There is no way to tell which line the clip contains, and a clip
  trained against a certainly-wrong target is worse than a missing clip, so the
  whole group is dropped.  Duplicates that agree are simply collapsed.

Text normalisation (``normalize_text``), in order:

0. protect Japanese punctuation (``！？。、：；…‥～〜``) behind private-use
   placeholders, so the NFKC in step 1 cannot fold it.  Pretrained
   SenseVoiceSmall emits full-width Japanese punctuation and its BPE vocab holds
   ASCII ``!``/``?`` and full-width ``！``/``？`` as *separate* tokens, so a
   blanket NFKC would train the model against an orthography it never used --
   measurably so: it folded ``…`` into 120,921 ASCII dots across this corpus.
   ASCII ``!``, ``?`` and ``~`` are converted *to* full-width for the same
   reason; and *runs* of ASCII dots are collapsed back to ``…`` while a lone
   dot is left alone, since every one in this corpus is a decimal point or an
   abbreviation (``Excuse me.``, ``Mr.N``, ``2.5次元``);
1. Unicode NFKC (folds full-width alphanumerics and spaces, and half-width
   katakana) with the punctuation above held out of its reach;
2. delete every backslash -- a backslash never occurs in legitimate Japanese
   transcript text, it is always a VN engine escape artifact (e.g. the observed
   ``に\\"ぇっ！？``);
3. repeatedly strip a quote/bracket pair that wraps the *whole* line
   (``「」『』（）“”…``), only when the closing character's first occurrence is
   the final character, so ``「A」と「B」`` is left intact;
4. delete residual ASCII double quotes, which are the leftovers of step 2's
   escape sequences;
5. collapse every run of whitespace to a single space and strip the ends.

Long-vowel marks, kana, kanji and sentence-final punctuation are preserved --
they are pronounced content and the transcript must match the audio.

Content note: this corpus is unfiltered adult visual-novel material and the
transcripts contain NSFW text.  No content filtering or censorship is applied,
because an ASR training pair is only valid when the transcript is a faithful
transcription of the audio.

Train/val split
---------------
The split is *speaker-disjoint*: whole speakers are held out, so no speaker
appears in both files.  Speaker leakage would let the model memorise voices and
make the validation loss an optimistic lie about generalisation.

The grouping key is the **bare speaker name across the whole corpus**, not
``(title, speaker)``.  These archives are franchises -- sequels and fandiscs of
the same series -- so the same character name in two titles is normally the same
voice actor, and a per-title key produced a formally disjoint split that still
leaked ~20% of val clips.  Names shared across titles are logged and recorded in
``manifest.json`` so the fusing is visible.

Val is additionally *stratified and capped*: every title gets a quota
proportional to its share of the corpus, and no speaker may contribute more than
``VAL_MAX_CLIPS_PER_SPEAKER`` clips.  Holding out whole speakers instead put 90%
of val on a single main character and left two titles unrepresented, which turns
held-out CER into a high-variance measurement of one voice -- bad, because that
number is what checkpoints are chosen by.  Surplus clips of a val speaker are
dropped rather than returned to train, so voice identity never straddles the
split; ``totals.val_surplus_clips_dropped`` reports the cost.

Usage
-----
    python scripts/prepare_vn_data.py                     # full ~54 h corpus
    python scripts/prepare_vn_data.py --limit-hours 0.5   # quick trial slice
    python scripts/prepare_vn_data.py --list-archives     # inventory, then exit

``--list-archives`` is a pure query: it lists every ``.7z`` in the dataset repo
with its size and a running cumulative total, then exits without downloading,
extracting or converting anything.  It is how a larger corpus is scoped -- pick
archives off the listing until the cumulative column reaches the size you want
and feed those paths back in via ``--archives`` (``--list-format plain`` emits
exactly that, shell-quoted, one per line).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import shlex
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_REPO = "OOPPEENN/56697375616C4E6F76656C5F44617461736574"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"

# The archives are password protected.  The password is published with the
# dataset, but it is still a secret by the repo's rules: it is read from the
# environment (or a local file), never hardcoded, never logged, and never placed
# in a subprocess argv where ``ps`` would expose it to every user on the box.
ARCHIVE_PASSWORD_ENV = "VN_ARCHIVE_PASSWORD"
ARCHIVE_PASSWORD_FILE = Path.home() / ".cache" / "sensevoice" / "vn_archive_password"

# Written into <out-dir>/raw/<stem>/ once an archive has been extracted in full.
# A dotfile so it can never be mistaken for dataset content: every later stage
# reads the archive's own index.json, and find_index_json only ever descends
# into *directories*, so nothing walks this file into a manifest.
EXTRACT_MARKER_NAME = ".extract_complete"

DEFAULT_ARCHIVES: tuple[str, ...] = (
    "GalGame/Studio e.go!_Meguru Sekai de Towanaru Chikai o!.7z",
    "GalGame/GIGA_Ai Kiss 2.7z",
    "GalGame/Purple software_Criminal Border 2nd offence.7z",
    "GalGame/Purple software_Criminal Border 3rd offence.7z",
    "GalGame/GIGA_Ai Kiss 2 Extra.7z",
)

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "vn"

# ``--list-format`` defaults to None rather than to this value so that passing
# it without ``--list-archives`` -- where it would do nothing -- is detectable
# and can be rejected instead of silently ignored.
DEFAULT_LIST_FORMAT = "text"

SAMPLE_RATE = 16_000
MIN_SECONDS = 0.5
MAX_SECONDS = 20.0
VAL_FRACTION = 0.025  # ~2.5% of clips, held out as whole speakers

# Per-speaker ceiling on val clips.  Without it a single main character with
# hundreds of lines swallows the whole validation budget and held-out CER
# becomes a measurement of that one voice.  ~40 keeps val spread over roughly
# 20-40 speakers at the default 2.5% fraction.
VAL_MAX_CLIPS_PER_SPEAKER = 40

# Surplus clips of a val speaker are dropped (never returned to train), so
# holding out a 2500-clip main character would cost ~2460 training clips.  A
# speaker larger than this multiple of the cap is passed over, and the total
# dropped surplus is additionally bounded by VAL_MAX_SURPLUS_FRACTION of the
# corpus.  Together these stop val stratification from quietly eating the
# training set.
VAL_MAX_SPEAKER_CLIPS_MULTIPLE = 10
VAL_MAX_SURPLUS_FRACTION = 0.03

# One fbank frame is 10 ms.  ``source_len`` therefore counts 100 frames per
# second -- see ``compute_source_len`` for why no LFR division is applied.
FRAME_MS = 10

# Fixed tags: this corpus is Japanese speech without ITN or emotion labels.
TEXT_LANGUAGE = "<|ja|>"
EMO_TARGET = "<|NEUTRAL|>"
EVENT_TARGET = "<|Speech|>"
WITH_OR_WO_ITN = "<|woitn|>"

# Characters that make a line "real" text.  A line consisting solely of
# punctuation, symbols or dashes carries no transcribable content.
_CONTENT_RE = re.compile(
    r"[0-9A-Za-z"
    r"぀-ゟ"  # hiragana
    r"゠-ヿ"  # katakana
    r"㐀-䶿一-鿿"  # kanji
    r"ｦ-ﾟ"  # half-width katakana (survives NFKC only if malformed)
    r"]"
)

_WHITESPACE_RE = re.compile(r"\s+")

# Quote/bracket pairs the VN engine wraps whole spoken lines in.
_WRAPPING_PAIRS: tuple[tuple[str, str], ...] = (
    ("「", "」"),
    ("『", "』"),
    ("（", "）"),
    ("(", ")"),
    ("【", "】"),
    ("〔", "〕"),
    ("〈", "〉"),
    ("《", "》"),
    ("［", "］"),
    ("[", "]"),
    ("“", "”"),
    ("‘", "’"),
    ("＜", "＞"),
    ('"', '"'),
    ("'", "'"),
)

_SLUG_RE = re.compile(r"[^0-9A-Za-z぀-鿿ｦ-ﾟ]+")

# Punctuation that must survive NFKC as-is.  Pretrained SenseVoiceSmall emits
# full-width Japanese punctuation, and its BPE vocab carries ASCII "!" "?" and
# full-width "！" "？" as *separate* tokens -- so a blanket NFKC would train the
# model against an orthography it never used.  These are swapped for private-use
# placeholders (which NFKC leaves alone), then restored afterwards, rather than
# hand-rolling a partial NFKC.
_PROTECTED_PUNCT = "！？。、：；…‥～〜"
_PROTECT_MAP = {
    char: chr(0xE000 + index) for index, char in enumerate(_PROTECTED_PUNCT)
}
_RESTORE_MAP = {holder: char for char, holder in _PROTECT_MAP.items()}

# ASCII punctuation arriving from the engine is converted to the model's
# convention.  The corpus is entirely Japanese, so this is unconditional.
_ASCII_TO_FULLWIDTH = {"!": "！", "?": "？", "~": "～"}

# NFKC decomposes "…" into three ASCII dots, so any run of dots in this corpus
# is a folded ellipsis rather than punctuation.
_DOT_RUN_RE = re.compile(r"\.{2,}")

# A lone "." is left exactly as it is.  Every one in this corpus is a decimal
# point or an abbreviation -- "Excuse me.", "Mr.N", "2.5次元" -- so folding it
# into "…" or rewriting it to "。" would corrupt real text.  Only *runs* of
# dots are treated as folded ellipses.


# --------------------------------------------------------------------------
# Pure helpers (unit-tested: normalize_text / build_record / split_by_speaker /
# compute_source_len -- keep these four signatures stable)
# --------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalise one VN transcript line.  See the module docstring for policy.

    Returns the empty string for input that is empty, whitespace-only or
    punctuation-only, which is the caller's signal to drop the clip.
    """
    if not text:
        return ""

    # NFKC for the useful part (full-width alphanumerics and spaces, half-width
    # katakana) while Japanese punctuation is held out of its reach.
    protected = text
    for char, holder in _PROTECT_MAP.items():
        protected = protected.replace(char, holder)
    normalized = unicodedata.normalize("NFKC", protected)
    for holder, char in _RESTORE_MAP.items():
        normalized = normalized.replace(holder, char)

    # Backslashes only ever arrive here as engine escape artifacts.
    normalized = normalized.replace("\\", "")

    # Strip pairs that wrap the entire line, innermost-last, repeatedly.
    changed = True
    while changed:
        changed = False
        stripped = normalized.strip()
        if len(stripped) < 2:
            normalized = stripped
            break
        for open_char, close_char in _WRAPPING_PAIRS:
            if not (stripped.startswith(open_char) and stripped.endswith(close_char)):
                continue
            # Only unwrap when the closer does not also appear mid-line, so
            # "「A」と「B」" is not mangled into "A」と「B".
            if stripped.index(close_char, 1) != len(stripped) - 1:
                continue
            normalized = stripped[len(open_char) : -len(close_char)]
            changed = True
            break
        else:
            normalized = stripped

    # Residue of the removed escape sequences.
    normalized = normalized.replace('"', "")

    # Adopt the model's punctuation convention.
    normalized = _DOT_RUN_RE.sub(
        lambda match: "…" * max(1, len(match.group(0)) // 3), normalized
    )
    for ascii_char, full_width in _ASCII_TO_FULLWIDTH.items():
        normalized = normalized.replace(ascii_char, full_width)

    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not _CONTENT_RE.search(normalized):
        return ""
    return normalized


def compute_source_len(duration_sec: float, *, lfr_n: int = 1) -> int:
    """Return the ``source_len`` frame count for a clip of ``duration_sec``.

    This matches ``scripts/make_smoke_data.py`` exactly -- one frame per 10 ms,
    i.e. ``int(duration_sec * 100)`` with **no** LFR downsampling -- because
    that is the scale the rest of the repo is calibrated against:

    * ``data/train_example.jsonl`` and ``data/smoke_train.jsonl`` both store
      100 frames per second (a 1.0 s smoke clip has ``source_len`` 100);
    * ``finetune_chunk.sh`` uses ``batch_type=token`` with ``batch_size=800``
      and its comment records that this packs "2-4 clips" per batch.  The smoke
      clips are 1.0-4.0 s, i.e. 100-400 at this scale (2-4 per 800-token batch,
      as documented) versus 17-67 after an LFR/6 division, which would pack
      ~12+ clips and silently change the effective batch size.

    ``lfr_n`` is exposed for callers that genuinely want post-LFR frame counts
    (``lfr_n=6`` for the model's lfr_m=7/lfr_n=6 stack); it defaults to 1 so the
    on-disk convention is unchanged.
    """
    if duration_sec <= 0:
        return 0
    frames = int(duration_sec * 1000 / FRAME_MS)
    if lfr_n > 1:
        frames = frames // lfr_n
    return frames


def compute_target_len(text: str) -> int:
    """Token count for ``target_len``.

    ``make_smoke_data.py`` uses ``len(text.split())`` because its dummy
    transcripts are English.  Japanese is unsegmented, so whitespace splitting
    would report 1 for every line and destroy the length signal the batch
    sampler uses.  We therefore count *characters* (whitespace excluded), which
    is the closest analogue of "one token per written unit" for Japanese and is
    what the CTC target length actually approximates.
    """
    return len(text.replace(" ", ""))


def build_record(
    key: str,
    text: str,
    wav_path: str | os.PathLike[str],
    duration_sec: float,
    *,
    text_language: str = TEXT_LANGUAGE,
    emo_target: str = EMO_TARGET,
    event_target: str = EVENT_TARGET,
    with_or_wo_itn: str = WITH_OR_WO_ITN,
) -> dict[str, Any]:
    """Build one jsonl record in the schema of ``data/train_example.jsonl``.

    ``text`` is expected to be already normalised.  ``wav_path`` is stored as an
    absolute path because the trainer resolves ``source`` relative to nothing.
    """
    return {
        "key": key,
        "text_language": text_language,
        "emo_target": emo_target,
        "event_target": event_target,
        "with_or_wo_itn": with_or_wo_itn,
        "target": text,
        "source": str(Path(wav_path).resolve()),
        "target_len": compute_target_len(text),
        "source_len": compute_source_len(duration_sec),
    }


def record_speaker(record: dict[str, Any]) -> str:
    """Title-qualified speaker id for a record, e.g. ``<title>/<speaker>``.

    Prefers an explicit ``speaker`` field; otherwise recovers it from the wav
    layout ``.../audio/<title>/<speaker>/<voice>.wav``.

    This is the *bookkeeping* id used for per-title speaker tallies.  It is NOT
    the split key -- see ``speaker_group_key``.
    """
    explicit = record.get("speaker")
    if explicit:
        return str(explicit)
    source = Path(str(record["source"]))
    return f"{source.parent.parent.name}/{source.parent.name}"


def record_title(record: dict[str, Any]) -> str:
    """Title (archive) a record came from."""
    explicit = record.get("title")
    if explicit:
        return str(explicit)
    return record_speaker(record).split("/", 1)[0]


def speaker_group_key(record: dict[str, Any]) -> str:
    """Split key: the *bare* speaker name, un-qualified by title.

    Voice identity, not ``(title, speaker)``, is what must not straddle the
    train/val boundary.  These archives are franchises -- sequels and fandiscs
    of the same series -- so the same character name in two titles is normally
    the same voice actor (observed: 七瀬 across Ai Kiss 2 / Ai Kiss 2 Extra,
    東雲 and 栞 across Criminal Border 2nd / 3rd offence).  Grouping on
    ``(title, speaker)`` therefore produced a formally disjoint split that still
    leaked ~20% of val clips.

    Grouping on the bare name is deliberately conservative: it may fuse
    genuinely distinct characters who share a common name (or a generic role
    label such as 老婆, "old woman"), costing a little val data.  That is the
    right direction to err -- an over-strict val set only understates quality,
    while a leaky one silently overstates it, which is exactly the measurement
    this fine-tune is judged on.
    """
    return record_speaker(record).split("/", 1)[-1]


def cross_title_speakers(records: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Bare speaker names that appear in more than one title, name -> titles.

    Surfaced in the log and the manifest so the fusing done by
    ``speaker_group_key`` is visible rather than silent.
    """
    titles_by_name: dict[str, set[str]] = {}
    for record in records:
        titles_by_name.setdefault(speaker_group_key(record), set()).add(
            record_title(record)
        )
    return {
        name: sorted(titles)
        for name, titles in sorted(titles_by_name.items())
        if len(titles) > 1
    }


def _capped_group(
    group: Sequence[dict[str, Any]],
    cap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Take at most ``cap`` clips from one speaker, spread across their titles.

    A speaker who appears in several titles has their allowance split
    proportionally with at least one clip per title, so capping never silently
    erases a title's only representation.  Clips are sampled at random rather
    than taken from the head of the list: file order follows scene order, so the
    first N clips of a speaker are correlated (one scene, one emotional
    register) and would make a biased val sample.
    """
    if len(group) <= cap:
        return sorted(group, key=lambda record: str(record.get("key", "")))

    by_title: dict[str, list[dict[str, Any]]] = {}
    for record in group:
        by_title.setdefault(record_title(record), []).append(record)
    titles = sorted(by_title)

    allowance = {
        title: max(1, int(cap * len(by_title[title]) / len(group))) for title in titles
    }
    while sum(allowance.values()) > cap:
        over = [t for t in titles if allowance[t] > 1]
        if not over:
            break
        title = max(over, key=lambda t: (allowance[t] / len(by_title[t]), t))
        allowance[title] -= 1
    while sum(allowance.values()) < cap:
        room = [t for t in titles if allowance[t] < len(by_title[t])]
        if not room:
            break
        title = min(room, key=lambda t: (allowance[t] / len(by_title[t]), t))
        allowance[title] += 1

    picked: list[dict[str, Any]] = []
    for title in titles:
        pool = sorted(by_title[title], key=lambda record: str(record.get("key", "")))
        rng.shuffle(pool)
        picked.extend(pool[: min(allowance[title], len(pool))])
    return sorted(picked, key=lambda record: str(record.get("key", "")))


def split_by_speaker(
    records: Sequence[dict[str, Any]],
    val_frac: float = VAL_FRACTION,
    seed: int = 0,
    *,
    max_clips_per_speaker: int = VAL_MAX_CLIPS_PER_SPEAKER,
    cover_all_titles: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``records`` into (train, val), holding out whole speakers.

    The split key is the **bare speaker name across the whole corpus** (see
    ``speaker_group_key``): if a name occurs in several titles, every one of its
    clips lands on the same side, so no voice identity straddles the boundary.

    Val is *stratified and capped* rather than "whole speakers until the budget
    is met".  Selecting whole speakers put 708 of 791 val clips (89.5%) on one
    main character and left 2 of 5 titles unrepresented, which makes held-out
    CER a high-variance measurement of one voice -- dangerous when the number is
    used to pick between checkpoints.  Instead:

    * every title gets a val quota proportional to its share of the corpus, so
      all titles are represented roughly proportionally;
    * each selected speaker contributes at most ``max_clips_per_speaker`` clips,
      so no single voice can dominate;
    * speakers larger than 4x the cap are passed over unless a quota cannot
      otherwise be filled, which keeps the discarded surplus small;
    * a coverage pass adds a speaker for any title still absent, which is how
      small titles whose speakers all belong primarily to a sibling title (the
      Ai Kiss 2 / Ai Kiss 2 Extra fandisc pair) stay represented.

    Surplus clips of a selected speaker are **dropped entirely**, not returned
    to train: returning them would put that voice on both sides, which is the
    leak this split exists to prevent.  ``train + val`` is therefore smaller
    than ``records``; the caller reports the difference.

    ``max_clips_per_speaker`` is keyword-only so the documented three-argument
    signature stays stable.
    """
    if not records:
        return [], []
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_speaker.setdefault(speaker_group_key(record), []).append(record)

    if len(by_speaker) < 2:
        # Cannot hold out a speaker without emptying train.
        return list(records), []

    rng = random.Random(seed)
    cap = max(1, int(max_clips_per_speaker))
    total = len(records)
    target = max(1, round(total * val_frac))

    titles_of: dict[str, set[str]] = {
        name: {record_title(record) for record in group}
        for name, group in by_speaker.items()
    }
    # Ties broken on the title name so the choice is deterministic.
    primary_title: dict[str, str] = {}
    for name, group in by_speaker.items():
        counts: dict[str, int] = {}
        for record in group:
            title = record_title(record)
            counts[title] = counts.get(title, 0) + 1
        primary_title[name] = max(sorted(counts), key=lambda t: (counts[t], t))

    clips_by_title: dict[str, int] = {}
    for record in records:
        title = record_title(record)
        clips_by_title[title] = clips_by_title.get(title, 0) + 1

    candidates_by_title: dict[str, list[str]] = {}
    for name in sorted(by_speaker):
        candidates_by_title.setdefault(primary_title[name], []).append(name)
    for names in candidates_by_title.values():
        rng.shuffle(names)

    selected: list[str] = []
    selected_clips = 0  # full group sizes, i.e. what train gives up
    val_clips = 0  # capped contributions, i.e. what val actually receives
    # Per-title quotas are granular (a speaker contributes several clips at
    # once) and round up, so they can overshoot the global budget.  This
    # ceiling keeps the total honest; the first pick is always allowed so val
    # is never empty.
    ceiling = target * 1.5

    size_limit = cap * VAL_MAX_SPEAKER_CLIPS_MULTIPLE
    surplus_budget = int(total * VAL_MAX_SURPLUS_FRACTION)
    surplus_used = 0

    def may_select(name: str, exempt: bool = False, force: bool = False) -> bool:
        size = len(by_speaker[name])
        if total - (selected_clips + size) <= 0:  # never let val empty train
            return False
        if force:
            return True
        if size > size_limit:  # a main character costs too much surplus
            return False
        if exempt or val_clips == 0:
            return True
        if surplus_used + max(0, size - cap) > surplus_budget:
            return False
        return val_clips + min(cap, size) <= ceiling

    def take(name: str) -> int:
        nonlocal selected_clips, val_clips, surplus_used
        size = len(by_speaker[name])
        selected.append(name)
        selected_clips += size
        surplus_used += max(0, size - cap)
        contribution = min(cap, size)
        val_clips += contribution
        return contribution

    # Pass 1 -- proportional per-title quotas.
    shortfall = 0
    for title in sorted(clips_by_title):
        quota = max(1, round(target * clips_by_title[title] / total))
        contributed = 0
        for name in candidates_by_title.get(title, []):
            if contributed >= quota:
                break
            if name in selected or not may_select(name):
                continue
            contributed += take(name)
        shortfall += max(0, quota - contributed)

    # Coverage pass: a title whose speakers all sit primarily in another title
    # (fandiscs) would otherwise never be picked.  ``_capped_group`` guarantees
    # at least one clip per title a selected speaker touches.
    covered: set[str] = set()
    for name in selected:
        covered |= titles_of[name]
    for title in sorted(clips_by_title):
        if title in covered:
            continue
        # Spanning at least two titles is non-negotiable -- a single-title val
        # set measures adaptation to one visual novel, not generalisation -- so
        # the first two titles ignore the ceiling.  Beyond that, full
        # proportional coverage is pursued only within budget; on a real corpus
        # the ceiling sits far above what covering every title costs.
        exempt = len(covered) < 2
        pool = [
            name
            for name in sorted(by_speaker)
            if title in titles_of[name]
            and name not in selected
            and may_select(name, exempt)
        ]
        if not pool and (len(covered) < 2 or cover_all_titles):
            # Two titles is the one guarantee worth overpaying for, so here the
            # size limit yields too (a title whose every speaker is a main
            # character has no cheap representative).  Beyond two titles the
            # same override is opt-in via ``cover_all_titles``, because the bill
            # can be steep: in this corpus the only voices in the Ai Kiss 2
            # Extra fandisc are Ai Kiss 2 main characters, so representing its
            # 156 clips costs ~493 dropped training clips for ~5 val clips of a
            # title whose voices val already covers.
            pool = [
                name
                for name in sorted(by_speaker)
                if title in titles_of[name]
                and name not in selected
                and may_select(name, force=True)
            ]
        if not pool:
            continue
        name = min(pool, key=lambda n: (len(by_speaker[n]), n))
        take(name)
        covered |= titles_of[name]

    # Pass 3 -- top up toward the target.  Titles whose speaker roster is a
    # handful of main characters (Ai Kiss 2: 22 voices, 8 of them over 400
    # clips) cannot fill a proportional quota within the surplus budget, so
    # their shortfall is spent wherever cheap speakers remain.  This trades
    # strict proportionality for val size and speaker diversity; the achieved
    # per-title mix is reported in manifest.json rather than assumed.
    if shortfall and val_clips < target:
        leftovers = [name for name in sorted(by_speaker) if name not in selected]
        rng.shuffle(leftovers)
        for name in leftovers:
            if val_clips >= target:
                break
            if may_select(name):
                take(name)

    if not selected:  # every speaker would empty train; take the smallest
        smallest = min(sorted(by_speaker), key=lambda n: (len(by_speaker[n]), n))
        selected.append(smallest)

    val_set = set(selected)
    train = [r for r in records if speaker_group_key(r) not in val_set]
    if not train:  # degenerate corpus; keep everything trainable
        return list(records), []

    val: list[dict[str, Any]] = []
    for name in sorted(val_set):
        val.extend(_capped_group(by_speaker[name], cap, rng))
    val.sort(key=lambda record: str(record.get("key", "")))
    return train, val


def slugify(name: str, fallback: str = "unknown") -> str:
    """Filesystem-safe token that keeps Japanese characters readable."""
    slug = _SLUG_RE.sub("_", unicodedata.normalize("NFKC", name)).strip("_")
    return slug or fallback


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class ConvertTask:
    """One index.json entry resolved to concrete input/output paths."""

    key: str
    title: str
    speaker: str
    src: str
    dst: str
    text: str


@dataclass
class FilterStats:
    index_entries: int = 0
    empty_text: int = 0
    missing_audio: int = 0
    too_short: int = 0
    too_long: int = 0
    decode_error: int = 0
    over_limit_hours: int = 0
    ambiguous_duplicate: int = 0
    duplicate_entry: int = 0
    kept: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "index_entries": self.index_entries,
            "dropped_empty_or_punctuation_only_text": self.empty_text,
            "dropped_missing_audio": self.missing_audio,
            "dropped_too_short": self.too_short,
            "dropped_too_long": self.too_long,
            "dropped_decode_error": self.decode_error,
            "dropped_conflicting_duplicate_transcripts": self.ambiguous_duplicate,
            "collapsed_identical_duplicate_entries": self.duplicate_entry,
            "skipped_over_limit_hours": self.over_limit_hours,
            "kept": self.kept,
        }


@dataclass
class ArchiveStats:
    archive: str
    title: str
    index_entries: int = 0
    kept: int = 0
    seconds: float = 0.0
    speakers: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "title": self.title,
            "index_entries": self.index_entries,
            "kept_clips": self.kept,
            "hours": round(self.seconds / 3600.0, 4),
            "speakers": len(self.speakers),
        }


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def fmt_bytes(num_bytes: float) -> str:
    """Human-readable byte count, e.g. ``783.8MiB`` / ``1.2GiB``.

    Deliberately contains **no space** between the number and the unit so that
    the ``--list-archives`` table stays splittable on whitespace: the archive
    path is the last column and is the only field allowed to contain spaces,
    which keeps ``awk '{print $2}'`` (bytes) and ``cut``/``awk`` on the path
    working on the same lines.
    """
    size = float(max(0.0, num_bytes))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TiB"


def eta(done: int, total: int, started: float) -> str:
    if done <= 0:
        return "?"
    elapsed = time.monotonic() - started
    return fmt_duration(elapsed / done * (total - done))


# --------------------------------------------------------------------------
# HuggingFace auth (shared by the repo listing and the downloader)
# --------------------------------------------------------------------------
def read_hf_token() -> str:
    """Load the HF token from the environment or the CLI cache.

    The token is never printed, and never included in an exception message.
    """
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    if HF_TOKEN_FILE.is_file():
        value = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise SystemExit(
        f"no HuggingFace token found (checked $HF_TOKEN and {HF_TOKEN_FILE}); "
        "run `huggingface-cli login` first"
    )


# --------------------------------------------------------------------------
# Repo listing (--list-archives)
# --------------------------------------------------------------------------
def list_repo_archives(token: str) -> list[tuple[str, int]]:
    """Every ``.7z`` in ``HF_REPO`` as ``(path, size_bytes)``, sorted by path.

    Uses the Hub API rather than the raw ``resolve/`` URLs the downloader
    speaks, because only the tree endpoint reports sizes.  Authentication is
    the *same* token ``read_hf_token`` supplies to the downloader -- the repo
    is gated, so an anonymous listing 404s.

    ``expand=False`` is deliberate and was verified against the installed
    huggingface_hub (1.27): the tree endpoint already returns ``size`` on every
    ``RepoFile``, and ``expand=True`` only adds ``last_commit``/``security`` at
    the cost of switching to a 50-entries-per-page crawl of a repo this large.
    Folder entries carry ``size=None`` and are filtered out here.
    """
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - listing path only
        from huggingface_hub.errors import (  # noqa: PLC0415
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required for --list-archives "
            "(pip install huggingface_hub)"
        ) from error

    try:
        entries = HfApi().list_repo_tree(
            HF_REPO,
            repo_type="dataset",
            recursive=True,
            expand=False,
            token=token,
        )
        archives = [
            (entry.path, int(entry.size))
            for entry in entries
            if getattr(entry, "size", None) is not None
            and entry.path.lower().endswith(".7z")
        ]
    except (GatedRepoError, RepositoryNotFoundError):
        # A gated repo the token cannot see answers 404, not 403, so these two
        # are the same actionable situation from the caller's side.
        raise SystemExit(
            f"cannot list {HF_REPO}: access denied or repo not found.  This "
            "dataset is gated -- accept its terms at "
            f"https://huggingface.co/datasets/{HF_REPO} using the same account "
            "the token belongs to, then re-run `huggingface-cli login`."
        ) from None
    except HfHubHTTPError as error:
        detail = str(error).replace(token, "***")
        raise SystemExit(f"failed to list {HF_REPO}: {detail[:400]}") from None
    except Exception as error:  # noqa: BLE001 - clean, secret-free message
        # Scrub defensively: transport errors can echo the request that carried
        # the Authorization header.
        detail = str(error).replace(token, "***")
        raise SystemExit(
            f"failed to list {HF_REPO}: {type(error).__name__}: {detail[:400]}"
        ) from None

    archives.sort(key=lambda item: item[0])
    return archives


def print_archive_listing(
    archives: Sequence[tuple[str, int]],
    list_format: str = "text",
) -> None:
    """Print the archive inventory to stdout.

    ``plain`` emits shell-quoted paths only, so the picked subset can be pasted
    straight after ``--archives``; the paths contain spaces ("GIGA_Ai Kiss
    2.7z"), so bare output could not survive word splitting.

    ``text`` is the table used to *choose* that subset: every row carries the
    exact byte count, a human-readable size and the running cumulative total,
    so a "first N archives under X GB" cut is read straight off the page.
    Header and summary lines start with ``#`` so ``grep -v '^#'`` leaves pure
    data, and the path is the final, whitespace-containing column.
    """
    if list_format == "plain":
        for path, _ in archives:
            print(shlex.quote(path))
        return

    total = sum(size for _, size in archives)
    print(f"# {HF_REPO}")
    print(f"# {'idx':>3}  {'bytes':>13}  {'size':>9}  {'cumulative':>10}  path")
    cumulative = 0
    for index, (path, size) in enumerate(archives, start=1):
        cumulative += size
        print(
            f"{index:>5}  {size:>13}  {fmt_bytes(size):>9}  "
            f"{fmt_bytes(cumulative):>10}  {path}"
        )
    print(f"# {len(archives)} archives, {total} bytes total ({fmt_bytes(total)})")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def check_archive_basenames(remote_paths: Sequence[str]) -> None:
    """Refuse an ``--archives`` list whose entries share a local filename.

    Every stage downstream keys an archive by its *basename*: the download
    writes ``<out-dir>/archives/<name>``, extraction unpacks to
    ``<out-dir>/raw/<stem>`` and the manifest records ``<stem>`` as the title.
    Two repo paths with the same basename therefore collapse onto one local
    file, and because :func:`download_archive` returns early for a file that is
    already present, the second archive silently resolves to the first one's
    contents -- a different corpus than was asked for, with a manifest
    fingerprint that says otherwise.  ``--list-archives`` spans the whole repo,
    so pasting paths off it can reach this; ``DEFAULT_ARCHIVES`` cannot.

    Failing here rather than slugifying the local name is deliberate: a
    slugified layout would not match the archives already on disk, so it would
    silently re-download an existing corpus instead of resuming it.

    Args:
        remote_paths: The resolved ``--archives`` list, repo-relative.

    Raises:
        SystemExit: If two or more entries share a basename, naming them.
    """
    by_name: dict[str, list[str]] = {}
    for remote in remote_paths:
        by_name.setdefault(Path(remote).name, []).append(remote)

    collisions = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if not collisions:
        return

    lines = [
        "--archives contains paths that share a filename, which would collapse "
        "onto one local file and silently build the wrong corpus:"
    ]
    for name in sorted(collisions):
        lines.append(f"  {name}")
        lines.extend(f"    from {path}" for path in collisions[name])
    lines.append("Drop or rename the duplicates and re-run.")
    raise SystemExit("\n".join(lines))


def download_archive(
    remote_path: str,
    dest: Path,
    token: str,
    attempts: int = 4,
) -> Path:
    """Download one archive with range-resume.  Returns the final path."""
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"download: already present, skipping {dest.name}")
        return dest

    url = HF_BASE_URL + urllib.parse.quote(remote_path)
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.is_file() else 0
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("User-Agent", "prepare_vn_data/1.0")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resuming = response.status == 206 and offset > 0
                if offset and not resuming:
                    offset = 0  # server ignored Range: start over
                declared = response.headers.get("Content-Length")
                total = (int(declared) + offset) if declared else 0
                mode = "ab" if resuming else "wb"
                downloaded = offset
                last_report = time.monotonic()
                started = time.monotonic()
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 15.0:
                            rate = (downloaded - offset) / max(now - started, 1e-6)
                            pct = f"{downloaded / total * 100:5.1f}%" if total else "  ?  "
                            log(
                                f"download: {dest.name} {pct} "
                                f"({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB, "
                                f"{rate / 1e6:.1f} MB/s)"
                            )
                            last_report = now
            if total and part.stat().st_size != total:
                raise OSError(
                    f"size mismatch for {dest.name}: "
                    f"{part.stat().st_size} != {total}"
                )
            part.replace(dest)
            log(f"download: done {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
            return dest
        except urllib.error.HTTPError as error:
            if error.code == 416 and part.is_file():  # already complete
                part.replace(dest)
                return dest
            if error.code in (401, 403):
                raise SystemExit(
                    f"HTTP {error.code} for {remote_path}: the HuggingFace token "
                    "does not have access to this gated dataset"
                ) from None
            last_error = error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = error
        if attempt < attempts:
            backoff = 2**attempt
            # Scrub defensively: transport errors can echo the request that
            # carried the Authorization header.  No empty-needle guard is
            # needed the way the extraction path guards its password --
            # read_hf_token raises rather than returning an empty string.
            detail = str(last_error).replace(token, "***")
            log(f"download: {dest.name} attempt {attempt} failed ({detail}); "
                f"retrying in {backoff}s")
            time.sleep(backoff)

    detail = str(last_error).replace(token, "***")
    raise SystemExit(f"failed to download {remote_path}: {detail}")


def download_all(
    remote_paths: Sequence[str],
    archive_dir: Path,
    token: str,
    workers: int,
) -> dict[str, Path]:
    """Fetch every archive, up to ``workers`` connections in parallel."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                download_archive,
                remote,
                archive_dir / Path(remote).name,
                token,
            ): remote
            for remote in remote_paths
        }
        for future in as_completed(futures):
            remote = futures[future]
            results[remote] = future.result()
    return results


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def find_archive_password() -> str | None:
    """Return the configured 7z password, or ``None`` when there is none.

    The tolerant half of :func:`read_archive_password`: the resume check wants
    the password to *verify* an existing extraction against the archive's entry
    list, but a machine without the secret must still be able to re-run over an
    already-extracted corpus rather than dying on a verification it only ever
    wanted to attempt.
    """
    value = os.environ.get(ARCHIVE_PASSWORD_ENV, "").strip()
    if value:
        return value
    if ARCHIVE_PASSWORD_FILE.is_file():
        value = ARCHIVE_PASSWORD_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def read_archive_password() -> str:
    """Load the 7z password from the environment or a local file.

    Mirrors ``read_hf_token``: never hardcoded, never printed.  Extraction runs
    in-process via ``py7zr`` rather than the ``7z`` CLI specifically so the
    password never appears in a subprocess argv, which ``ps`` exposes to every
    user on the machine for the whole extraction.

    The value is published alongside the dataset on HuggingFace; it is supplied
    per-machine rather than committed.
    """
    value = find_archive_password()
    if value:
        return value
    raise SystemExit(
        f"no archive password found: set ${ARCHIVE_PASSWORD_ENV} or write it to "
        f"{ARCHIVE_PASSWORD_FILE}.  The password is published with the dataset "
        f"on https://huggingface.co/datasets/{HF_REPO}"
    )


def count_extracted_files(target: Path) -> int:
    """Number of real files under ``target``, ignoring dotfiles.

    Dotfiles are excluded on both sides of the completeness comparison (here and
    in :func:`count_archive_entries`) so that this script's own marker -- and
    any ``.DS_Store``-style litter the filesystem drops in -- cannot make a
    complete extraction look over-full and trigger a needless re-extraction.
    """
    if not target.is_dir():
        return 0
    return sum(
        1
        for path in target.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )


def count_archive_entries(archive: Path, password: str) -> int | None:
    """Number of file entries inside ``archive``, or ``None`` if unreadable.

    Reading the entry list needs the password, so ``None`` is the honest answer
    on the no-password path (and when ``py7zr`` is missing, or the archive
    cannot be opened).  Callers must treat ``None`` as "cannot verify", never as
    "zero files".
    """
    if not password:
        return None
    try:
        import py7zr  # noqa: PLC0415 - only needed on the extraction path

        with py7zr.SevenZipFile(archive, mode="r", password=password) as handle:
            entries = handle.list()
    except ImportError:
        return None
    except Exception as error:  # noqa: BLE001 - verification is best-effort
        detail = str(error).replace(password, "***")
        log(f"extract: cannot read entry list of {archive.name}: {detail[:200]}")
        return None
    return sum(
        1
        for entry in entries
        if not entry.is_directory
        and not Path(entry.filename).name.startswith(".")
    )


def read_extract_marker(target: Path) -> dict[str, Any] | None:
    """Load ``target``'s extraction marker; ``None`` if absent or unreadable."""
    path = target / EXTRACT_MARKER_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_extract_marker(archive: Path, target: Path, file_count: int) -> None:
    """Record that ``archive`` is extracted into ``target`` in full.

    Deliberately records enough to be falsifiable on the next run: the archive
    it came from, that archive's size (so a re-published archive invalidates the
    marker) and the file count the extraction produced.  Never the password.
    """
    payload = {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size if archive.is_file() else None,
        "file_count": file_count,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (target / EXTRACT_MARKER_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def classify_extraction(archive: Path, target: Path, password: str) -> tuple[str, str]:
    """Decide whether ``target`` holds a *complete* extraction of ``archive``.

    Returns ``(status, message)`` where status is one of:

    ``complete-by-marker``
        A marker is present and still agrees with the archive and the disk.
    ``complete-by-count``
        No marker (the pre-existing corpora), but the file count on disk matches
        the archive's entry count.  The caller backfills the marker.
    ``unverified``
        No marker and no way to read the archive's entry list (no password, no
        ``py7zr``).  Falls back to the historical "directory exists means done"
        behaviour, but says so out loud.
    ``incomplete``
        A genuine mismatch.  The caller must re-extract.

    Directory existence alone is *not* evidence of completeness: an extraction
    killed halfway leaves a directory indistinguishable from a finished one, and
    the pipeline then builds a silently truncated corpus whose only symptom is a
    raised ``dropped_missing_audio`` count in the filter summary.
    """
    on_disk = count_extracted_files(target)
    marker = read_extract_marker(target)
    if marker is not None:
        size = archive.stat().st_size if archive.is_file() else None
        agrees = (
            marker.get("archive") == archive.name
            and marker.get("archive_bytes") == size
            and marker.get("file_count") == on_disk
        )
        if agrees:
            return (
                "complete-by-marker",
                f"extract: already extracted, skipping {archive.name} "
                f"(marker verified: {on_disk} files)",
            )
        return (
            "incomplete",
            f"extract: marker disagrees with disk for {archive.name} "
            f"(marker: archive={marker.get('archive')!r} "
            f"bytes={marker.get('archive_bytes')} files={marker.get('file_count')}; "
            f"disk: archive={archive.name!r} bytes={size} files={on_disk}); "
            f"re-extracting",
        )

    expected = count_archive_entries(archive, password)
    if expected is None:
        return (
            "unverified",
            f"extract: already extracted, skipping {archive.name} "
            f"(no marker and no readable entry list: completeness NOT verified, "
            f"{on_disk} files on disk)",
        )
    if expected == on_disk:
        return (
            "complete-by-count",
            f"extract: already extracted, skipping {archive.name} "
            f"(no marker; verified by count: {on_disk} files == {expected} "
            f"archive entries, writing marker)",
        )
    return (
        "incomplete",
        f"extract: incomplete extraction of {archive.name} "
        f"(archive holds {expected} files, {on_disk} on disk); re-extracting",
    )


def discard_partial_extraction(target: Path, raw_dir: Path) -> None:
    """Delete ``target`` -- and only ``target`` -- before a re-extraction.

    ``py7zr`` writes into the directory in place, so re-extracting on top of a
    partial tree would silently merge the two: files the previous attempt wrote
    from a *different* archive revision, or paths no longer present in the
    archive, would survive as orphans and get indexed as if they were current.

    The deletion is scoped hard: ``target`` must be a direct child of ``raw_dir``
    (i.e. one archive's own raw directory), and never ``raw_dir`` itself.
    """
    if target == raw_dir or target.parent != raw_dir:
        raise SystemExit(
            f"refusing to delete {target}: not a single archive's raw directory "
            f"under {raw_dir}"
        )
    if not target.is_dir() or not any(target.iterdir()):
        return
    log(f"extract: removing stale partial extraction {target}")
    shutil.rmtree(target)


def extraction_password_need(archive: Path, raw_dir: Path) -> str:
    """How badly the password is needed for ``archive``: skip/verify/extract.

    ``extract`` means the run cannot proceed without the secret; ``verify``
    means it is only wanted to check an existing extraction and its absence is
    survivable; ``skip`` means the marker already settles the question.
    """
    target = raw_dir / archive.stem
    if find_index_json(target) is None:
        return "extract"
    status, _ = classify_extraction(archive, target, "")
    if status == "complete-by-marker":
        return "skip"
    if status == "incomplete":
        return "extract"
    return "verify"


def extract_archive(archive: Path, raw_dir: Path, password: str) -> Path:
    """Extract ``archive`` and return the directory holding ``index.json``.

    Idempotent, but resume is decided by *verified completeness*, not by the
    output directory existing: see :func:`classify_extraction`.  An archive with
    a valid marker is skipped without needing the password at all.
    """
    target = raw_dir / archive.stem
    index = find_index_json(target)
    if index is not None:
        status, message = classify_extraction(archive, target, password)
        log(message)
        if status == "complete-by-count":
            write_extract_marker(archive, target, count_extracted_files(target))
            return index.parent
        if status != "incomplete":
            return index.parent

    # Either nothing is extracted yet, or what is there is provably incomplete.
    # Both cases must start from an empty directory.
    discard_partial_extraction(target, raw_dir)
    target.mkdir(parents=True, exist_ok=True)
    log(f"extract: {archive.name} -> {target}")
    try:
        import py7zr  # noqa: PLC0415 - only needed on the extraction path

        with py7zr.SevenZipFile(archive, mode="r", password=password) as handle:
            handle.extractall(path=target)
    except ImportError as error:
        raise SystemExit(
            "py7zr is required to extract the archives "
            "(pip install py7zr); the 7z CLI is not used because it would "
            "expose the password in the process argv"
        ) from error
    except Exception as error:  # noqa: BLE001 - surface a clean, secret-free message
        # Scrub defensively: some py7zr errors echo constructor arguments.  The
        # empty-password guard matters: "".replace() would splice the mask
        # between every character of the message.
        detail = str(error).replace(password, "***") if password else str(error)
        raise SystemExit(
            f"failed to extract {archive.name}: {type(error).__name__}: "
            f"{detail[:400]}"
        ) from None

    index = find_index_json(target)
    if index is None:
        raise SystemExit(f"no index.json found after extracting {archive.name}")
    # Only now -- after extractall returned and index.json is on disk -- is the
    # extraction complete.  An interrupted run never reaches this line, so the
    # next run cannot mistake its leftovers for a finished extraction.
    write_extract_marker(archive, target, count_extracted_files(target))
    return index.parent


def find_index_json(root: Path) -> Path | None:
    """Locate ``index.json`` at the root or one level down (archive layout)."""
    if not root.is_dir():
        return None
    direct = root / "index.json"
    if direct.is_file():
        return direct
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "index.json").is_file():
            return child / "index.json"
    return None


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def resolve_entry_paths(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(speaker, voice, relative_ogg_path)`` for one index.json entry.

    The archives do not share one schema, so both observed shapes are handled:

    * ``{"Speaker", "Voice", "Text"}`` -- ``Voice`` carries no extension and the
      audio is at ``<Speaker>/<Voice>.ogg`` (GIGA, Studio e.go!);
    * ``{"Speaker", "FilePath", "Text"}`` -- ``FilePath`` is a Windows-style
      relative path including the extension, e.g. ``琴子\\ep2_kot0002.ogg``
      (Purple software / Criminal Border).

    The relative path is empty when the entry names no audio at all, which the
    caller counts as a missing-audio drop.
    """
    speaker = str(entry.get("Speaker", "")).strip()

    voice = str(entry.get("Voice", "") or "").strip()
    if voice:
        relative = f"{speaker}/{voice}.ogg" if speaker else f"{voice}.ogg"
        return speaker, voice, relative

    file_path = str(entry.get("FilePath", "") or "").strip().replace("\\", "/")
    if file_path:
        as_path = Path(file_path)
        if not speaker and as_path.parent.name:
            speaker = as_path.parent.name
        return speaker, as_path.stem, file_path

    return speaker, "", ""


def plan_archive(
    index_path: Path,
    title: str,
    audio_root: Path,
    stats: FilterStats,
    archive_stats: ArchiveStats,
) -> list[ConvertTask]:
    """Turn one ``index.json`` into conversion tasks, applying text filters."""
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"{index_path} is not a JSON list")

    stats.index_entries += len(entries)
    archive_stats.index_entries = len(entries)

    source_root = index_path.parent

    # Pass 1 -- resolve entries and apply the per-entry filters.
    resolved: dict[Path, list[tuple[str, str, str]]] = {}
    for entry in entries:
        speaker, voice, relative = resolve_entry_paths(entry)
        text = normalize_text(str(entry.get("Text", "")))
        if not text:
            stats.empty_text += 1
            continue
        if not relative:
            stats.missing_audio += 1
            continue

        src = source_root / relative
        if not src.is_file():
            stats.missing_audio += 1
            continue
        resolved.setdefault(src, []).append((speaker, voice, text))

    # Pass 2 -- one record per audio file.  Some indexes list the same
    # ``(Speaker, Voice)`` twice: harmless when the transcripts agree, but two
    # observed pairs disagree outright, and there is no way to tell which line
    # the audio actually contains.  A clip with two contradictory targets is
    # worse than no clip -- CTC would be trained against a transcript that is
    # certainly wrong for one of them -- so the whole group is dropped.
    tasks: list[ConvertTask] = []
    seen_keys: set[str] = set()
    seen_dsts: set[Path] = set()
    for src, group in resolved.items():
        texts = {text for _, _, text in group}
        if len(texts) > 1:
            stats.ambiguous_duplicate += len(group)
            continue
        if len(group) > 1:
            stats.duplicate_entry += len(group) - 1
        speaker, voice, text = group[0]

        speaker_slug = slugify(speaker, "unknown_speaker")
        voice_slug = slugify(voice, "unknown_voice")
        # Distinct source files must never collapse onto one wav, so the key
        # and the destination are disambiguated together.
        suffix = 0
        key = f"{title}__{speaker_slug}__{voice_slug}"
        dst = audio_root / title / speaker_slug / f"{voice_slug}.wav"
        while key in seen_keys or dst in seen_dsts:
            suffix += 1
            key = f"{title}__{speaker_slug}__{voice_slug}_{suffix}"
            dst = audio_root / title / speaker_slug / f"{voice_slug}_{suffix}.wav"
        seen_keys.add(key)
        seen_dsts.add(dst)

        tasks.append(
            ConvertTask(
                key=key,
                title=title,
                speaker=f"{title}/{speaker_slug}",
                src=str(src),
                dst=str(dst),
                text=text,
            )
        )
    return tasks


def interleave(groups: Sequence[Sequence[ConvertTask]]) -> list[ConvertTask]:
    """Round-robin the per-archive task lists.

    With ``--limit-hours`` the tail is cut off, so interleaving is what keeps a
    small slice spanning every title (and therefore keeps a speaker-disjoint val
    split able to cover several archives).
    """
    merged: list[ConvertTask] = []
    for row in zip(*[iter_pad(group, max(len(g) for g in groups)) for group in groups]):
        merged.extend(task for task in row if task is not None)
    return merged


def iter_pad(items: Sequence[ConvertTask], length: int) -> Iterator[ConvertTask | None]:
    for index in range(length):
        yield items[index] if index < len(items) else None


# --------------------------------------------------------------------------
# Conversion (multiprocessing worker)
# --------------------------------------------------------------------------
def convert_one(task: tuple[str, str, float, float]) -> tuple[str, float, str, str]:
    """Decode one ogg to 16 kHz mono PCM16 wav.

    Returns ``(dst, duration_seconds, status, detail)`` where status is one of
    ``ok`` / ``too_short`` / ``too_long`` / ``missing`` / ``error`` and
    ``detail`` carries the exception text for ``error``.  Runs in a worker
    process, so imports are local and nothing is printed.
    """
    src, dst, min_seconds, max_seconds = task
    import librosa  # noqa: PLC0415 - worker-local, keeps parent import cheap
    import soundfile as sf  # noqa: PLC0415

    dst_path = Path(dst)
    try:
        if dst_path.is_file():
            info = sf.info(dst)
            duration = info.frames / float(info.samplerate)
            if duration < min_seconds:
                return dst, duration, "too_short", ""
            if duration > max_seconds:
                return dst, duration, "too_long", ""
            return dst, duration, "ok", ""

        if not Path(src).is_file():
            return dst, 0.0, "missing", ""

        # Header-only probe first: an out-of-range clip never gets decoded.
        info = sf.info(src)
        duration = info.frames / float(info.samplerate)
        if duration < min_seconds:
            return dst, duration, "too_short", ""
        if duration > max_seconds:
            return dst, duration, "too_long", ""

        audio, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
        if audio.size == 0:
            return dst, 0.0, "error", "decoded to zero samples"
        duration = audio.size / float(SAMPLE_RATE)
        if duration < min_seconds:
            return dst, duration, "too_short", ""
        if duration > max_seconds:
            return dst, duration, "too_long", ""

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file so an interrupted run never leaves a
        # truncated wav that the next run would trust.  ``format`` must be
        # explicit: soundfile infers it from the extension, and the extension
        # here is ``.tmp``.
        tmp = dst_path.with_name(dst_path.name + ".tmp")
        sf.write(tmp, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        tmp.replace(dst_path)
        return dst, duration, "ok", ""
    except Exception as error:  # noqa: BLE001 - one bad ogg must not kill the run
        return dst, 0.0, "error", f"{type(error).__name__}: {error}"


def convert_all(
    tasks: Sequence[ConvertTask],
    stats: FilterStats,
    archives: dict[str, ArchiveStats],
    min_seconds: float,
    max_seconds: float,
    limit_hours: float | None,
    workers: int,
) -> list[dict[str, Any]]:
    """Convert every task in order, stopping once ``limit_hours`` is reached."""
    if not tasks:
        return []

    limit_seconds = limit_hours * 3600.0 if limit_hours else None
    payload = [
        (task.src, task.dst, min_seconds, max_seconds) for task in tasks
    ]

    records: list[dict[str, Any]] = []
    total_seconds = 0.0
    started = time.monotonic()
    processed = 0

    reported_errors: set[str] = set()

    context = mp.get_context("spawn")
    with context.Pool(processes=max(1, workers)) as pool:
        results = pool.imap(convert_one, payload, chunksize=8)
        for task, (dst, duration, status, detail) in zip(tasks, results):
            processed += 1
            if status == "error" and detail and detail not in reported_errors:
                # Surface each distinct failure once: a systematic decode bug
                # otherwise hides as a slowly rising "dropped" counter.
                reported_errors.add(detail)
                log(f"convert: error on {Path(task.src).name}: {detail}")
            if status == "ok":
                records.append(build_record(task.key, task.text, dst, duration))
                stats.kept += 1
                total_seconds += duration
                archive = archives[task.title]
                archive.kept += 1
                archive.seconds += duration
                archive.speakers.add(task.speaker)
            elif status == "too_short":
                stats.too_short += 1
            elif status == "too_long":
                stats.too_long += 1
            elif status == "missing":
                stats.missing_audio += 1
            else:
                stats.decode_error += 1

            if processed % 500 == 0 or processed == len(tasks):
                log(
                    f"convert: {processed}/{len(tasks)} clips "
                    f"({stats.kept} kept, {total_seconds / 3600:.2f} h, "
                    f"eta {eta(processed, len(tasks), started)})"
                )

            if limit_seconds is not None and total_seconds >= limit_seconds:
                stats.over_limit_hours = len(tasks) - processed
                log(
                    f"convert: reached --limit-hours {limit_hours} "
                    f"({total_seconds / 3600:.3f} h); skipping "
                    f"{stats.over_limit_hours} remaining clips"
                )
                pool.terminate()
                break
    return records


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def total_hours(records: Sequence[dict[str, Any]]) -> float:
    """Recover audio hours from ``source_len`` (100 frames == 1 s)."""
    frames = sum(int(record["source_len"]) for record in records)
    return frames * FRAME_MS / 1000.0 / 3600.0


def build_manifest(
    args: argparse.Namespace,
    archives: dict[str, ArchiveStats],
    stats: FilterStats,
    train: Sequence[dict[str, Any]],
    val: Sequence[dict[str, Any]],
    val_surplus_dropped: int = 0,
) -> dict[str, Any]:
    all_records = list(train) + list(val)
    speaker_counts: dict[str, int] = {}
    for record in all_records:
        speaker = record_speaker(record)
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    val_speakers = sorted({record_speaker(record) for record in val})
    val_speaker_names = sorted({speaker_group_key(record) for record in val})
    shared_names = cross_title_speakers(all_records)

    val_by_title: dict[str, int] = {}
    for record in val:
        title = record_title(record)
        val_by_title[title] = val_by_title.get(title, 0) + 1
    val_by_speaker: dict[str, int] = {}
    for record in val:
        name = speaker_group_key(record)
        val_by_speaker[name] = val_by_speaker.get(name, 0) + 1
    train_by_title: dict[str, int] = {}
    for record in train:
        title = record_title(record)
        train_by_title[title] = train_by_title.get(title, 0) + 1

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": HF_REPO,
        "config": {
            "seed": args.seed,
            "out_dir": str(args.out_dir.resolve()),
            "sample_rate": SAMPLE_RATE,
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "val_fraction": args.val_frac,
            "limit_hours": args.limit_hours,
            "archives": list(args.archives),
            "source_len_convention": (
                "fbank frames at 100 fps (10 ms/frame), no LFR downsampling -- "
                "matches scripts/make_smoke_data.py and data/train_example.jsonl"
            ),
            "target_len_convention": (
                "character count excluding spaces (Japanese is unsegmented, so "
                "the whitespace word count used for English would always be 1)"
            ),
            "val_max_clips_per_speaker": args.val_max_clips_per_speaker,
            "val_cover_all_titles": args.val_cover_all_titles,
            "val_max_surplus_fraction": VAL_MAX_SURPLUS_FRACTION,
            "val_selection": (
                "stratified: each title receives a val quota proportional to "
                "its share of the corpus, and each speaker contributes at most "
                "val_max_clips_per_speaker clips (sampled at random, spread "
                "across the titles that speaker appears in).  Surplus clips of "
                "a val speaker are DROPPED, not returned to train, so voice "
                "identity stays strictly on one side -- see "
                "totals.val_surplus_clips_dropped."
            ),
            "split_grouping": (
                "bare speaker name across the whole corpus (NOT per-title): a "
                "name occurring in several titles is one group, because these "
                "archives are franchises and the same character name in a "
                "sequel is normally the same voice actor.  Conservative by "
                "design -- it may fuse distinct characters sharing a common "
                "name or a generic role label, which costs a little val data "
                "but cannot leak voice identity across the split."
            ),
        },
        "totals": {
            "clips": len(all_records),
            "hours": round(total_hours(all_records), 4),
            # Distinct voices.  ``title_speaker_pairs`` is the older, larger
            # number (the same name in two titles counts twice); keeping both
            # under explicit names avoids a field called "speakers" that
            # silently means something else.
            "speakers": len({speaker_group_key(r) for r in all_records}),
            "title_speaker_pairs": len(speaker_counts),
            "val_surplus_clips_dropped": val_surplus_dropped,
            "train_clips": len(train),
            "train_hours": round(total_hours(train), 4),
            "val_clips": len(val),
            "val_hours": round(total_hours(val), 4),
            "val_clip_fraction": (
                round(len(val) / len(all_records), 5) if all_records else 0.0
            ),
        },
        "archives": [archives[title].as_dict() for title in sorted(archives)],
        "filter_stats": stats.as_dict(),
        "val_speakers": val_speakers,
        "val_speaker_names": val_speaker_names,
        "val_titles": sorted({record_title(record) for record in val}),
        "val_clips_by_title": dict(sorted(val_by_title.items())),
        "val_uncovered_titles": sorted(
            title for title in train_by_title if title not in val_by_title
        ),
        "val_clips_by_speaker": dict(
            sorted(val_by_speaker.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "train_clips_by_title": dict(sorted(train_by_title.items())),
        "cross_title_speaker_names": shared_names,
        "speaker_counts": dict(sorted(speaker_counts.items())),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root (default: %(default)s)",
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        default=list(DEFAULT_ARCHIVES),
        metavar="PATH",
        help=(
            "Override the .7z paths to fetch, repo-relative, as printed by "
            "--list-archives (most but not all live under GalGame/).  Their "
            "filenames must be unique: the local layout is keyed by basename "
            f"(default: the {len(DEFAULT_ARCHIVES)} GalGame titles in "
            "DEFAULT_ARCHIVES)"
        ),
    )
    parser.add_argument(
        "--list-archives",
        action="store_true",
        help=(
            "List every .7z in the dataset repo (byte size, human size, running "
            "cumulative total) and exit without downloading or extracting "
            "anything.  Use it to pick an --archives subset by total size"
        ),
    )
    parser.add_argument(
        "--list-format",
        choices=("text", "plain"),
        default=None,
        help=(
            "Output style for --list-archives, which it requires: 'text' is "
            "the sizes table, 'plain' is shell-quoted paths one per line, "
            f"ready to paste after --archives (default: {DEFAULT_LIST_FORMAT})"
        ),
    )
    parser.add_argument(
        "--limit-hours",
        type=float,
        default=None,
        help="Stop after roughly this many hours of kept audio (trial runs)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=MIN_SECONDS,
        help="Drop clips shorter than this (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_SECONDS,
        help="Drop clips longer than this (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=VAL_FRACTION,
        help="Target validation clip fraction (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the speaker-disjoint split (default: %(default)s)",
    )
    parser.add_argument(
        "--val-max-clips-per-speaker",
        type=int,
        default=VAL_MAX_CLIPS_PER_SPEAKER,
        help="Per-speaker ceiling on val clips (default: %(default)s)",
    )
    parser.add_argument(
        "--val-cover-all-titles",
        action="store_true",
        help=(
            "Force every title into val even when its only speakers are main "
            "characters, whose surplus clips must then be dropped from train"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Audio conversion processes (default: %(default)s)",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Parallel download connections (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use archives already present in <out-dir>/archives",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # A pure query: it reads the repo tree and exits.  Handled before every
    # other check so it needs no archive password, creates no --out-dir, and is
    # unaffected by the pipeline's argument validation.
    if args.list_archives:
        print_archive_listing(
            list_repo_archives(read_hf_token()),
            args.list_format or DEFAULT_LIST_FORMAT,
        )
        return 0

    if args.list_format is not None:
        raise SystemExit("--list-format only applies to --list-archives")

    if args.max_seconds <= args.min_seconds:
        raise SystemExit("--max-seconds must be greater than --min-seconds")
    if args.limit_hours is not None and args.limit_hours <= 0:
        raise SystemExit("--limit-hours must be > 0")

    # Before anything touches the network or the disk: two archives sharing a
    # basename would collapse onto one local file and build the wrong corpus.
    check_archive_basenames(args.archives)

    out_dir: Path = args.out_dir
    archive_dir = out_dir / "archives"
    raw_dir = out_dir / "raw"
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.monotonic()
    log(f"out dir : {out_dir.resolve()}")
    log(f"archives: {len(args.archives)}")

    # 1. download -------------------------------------------------------
    downloaded: dict[str, Path] = {}
    if args.skip_download:
        for remote in args.archives:
            local = archive_dir / Path(remote).name
            if not local.is_file():
                raise SystemExit(f"--skip-download but missing: {local}")
            downloaded[remote] = local
    else:
        token = read_hf_token()
        downloaded = download_all(
            args.archives, archive_dir, token, args.download_workers
        )
        del token

    # 2. extract --------------------------------------------------------
    # The password is only read if something actually needs extracting or
    # verifying, so a re-run on a marker-verified corpus needs no secret at all.
    # An extraction that still has to be *checked* (no marker: the corpora built
    # before markers existed) asks for the password too, but tolerates its
    # absence -- that check is a safety net, not a hard requirement.
    password: str | None = None
    index_paths: list[tuple[str, Path]] = []
    for remote in args.archives:
        archive = downloaded[remote]
        if password is None:
            need = extraction_password_need(archive, raw_dir)
            if need == "extract":
                password = read_archive_password()
            elif need == "verify":
                password = find_archive_password()
        root = extract_archive(archive, raw_dir, password or "")
        index_paths.append((archive.stem, root / "index.json"))
    del password

    # 3. plan -----------------------------------------------------------
    stats = FilterStats()
    archives: dict[str, ArchiveStats] = {}
    groups: list[list[ConvertTask]] = []
    for archive_stem, index_path in index_paths:
        title = slugify(archive_stem, "unknown_title")
        archive_stats = ArchiveStats(archive=archive_stem, title=title)
        archives[title] = archive_stats
        tasks = plan_archive(index_path, title, audio_dir, stats, archive_stats)
        groups.append(tasks)
        log(f"plan    : {title}: {len(tasks)} clips with usable text")

    tasks = interleave(groups) if groups else []
    log(f"plan    : {len(tasks)} clips queued for conversion")

    # 4. convert --------------------------------------------------------
    records = convert_all(
        tasks,
        stats,
        archives,
        args.min_seconds,
        args.max_seconds,
        args.limit_hours,
        args.workers,
    )
    if not records:
        raise SystemExit("no clips survived filtering; nothing to write")

    # 5. split and write ------------------------------------------------
    shared = cross_title_speakers(records)
    if shared:
        log(
            f"split   : {len(shared)} speaker name(s) span multiple titles; "
            "each is kept whole on one side of the split"
        )
        for name, titles in shared.items():
            log(f"split   :   {name}: {', '.join(titles)}")
    train, val = split_by_speaker(
        records,
        args.val_frac,
        args.seed,
        max_clips_per_speaker=args.val_max_clips_per_speaker,
        cover_all_titles=args.val_cover_all_titles,
    )
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    val_surplus_dropped = len(records) - len(train) - len(val)
    manifest = build_manifest(args, archives, stats, train, val, val_surplus_dropped)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    totals = manifest["totals"]
    log(f"train   : {train_path} ({totals['train_clips']} clips, "
        f"{totals['train_hours']} h)")
    log(f"val     : {val_path} ({totals['val_clips']} clips, "
        f"{totals['val_hours']} h, {len(manifest['val_speaker_names'])} speakers, "
        f"{len(manifest['val_titles'])}/{len(archives)} titles, "
        # default=0: a single-speaker corpus yields an empty val set, and a
        # bare max() would raise here -- after every output is already written.
        f"max {max(manifest['val_clips_by_speaker'].values(), default=0)} "
        "clips/speaker)")
    log(f"val/ttl : {json.dumps(manifest['val_clips_by_title'], ensure_ascii=False)}")
    if manifest["val_uncovered_titles"]:
        log(
            "val     : no val representation for "
            f"{', '.join(manifest['val_uncovered_titles'])} -- its speakers are "
            "all main characters elsewhere; use --val-cover-all-titles to force "
            "coverage at the cost of dropping their clips from train"
        )
    if val_surplus_dropped:
        log(
            f"val     : dropped {val_surplus_dropped} surplus clips from val "
            "speakers (kept off both sides to protect voice disjointness)"
        )
    log(f"manifest: {manifest_path}")
    log(f"filters : {json.dumps(stats.as_dict(), ensure_ascii=False)}")
    log(f"elapsed : {fmt_duration(time.monotonic() - run_started)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
