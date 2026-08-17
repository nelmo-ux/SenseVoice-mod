#!/usr/bin/env python3
"""Merge the two emotion labellers into the file ``prepare_vn_data.py`` consumes.

``scripts/label_emotions_audio.py`` labels the waveform and
``scripts/label_emotions_text.py`` labels the transcript.  This script joins
them on ``key`` and decides, per clip, whether their agreement is strong enough
to be used as supervision.  Its output is what
``prepare_vn_data.py --emo-labels`` reads to fill ``emo_target``.

    .venv/bin/python scripts/merge_emo_labels.py \\
        --audio outputs/emo_audio.jsonl \\
        --text outputs/emo_text.jsonl \\
        --out data/vn/emo_labels.jsonl \\
        --stats-out outputs/emo_merge_stats.json

**The two labellers must have been run over the same clips.**  If a pilot used
``--limit`` or ``--sample N --seed S``, the identical selection has to have been
passed to *both* labellers.  Nothing in the join can repair a mismatch: two runs
over different subsets produce a merge in which almost every clip is missing
from one side, which reads as "the labellers never agree" rather than "the
labellers never met".  ``--min-overlap`` (default 0.95) is the guard, and it
fails the run rather than writing a plausible-looking file with no supervision
in it.

Every clip in the union of the two inputs appears in the output.  Clips whose
labellers disagree, or that either labeller declined to classify, get
``<|SER|>`` -- the sentinel the model turns into ``ignore_id`` on the emotion
position.  **A masked clip is not a dropped clip**: it still trains the ASR and
CTC branches at full weight, it simply contributes nothing to the emotion head.
That is why this script can afford to be strict.  The expensive error is a
wrong emotion label, which teaches the head something false; the cheap error is
a mask, which teaches it nothing.

The NEUTRAL cap
---------------

Rounds 1 and 2 labelled every clip ``<|NEUTRAL|>`` and the emotion head learned
the constant.  Real labels do not fully fix that on their own: VN dialogue is
genuinely majority-neutral, and both labellers additionally *default* towards
neutral when unsure, so the two agree on ``neutral`` far more often than they
agree on anything else.  Agreement-filtered labels are therefore still
neutral-dominated, and a head trained on them can still reach a good loss by
predicting neutral almost always.

``--neutral-cap`` bounds ``<|NEUTRAL|>`` at a fraction of the supervised labels
and masks the excess.  The demotion order is the least-confident neutrals
first, by ``(audio_score ascending, key ascending)``: the clips the acoustic
labeller was least sure about are the ones whose neutrality is least likely to
be real, and the ``key`` tiebreak makes two runs over the same inputs produce
byte-identical output.

The optional confidence fallback
--------------------------------

There was one, ``--audio-conf-fallback TAU``: when the text said ``neutral`` and
the audio disagreed with confidence at or above TAU, adopt the audio label.  The
reasoning was that a line can be textually flat and delivered with obvious
affect ("そう", shouted), which a transcript labeller cannot see.

**Measured on 2026-08-14 over a 5,000-clip pilot, and removed.**  emotion2vec+
large has a corpus-wide domain shift on visual-novel dialogue: it returns
``surprised`` for 32.1% of clips and ``happy`` for 24.0%, against ``neutral``
for 1.64% -- ordinary conversational lines come back as ``surprised`` at a score
of 1.000.  Roughly 85% of all clips score above 0.7.

That makes the rule actively harmful rather than merely useless.  **Confidence
is not a proxy for correctness on this data**: any threshold worth the name
selects nearly the whole corpus, and confident mislabels are abundant inside the
selection, so the fallback would have imported the domain shift wholesale into
exactly the clips the text labeller had judged neutral.  The branch is gone
rather than defaulted-off, because a flag whose only honest documentation is
"measured, does not work, do not enable" reads as an invitation.

The disagreement confusion matrix stays in the stats, and is now *more* useful
than when it was the fallback's evidence: it is how the domain shift gets read
off the full corpus, and it is the input to whichever merge rule replaces this
one.

A second hypothesis is still open and points the other way.  Audio ``neutral``
is rare (1.64%) but appeared in the pilot to land correctly on short flat
utterances, so the acoustic labeller may be trustworthy in that one direction
while being useless in the other.  That is **unmeasured** -- the merge analysis
after the text pilot decides it -- so no rule is built on it here.  What this
file does guarantee is that the raw material survives: ``audio_label`` and
``audio_score`` stay on every output row and the per-decision counts stay in the
stats, so the question can be answered from a merge output without relabelling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, TextIO

__all__ = [
    "MASK_TOKEN",
    "NEUTRAL_TOKEN",
    "EMOTION_TOKENS",
    "AUDIO_LABEL_TO_TOKEN",
    "TEXT_LABEL_TO_TOKEN",
    "DEFAULT_MIN_OVERLAP",
    "LabelFile",
    "MergedRow",
    "apply_neutral_cap",
    "decide",
    "load_label_file",
    "merge",
    "overlap_fraction",
    "summarise",
    "main",
]

#: The mask sentinel.  A single sentencepiece token (24991) that the model maps
#: to ``ignore_id`` on the emotion position, so the emotion loss skips the clip
#: while every other loss term still sees it.
MASK_TOKEN = "<|SER|>"

NEUTRAL_TOKEN = "<|NEUTRAL|>"

#: The seven emotion tokens this pipeline may emit, in vocabulary-id order.
#:
#: ``<|EMO_UNKNOWN|>`` (25009) is deliberately absent and must stay absent.
#: Inference bans it (``ban_emo_unk``), so a target the decoder can never emit
#: is capacity spent on nothing -- and worse, it is a second high-frequency
#: catch-all class, which is how the constant-label collapse happened the first
#: time.  Uncertainty belongs in :data:`MASK_TOKEN`, which costs the head
#: nothing, not in a class.
EMOTION_TOKENS: tuple[str, ...] = (
    "<|HAPPY|>",
    "<|SAD|>",
    "<|ANGRY|>",
    "<|NEUTRAL|>",
    "<|FEARFUL|>",
    "<|DISGUSTED|>",
    "<|SURPRISED|>",
)

#: emotion2vec+ large's nine classes -> target token.  Total over the model's
#: label set: an unmapped class would raise rather than default, but the tests
#: assert totality so that never reaches a 550k-clip run.
AUDIO_LABEL_TO_TOKEN: Dict[str, str] = {
    "angry": "<|ANGRY|>",
    "disgusted": "<|DISGUSTED|>",
    "fearful": "<|FEARFUL|>",
    "happy": "<|HAPPY|>",
    "neutral": "<|NEUTRAL|>",
    "sad": "<|SAD|>",
    "surprised": "<|SURPRISED|>",
    "other": MASK_TOKEN,
    "unknown": MASK_TOKEN,
}

#: The text labeller's ten classes -> target token.  ``embarrassed`` and
#: ``sexual`` are auxiliary buckets: they exist so those lines do not have to be
#: forced into one of the seven, and they map to the mask here.  See
#: ``scripts/label_emotions_text.py`` for why that is better than dropping the
#: clips or folding them into ``happy``.
TEXT_LABEL_TO_TOKEN: Dict[str, str] = {
    "happy": "<|HAPPY|>",
    "sad": "<|SAD|>",
    "angry": "<|ANGRY|>",
    "neutral": "<|NEUTRAL|>",
    "fearful": "<|FEARFUL|>",
    "disgusted": "<|DISGUSTED|>",
    "surprised": "<|SURPRISED|>",
    "other": MASK_TOKEN,
    "embarrassed": MASK_TOKEN,
    "sexual": MASK_TOKEN,
}

#: Text classes that are auxiliary buckets rather than a refusal.  Separated
#: from ``other`` in the decision log so the pilot can tell "the model had no
#: opinion" apart from "the model recognised a category we do not train".
AUX_TEXT_LABELS = frozenset({"embarrassed", "sexual"})

#: The decisions :func:`decide` and :func:`apply_neutral_cap` may record.
DECISIONS: tuple[str, ...] = (
    "agree",
    "disagree_masked",
    "aux_masked",
    "other_masked",
    "missing_masked",
    "neutral_capped",
)

DEFAULT_NEUTRAL_CAP = 0.5

#: Minimum Jaccard overlap between the two labellers' key sets.  Nothing else
#: enforces that they were run over the same clips: the join silently succeeds
#: on disjoint inputs and reports every clip as ``missing_masked``, which looks
#: like two labellers that never agree rather than two runs that never met.
#: Not 1.0, because a legitimately pre-empted labeller leaves a small tail
#: behind; anything below this is an operator error, not a tail.
DEFAULT_MIN_OVERLAP = 0.95


@dataclass
class MergedRow:
    """One clip's merged label.

    Attributes:
        key: The join identifier.
        emo_target: One of :data:`EMOTION_TOKENS` or :data:`MASK_TOKEN`.
        decision: The final decision, one of :data:`DECISIONS`.  Mutated to
            ``neutral_capped`` by :func:`apply_neutral_cap`.
        audio_label: The raw audio class, or ``None`` when absent.
        audio_score: The audio labeller's confidence, or ``None``.
        text_label: The raw text class, or ``None`` when absent.
        original_decision: The decision before the neutral cap.  Not emitted;
            the cap runs after the fact, and folding its demotions into the
            agreement rate would make the reported agreement depend on the cap
            rather than on the labellers.
    """

    key: str
    emo_target: str
    decision: str
    audio_label: Optional[str] = None
    audio_score: Optional[float] = None
    text_label: Optional[str] = None
    original_decision: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.original_decision:
            self.original_decision = self.decision

    def to_json(self) -> Dict[str, Any]:
        """Render the row for the output jsonl."""
        return {
            "key": self.key,
            "emo_target": self.emo_target,
            "decision": self.decision,
            "audio_label": self.audio_label,
            "audio_score": self.audio_score,
            "text_label": self.text_label,
        }


# ------------------------------------------------------------------- label I/O


@dataclass(frozen=True)
class LabelFile:
    """One labeller's output, split into what it labelled and what it saw.

    The two sets differ, and the difference is the diagnosis.  ``keys`` is every
    clip the labeller *processed*; ``labels`` is the subset it produced a usable
    label for.  A clip missing from ``keys`` means the two labellers were run
    over different subsets -- an operator error.  A clip in ``keys`` but not in
    ``labels`` means the labeller ran and declined -- a model or prompt problem.
    Both end up as ``missing_masked``, so without this split the report cannot
    tell them apart, and they need completely different fixes.

    Attributes:
        labels: ``key -> {"label", "score"}`` for usable labels only.
        keys: Every key seen in the file, usable or not.
    """

    labels: Dict[str, Dict[str, Any]]
    keys: Set[str]

    @classmethod
    def from_labels(cls, labels: Dict[str, Dict[str, Any]]) -> "LabelFile":
        """Build one from a bare label map, assuming it saw exactly those keys."""
        return cls(labels=labels, keys=set(labels))


def load_label_file(path: Path) -> LabelFile:
    """Read one labeller's jsonl.

    Both labellers append rather than truncate, so a resumed job can legally
    leave a duplicate key behind (a batch flushed just before the kill, then
    redone).  The last occurrence wins, which is the completed one.

    Only ``label`` and ``score`` are retained.  The rest of each row -- the
    audio labeller's 9-element ``probs`` dict, the text labeller's ``raw``
    string -- is diagnostic output for a human reading the labeller's file, and
    nothing here reads it.  At 550k clips x 2 files, holding those two fields
    costs on the order of a gigabyte of resident memory for no purpose.

    Args:
        path: A labeller's output jsonl.

    Returns:
        A :class:`LabelFile`.  Records whose ``label`` is ``None`` -- the text
        labeller's "did not parse" -- are kept out of ``labels`` (so they reach
        :func:`decide` as missing and mask the clip) but stay in ``keys``, which
        is what lets the report attribute them to a parse failure rather than to
        a mismatched subset.

    Raises:
        ValueError: On malformed JSON or a record with no ``key``.
    """
    labels: Dict[str, Dict[str, Any]] = {}
    keys: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            key = record.get("key")
            if key is None:
                raise ValueError(f"{path}:{lineno}: record missing key")
            key = str(key)
            keys.add(key)
            if record.get("label") is None:
                labels.pop(key, None)
                continue
            labels[key] = {"label": record["label"], "score": record.get("score")}
    return LabelFile(labels=labels, keys=keys)


def overlap_fraction(audio: LabelFile, text: LabelFile) -> float:
    """Jaccard overlap of the two labellers' processed key sets.

    Computed over ``keys`` rather than ``labels`` so that a labeller which ran
    everywhere but parsed nothing reads as full overlap with zero usable labels
    -- a different fault, caught by the text labeller's own unparsed-fraction
    guard -- instead of masquerading as a subset mismatch.

    Args:
        audio: The audio labeller's file.
        text: The text labeller's file.

    Returns:
        ``|A n T| / |A u T|``, or ``0.0`` when both are empty.
    """
    union = audio.keys | text.keys
    if not union:
        return 0.0
    return len(audio.keys & text.keys) / len(union)


# -------------------------------------------------------------------- decisions


def decide(
    key: str,
    audio: Optional[Dict[str, Any]],
    text: Optional[Dict[str, Any]],
) -> MergedRow:
    """Decide one clip's emotion target from the two labellers.

    Rules, in order:

    1. Either labeller missing -> mask (``missing_masked``).
    2. Either label maps to the mask -> mask (``aux_masked`` when the text
       labeller chose an auxiliary bucket, otherwise ``other_masked``).  The two
       are distinguished only for the report; the target is the same.
    3. Both map to the same token -> adopt it (``agree``).
    4. Otherwise mask (``disagree_masked``).  There is no override, by
       measurement rather than by caution -- see the module docstring on the
       removed confidence fallback.

    Args:
        key: The clip key.
        audio: The audio labeller's record, or ``None``.
        text: The text labeller's record, or ``None``.

    Returns:
        The decided row.

    Raises:
        KeyError: If either record carries a class name absent from its mapping
            table.  Not defaulted to the mask: a class this script does not know
            means the labeller changed under it, and silently masking those
            clips would quietly delete a whole emotion from the training signal.
    """
    audio_label = audio.get("label") if audio else None
    text_label = text.get("label") if text else None
    audio_score = audio.get("score") if audio else None

    base = {
        "key": key,
        "audio_label": audio_label,
        "audio_score": audio_score,
        "text_label": text_label,
    }

    if audio_label is None or text_label is None:
        return MergedRow(emo_target=MASK_TOKEN, decision="missing_masked", **base)

    if audio_label not in AUDIO_LABEL_TO_TOKEN:
        raise KeyError(f"{key}: unknown audio class {audio_label!r}")
    if text_label not in TEXT_LABEL_TO_TOKEN:
        raise KeyError(f"{key}: unknown text class {text_label!r}")

    audio_token = AUDIO_LABEL_TO_TOKEN[audio_label]
    text_token = TEXT_LABEL_TO_TOKEN[text_label]

    if audio_token == MASK_TOKEN or text_token == MASK_TOKEN:
        # When both sides abstain at once -- audio ``other``, text ``sexual`` --
        # the aux bucket wins the attribution. Deliberate: "the text labeller
        # recognised a category we do not train" is a specific, actionable fact
        # about the corpus, whereas "a labeller had no opinion" is not. The
        # target is MASK either way, so this choice only moves a number between
        # two report lines, and it moves it to the more informative one.
        decision = "aux_masked" if text_label in AUX_TEXT_LABELS else "other_masked"
        return MergedRow(emo_target=MASK_TOKEN, decision=decision, **base)

    if audio_token == text_token:
        return MergedRow(emo_target=audio_token, decision="agree", **base)

    # Every disagreement masks. A confidence-based override used to live here --
    # "text says neutral, audio is sure it is X, take X" -- and the 5,000-clip
    # pilot of 2026-08-14 killed it: emotion2vec+ large returns surprised for
    # 32.1% of this corpus and happy for 24.0% against neutral for 1.64%,
    # scoring 1.000 on ordinary conversational lines, with ~85% of all clips
    # above 0.7. A threshold therefore selects nearly everything and the
    # selection is full of confident mislabels, so the override imported that
    # domain shift into precisely the clips the text labeller had called
    # neutral. audio_score survives on the row (the neutral cap orders by it,
    # and the open "is audio trustworthy when it says neutral" question needs
    # it) but nothing here reads it as evidence of correctness.
    return MergedRow(emo_target=MASK_TOKEN, decision="disagree_masked", **base)


def _neutral_sort_key(row: MergedRow) -> tuple[float, str]:
    """Order neutrals least-confident-first, deterministically.

    A missing ``audio_score`` sorts below every real probability so it is
    demoted first: no confidence is not the same as high confidence, and the cap
    should spend its budget on the labels it can actually vouch for.
    """
    score = row.audio_score
    return (float(score) if score is not None else -1.0, row.key)


def apply_neutral_cap(
    rows: Sequence[MergedRow],
    cap: float,
    warn: Optional[TextIO] = None,
) -> int:
    """Demote the excess ``<|NEUTRAL|>`` labels to the mask, in place.

    The cap is on the *final* composition, which makes it a fixed point rather
    than a one-shot trim: demoting a neutral shrinks the supervised set too, so
    keeping ``k`` neutrals alongside ``m`` non-neutrals satisfies the cap only
    when ``k <= cap * (k + m)``, i.e. ``k <= floor(cap * m / (1 - cap))``.
    Trimming to ``cap * (current total)`` instead would leave the result above
    the cap, which is the easy mistake here.

    Args:
        rows: All merged rows; mutated in place.
        cap: Maximum fraction of the supervised labels that may be
            ``<|NEUTRAL|>``.  ``>= 1.0`` disables the cap.
        warn: Stream for the zero-non-neutral warning; ``None`` silences it.

    Returns:
        How many rows were demoted.
    """
    if cap >= 1.0:
        return 0

    neutrals = [row for row in rows if row.emo_target == NEUTRAL_TOKEN]
    others = sum(
        1 for row in rows if row.emo_target not in (NEUTRAL_TOKEN, MASK_TOKEN)
    )
    if not neutrals:
        return 0

    if others == 0 and warn is not None:
        # The formula is correct here -- with no non-neutral labels to be a
        # majority of, the cap allows zero neutrals -- but the consequence is
        # that a label file with real agreement in it becomes a file with no
        # supervision at all, and the per-decision counts would show that as an
        # ordinary cap effect.
        print(
            f"warning: every one of the {len(neutrals)} supervised labels is "
            "<|NEUTRAL|>, so the cap demotes all of them and the output will "
            "carry no emotion supervision at all. Check the labellers before "
            "using this file, or pass --neutral-cap 1.0 to disable the cap.",
            file=warn,
        )

    allowed = 0 if cap <= 0.0 else int(math.floor(cap * others / (1.0 - cap)))
    excess = len(neutrals) - allowed
    if excess <= 0:
        return 0

    neutrals.sort(key=_neutral_sort_key)
    for row in neutrals[:excess]:
        row.emo_target = MASK_TOKEN
        row.decision = "neutral_capped"
    return excess


def merge(
    audio: LabelFile,
    text: LabelFile,
    neutral_cap: float = DEFAULT_NEUTRAL_CAP,
    sample: Optional[int] = None,
    seed: int = 0,
    warn: Optional[TextIO] = None,
) -> List[MergedRow]:
    """Join, decide and cap.

    Args:
        audio: The audio labeller's file.
        text: The text labeller's file.
        neutral_cap: See :func:`apply_neutral_cap`.
        sample: Restrict to N keys drawn uniformly at random, for the pilot.
            Drawn from the sorted union so the draw does not depend on file
            order.
        seed: Seed for ``sample``.
        warn: Stream for :func:`apply_neutral_cap`'s warning.

    Returns:
        One row per key, sorted by key.  The union is taken over the labellers'
        *processed* keys, not their usable labels, so a clip whose text response
        failed to parse still gets a row rather than vanishing.  Sorted rather
        than left in input order so that two runs over the same inputs -- in
        either file order -- produce byte-identical output.
    """
    keys = sorted(audio.keys | text.keys)
    if sample is not None and sample < len(keys):
        keys = sorted(random.Random(seed).sample(keys, sample))

    rows = [decide(key, audio.labels.get(key), text.labels.get(key)) for key in keys]
    apply_neutral_cap(rows, neutral_cap, warn=warn)
    return rows


# ---------------------------------------------------------------------- report


def _missing_breakdown(
    rows: Sequence[MergedRow],
    audio: Optional[LabelFile],
    text: Optional[LabelFile],
) -> Optional[Dict[str, int]]:
    """Split ``missing_masked`` into the causes that need different fixes.

    ``missing_masked`` conflates "this clip was never given to that labeller"
    with "that labeller ran and produced nothing usable".  The first is an
    operator error in how the two jobs were launched; the second is a model or
    prompt problem.  A merge report that cannot separate them sends whoever
    reads it to the wrong place.

    Args:
        rows: The merged rows.
        audio: The audio labeller's file, or ``None`` if unavailable.
        text: The text labeller's file, or ``None``.

    Returns:
        The four counts, or ``None`` when the key sets were not supplied.
    """
    if audio is None or text is None:
        return None
    breakdown = {"audio_only": 0, "text_only": 0, "audio_unusable": 0, "text_unusable": 0}
    for row in rows:
        if row.original_decision != "missing_masked":
            continue
        in_audio = row.key in audio.keys
        in_text = row.key in text.keys
        if in_audio and not in_text:
            breakdown["audio_only"] += 1
        elif in_text and not in_audio:
            breakdown["text_only"] += 1
        else:
            if row.audio_label is None:
                breakdown["audio_unusable"] += 1
            if row.text_label is None:
                breakdown["text_unusable"] += 1
    return breakdown


def summarise(
    rows: Sequence[MergedRow],
    neutral_cap: float = DEFAULT_NEUTRAL_CAP,
    audio: Optional[LabelFile] = None,
    text: Optional[LabelFile] = None,
) -> Dict[str, Any]:
    """Aggregate the merged rows into the stats report.

    Args:
        rows: The merged rows, after the cap.
        neutral_cap: The cap that was in force.
        audio: The audio labeller's file, for the join diagnostics.
        text: The text labeller's file, for the join diagnostics.

    Returns:
        A JSON-serialisable report.  The agreement rate and the confusion
        matrix are computed from ``original_decision``, so the neutral cap --
        which is a policy applied on top, not a labeller behaviour -- does not
        move them.
    """
    decisions = Counter(row.decision for row in rows)
    distribution = Counter(row.emo_target for row in rows)
    usable = sum(count for token, count in distribution.items() if token != MASK_TOKEN)

    # Denominator: pairs where both labellers produced a token, i.e. the pairs
    # that could have agreed. Masked-by-construction pairs are excluded because
    # counting them as disagreements would make the rate a measure of the
    # auxiliary buckets' frequency instead of of labeller agreement.
    comparable = [
        row
        for row in rows
        if row.original_decision in ("agree", "disagree_masked")
    ]
    agreed = sum(1 for row in comparable if row.original_decision == "agree")

    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in comparable:
        if row.original_decision == "agree":
            continue
        # Subscript, never ``.get(..., MASK_TOKEN)``. Every row reaching here
        # already passed ``decide``'s membership check, so a KeyError is
        # impossible today -- but a silent MASK_TOKEN default directly
        # contradicts the strictness three functions up, and would quietly
        # bucket a renamed class into the confusion matrix's mask column if a
        # future refactor ever let an unmapped label through.
        audio_token = AUDIO_LABEL_TO_TOKEN[row.audio_label or ""]
        text_token = TEXT_LABEL_TO_TOKEN[row.text_label or ""]
        confusion[audio_token][text_token] += 1

    total = len(rows)
    neutral_capped = decisions.get("neutral_capped", 0)
    return {
        "total_keys": total,
        "decisions": {name: decisions.get(name, 0) for name in DECISIONS},
        "label_distribution": {
            "counts": {
                token: distribution.get(token, 0)
                for token in (*EMOTION_TOKENS, MASK_TOKEN)
            },
            "fractions": {
                token: (distribution.get(token, 0) / total if total else 0.0)
                for token in (*EMOTION_TOKENS, MASK_TOKEN)
            },
        },
        "num_comparable": len(comparable),
        "num_agreed": agreed,
        "agreement_rate": (agreed / len(comparable)) if comparable else None,
        "disagreement_confusion": {
            audio_token: dict(sorted(by_text.items()))
            for audio_token, by_text in sorted(confusion.items())
        },
        "neutral_cap": {
            "cap": neutral_cap,
            "enabled": neutral_cap < 1.0,
            "neutral_before": distribution.get(NEUTRAL_TOKEN, 0) + neutral_capped,
            "neutral_after": distribution.get(NEUTRAL_TOKEN, 0),
            "demoted": neutral_capped,
        },
        "num_usable": usable,
        "usable_fraction": (usable / total) if total else 0.0,
        "num_audio_keys": len(audio.keys) if audio is not None else None,
        "num_text_keys": len(text.keys) if text is not None else None,
        "num_audio_only": (
            len(audio.keys - text.keys) if audio is not None and text is not None else None
        ),
        "num_text_only": (
            len(text.keys - audio.keys) if audio is not None and text is not None else None
        ),
        "overlap_fraction": (
            overlap_fraction(audio, text) if audio is not None and text is not None else None
        ),
        "missing_breakdown": _missing_breakdown(rows, audio, text),
    }


def format_summary(stats: Dict[str, Any]) -> str:
    """Render the stats report as a readable block.

    Args:
        stats: Output of :func:`summarise`.

    Returns:
        A multi-line plain-text block.
    """
    total = stats["total_keys"]
    lines = [
        "emotion label merge",
        f"  clips            : {total}",
        f"  usable labels    : {stats['num_usable']} ({stats['usable_fraction']:.1%})",
    ]
    if stats.get("overlap_fraction") is not None:
        lines.append(
            f"  join             : {stats['overlap_fraction']:.1%} overlap"
            f"  (audio {stats['num_audio_keys']}, text {stats['num_text_keys']},"
            f" audio-only {stats['num_audio_only']}, text-only {stats['num_text_only']})"
        )
    breakdown = stats.get("missing_breakdown")
    if breakdown and any(breakdown.values()):
        rendered = "  ".join(f"{name}:{count}" for name, count in breakdown.items())
        lines.append(f"  missing causes   : {rendered}")
    rate = stats["agreement_rate"]
    lines.append(
        "  agreement        : "
        + (
            f"{rate:.1%} ({stats['num_agreed']}/{stats['num_comparable']} comparable pairs)"
            if rate is not None
            else "no comparable pairs"
        )
    )
    lines.append("  decisions        :")
    for name, count in stats["decisions"].items():
        share = count / total if total else 0.0
        lines.append(f"    {name:<16} {count:>9}  {share:6.1%}")
    lines.append("  final labels     :")
    counts = stats["label_distribution"]["counts"]
    fractions = stats["label_distribution"]["fractions"]
    for token, count in counts.items():
        lines.append(f"    {token:<16} {count:>9}  {fractions[token]:6.1%}")

    cap = stats["neutral_cap"]
    lines.append(
        f"  neutral cap      : {cap['cap']:.2f}"
        + (
            f"  {cap['neutral_before']} -> {cap['neutral_after']} (demoted {cap['demoted']})"
            if cap["enabled"]
            else "  disabled"
        )
    )
    confusion = stats["disagreement_confusion"]
    if confusion:
        # Printed, not just written to the stats JSON: this matrix is how the
        # acoustic labeller's domain shift is read off a run, and it is the
        # input to whatever merge rule replaces the removed fallback.
        lines.append("  disagreements (audio -> text):")
        for audio_token, by_text in confusion.items():
            rendered = "  ".join(f"{text}:{n}" for text, n in by_text.items())
            lines.append(f"    {audio_token:<16} {rendered}")
    return "\n".join(lines)


def write_rows(rows: Sequence[MergedRow], path: Path) -> None:
    """Write the merged label jsonl.

    Args:
        rows: Rows, already sorted by :func:`merge`.
        path: Destination; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="jsonl from scripts/label_emotions_audio.py",
    )
    parser.add_argument(
        "--text",
        type=Path,
        required=True,
        help="jsonl from scripts/label_emotions_text.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="destination jsonl consumed by prepare_vn_data.py --emo-labels",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="stats JSON destination (default: <out>.stats.json)",
    )
    parser.add_argument(
        "--neutral-cap",
        type=float,
        default=DEFAULT_NEUTRAL_CAP,
        help=(
            "max fraction of supervised labels that may be <|NEUTRAL|>; the "
            "least-confident excess is masked. 1.0 disables (default: "
            f"{DEFAULT_NEUTRAL_CAP})"
        ),
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP,
        help=(
            "minimum |audio n text| / |audio u text| over the labellers' key "
            "sets; below this the run aborts rather than reporting non-overlap "
            f"as disagreement. 0 disables (default: {DEFAULT_MIN_OVERLAP})"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="merge only N keys drawn uniformly at random, for the pilot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for --sample (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the two labellers and write the emotion target file.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when an input is unusable.
    """
    args = parse_args(argv)

    for path in (args.audio, args.text):
        if not path.is_file():
            print(f"label file not found: {path}", file=sys.stderr)
            return 1
    if not 0.0 <= args.neutral_cap <= 1.0:
        print("--neutral-cap must be between 0 and 1", file=sys.stderr)
        return 1
    if not 0.0 <= args.min_overlap <= 1.0:
        print("--min-overlap must be between 0 and 1", file=sys.stderr)
        return 1

    try:
        audio = load_label_file(args.audio)
        text = load_label_file(args.text)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not audio.keys and not text.keys:
        print("both label files are empty", file=sys.stderr)
        return 1

    overlap = overlap_fraction(audio, text)
    if overlap < args.min_overlap:
        print(
            f"aborting: the two labellers overlap on only {overlap:.1%} of their "
            f"keys ({len(audio.keys & text.keys)} shared, {len(audio.keys - text.keys)} "
            f"audio-only, {len(text.keys - audio.keys)} text-only), below "
            f"--min-overlap {args.min_overlap:.2f}.\n"
            "The two labellers were almost certainly run over different subsets: "
            "check that --limit / --sample / --seed matched between the audio and "
            "text runs, and that both were pointed at the same manifest.\n"
            "Merging anyway would measure non-overlap and report it as "
            "disagreement, producing a plausible-looking file with little or no "
            "emotion supervision in it.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = merge(
            audio,
            text,
            neutral_cap=args.neutral_cap,
            sample=args.sample,
            seed=args.seed,
            warn=sys.stderr,
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_rows(rows, args.out)
    stats = summarise(rows, args.neutral_cap, audio, text)
    stats_path = args.stats_out or args.out.with_suffix(args.out.suffix + ".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "audio": str(args.audio),
                "text": str(args.text),
                "out": str(args.out),
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(format_summary(stats))
    print(f"  labels           : {args.out}")
    print(f"  stats            : {stats_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
