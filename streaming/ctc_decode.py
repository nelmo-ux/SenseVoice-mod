"""Greedy CTC decoding for the streaming pipeline.

This is a standalone reimplementation of the decode path inside
``SenseVoiceSmall.inference`` (``model.py``): argmax over the vocabulary,
collapse of consecutive duplicates, removal of the blank symbol, and a
tokenizer lookup.  It is kept here so the streaming loop can decode an encoder
output directly, without going through ``inference`` (which owns audio loading,
batching and file writing) and without pulling in an external decoder package.
"""

from __future__ import annotations

import re
from typing import Any, List

import torch

__all__ = ["ctc_greedy_token_ids", "ctc_greedy_decode", "strip_rich_tags"]

#: Matches a single ``<|...|>`` rich tag.  The character class ``[^|]*`` keeps
#: the match non-greedy *and* prevents it from spanning the ``|`` of a later
#: tag, so ``<|a|><|b|>`` is removed as two separate tags.
_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")


def ctc_greedy_token_ids(
    ctc_logits: "torch.Tensor",
    blank_id: int = 0,
) -> List[int]:
    """Greedily decode CTC output for one utterance into token ids.

    Applies the same three steps as ``model.py``: ``argmax(dim=-1)``,
    ``torch.unique_consecutive(dim=-1)``, then a ``!= blank_id`` mask.

    Args:
        ctc_logits: A 2-D tensor of shape ``(T, V)`` holding log-probabilities
            (or raw logits - argmax is invariant to the softmax) for a *single*
            utterance.  The caller is responsible for stripping the batch
            dimension and trimming to the true length.
        blank_id: Vocabulary index of the CTC blank symbol.

    Returns:
        The decoded token ids, blanks removed.  Empty when ``T == 0`` or when
        every frame decodes to blank.

    Raises:
        ValueError: If ``ctc_logits`` is not 2-D.
    """
    if ctc_logits.dim() != 2:
        raise ValueError(
            f"ctc_logits must be 2-D (T, V) for a single utterance, "
            f"got shape {tuple(ctc_logits.shape)}"
        )
    if ctc_logits.size(0) == 0:
        return []

    yseq = ctc_logits.argmax(dim=-1)
    yseq = torch.unique_consecutive(yseq, dim=-1)
    mask = yseq != blank_id
    return yseq[mask].tolist()


def ctc_greedy_decode(
    ctc_logits: "torch.Tensor",
    tokenizer: Any,
    blank_id: int = 0,
) -> str:
    """Greedily decode CTC output for one utterance into text.

    Args:
        ctc_logits: A 2-D ``(T, V)`` tensor for a single utterance, as in
            :func:`ctc_greedy_token_ids`.
        tokenizer: Any object exposing ``decode(token_ids) -> str``; in this
            repo it is the tokenizer returned by ``SenseVoiceSmall.from_pretrained``.
        blank_id: Vocabulary index of the CTC blank symbol.

    Returns:
        The decoded text, still carrying its ``<|...|>`` rich tags.  Empty
        string when no non-blank token survives, without calling the tokenizer.
    """
    token_ids = ctc_greedy_token_ids(ctc_logits, blank_id=blank_id)
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids)


def strip_rich_tags(text: str) -> str:
    """Remove ``<|...|>`` rich tags from decoded text.

    SenseVoice prefixes its output with language, emotion, event and
    text-normalisation markers such as
    ``<|zh|><|NEUTRAL|><|Speech|><|withitn|>``.  Partial (in-progress) results
    are displayed without them, since the markers flicker as more audio
    arrives.

    Args:
        text: Decoded text, with or without tags.

    Returns:
        ``text`` with every tag removed.  Each tag is removed individually, so
        the text *between* two tags is preserved.
    """
    return _RICH_TAG_RE.sub("", text)
