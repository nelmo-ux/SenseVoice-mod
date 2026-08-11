"""Unit tests for streaming/ctc_decode.py.

The module needs torch, which the other tests in this directory deliberately do
not require; ``importorskip`` keeps the suite green on a torch-less checkout.
"""

import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from streaming.ctc_decode import (  # noqa: E402
    ctc_greedy_decode,
    ctc_greedy_token_ids,
    strip_rich_tags,
)

VOCAB_SIZE = 6


def one_hot_logits(frame_ids: List[int], vocab_size: int = VOCAB_SIZE):
    """Build a ``(T, V)`` log-prob tensor whose argmax is exactly ``frame_ids``."""
    logits = torch.full((len(frame_ids), vocab_size), -10.0)
    for step, token_id in enumerate(frame_ids):
        logits[step, token_id] = 0.0
    return logits


class FakeTokenizer:
    """Minimal stand-in for the FunASR tokenizer used by ``model.py``."""

    def decode(self, token_ids: List[int]) -> str:
        return "".join(f"t{token_id}" for token_id in token_ids)


def test_collapses_consecutive_duplicates():
    logits = one_hot_logits([3, 3, 3, 4, 4, 5])
    assert ctc_greedy_token_ids(logits) == [3, 4, 5]


def test_drops_blank_frames():
    logits = one_hot_logits([0, 3, 0, 0, 4, 0])
    assert ctc_greedy_token_ids(logits) == [3, 4]


def test_blank_separates_repeated_tokens():
    # Standard CTC semantics: a blank between two identical tokens keeps both.
    logits = one_hot_logits([2, 2, 0, 2, 2])
    assert ctc_greedy_token_ids(logits) == [2, 2]


def test_empty_input_returns_empty_list():
    logits = torch.zeros((0, VOCAB_SIZE))
    assert ctc_greedy_token_ids(logits) == []


def test_all_blank_input_returns_empty_list():
    logits = one_hot_logits([0, 0, 0, 0])
    assert ctc_greedy_token_ids(logits) == []


def test_custom_blank_id_is_honoured():
    logits = one_hot_logits([0, 5, 5, 1])
    assert ctc_greedy_token_ids(logits, blank_id=5) == [0, 1]


def test_non_2d_input_raises_value_error():
    logits = one_hot_logits([1, 2]).unsqueeze(0)  # (1, T, V)
    with pytest.raises(ValueError):
        ctc_greedy_token_ids(logits)


def test_matches_model_py_reference_implementation():
    """The result must equal the argmax/unique_consecutive/mask path in model.py."""
    torch.manual_seed(0)
    logits = torch.randn(64, VOCAB_SIZE).log_softmax(dim=-1)

    yseq = logits.argmax(dim=-1)
    yseq = torch.unique_consecutive(yseq, dim=-1)
    expected = yseq[yseq != 0].tolist()

    assert ctc_greedy_token_ids(logits) == expected


def test_greedy_decode_uses_tokenizer():
    logits = one_hot_logits([3, 3, 0, 4])
    assert ctc_greedy_decode(logits, FakeTokenizer()) == "t3t4"


def test_greedy_decode_empty_returns_empty_string():
    logits = torch.zeros((0, VOCAB_SIZE))
    assert ctc_greedy_decode(logits, FakeTokenizer()) == ""


def test_greedy_decode_skips_tokenizer_when_all_blank():
    class ExplodingTokenizer:
        def decode(self, token_ids):
            raise AssertionError("tokenizer must not be called for empty output")

    logits = one_hot_logits([0, 0, 0])
    assert ctc_greedy_decode(logits, ExplodingTokenizer()) == ""


def test_strip_rich_tags_removes_leading_marker_block():
    text = "<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好世界"
    assert strip_rich_tags(text) == "你好世界"


def test_strip_rich_tags_does_not_swallow_text_between_tags():
    """A greedy ``<\\|.*\\|>`` would delete everything up to the last tag."""
    text = "<|zh|>hello<|en|>world<|withitn|>!"
    assert strip_rich_tags(text) == "helloworld!"


def test_strip_rich_tags_removes_each_tag_individually():
    text = "a<|x|>b<|y|>c"
    assert strip_rich_tags(text) == "abc"


def test_strip_rich_tags_is_a_noop_without_tags():
    assert strip_rich_tags("plain text") == "plain text"
    assert strip_rich_tags("") == ""
