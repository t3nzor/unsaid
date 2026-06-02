"""Tests for Session logic using a fake engine (no torch/model)."""

from unsaid.engine import Candidate, CompletionEngine
from unsaid.session import Session


class FakeEngine(CompletionEngine):
    """Deterministic engine: candidates depend only on the encoded length."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids)

    def topk(self, token_ids: list[int], k: int) -> list[Candidate]:
        # Always offer " a", " b", ... so accept() appends predictably.
        letters = "abcdefghij"
        return [Candidate(ord(letters[i]), f" {letters[i]}", 1.0 / (i + 1)) for i in range(k)]


def test_set_text_recomputes():
    s = Session(FakeEngine(), top_k=5)
    cands = s.set_text("hi")
    assert len(cands) == 5
    assert s.text == "hi"


def test_accept_appends_candidate_text():
    s = Session(FakeEngine(), top_k=5)
    s.set_text("hi")
    new = s.accept(0)
    assert new == "hi a"
    assert s.text == "hi a"


def test_accept_top_is_index_zero():
    s = Session(FakeEngine(), top_k=5)
    s.set_text("x")
    assert s.accept_top() == "x a"


def test_accept_out_of_range_is_noop():
    s = Session(FakeEngine(), top_k=3)
    s.set_text("x")
    assert s.accept(99) == "x"
    assert s.text == "x"


def test_empty_buffer_allowed():
    s = Session(FakeEngine(), top_k=3)
    cands = s.set_text("")
    assert len(cands) == 3
