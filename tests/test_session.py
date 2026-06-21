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
        # Always offer " a", " b", ... cycling, so accept() is predictable and
        # pools larger than the alphabet still return k candidates.
        letters = "abcdefghij"
        return [
            Candidate(ord(letters[i % 10]), f" {letters[i % 10]}", 1.0 / (i + 1)) for i in range(k)
        ]

    def surprisal(self, text: str) -> float:
        return float(len(text))  # deterministic stand-in for wiring tests


class FakeLetterEngine(FakeEngine):
    def complete(self, text: str, k: int) -> list[Candidate]:
        return [Candidate(1, "l", 0.8, continuation="lo")]


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


def test_accept_letter_candidate_appends_only_next_character():
    s = Session(FakeLetterEngine(), top_k=5)
    s.set_text("Hel")

    assert s.accept_top() == "Hell"


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


def test_next_page_advances_slice():
    s = Session(FakeEngine(), top_k=3, max_pages=4)
    page0 = list(s.set_text("x"))
    page1 = s.next_page()
    assert s.page == 1
    assert page1 != page0
    assert len(page1) == 3


def test_prev_page_goes_back():
    s = Session(FakeEngine(), top_k=3, max_pages=4)
    s.set_text("x")
    s.next_page()
    back = s.prev_page()
    assert s.page == 0
    assert back == s.pool[0:3]


def test_prev_page_at_top_is_noop():
    s = Session(FakeEngine(), top_k=3, max_pages=4)
    s.set_text("x")
    s.prev_page()
    assert s.page == 0


def test_next_page_clamps_at_last():
    s = Session(FakeEngine(), top_k=3, max_pages=2)  # pool of 6 -> pages 0,1
    s.set_text("x")
    s.next_page()
    s.next_page()
    s.next_page()
    assert s.page == s.last_page == 1


def test_editing_resets_to_first_page():
    s = Session(FakeEngine(), top_k=3, max_pages=4)
    s.set_text("x")
    s.next_page()
    s.set_text("xy")
    assert s.page == 0


def test_accept_resets_to_first_page():
    s = Session(FakeEngine(), top_k=3, max_pages=4)
    s.set_text("x")
    s.next_page()
    s.accept(0)
    assert s.page == 0


def test_session_stores_surprisal_and_token_count():
    s = Session(FakeEngine(), top_k=3)
    s.set_text("hello")
    assert s.surprisal == 5.0  # FakeEngine returns len(text)
    assert s.n_tokens == 5  # FakeEngine encodes one id per char


def test_preamble_seeds_buffer_and_recomputes():
    s = Session(FakeEngine(), top_k=3, preamble="hi")
    assert s.text == "hi"
    assert s.surprisal == 2.0  # len("hi")
    assert s.n_tokens == 2
    assert len(s.candidates) == 3


def test_preamble_included_in_surprisal_after_set_text():
    s = Session(FakeEngine(), top_k=3, preamble="hi")
    s.set_text("hi world")
    assert s.surprisal == 8.0  # len("hi world")
    assert s.n_tokens == 8


def test_empty_preamble_no_initial_recompute():
    s = Session(FakeEngine(), top_k=3, preamble="")
    assert s.text == ""
    assert s.pool == []  # no recompute happened
    assert s.candidates == []
