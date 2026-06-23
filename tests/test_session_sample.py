"""Tests for Session.sample_reply using fake engines (no torch/model)."""

from unsaid.engine import Candidate, CompletionEngine
from unsaid.session import Session


class _FakeEngine(CompletionEngine):
    """Minimal fake: encode is 1-char→1-id, surprisal returns len(text)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids)

    def topk(self, token_ids: list[int], k: int) -> list[Candidate]:
        letters = "abcdefghij"
        return [
            Candidate(ord(letters[i % 10]), f" {letters[i % 10]}", 1.0 / (i + 1))
            for i in range(k)
        ]

    def surprisal(self, text: str) -> float:
        return float(len(text))


class _FakeReplyEngine(_FakeEngine):
    """Fake that actually produces a sample."""

    def sample_reply(
        self,
        text: str,
        *,
        max_tokens: int = 40,
        rng: object = None,
        on_token: object = None,
        is_cancelled: object = None,
    ) -> str:
        reply = " ok."
        if on_token is not None:
            on_token(reply, " ok.")
        return reply


def test_sample_reply_appends_and_updates_surprisal():
    s = Session(_FakeReplyEngine(), top_k=3)
    s.set_text("hello")
    result = s.sample_reply()
    assert result == "hello ok."
    assert s.text == "hello ok."
    assert s.surprisal == len("hello ok.")
    assert s.n_tokens == len("hello ok.")
    assert s.page == 0


def test_sample_reply_calls_on_token():
    tokens = []

    def on_token(acc: str, tok: str) -> None:
        tokens.append(tok)

    s = Session(_FakeReplyEngine(), top_k=3)
    s.set_text("hello")
    s.sample_reply(on_token=on_token)
    assert tokens == [" ok."]


def test_sample_reply_noop_on_bare_engine():
    s = Session(_FakeEngine(), top_k=3)
    s.set_text("hello")
    before = s.text
    result = s.sample_reply()
    assert result == before
    assert s.text == before
    # surprisal / n_tokens unchanged
    assert s.surprisal == len("hello")
    assert s.n_tokens == len("hello")


def test_sample_reply_on_empty_buffer():
    s = Session(_FakeReplyEngine(), top_k=3)
    result = s.sample_reply()
    assert result == " ok."
    assert s.text == " ok."
    assert s.surprisal == 4.0
    assert s.n_tokens == 4
