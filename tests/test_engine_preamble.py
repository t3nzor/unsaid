"""Tests for HFEngine preamble (initial_prompt) behavior."""

from unsaid.engine import HFEngine


class FakeProbs:
    def __init__(self, probs: list[float]) -> None:
        self.probs = probs

    def cpu(self):
        return self

    def tolist(self) -> list[float]:
        return self.probs


def _make_engine(*, heal: bool = True, preamble: str = "") -> HFEngine:
    """Build a minimal HFEngine via __new__ (no model / tokenizer)."""
    engine = HFEngine.__new__(HFEngine)
    engine.heal = heal
    engine.initial_prompt = preamble
    engine._preamble_ids = [ord(c) for c in preamble]
    engine._vocab_strings = None
    return engine


# --- _context_ids ---


def test_context_ids_without_preamble():
    engine = _make_engine()
    engine.encode = lambda text: [ord(c) for c in text]

    assert engine._context_ids("ab") == [ord("a"), ord("b")]


def test_context_ids_with_preamble():
    engine = _make_engine(preamble="XY")
    engine.encode = lambda text: [ord(c) for c in text]

    assert engine._context_ids("ab") == [ord("X"), ord("Y"), ord("a"), ord("b")]


# --- complete (no-heal raw path) ---


def test_complete_raw_path_uses_context_ids():
    captured_ids: list[list[int]] = []
    engine = _make_engine(heal=False, preamble="Q")
    engine.encode = lambda text: [ord(c) for c in text]
    engine._vocab_strings = ["A", "B"]

    def fake_probs(token_ids):
        captured_ids.append(token_ids)
        return FakeProbs([0.4, 0.6])

    engine._next_token_probs = fake_probs

    engine.complete("x", 2)

    assert captured_ids == [[ord("Q"), ord("x")]]


# --- complete (healing path) ---


def test_complete_healing_path_uses_context_ids():
    captured_ids: list[list[int]] = []
    engine = _make_engine(heal=True, preamble="Pre ")
    engine.encode = lambda text: [ord(c) for c in text]
    engine._vocab_strings = ["Hello", "Help", "Hey"]

    def fake_probs(token_ids):
        captured_ids.append(token_ids)
        return FakeProbs([0.5, 0.3, 0.2])

    engine._next_token_probs = fake_probs

    engine.complete("He", 10)

    # Healing: stripped="" (no space before "He"), so context is preamble only
    assert [ord(c) for c in "Pre "] in captured_ids


def test_complete_healing_empty_text_with_preamble():
    captured_ids: list[list[int]] = []
    engine = _make_engine(heal=True, preamble="CTX")
    engine.encode = lambda text: [ord(c) for c in text]
    engine._vocab_strings = ["A"]

    def fake_probs(token_ids):
        captured_ids.append(token_ids)
        return FakeProbs([1.0])

    engine._next_token_probs = fake_probs

    cands = engine.complete("", 1)

    # Empty text falls to _raw_letter_candidates with context_ids("")
    # which is preamble_ids + encode("") = preamble_ids
    assert [ord(c) for c in "CTX"] in captured_ids
    assert len(cands) == 1


# --- complete (healing path, empty preamble — regression) ---


def test_complete_healing_without_preamble_matches_original():
    engine = _make_engine(heal=True, preamble="")
    engine.encode = lambda text: [ord(c) for c in text]
    engine._vocab_strings = ["Hello", "Help"]

    def fake_probs(token_ids):
        return FakeProbs([0.6, 0.4])

    engine._next_token_probs = fake_probs

    cands = engine.complete("Hel", 10)

    assert [(c.text, c.continuation) for c in cands] == [("l", "lo"), ("p", "p")]


# --- CompletionEngine ABC default ---


def test_completion_engine_default_initial_prompt():
    from unsaid.engine import CompletionEngine

    assert CompletionEngine.initial_prompt == ""
