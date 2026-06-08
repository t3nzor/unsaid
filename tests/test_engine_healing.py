"""Pure tests for HFEngine token-healing edge cases."""

import pytest

from unsaid.engine import HFEngine


class FakeProbs:
    def __init__(self, probs: list[float]) -> None:
        self.probs = probs

    def cpu(self):
        return self

    def tolist(self) -> list[float]:
        return self.probs


def test_healing_ignores_logits_beyond_tokenizer_vocab():
    engine = HFEngine.__new__(HFEngine)
    engine.heal = True
    engine._vocab_strings = ["A", "Hello", "Help"]
    engine._next_token_probs = lambda _token_ids: FakeProbs([0.1, 0.6, 0.3, 0.9, 0.8])

    cands = engine.complete("Hel", 10)

    assert [(c.token_id, c.text) for c in cands] == [(1, "lo"), (2, "p")]
    assert [c.prob for c in cands] == pytest.approx([0.6 / 0.9, 0.3 / 0.9])
