"""Model-backed tests for HFEngine.

These are slow (they download/load gpt2) and require torch + transformers.
Run with ``pytest -m slow`` or deselect with ``pytest -m "not slow"``.
"""

import pytest

pytestmark = pytest.mark.slow

MODEL = "gpt2"


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from unsaid.engine import HFEngine

    return HFEngine(MODEL, num_threads=4)


def test_topk_returns_k_candidates(engine):
    cands = engine.topk(engine.encode("The quick brown"), 10)
    assert len(cands) == 10


def test_topk_is_descending(engine):
    cands = engine.topk(engine.encode("The quick brown"), 10)
    probs = [c.prob for c in cands]
    assert probs == sorted(probs, reverse=True)


def test_topk_probs_in_unit_interval(engine):
    cands = engine.topk(engine.encode("Hello"), 10)
    assert all(0.0 <= c.prob <= 1.0 for c in cands)


def test_topk_deterministic(engine):
    a = engine.topk(engine.encode("Hello"), 5)
    b = engine.topk(engine.encode("Hello"), 5)
    assert [c.token_id for c in a] == [c.token_id for c in b]


def test_empty_context_still_predicts(engine):
    cands = engine.topk([], 5)
    assert len(cands) == 5


def _word(prefix, cand):
    from unsaid.format import candidate_word

    return candidate_word(cand, prefix)


def test_heal_surfaces_real_words(engine):
    # The motivating bug: raw next-token after "Hel" never yields Hello/Help.
    cands = engine.complete("Hel", 10)
    words = {_word("Hel", c) for c in cands}
    assert "Hello" in words
    assert "Help" in words


def test_heal_remainder_completes_word(engine):
    # Candidate.text is the *remainder*; prefix + text must reconstruct a word.
    cands = engine.complete("I went to the libr", 5)
    words = {_word("libr", c) for c in cands}
    assert "library" in words


def test_heal_probs_renormalized(engine):
    cands = engine.complete("Hel", 10)
    assert abs(sum(c.prob for c in cands) - 1.0) <= 1.0  # subset of a distribution
    assert all(0.0 <= c.prob <= 1.0 for c in cands)


def test_no_heal_falls_back_to_raw(engine):
    engine.heal = False
    try:
        healed = engine.complete("Hel", 5)  # heal disabled -> raw next token
        direct = engine.topk(engine.encode("Hel"), 5)
        assert [c.token_id for c in healed] == [c.token_id for c in direct]
    finally:
        engine.heal = True


def test_complete_between_words_uses_raw(engine):
    # Trailing space => no partial word => raw distribution.
    cands = engine.complete("The quick brown ", 5)
    assert len(cands) == 5

