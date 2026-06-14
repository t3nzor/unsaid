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
    # Candidate.continuation is display context; prefix + continuation must
    # reconstruct a word, while Candidate.text is only the next character.
    cands = engine.complete("I went to the libr", 5)
    words = {_word("libr", c) for c in cands}
    assert "library" in words
    assert all(len(c.text) == 1 for c in cands)


def test_heal_probs_renormalized(engine):
    cands = engine.complete("Hel", 10)
    assert sum(c.prob for c in cands) == pytest.approx(1.0)
    assert all(0.0 <= c.prob <= 1.0 for c in cands)


def test_no_heal_uses_raw_next_token_context(engine):
    engine.heal = False
    try:
        cands = engine.complete("Hel", 5)  # heal disabled -> raw next characters
        assert len(cands) == 5
        assert all(len(c.text) == 1 for c in cands)
        assert all(c.continuation is not None for c in cands)
    finally:
        engine.heal = True


def test_trailing_space_heals_to_new_words(engine):
    # A trailing space is a word boundary: completions should be whole new
    # words (leading-space tokens), not garbage from the dangling space token.
    cands = engine.complete("The quick brown fox ", 10)
    assert len(cands) == 10
    words = [_word("", c) for c in cands]
    # Healed remainders have the leading space stripped and are real words.
    assert any(w.strip().isalpha() for w in words)
    # No empty/blank accepted characters (the bare-space token is filtered out).
    assert all(c.text for c in cands)


def test_surprisal_empty_is_zero(engine):
    assert engine.surprisal("") == 0.0


def test_surprisal_nonnegative_and_monotonic(engine):
    # Each additional token adds non-negative surprisal, so a longer prefix of
    # the same string has total surprisal >= the shorter one.
    short = engine.surprisal("The cat")
    long = engine.surprisal("The cat sat on the mat")
    assert short >= 0.0
    assert long >= short


def test_surprisal_higher_for_unpredictable_text(engine):
    def per_token(t):
        return engine.surprisal(t) / max(1, len(engine.encode(t)))

    predictable = per_token("The United States of America")
    gibberish = per_token("The qwx zzfp blorgon")
    assert gibberish > predictable
