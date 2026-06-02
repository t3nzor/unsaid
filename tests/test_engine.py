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
