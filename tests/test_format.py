"""Tests for the pure rendering helpers (no torch/model)."""

from unsaid.engine import Candidate
from unsaid.format import format_candidate, prob_bar, visible_token


def test_visible_token_leading_space():
    assert visible_token(" and") == "·and"


def test_visible_token_newline_and_tab():
    assert visible_token("\n") == "⏎"
    assert visible_token("\t") == "⇥"


def test_visible_token_plain():
    assert visible_token("ies") == "ies"


def test_prob_bar_bounds():
    assert prob_bar(0.0, width=10).strip() == ""
    assert prob_bar(1.0, width=10) == "█" * 10
    assert len(prob_bar(0.37, width=20)) == 20


def test_prob_bar_clamps_out_of_range():
    assert prob_bar(5.0, width=4) == "█" * 4
    assert prob_bar(-1.0, width=4).strip() == ""


def test_format_candidate_uses_zero_for_tenth():
    row = format_candidate(10, Candidate(1, " x", 0.5))
    assert row.startswith("0  ")


def test_format_candidate_includes_numeric_prob():
    row = format_candidate(1, Candidate(1, "ie", 0.1921))
    assert row.endswith("0.1921")
    assert row.startswith("1  ie")
