"""Tests for the pure rendering helpers (no torch/model)."""

from unsaid.engine import Candidate
from unsaid.format import (
    candidate_word,
    current_word_prefix,
    format_candidate,
    format_candidates_fragments,
    prob_bar,
    visible_token,
)


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


def test_current_word_prefix_partial_word():
    assert current_word_prefix("The quick br") == "br"


def test_current_word_prefix_empty_after_space():
    assert current_word_prefix("The quick ") == ""
    assert current_word_prefix("") == ""


def test_candidate_word_prepends_prefix_when_continuing():
    assert candidate_word(Candidate(1, "ie", 0.1), "brown") == "brownie"


def test_candidate_word_skips_prefix_for_new_word():
    # leading-space token starts a new word -> shown with marker, no prefix
    assert candidate_word(Candidate(1, " and", 0.1), "brown") == "·and"


def test_candidate_word_no_prefix_passthrough():
    assert candidate_word(Candidate(1, "ie", 0.1)) == "ie"


def test_candidate_word_empty_remainder_shows_prefix():
    # Healed candidate where the typed word is itself a complete token.
    assert candidate_word(Candidate(1, "", 0.1), "Hel") == "Hel"


def test_format_candidate_shows_full_word():
    row = format_candidate(1, Candidate(1, "ie", 0.1921), "brown")
    assert row.startswith("1  brownie")


def test_fragments_dim_prefix_highlight_token():
    frags = format_candidates_fragments([Candidate(1, "ie", 0.5)], "brown")
    styles = {style: text for style, text in frags if style}
    assert styles.get("class:prefix") == "brown"
    assert styles.get("class:token") == "ie"

