"""Rendering helpers for next-token candidates.

These are pure functions (no torch, no model) so they're cheap to unit-test.

Display conventions
-------------------
Candidate text comes from the tokenizer already decoded, so a leading space is
a real space character and a newline is a real ``\\n``. To keep single-token
fragments readable in a list we translate whitespace into visible markers:

* a leading space -> ``·`` (middle dot)
* a newline       -> ``⏎``
* a tab           -> ``⇥``
"""

from __future__ import annotations

from .engine import Candidate

_LEADING_SPACE = "·"
_NEWLINE = "⏎"
_TAB = "⇥"

_BAR_FULL = "█"
_BAR_PARTIALS = " ▏▎▍▌▋▊▉"  # 1/8th increments


def visible_token(text: str) -> str:
    """Make a decoded token's whitespace visible for display."""
    out = text.replace("\n", _NEWLINE).replace("\t", _TAB)
    if out.startswith(" "):
        out = _LEADING_SPACE + out[1:]
    return out


def prob_bar(prob: float, width: int = 20) -> str:
    """Render ``prob`` (0..1) as a fixed-``width`` unicode bar."""
    prob = max(0.0, min(1.0, prob))
    units = prob * width * 8  # eighths of a cell
    full = int(units // 8)
    rem = int(units % 8)
    bar = _BAR_FULL * full
    if rem:
        bar += _BAR_PARTIALS[rem]
    return bar.ljust(width)


def format_candidate(rank: int, cand: Candidate, *, bar_width: int = 20) -> str:
    """Format one candidate row: ``rank  token  bar  prob``."""
    key = rank % 10  # 1..9 then 0 for the tenth
    token = visible_token(cand.text)
    bar = prob_bar(cand.prob, bar_width)
    return f"{key}  {token:<14.14}  {bar}  {cand.prob:.4f}"


def format_candidates(cands: list[Candidate], *, bar_width: int = 20) -> str:
    """Format a full top-k list, one candidate per line."""
    return "\n".join(
        format_candidate(i, c, bar_width=bar_width) for i, c in enumerate(cands, start=1)
    )
