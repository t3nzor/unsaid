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

Word prefix
-----------
A predicted token is often a subword fragment (``ie``, ``ing``). On its own it
is hard to read, so we prepend the *current word prefix* — the trailing run of
non-whitespace the user is mid-typing — to any candidate that continues that
word (i.e. does not start with whitespace). ``brown`` + ``ie`` -> ``brownie``.
Candidates that begin a new token (leading space/newline) are shown as-is. In
the styled (TUI) output the prefix is dimmed so the predicted token still
stands out.
"""

from __future__ import annotations

import re

from .engine import Candidate

_LEADING_SPACE = "·"
_NEWLINE = "⏎"
_TAB = "⇥"

_BAR_FULL = "█"
_BAR_PARTIALS = " ▏▎▍▌▋▊▉"  # 1/8th increments

_WORD_COL = 18  # min width of the word column, for bar alignment

_TRAILING_WORD = re.compile(r"\S+$")


def current_word_prefix(text: str) -> str:
    """Return the partial word the user is currently typing.

    This is the trailing run of non-whitespace characters. Empty if the buffer
    is empty or ends in whitespace (i.e. the next token starts a new word).
    """
    match = _TRAILING_WORD.search(text)
    return match.group(0) if match else ""


def visible_token(text: str) -> str:
    """Make a decoded token's whitespace visible for display."""
    out = text.replace("\n", _NEWLINE).replace("\t", _TAB)
    if out.startswith(" "):
        out = _LEADING_SPACE + out[1:]
    return out


def _continues_word(text: str) -> bool:
    """True if ``text`` extends the current word rather than starting a new one."""
    return bool(text) and not text[0].isspace()


def candidate_word(cand: Candidate, prefix: str = "") -> str:
    """The full word a candidate forms, prepending ``prefix`` when it continues it."""
    token = visible_token(cand.text)
    if prefix and _continues_word(cand.text):
        return prefix + token
    return token


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


def format_candidate(rank: int, cand: Candidate, prefix: str = "", *, bar_width: int = 20) -> str:
    """Format one candidate row: ``rank  word  bar  prob`` (plain text)."""
    key = rank % 10  # 1..9 then 0 for the tenth
    word = candidate_word(cand, prefix)
    bar = prob_bar(cand.prob, bar_width)
    return f"{key}  {word.ljust(_WORD_COL)}  {bar}  {cand.prob:.4f}"


def format_candidates(cands: list[Candidate], prefix: str = "", *, bar_width: int = 20) -> str:
    """Format a full top-k list as plain text, one candidate per line."""
    return "\n".join(
        format_candidate(i, c, prefix, bar_width=bar_width) for i, c in enumerate(cands, start=1)
    )


# Styles used by the fragment (TUI) renderer below.
StyledFragments = list[tuple[str, str]]


def format_candidates_fragments(
    cands: list[Candidate], prefix: str = "", *, bar_width: int = 20
) -> StyledFragments:
    """Format the top-k list as prompt_toolkit ``(style, text)`` fragments.

    The already-typed ``prefix`` is given the ``class:prefix`` style (dim) and
    the predicted token ``class:token`` (highlighted) so the model's actual
    next token stands out from the word context.
    """
    frags: StyledFragments = []
    for rank, cand in enumerate(cands, start=1):
        key = rank % 10
        token = visible_token(cand.text)
        show_prefix = bool(prefix) and _continues_word(cand.text)
        word_len = (len(prefix) if show_prefix else 0) + len(token)

        frags.append(("", f"{key}  "))
        if show_prefix:
            frags.append(("class:prefix", prefix))
        frags.append(("class:token", token))
        frags.append(("", " " * max(2, _WORD_COL - word_len + 2)))
        frags.append(("class:bar", prob_bar(cand.prob, bar_width)))
        frags.append(("", f"  {cand.prob:.4f}\n"))
    return frags
