"""Rendering helpers for completion candidates.

These are pure functions (no torch, no model) so they're cheap to unit-test.

Display conventions
-------------------
Candidate text comes already decoded, so a leading space is a real space
character and a newline is a real ``\\n``. To keep fragments readable in a list
we translate whitespace into visible markers:

* a leading space -> ``·`` (middle dot)
* a newline       -> ``⏎``
* a tab           -> ``⇥``

Word prefix
-----------
A predicted continuation is often a subword fragment (``ie``, ``ing``). On its
own it is hard to read, so we prepend the *current word prefix* — the trailing
run of non-whitespace the user is mid-typing — to any candidate that continues
that word (i.e. does not start with whitespace). ``brown`` + ``ie`` ->
``brownie``. In the styled (TUI) output, only the next character that accepting
the candidate will type is highlighted.
"""

from __future__ import annotations

from .engine import Candidate, current_word_prefix

__all__ = [
    "current_word_prefix",
    "visible_token",
    "candidate_completion",
    "candidate_word",
    "prob_bar",
    "format_candidate",
    "format_candidates",
    "format_candidates_fragments",
    "format_surprisal",
    "StyledFragments",
]

_LEADING_SPACE = "·"
_NEWLINE = "⏎"
_TAB = "⇥"

_BAR_FULL = "█"
_BAR_PARTIALS = " ▏▎▍▌▋▊▉"  # 1/8th increments

_WORD_COL = 18  # min width of the word column, for bar alignment


def visible_token(text: str) -> str:
    """Make a decoded token's whitespace visible for display."""
    out = text.replace("\n", _NEWLINE).replace("\t", _TAB)
    if out.startswith(" "):
        out = _LEADING_SPACE + out[1:]
    return out


def _continues_word(text: str) -> bool:
    """True if ``text`` extends the current word rather than starting a new one.

    A new word starts only when the token begins with whitespace. An empty
    remainder (the typed word is itself a complete token) counts as continuing,
    so the prefix is still shown.
    """
    return not text[:1].isspace()


def candidate_completion(cand: Candidate) -> str:
    """Return the display continuation for ``cand``.

    Character candidates use ``continuation`` to show the most likely token they
    can continue to, while ``text`` remains the one-character accepted input.
    """
    return cand.continuation if cand.continuation is not None else cand.text


def candidate_word(cand: Candidate, prefix: str = "") -> str:
    """The full word a candidate forms, prepending ``prefix`` when it continues it."""
    completion = candidate_completion(cand)
    token = visible_token(completion)
    if prefix and _continues_word(completion):
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


def format_surprisal(bits: float, n_tokens: int) -> str:
    """One-line summary of the running string's surprisal."""
    if n_tokens <= 0:
        return "surprisal: 0.00 bits"
    return f"surprisal: {bits:.2f} bits  ({bits / n_tokens:.2f}/token, {n_tokens} tok)"


# Styles used by the fragment (TUI) renderer below.
StyledFragments = list[tuple[str, str]]


def format_candidates_fragments(
    cands: list[Candidate], prefix: str = "", *, bar_width: int = 20
) -> StyledFragments:
    """Format the top-k list as prompt_toolkit ``(style, text)`` fragments.

    The already-typed ``prefix`` is given the ``class:prefix`` style (dim) and
    the accepted next character uses ``class:token`` (highlighted). Any remaining
    best-continuation context is shown normally.
    """
    frags: StyledFragments = []
    for rank, cand in enumerate(cands, start=1):
        key = rank % 10
        completion = candidate_completion(cand)
        token = visible_token(cand.text)
        suffix = ""
        if cand.continuation is not None:
            suffix = visible_token(completion[len(cand.text) :])
        show_prefix = bool(prefix) and _continues_word(completion)
        word_len = (len(prefix) if show_prefix else 0) + len(token) + len(suffix)

        frags.append(("", f"{key}  "))
        if show_prefix:
            frags.append(("class:prefix", prefix))
        frags.append(("class:token", token))
        if suffix:
            frags.append(("", suffix))
        frags.append(("", " " * max(2, _WORD_COL - word_len + 2)))
        frags.append(("class:bar", prob_bar(cand.prob, bar_width)))
        frags.append(("", f"  {cand.prob:.4f}\n"))
    return frags
