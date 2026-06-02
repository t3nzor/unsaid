"""Session state: the typed text and its derived completion candidates.

Keeps UI concerns out of the engine. The session owns the current text buffer,
recomputes the distribution, and implements "accept candidate" (Tab / number
keys) by appending a chosen token's text to the buffer.

Paging
------
The session fetches a *pool* of up to ``top_k * max_pages`` candidates once per
text change and exposes one page of ``top_k`` at a time via ``candidates``.
``next_page``/``prev_page`` re-slice the cached pool without re-running the
model. Editing the text (or accepting a candidate) resets to the first page.
"""

from __future__ import annotations

from .engine import Candidate, CompletionEngine


class Session:
    def __init__(self, engine: CompletionEngine, *, top_k: int = 10, max_pages: int = 10) -> None:
        self.engine = engine
        self.top_k = top_k
        self.max_pages = max_pages
        self.text: str = ""
        self.pool: list[Candidate] = []
        self.page: int = 0
        self.candidates: list[Candidate] = []

    def set_text(self, text: str) -> list[Candidate]:
        """Replace the buffer text, reset to the first page, and recompute."""
        self.text = text
        self.page = 0
        return self.recompute()

    def recompute(self) -> list[Candidate]:
        self.pool = self.engine.complete(self.text, self.top_k * self.max_pages)
        self._update_page()
        return self.candidates

    @property
    def last_page(self) -> int:
        if not self.pool:
            return 0
        return (len(self.pool) - 1) // self.top_k

    def _update_page(self) -> list[Candidate]:
        self.page = max(0, min(self.page, self.last_page))
        start = self.page * self.top_k
        self.candidates = self.pool[start : start + self.top_k]
        return self.candidates

    def next_page(self) -> list[Candidate]:
        """Show the next ``top_k`` candidates (lower-ranked), if any."""
        if self.page < self.last_page:
            self.page += 1
            self._update_page()
        return self.candidates

    def prev_page(self) -> list[Candidate]:
        """Show the previous ``top_k`` candidates (higher-ranked), if any."""
        if self.page > 0:
            self.page -= 1
            self._update_page()
        return self.candidates

    def accept(self, index: int) -> str:
        """Append the candidate at ``index`` (0-based, within the current page).

        Returns the new buffer text and resets to the first page. Out-of-range
        indices are ignored.
        """
        if not (0 <= index < len(self.candidates)):
            return self.text
        self.set_text(self.text + self.candidates[index].text)
        return self.text

    def accept_top(self) -> str:
        """Append the most likely candidate on the current page (used by Tab)."""
        return self.accept(0)
