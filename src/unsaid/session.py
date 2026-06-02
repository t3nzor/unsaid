"""Session state: the typed text and its derived next-token candidates.

Keeps UI concerns out of the engine. The session owns the current text buffer,
recomputes the distribution, and implements "accept candidate" (Tab / number
keys) by appending a chosen token's text to the buffer.
"""

from __future__ import annotations

from .engine import Candidate, CompletionEngine


class Session:
    def __init__(self, engine: CompletionEngine, *, top_k: int = 10) -> None:
        self.engine = engine
        self.top_k = top_k
        self.text: str = ""
        self.candidates: list[Candidate] = []

    def set_text(self, text: str) -> list[Candidate]:
        """Replace the buffer text and recompute candidates."""
        self.text = text
        return self.recompute()

    def recompute(self) -> list[Candidate]:
        self.candidates = self.engine.complete(self.text, self.top_k)
        return self.candidates

    def accept(self, index: int) -> str:
        """Append the candidate at ``index`` (0-based) and recompute.

        Returns the new buffer text. Out-of-range indices are ignored.
        """
        if not (0 <= index < len(self.candidates)):
            return self.text
        self.text += self.candidates[index].text
        self.recompute()
        return self.text

    def accept_top(self) -> str:
        """Append the most likely candidate (used by Tab)."""
        return self.accept(0)
