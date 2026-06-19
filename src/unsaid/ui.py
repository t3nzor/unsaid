"""Live full-screen TUI for exploring next-token distributions.

Layout: an editable input buffer on top, a live top-k panel below. The
distribution recomputes (debounced) as you type. Tab accepts the top
candidate; Alt+1..9/Alt+0 accept the Nth listed candidate. PageDown/PageUp
scroll through the next/previous page of lower-ranked completions.
"""

from __future__ import annotations

import threading

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style

from .format import (
    StyledFragments,
    current_word_prefix,
    format_candidates_fragments,
    format_surprisal,
)
from .session import Session

_STYLE = Style.from_dict(
    {
        "prefix": "#888888",  # already-typed word context: dimmed
        "token": "bold #00afff",  # the accepted next character: highlighted
        "bar": "#5f8700",
        "header": "#888888",
        "surprisal": "#d78700",
    }
)


class UnsaidApp:
    def __init__(self, session: Session, *, debounce_s: float = 0.08) -> None:
        self.session = session
        self.debounce_s = debounce_s
        self._timer: threading.Timer | None = None

        self.input = Buffer(multiline=False, on_text_changed=self._on_text_changed)

        kb = self._make_keybindings()
        body = HSplit(
            [
                Window(
                    BufferControl(buffer=self.input),
                    height=Dimension(min=1, max=3),
                ),
                Window(height=1, char="─"),
                Window(FormattedTextControl(self._get_panel_fragments)),
            ]
        )
        self.app: Application = Application(
            layout=Layout(body, focused_element=self.input),
            key_bindings=kb,
            style=_STYLE,
            full_screen=True,
        )

    def _get_panel_fragments(self) -> StyledFragments:
        frags: StyledFragments = []
        preamble = self.session.engine.initial_prompt
        if preamble:
            try:
                from prompt_toolkit.application import get_app
                cols = get_app().output.get_size().columns
            except Exception:
                cols = 120
            label = "preamble: "
            budget = max(20, cols - len(label))
            display = preamble if len(preamble) <= budget else preamble[: budget - 1] + "\u2026"
            frags.append(("class:header", label + display + "\n"))

        surprisal = format_surprisal(self.session.surprisal, self.session.n_tokens)
        cands = self.session.candidates
        if not cands:
            frags.append(("class:surprisal", surprisal + "\n"))
            frags.append(("", "(type to begin)"))
            return frags
        prefix = current_word_prefix(self.session.text)
        start = self.session.page * self.session.top_k + 1
        end = start + len(cands) - 1
        header = f"ranks {start}-{end} of {len(self.session.pool)}  \u00b7  PgUp/PgDn\n"
        frags.append(("class:surprisal", surprisal + "\n"))
        frags.append(("class:header", header))
        frags.extend(format_candidates_fragments(cands, prefix))
        return frags

    def _on_text_changed(self, _buf: Buffer) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_s, self._recompute_async)
        self._timer.daemon = True
        self._timer.start()

    def _recompute_async(self) -> None:
        self.session.set_text(self.input.text)
        # Repaint from this background thread safely.
        self.app.invalidate()

    def _accept_and_sync(self, index: int) -> None:
        new_text = self.session.accept(index)
        # Reflect accepted token in the input buffer (cursor at end).
        self.input.set_document(
            self.input.document.__class__(new_text, len(new_text)),
            bypass_readonly=True,
        )

    def _make_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-q")
        def _quit(event) -> None:
            event.app.exit()

        @kb.add("tab")
        def _accept_top(event) -> None:
            self._accept_and_sync(0)

        @kb.add("pagedown")
        def _next_page(event) -> None:
            self.session.next_page()

        @kb.add("pageup")
        def _prev_page(event) -> None:
            self.session.prev_page()

        # Alt+digit accepts the Nth candidate without stealing plain digit
        # typing from the input buffer. 1..9 -> index 0..8, 0 -> index 9.
        for digit in range(10):
            index = 9 if digit == 0 else digit - 1

            @kb.add("escape", str(digit))
            def _accept_n(event, index: int = index) -> None:
                self._accept_and_sync(index)

        return kb

    def run(self) -> None:
        self.app.run()
