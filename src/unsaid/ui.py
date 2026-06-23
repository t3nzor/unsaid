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
from prompt_toolkit.document import Document
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
        self._sampling: bool = False

        initial = self.session.text
        self.input = Buffer(
            multiline=False,
            on_text_changed=self._on_text_changed,
            document=Document(initial, len(initial)) if initial else None,
        )

        kb = self._make_keybindings()
        body = HSplit(
            [
                Window(
                    BufferControl(buffer=self.input),
                    height=Dimension(min=1, max=3),
                    wrap_lines=True,
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
        if self._sampling:
            return [("class:surprisal", "sampling…\n")]
        surprisal = format_surprisal(self.session.surprisal, self.session.n_tokens)
        cands = self.session.candidates
        if not cands:
            return [("class:surprisal", surprisal + "\n"), ("", "(type to begin)")]
        prefix = current_word_prefix(self.session.text)
        start = self.session.page * self.session.top_k + 1
        end = start + len(cands) - 1
        header = f"ranks {start}-{end} of {len(self.session.pool)}  \u00b7  PgUp/PgDn\n"
        return [
            ("class:surprisal", surprisal + "\n"),
            ("class:header", header),
            *format_candidates_fragments(cands, prefix),
        ]

    def _on_text_changed(self, _buf: Buffer) -> None:
        if self._sampling:
            return
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

    def _move_visual_line(self, event, *, down: bool) -> None:
        """Move the cursor one visual (wrapped) line up or down.

        prompt_toolkit's ``auto_up``/``auto_down`` operate on *logical*
        lines (real ``\\n``).  Here we model the character-wrap layout of the
        input window so that Up/Down arrows traverse the visual rows produced
        by ``wrap_lines=True``.
        """
        from prompt_toolkit.application import get_app

        try:
            w = get_app().output.get_size().columns
        except Exception:
            w = 80

        c0 = max(1, w)
        buf = self.input
        text = buf.text
        n = len(text)
        i = buf.cursor_position

        # Current visual (row, screen_col).
        if i < c0:
            row, screen_col = 0, i
        else:
            rem = i - c0
            row, screen_col = 1 + rem // w, rem % w

        # Last visual row (cursor at end of text).
        if n < c0:
            last_row = 0
        else:
            last_row = 1 + (n - c0) // w

        pc = buf.preferred_column if buf.preferred_column is not None else screen_col

        if not down:
            target_row = row - 1
        else:
            target_row = row + 1

        target_row = max(0, min(target_row, last_row))

        if target_row <= 0:
            user_col = max(0, pc)
            target_i = min(user_col, c0 - 1, n)
        else:
            start = c0 + (target_row - 1) * w
            target_i = min(start + pc, start + w - 1, n)

        buf.cursor_position = target_i
        buf.preferred_column = pc

    def _start_sampling(self) -> None:
        """Called by the Enter keybinding; spawns a streaming sampler."""
        if self._sampling:
            return
        self._sampling = True
        self.app.invalidate()

        t = threading.Thread(target=self._sample_worker, daemon=True)
        t.start()

    def _sample_worker(self) -> None:
        """Background thread: sample a reply with per-token streaming."""
        loop = self.app.loop
        base_text = self.session.text

        def on_token(accumulated: str, _token_text: str) -> None:
            full = base_text + accumulated

            def update() -> None:
                self._apply_buffer_text(full)

            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(update)

        self.session.sample_reply(on_token=on_token)

        self._sampling = False

        def finish() -> None:
            self.app.invalidate()

        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(finish)

    def _apply_buffer_text(self, text: str) -> None:
        """Set the input buffer text (cursor at end) — runs on UI thread."""
        self.input.set_document(
            self.input.document.__class__(text, len(text)),
            bypass_readonly=True,
        )
        self.app.invalidate()

    def _make_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-q")
        def _quit(event) -> None:
            event.app.exit()

        @kb.add("tab")
        def _accept_top(event) -> None:
            self._accept_and_sync(0)

        @kb.add("enter")
        def _sample(event) -> None:
            self._start_sampling()

        @kb.add("pagedown")
        def _next_page(event) -> None:
            self.session.next_page()

        @kb.add("pageup")
        def _prev_page(event) -> None:
            self.session.prev_page()

        @kb.add("up")
        def _up(event) -> None:
            self._move_visual_line(event, down=False)

        @kb.add("down")
        def _down(event) -> None:
            self._move_visual_line(event, down=True)

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
