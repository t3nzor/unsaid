# AGENTS.md

`unsaid` — "Explore the web of the words left unsaid." A CLI that shows the
LLM's top-k next-token distribution live as you type. Python project.

## Environment (READ FIRST — non-obvious)
Use this interpreter for everything; do NOT use system `python`:

    /home/t3nzor/venv3.11/bin/python   # Python 3.11.13, torch 2.7.0+cu126, transformers 4.51.3

- **CPU-only by design.** The machine's GPU is an RTX 5070 (`sm_120`/Blackwell).
  The installed torch (`2.7.0+cu126`) ships kernels only up to `sm_90`, so
  `torch.cuda.is_available()` returns `True` but any real CUDA op fails with
  `no kernel image is available for execution on the device`. `HFEngine` pins
  CPU; never add `.cuda()`/device=auto paths unless torch is upgraded to a
  cu128 build with `sm_120` support.
- **numpy must be `<2`.** torch 2.7.0 was built against the numpy 1.x ABI;
  numpy 2.x triggers `numpy.dtype size changed` import errors in transformers.
  Pinned to `1.26.4`. (A shared, unrelated `opencv-python` in this venv wants
  numpy>=2 — ignore that resolver warning.)
- The other venvs are dead ends: `/home/t3nzor/venv` and the system python are
  Python 3.14, which has no torch wheels.

## Commands
Install (deps already in the venv; this adds the project + dev tools):

    /home/t3nzor/venv3.11/bin/python -m pip install -e ".[dev]"

Run:

    /home/t3nzor/venv3.11/bin/python -m unsaid.cli                 # live TUI
    /home/t3nzor/venv3.11/bin/python -m unsaid.cli -p "The quick brown"   # non-interactive top-k

Lint / typecheck / test:

    /home/t3nzor/venv3.11/bin/python -m ruff check .
    /home/t3nzor/venv3.11/bin/python -m mypy src
    /home/t3nzor/venv3.11/bin/python -m pytest -m "not slow"   # fast, no model
    /home/t3nzor/venv3.11/bin/python -m pytest -m slow         # loads gpt2 (~7s)
    /home/t3nzor/venv3.11/bin/python -m pytest tests/test_session.py::test_accept_top_is_index_zero   # single test

- **mypy** would hang walking the installed torch/transformers trees; it's
  configured with `follow_imports = skip` for those modules. Keep that.
- First gpt2 use downloads weights (slow, >2 min) and the first forward pass
  includes a multi-second warm-up; later passes are fast.

## Architecture
`src/unsaid/` (entrypoint `unsaid = unsaid.cli:app`):
- `engine.py` — `CompletionEngine` ABC + `HFEngine`. Forward pass → last-pos
  logits → temperature → softmax → `topk` → decode each token id individually.
  torch/transformers are imported lazily inside methods so `--help` and the
  pure-module tests stay fast.
- `session.py` — buffer text + recompute + accept-candidate logic. No UI/torch.
- `format.py` — pure rendering. Tokens are already decoded, so whitespace is
  made visible: leading space → `·`, newline → `⏎`, tab → `⇥`.
- `ui.py` — prompt_toolkit full-screen TUI, debounced background recompute.
- `cli.py` — typer entrypoint.

## Conventions
- Completions are **raw next tokens** (often subword fragments), not whole
  words — that's intentional ("implicit lexical knowledge").
- TUI keys: `Tab` accepts top-1; `Alt+1..9`/`Alt+0` accept the Nth candidate.
  Plain digits are reserved for typing into the buffer, so accept is on Alt.
- Tests that need a model are marked `slow`; pure logic tests use a `FakeEngine`
  / no torch and must stay fast.
