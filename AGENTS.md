# AGENTS.md

`unsaid` — "Explore the web of the words left unsaid." A CLI that shows the
LLM's top-k next-character distribution live as you type. Python project.

## Environment (READ FIRST — non-obvious)
Use this interpreter for everything; do NOT use system `python`:

    /home/t3nzor/venv/bin/python   # Python 3.14.5, torch 2.12.0, transformers 5.10.2

- **GPU by default.** The machine has multiple NVIDIA GPUs. `HFEngine` defaults
  to `device="auto"`, which chooses the CUDA device with the most VRAM (currently
  the RTX 5060 Ti at `cuda:1`) and falls back to CPU if CUDA is unavailable.
- Use `--device cpu`, `--device cuda:0`, or `--device cuda:1` to override the
  automatic choice.
- `--dtype auto` is the default: BF16 if CUDA supports it, otherwise FP16 on
  CUDA, and FP32 on CPU. Override with `--dtype float32`, `--dtype float16`, or
  `--dtype bfloat16`.
- 4-bit loading is opt-in via `--load-in-4bit`; it uses BitsAndBytes NF4 with
  double quantization and requires the `quant` extra.
- Hugging Face tokens are resolved from `HF_TOKEN`, then
  `HUGGING_FACE_HUB_TOKEN`, then `~/.config/unsaid/config.toml` (or `--config`).
  Never commit a real token; `config.example.toml` is a safe placeholder.

## Commands
Install (deps already in the venv; this adds the project, dev tools, and 4-bit support):

    /home/t3nzor/venv/bin/python -m pip install -e ".[dev,quant]"

Run:

    /home/t3nzor/venv/bin/python -m unsaid.cli                 # live TUI
    /home/t3nzor/venv/bin/python -m unsaid.cli -p "The quick brown"   # non-interactive top-k
    /home/t3nzor/venv/bin/python -m unsaid.cli --load-in-4bit -p "The quick brown"
    /home/t3nzor/venv/bin/python -m unsaid.cli --config ~/.config/unsaid/config.toml -p "The quick brown"

Lint / typecheck / test:

    /home/t3nzor/venv/bin/python -m ruff check .
    /home/t3nzor/venv/bin/python -m mypy src
    /home/t3nzor/venv/bin/python -m pytest -m "not slow"   # fast, no model
    /home/t3nzor/venv/bin/python -m pytest -m slow         # loads gpt2 (~7s)
    /home/t3nzor/venv/bin/python -m pytest tests/test_session.py::test_accept_top_is_index_zero   # single test

- **mypy** would hang walking the installed torch/transformers trees; it's
  configured with `follow_imports = skip` for those modules. Keep that.
- First gpt2 use downloads weights (slow, >2 min) and the first forward pass
  includes a multi-second warm-up; later passes are fast.

## Architecture
`src/unsaid/` (entrypoint `unsaid = unsaid.cli:app`):
- `engine.py` — `CompletionEngine` ABC + `HFEngine`. Forward pass → last-pos
  logits → temperature → softmax → aggregate next-character probabilities over
  decoded vocabulary entries.
  torch/transformers are imported lazily inside methods so `--help` and the
  pure-module tests stay fast. `surprisal(text)` scores the whole string
  (`-sum log2 P(token|context)`, BOS-primed, in **bits**, at temperature 1).
- `session.py` — buffer text + recompute + accept-candidate logic. No UI/torch.
- `format.py` — pure rendering. Tokens are already decoded, so whitespace is
  made visible: leading space → `·`, newline → `⏎`, tab → `⇥`.
- `ui.py` — prompt_toolkit full-screen TUI, debounced background recompute.
- `cli.py` — typer entrypoint.

## Conventions
- **Token healing is the default.** Typing mid-word (`Hel`) shows next-character
  probabilities with real word context (`Hello`, `Help`) instead of raw BPE
  fragments. `HFEngine.complete` conditions on the text *before* the trailing
  fragment, filters the vocabulary to tokens continuing what's typed, groups
  those tokens by the next character, and returns that character in
  `Candidate.text`. `Candidate.continuation` stores the most likely token
  remainder for display only. Accepting a candidate appends only
  `Candidate.text`. Healed probabilities are renormalized over the matching
  set. `--no-heal` / `heal=False` aggregates raw next-token first characters.
  - The same machinery heals a **trailing space** (`Hello `): GPT-2 tokenizes a
    dangling space as its own token and predicts garbage (`_`, `!!!`) after it,
    so a word boundary is healed by conditioning on the preceding word and
    matching leading-space (new-word) tokens.
  - Raw next tokens are subword fragments and mid-word look broken because of
    BPE (`Hel` is one token; `Hello` is a different single token, so the model
    rarely emits `lo` after `Hel`). That's expected, not a ranking bug.
- TUI keys: `Tab` accepts top-1; `Alt+1..9`/`Alt+0` accept the Nth candidate.
  Plain digits are reserved for typing into the buffer, so accept is on Alt.
  `PageDown`/`PageUp` scroll through pages of lower/higher-ranked completions.
- Paging lives in `Session`: it fetches a pool of `top_k * max_pages` candidates
  per keystroke and slices one page at a time; the model runs once per text
  change, not per page. Editing or accepting resets to page 0. Accept indices
  are relative to the *current page*. (Token healing often yields a tiny pool,
  so paging mainly matters between words / with `--no-heal`.)
- **`--preamble` / `[unsaid] preamble`** seeds the initial editable buffer text.
  It conditions every `complete` call and is **included** in `surprisal` (preamble
  tokens count toward `n_tokens` and bits, since the preamble is part of the
  buffer).  The TUI opens with the preamble pre-filled and editable like any other
  buffer content.  CLI `--preamble` overrides TOML; both TUI and `--prompt`
  one-shot mode apply it.
- **Enter streams a sampled reply.** Pressing Enter triggers full-distribution
  multinomial sampling (``random.SystemRandom``) that streams token-by-token
  into the buffer, stopping at the first sentence-ending punctuation
  (``.``/``!``/``?``, minimum 2 tokens) or 40-token cap; ``\\n`` tokens are
  silently dropped.  The existing ``--temperature`` flag is reused.  Surprisal
  and ``n_tokens`` update once after the stream completes.  The panel shows
  ``sampling…`` during generation; keystrokes are ignored until it finishes.
- Tests that need a model are marked ``slow``; pure logic tests use a ``FakeEngine``
  / no torch and must stay fast.
