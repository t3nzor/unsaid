# unsaid

Explore the web of the words left unsaid.

`unsaid` is a CLI that shows an LLM's top-k next-character distribution **live
as you type**. Every keystroke runs a forward pass and renders the model's most
likely continuations, so you can watch the lexical structure of language unfold
one character at a time.

## What it does

- **Live next-token explorer.** A full-screen TUI shows the top-k candidates for
  the next character along with their probabilities, refreshed on every edit.
- **Token healing (on by default).** Typing mid-word (`Hel`) shows
  next-character probabilities with real word context (`Hello`, `Help`) instead
  of raw BPE fragments. `--no-heal` aggregates raw next-token first characters.
- **Accept completions.** `Tab` accepts the top-1 candidate; `Alt+1..9`/`Alt+0`
  accept the Nth. `PageDown`/`PageUp` page through more candidates.
- **Stream a sampled reply.** Press `Enter` to sample a reply token-by-token
  from the full distribution, stopping at sentence-ending punctuation
  (`.`/`!`/`?`) or a 40-token cap.
- **Surprisal scoring.** The panel reports the buffer's surprisal
  (`-sum log2 P(token|context)`, in bits) and token count.
- **Preamble.** `--preamble` seeds the initial editable buffer text and
  conditions every completion.
- **GPU by default.** Auto-selects the CUDA device with the most VRAM, falls
  back to CPU. Supports BF16/FP16 on CUDA and optional 4-bit NF4 quantization.

## Requirements

- Python >= 3.11
- PyTorch and Hugging Face Transformers (expected to be provided by the
  environment — not pinned in `pyproject.toml` to avoid reinstalling heavy
  torch builds)
- An optional Hugging Face token for gated models, resolved from `HF_TOKEN`,
  then `HUGGING_FACE_HUB_TOKEN`, then `~/.config/unsaid/config.toml`.

## Install

```bash
pip install -e ".[dev,quant]"
```

The `quant` extra pulls in `accelerate` and `bitsandbytes` for 4-bit loading.
The `dev` extra pulls in `pytest`, `ruff`, and `mypy`.

## Run

```bash
unsaid                                # live TUI (defaults to gpt2)
unsaid -p "The quick brown"           # non-interactive: print top-k and exit
unsaid --model gpt2 --top-k 15        # bigger candidate list
unsaid --load-in-4bit -p "The quick brown"
unsaid --preamble "You are a poet."   # seed the buffer
```

As a module:

```bash
python -m unsaid.cli
python -m unsaid.cli -p "The quick brown"
```

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `gpt2` | Hugging Face causal-LM name. |
| `--top-k`, `-k` | `10` | Number of candidates to show. |
| `--temperature`, `-t` | `1.0` | Softmax temperature. |
| `--threads` | `0` | CPU threads for torch (`0` = library default). |
| `--device` | `auto` | Torch device: `auto`, `cpu`, `cuda:0`, `cuda:1`. |
| `--dtype` | `auto` | Model dtype: `auto`, `float32`, `float16`, `bfloat16`. |
| `--load-in-4bit` | off | Load with BitsAndBytes 4-bit NF4 quantization. |
| `--config` | `~/.config/unsaid/config.toml` | Config TOML path for secrets. |
| `--heal` / `--no-heal` | on | Token healing of the partial word you're typing. |
| `--prompt`, `-p` | — | Non-interactive: print top-k for this text and exit. |
| `--preamble` | — | Editable buffer text prepended to the model's context. |

## Configuration

Copy `config.example.toml` to `~/.config/unsaid/config.toml` and fill in your
Hugging Face token. Keep the real token out of git.

```toml
[huggingface]
token = "hf_your_token_here"

[unsaid]
preamble = ""   # seeds the buffer at startup; overridden by --preamble
```

## TUI keys

- `Tab` — accept top-1 candidate
- `Alt+1..9` / `Alt+0` — accept the Nth candidate
- `PageDown` / `PageUp` — page through more candidates
- `Enter` — stream a sampled reply into the buffer
- Plain digits type into the buffer (accept is on `Alt`)

## Lint / typecheck / test

```bash
ruff check .
mypy src
pytest -m "not slow"     # fast, no model
pytest -m slow           # loads gpt2 (~7s)
```

First gpt2 use downloads weights (slow, >2 min) and the first forward pass
includes a multi-second warm-up; later passes are fast.

## Architecture

`src/unsaid/` (entrypoint `unsaid = unsaid.cli:app`):

- `engine.py` — `CompletionEngine` ABC + `HFEngine`. Forward pass → last-pos
  logits → temperature → softmax → aggregate next-character probabilities over
  decoded vocabulary entries. `surprisal(text)` scores the whole string.
- `session.py` — buffer text + recompute + accept-candidate + paging logic. No
  UI/torch.
- `format.py` — pure rendering. Whitespace is made visible: leading space →
  `·`, newline → `⏎`, tab → `⇥`.
- `ui.py` — prompt_toolkit full-screen TUI, debounced background recompute.
- `cli.py` — typer entrypoint.

## License

See [LICENSE](LICENSE).
