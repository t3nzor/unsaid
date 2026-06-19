"""Command-line entrypoint for unsaid."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="Explore an LLM's implicit lexical knowledge via live next-token completions.",
)


@app.command()
def main(
    model: str = typer.Option("gpt2", help="Hugging Face causal-LM name."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of candidates to show."),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Softmax temperature."),
    threads: int = typer.Option(
        0, "--threads", help="CPU threads for torch (0 = library default)."
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Torch device: auto picks the CUDA GPU with the most VRAM, else CPU.",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        help="Model dtype: auto uses BF16/FP16 on CUDA and FP32 on CPU.",
    ),
    load_in_4bit: bool = typer.Option(
        False,
        "--load-in-4bit/--no-load-in-4bit",
        help="Load model with BitsAndBytes 4-bit NF4 quantization.",
    ),
    config: str = typer.Option(
        "",
        "--config",
        help="Config TOML path for secrets (default: ~/.config/unsaid/config.toml).",
    ),
    heal: bool = typer.Option(
        True,
        "--heal/--no-heal",
        help="Token healing: complete the partial word you're typing "
        "(--no-heal aggregates raw next-token first characters).",
    ),
    prompt: str = typer.Option(
        "", "--prompt", "-p", help="Non-interactive: print top-k for this text and exit."
    ),
    preamble: str = typer.Option(
        "",
        "--preamble",
        help="Hidden text prepended to model context (system prompt). "
        "Defaults to [unsaid] preamble in the config TOML; this flag overrides it.",
    ),
) -> None:
    """Run the live explorer, or print a single distribution with --prompt."""
    from .config import load_preamble, resolve_hf_token
    from .engine import HFEngine
    from .session import Session

    try:
        hf_token = resolve_hf_token(config or None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc

    effective_preamble = preamble
    if not effective_preamble:
        try:
            toml_preamble = load_preamble(config or None)
            if toml_preamble is not None:
                effective_preamble = toml_preamble
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--config") from exc

    engine = HFEngine(
        model,
        temperature=temperature,
        num_threads=threads or None,
        heal=heal,
        device=device,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        hf_token=hf_token,
        initial_prompt=effective_preamble,
    )
    session = Session(engine, top_k=top_k)

    if prompt:
        from .format import current_word_prefix, format_candidates, format_surprisal

        cands = session.set_text(prompt)
        if effective_preamble:
            typer.echo(f"preamble: {effective_preamble!r}")
        typer.echo(f"prefix: {prompt!r}")
        typer.echo(format_surprisal(session.surprisal, session.n_tokens))
        typer.echo(format_candidates(cands, current_word_prefix(prompt)))
        raise typer.Exit()

    from .ui import UnsaidApp

    UnsaidApp(session).run()


if __name__ == "__main__":
    app()
