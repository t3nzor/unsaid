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
    prompt: str = typer.Option(
        "", "--prompt", "-p", help="Non-interactive: print top-k for this text and exit."
    ),
) -> None:
    """Run the live explorer, or print a single distribution with --prompt."""
    from .engine import HFEngine
    from .session import Session

    engine = HFEngine(
        model,
        temperature=temperature,
        num_threads=threads or None,
    )
    session = Session(engine, top_k=top_k)

    if prompt:
        from .format import current_word_prefix, format_candidates

        cands = session.set_text(prompt)
        typer.echo(f"prefix: {prompt!r}")
        typer.echo(format_candidates(cands, current_word_prefix(prompt)))
        raise typer.Exit()

    from .ui import UnsaidApp

    UnsaidApp(session).run()


if __name__ == "__main__":
    app()
