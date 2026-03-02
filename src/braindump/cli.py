"""Braindump CLI — synthesize agent context from code review history."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="braindump",
    help="Synthesize agent context from code review history.",
    no_args_is_help=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    repo: str = typer.Option("", envvar="BRAINDUMP_REPO", help="GitHub repo (owner/repo)"),
    model: str = typer.Option(
        "", envvar="BRAINDUMP_MODEL", help="LLM model (e.g. gateway/anthropic:claude-sonnet-4-5, openai:gpt-4o)"
    ),
    embedding_model: str = typer.Option(
        "", envvar="BRAINDUMP_EMBEDDING_MODEL", help="Embedding model (e.g. gateway/openai:text-embedding-3-small)"
    ),
) -> None:
    """Braindump: extract coding rules from PR review comments."""
    import sys

    ctx.ensure_object(dict)

    # Skip setup when just showing help or doing shell completion
    if ctx.resilient_parsing or "--help" in sys.argv or "-h" in sys.argv:
        return

    from braindump.config import RepoConfig, configure_models, init_env

    init_env()
    configure_models(model=model or None, embedding_model=embedding_model or None)

    if not repo:
        repo = typer.prompt("GitHub repository (owner/repo)")
    if "/" not in repo:
        typer.echo("Error: repo must be in owner/repo format", err=True)
        raise typer.Exit(1)

    ctx.obj["repo"] = repo
    ctx.obj["config"] = RepoConfig(repo)


# Register all commands
from braindump.commands.dedupe import dedupe as _dedupe  # noqa: E402
from braindump.commands.download import download as _download  # noqa: E402
from braindump.commands.extract import extract as _extract  # noqa: E402
from braindump.commands.generate import generate as _generate  # noqa: E402
from braindump.commands.group import group as _group  # noqa: E402
from braindump.commands.lookup import lookup as _lookup  # noqa: E402
from braindump.commands.place import place as _place  # noqa: E402
from braindump.commands.run import run as _run  # noqa: E402
from braindump.commands.status import status as _status  # noqa: E402
from braindump.commands.synthesize import synthesize as _synthesize  # noqa: E402

app.command(name="download")(_download)
app.command(name="extract")(_extract)
app.command(name="synthesize")(_synthesize)
app.command(name="dedupe")(_dedupe)
app.command(name="place")(_place)
app.command(name="group")(_group)
app.command(name="generate")(_generate)
app.command(name="run")(_run)
app.command(name="lookup")(_lookup)
app.command(name="status")(_status)
