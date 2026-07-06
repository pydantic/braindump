"""Apply generated files into a target repo via non-destructive block merge."""

from __future__ import annotations

from pathlib import Path

import typer

from braindump.commands.generate import merge_braindump_block
from braindump.config import RepoConfig
from braindump.progress import console


def _run(config: RepoConfig, into: Path, dry_run: bool = False) -> None:
    output_dir = config.output_dir
    generated = sorted(output_dir.rglob("*.md")) if output_dir.exists() else []
    if not generated:
        console.print(
            f"[red]No generated .md files under {output_dir}. Run `generate` first.[/red]"
        )
        raise typer.Exit(1)

    into = into.resolve()
    created = merged = unchanged = 0
    for src in generated:
        rel = src.relative_to(output_dir)
        target = into / rel
        content = src.read_text()

        if target.exists():
            existing = target.read_text()
            new_content = merge_braindump_block(existing, content)
            if new_content == existing:
                verb, unchanged = "unchanged", unchanged + 1
            else:
                verb, merged = "merge", merged + 1
        else:
            new_content, verb, created = content, "create", created + 1

        console.print(f"[dim]{verb:9}[/dim] {target}")
        if not dry_run and verb != "unchanged":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_content)

    prefix = "[yellow]Dry run — nothing written.[/yellow] " if dry_run else ""
    console.print(
        f"\n{prefix}{created} created, {merged} merged, {unchanged} unchanged (into {into})"
    )


def apply(
    ctx: typer.Context,
    into: Path = typer.Option(..., "--into", help="Target repo root to merge generated files into"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
) -> None:
    """Merge generated AGENTS.md files into a repo, preserving hand-written content."""
    config: RepoConfig = ctx.obj["config"]
    _run(config, into=into, dry_run=dry_run)
