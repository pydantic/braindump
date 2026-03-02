"""Pipeline orchestrator: run full pipeline or a subset of stages."""

from __future__ import annotations

from typing import Any

import logfire
import typer

from braindump.config import RepoConfig
from braindump.progress import (
    console,
    format_cost,
    print_pipeline_header,
)

STAGES = ["download", "extract", "synthesize", "dedupe", "place", "group", "generate"]


def _reset_agents() -> None:
    """Reset cached Agent/Embedder singletons between pipeline stages.

    Each stage calls asyncio.run() which creates a new event loop.
    Agents created on a previous loop hold stale httpx connections,
    causing APIConnectionError in subsequent stages.
    """
    import braindump.commands.dedupe as dedupe_mod
    import braindump.commands.extract as extract_mod
    import braindump.commands.generate as generate_mod
    import braindump.commands.group as group_mod
    import braindump.commands.place as place_mod
    import braindump.commands.synthesize as synthesize_mod

    extract_mod._agent = None
    synthesize_mod._agent = None
    synthesize_mod._embedder = None
    dedupe_mod._consolidate_agent = None
    dedupe_mod._category_review_agent = None
    dedupe_mod._embedder = None
    place_mod._agent = None
    group_mod._clustering_agent = None
    generate_mod._rephrase_agent = None


def run(
    ctx: typer.Context,
    from_stage: str | None = typer.Option(
        None, "--from", help=f"Start from this stage ({', '.join(STAGES)})"
    ),
    skip: list[str] | None = typer.Option(None, help="Skip specific stages (repeatable)"),
    authors: str = typer.Option("all", help="Author filter for extract stage"),
    since: str | None = typer.Option(None, help="Date filter for download stage (YYYY-MM-DD)"),
    min_score: float | None = typer.Option(
        None, help="Minimum rule score (default: 0.5)"
    ),
    max_rules: int | None = typer.Option(
        None, "--max-rules", help="Maximum number of rules to include in group stage (top-scored)"
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Wipe all stage outputs and start from scratch"
    ),
) -> None:
    """Run full pipeline (or subset via --from/--skip)."""
    config: RepoConfig = ctx.obj["config"]
    skip_set = set(skip) if skip else set()

    # Determine which stages to run
    if from_stage:
        if from_stage not in STAGES:
            console.print(f"[red]Error: --from must be one of: {', '.join(STAGES)}[/red]")
            raise typer.Exit(1)
        start_idx = STAGES.index(from_stage)
        stages_to_run = STAGES[start_idx:]
    else:
        stages_to_run = list(STAGES)

    stages_to_run = [s for s in stages_to_run if s not in skip_set]

    print_pipeline_header(stages_to_run, config.repo)

    with logfire.span("braindump: {repo}", repo=config.repo, stages=stages_to_run):
        for stage in stages_to_run:
            console.rule(f"[bold]{stage.capitalize()}[/bold]")

            with logfire.span("stage: {stage_name}", stage_name=stage):
                stats = None

                if stage == "download":
                    from braindump.commands.download import _run as download_run

                    stats = download_run(config, since=since, is_pipeline=True, fresh=fresh)

                elif stage == "extract":
                    from braindump.commands.extract import _run as extract_run

                    stats = extract_run(config, authors=authors, is_pipeline=True, fresh=fresh)

                elif stage == "synthesize":
                    from braindump.commands.synthesize import _run as synthesize_run

                    stats = synthesize_run(config, is_pipeline=True, fresh=fresh)

                elif stage == "dedupe":
                    from braindump.commands.dedupe import _run as dedupe_run

                    stats = dedupe_run(config, is_pipeline=True, fresh=fresh)

                elif stage == "place":
                    from braindump.commands.place import _run as place_run

                    place_kwargs: dict = {"is_pipeline": True, "fresh": fresh}
                    if min_score is not None:
                        place_kwargs["min_score"] = min_score
                    stats = place_run(config, **place_kwargs)

                elif stage == "group":
                    from braindump.commands.group import _run as group_run

                    group_kwargs: dict = {"is_pipeline": True, "fresh": fresh}
                    if min_score is not None:
                        group_kwargs["min_score"] = min_score
                    if max_rules is not None:
                        group_kwargs["max_rules"] = max_rules
                    stats = group_run(config, **group_kwargs)

                elif stage == "generate":
                    from braindump.commands.generate import _run as generate_run

                    stats = generate_run(config, is_pipeline=True, fresh=fresh)

                if stats:
                    stats_dict: dict[str, Any] = stats.as_dict()
                    logfire.info("stage complete: {stage_name}", stage_name=stage, **stats_dict)

            _reset_agents()

    # Pipeline summary: reuse the status table
    from braindump.commands.status import build_status_table, get_total_cost

    table, _ = build_status_table(config)
    console.print()
    console.print(table)
    total_cost = get_total_cost(config)
    if total_cost > 0:
        console.print(f"\n[dim]Total cost: [cyan]{format_cost(total_cost)}[/cyan][/dim]")
    console.print("\n[bold green]Pipeline complete![/bold green]")

    # Generated files at the end
    if "generate" in stages_to_run and config.output_dir.exists():
        from pathlib import Path

        gen_files = sorted(config.output_dir.rglob("*.md"))
        if gen_files:
            console.print("\n[bold]Generated files:[/bold]")
            for f in gen_files:
                try:
                    rel = f.relative_to(Path.cwd())
                except ValueError:
                    rel = f
                console.print(f"  {rel}")
