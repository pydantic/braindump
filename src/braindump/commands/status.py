"""Show pipeline status: what data exists per stage."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from braindump.config import RepoConfig, compute_file_hash, compute_multi_file_hash
from braindump.progress import console, format_cost, load_stage_cost


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching a glob pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def _count_items_in_json_dir(directory: Path, pattern: str = "*.json") -> int:
    """Count total items across JSON array files in a directory."""
    if not directory.exists():
        return 0
    total = 0
    for f in directory.glob(pattern):
        try:
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    total += len(data)
        except (json.JSONDecodeError, OSError):
            pass
    return total


def _fmt(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def _format_age(mtime: datetime) -> str:
    """Format a datetime as a human-readable age string."""
    delta = datetime.now() - mtime
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    return f"{minutes}m ago"


def _file_age(path: Path) -> str:
    """Return human-readable age of a file."""
    if not path.exists():
        return ""
    return _format_age(datetime.fromtimestamp(path.stat().st_mtime))


def _newest_age(*paths: Path) -> str:
    """Return human-readable age of the most recently modified path."""
    newest_mtime: float | None = None
    for path in paths:
        if path.exists():
            mtime = path.stat().st_mtime
            if newest_mtime is None or mtime > newest_mtime:
                newest_mtime = mtime
    if newest_mtime is None:
        return ""
    return _format_age(datetime.fromtimestamp(newest_mtime))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _is_stale(hash_path: Path, *input_paths: Path) -> bool | None:
    """Check if stored input hash matches current input file(s).

    Returns True if stale, False if current, None if can't determine.
    """
    if not hash_path.exists():
        return None
    existing = [p for p in input_paths if p.exists()]
    if not existing:
        return None
    stored = hash_path.read_text().strip()
    if len(existing) == 1:
        current = compute_file_hash(existing[0])
    else:
        current = compute_multi_file_hash(existing)
    return current != stored


# ============================================================================
# Per-stage checks
# ============================================================================

DONE = "[green]done[/green]"
PARTIAL = "[yellow]partial[/yellow]"
STALE = "[yellow]stale[/yellow]"
MISSING = "[red]missing[/red]"


def _check_download(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check download stage status."""
    pr_list = config.pr_data_dir / "pr_list.json"
    if not pr_list.exists():
        return MISSING, "No PR data downloaded", "", ""

    prs = _load_json(pr_list)
    pr_count = len(prs) if isinstance(prs, list) else 0
    review_comments = _count_items_in_json_dir(config.pr_data_dir / "review_comments")
    diffs = _count_files(config.pr_data_dir / "diffs", "*.diff")

    # Quick completeness check: compare file counts per subdirectory
    output_dir = config.pr_data_dir
    min_files = (
        min(
            _count_files(output_dir / "prs", "*.json"),
            _count_files(output_dir / "reviews", "*.json"),
            _count_files(output_dir / "review_comments", "*.json"),
            _count_files(output_dir / "diffs", "*.diff"),
            _count_files(output_dir / "issue_comments", "*.json"),
        )
        if pr_count > 0
        else 0
    )

    age = _newest_age(pr_list, output_dir)

    if pr_count > 0 and min_files < pr_count:
        details = (
            f"{_fmt(min_files)}/{_fmt(pr_count)} PRs downloaded | "
            f"{_fmt(review_comments)} review comments"
        )
        return PARTIAL, details, age, ""

    details = f"{_fmt(pr_count)} PRs | {_fmt(review_comments)} review comments, {_fmt(diffs)} diffs"
    return DONE, details, age, ""


def _check_extract(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check extract stage status."""
    extractions_path = config.extractions_path
    stage_dir = config.stage_dir("2-extract")
    checkpoint_path = stage_dir / "checkpoint.json"
    stats_path = stage_dir / "stats.json"
    hash_path = stage_dir / "input_hash.txt"
    cost = format_cost(load_stage_cost(stage_dir / "cost.json"))

    if not extractions_path.exists():
        if checkpoint_path.exists() or hash_path.exists():
            return PARTIAL, "Stage started, no extractions yet", _file_age(hash_path), cost
        return MISSING, "No extractions", "", ""

    stats = _load_json(stats_path)
    if isinstance(stats, dict):
        total_processed = stats.get("total_processed", 0)
        actionable = stats.get("actionable", 0)
        non_actionable = stats.get("non_actionable", 0)
        generalizations = stats.get("total_potential_rules", 0)

        total_available = _count_items_in_json_dir(config.review_comments_dir)
        if total_available and total_processed < total_available:
            prefix = f"{_fmt(total_processed)}/{_fmt(total_available)} comments"
        else:
            prefix = f"{_fmt(total_processed)} comments"

        details = (
            f"{prefix} \u2192 "
            f"{_fmt(actionable)} actionable, {_fmt(non_actionable)} rejected \u2192 "
            f"{_fmt(generalizations)} generalizations"
        )
    else:
        extraction_count = _count_lines(extractions_path)
        details = f"{_fmt(extraction_count)} extractions"

    age = _newest_age(extractions_path, hash_path, checkpoint_path, stats_path)

    # Stale check: input_hash vs review_comments
    input_files = sorted(config.review_comments_dir.glob("*.json"))
    if input_files:
        stale = _is_stale(hash_path, *input_files)
        if stale:
            return STALE, details + " (input changed)", age, cost

    return DONE, details, age, cost


def _check_synthesize(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check synthesize stage status."""
    rules_path = config.rules_path
    stage_dir = config.stage_dir("3-synthesize")
    stats_path = stage_dir / "stats.json"
    hash_path = stage_dir / "input_hash.txt"
    clusters_path = stage_dir / "clusters.json"
    unclustered_path = stage_dir / "unclustered.json"
    cost = format_cost(load_stage_cost(stage_dir / "cost.json"))

    if not rules_path.exists():
        if hash_path.exists():
            return PARTIAL, "Stage started, analyzing clusters...", _file_age(hash_path), cost
        return MISSING, "No rules extracted", "", ""

    rule_count = _count_lines(rules_path)
    stats = _load_json(stats_path)

    # Check completion: stats.rules_extracted should match actual rule count
    is_complete = isinstance(stats, dict) and stats.get("rules_extracted", -1) == rule_count

    if not is_complete:
        # Show progress from clusters.json
        clusters = _load_json(clusters_path)
        if isinstance(clusters, list):
            analyzed = sum(1 for c in clusters if "coherence" in c)
            total = len(clusters)
            details = f"{_fmt(rule_count)} rules from {_fmt(analyzed)}/{_fmt(total)} clusters"
        else:
            details = f"{_fmt(rule_count)} rules so far"
        return PARTIAL, details, _newest_age(rules_path, hash_path), cost

    # Stage completed — build full details from stats
    clusters_count = stats.get("total_clusters", 0)
    unclustered = stats.get("unclustered", 0)
    coherence = stats.get("avg_coherence", 0)
    sim_thresh = stats.get("similarity_threshold", 0.65)
    min_size = stats.get("min_cluster_size", 3)
    input_count = stats.get("total_generalizations", 0)
    clustered = input_count - unclustered if input_count else 0

    if not unclustered:
        unclustered_data = _load_json(unclustered_path)
        if isinstance(unclustered_data, list):
            unclustered = len(unclustered_data)
            clustered = input_count - unclustered if input_count else 0

    if input_count:
        details = (
            f"{_fmt(input_count)} generalizations \u2192 "
            f"{_fmt(clustered)} in {_fmt(clusters_count)} clusters, "
            f"{_fmt(unclustered)} unclustered \u2192 "
            f"{_fmt(rule_count)} rules"
            f"\n              (similarity \u2265 {sim_thresh}, min cluster size {min_size}, coherence: {coherence:.2f})"
        )
    else:
        details = (
            f"{_fmt(rule_count)} rules from {_fmt(clusters_count)} clusters, "
            f"{_fmt(unclustered)} unclustered"
            f"\n              (similarity \u2265 {sim_thresh}, min cluster size {min_size}, coherence: {coherence:.2f})"
        )

    age = _newest_age(rules_path, hash_path, stats_path, clusters_path)

    # Stale check
    stale = _is_stale(hash_path, config.extractions_path)
    if stale:
        return STALE, details + " (input changed)", age, cost

    return DONE, details, age, cost


def _check_dedupe(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check dedupe stage status."""
    deduped_path = config.rules_deduped_path
    stage_dir = config.stage_dir("4-dedupe")
    hash_path = stage_dir / "input_hash.txt"
    cost = format_cost(load_stage_cost(stage_dir / "cost.json"))

    if not deduped_path.exists():
        if hash_path.exists():
            return PARTIAL, "Stage started, deduplicating...", _file_age(hash_path), cost
        return MISSING, "No deduped rules", "", ""

    deduped_count = _count_lines(deduped_path)
    original_count = _count_lines(config.rules_path)
    merged = original_count - deduped_count

    details = f"{_fmt(original_count)} \u2192 {_fmt(deduped_count)} rules ({_fmt(merged)} merged)"
    age = _newest_age(deduped_path, hash_path, stage_dir / "merge_log.json")

    # Stale check: hash vs rules.jsonl + overrides
    input_paths = [config.rules_path]
    if config.overrides_path.exists():
        input_paths.append(config.overrides_path)
    stale = _is_stale(hash_path, *input_paths)
    if stale:
        return STALE, details + " (input changed)", age, cost

    return DONE, details, age, cost


def _check_place(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check place stage status."""
    placements_path = config.placements_path
    stage_dir = config.stage_dir("5-place")
    stats_path = stage_dir / "stats.json"
    hash_path = stage_dir / "input_hash.txt"
    cost = format_cost(load_stage_cost(stage_dir / "cost.json"))

    if not placements_path.exists():
        if hash_path.exists():
            return PARTIAL, "Stage started, no placements yet", _file_age(hash_path), cost
        return MISSING, "No placements", "", ""

    placement_count = _count_lines(placements_path)
    place_stats = _load_json(stats_path)

    # Check completion: stats.placed should match actual placement count
    is_complete = isinstance(place_stats, dict) and place_stats.get("placed", -1) == placement_count

    if not is_complete:
        # Partial: placements exist but stage didn't finish
        input_count = _count_lines(config.rules_deduped_path)
        details = f"{_fmt(placement_count)} rules placed so far (input: {_fmt(input_count)})"
        return PARTIAL, details, _newest_age(placements_path, hash_path), cost

    # Stage completed — build full details
    input_count = place_stats.get("input_rules", 0) or place_stats.get("filtered_rules", 0)
    min_score = place_stats.get("min_score", 0.5)
    if not input_count:
        input_count = _count_lines(config.rules_deduped_path)

    by_type: dict[str, int] = place_stats.get("by_type", {})
    if not by_type:
        with open(placements_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    ptype = p.get("placement", {}).get("placement_type", "unknown")
                    by_type[ptype] = by_type.get(ptype, 0) + 1

    type_str = ", ".join(f"{t}: {n}" for t, n in sorted(by_type.items(), key=lambda x: -x[1]))
    filter_str = f" (score \u2265 {min_score})"

    details = (
        f"{_fmt(input_count)} \u2192 {_fmt(placement_count)} rules placed{filter_str} | {type_str}"
    )
    age = _newest_age(placements_path, hash_path, stats_path)

    # Stale check
    stale = _is_stale(hash_path, config.rules_deduped_path)
    if stale:
        return STALE, details + " (input changed)", age, cost

    return DONE, details, age, cost


def _check_group(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check group stage status."""
    org_path = config.organization_path
    stage_dir = config.stage_dir("6-group")
    hash_path = stage_dir / "input_hash.txt"
    cost = format_cost(load_stage_cost(stage_dir / "cost.json"))

    if not org_path.exists():
        if hash_path.exists():
            return PARTIAL, "Stage started, organizing...", _file_age(hash_path), cost
        return MISSING, "No organization data", "", ""

    org = _load_json(org_path)
    if not isinstance(org, dict):
        return "[yellow]error[/yellow]", "Invalid organization file", _file_age(org_path), cost

    locations = org.get("locations", [])
    non_file_locs = [loc for loc in locations if not loc.get("location", "").startswith("file:")]

    total_inline = sum(len(loc.get("inline_rules", [])) for loc in non_file_locs)
    total_topics = sum(
        sum(len(t.get("rule_ids", [])) for t in loc.get("topic_references", []))
        for loc in non_file_locs
    )

    input_count = _count_lines(config.placements_path)
    group_stats = org.get("stats", {})
    min_score = group_stats.get("min_score")
    filtered_count = group_stats.get("filtered_placements")

    score_str = f" (score \u2265 {min_score})" if min_score is not None else ""
    input_str = (
        f"{_fmt(filtered_count)}/{_fmt(input_count)}"
        if filtered_count is not None and filtered_count != input_count
        else f"{_fmt(input_count)}"
    )

    details = (
        f"{input_str} rules{score_str} \u2192 {len(non_file_locs)} locations | "
        f"{_fmt(total_inline)} inline, {_fmt(total_topics)} in topics"
    )
    age = _newest_age(org_path, hash_path)

    # Stale check
    stale = _is_stale(hash_path, config.placements_path)
    if stale:
        return STALE, details + " (input changed)", age, cost

    return DONE, details, age, cost


def _check_generate(config: RepoConfig) -> tuple[str, str, str, str]:
    """Check generate stage status."""
    output_dir = config.output_dir
    hash_path = output_dir / "input_hash.txt"
    cost = format_cost(load_stage_cost(output_dir / "cost.json"))

    if not output_dir.exists():
        return MISSING, "No output directory", "", ""

    agents_files = list(output_dir.rglob("AGENTS.md"))
    topic_files = list(output_dir.rglob("agent_docs/*.md"))

    if not agents_files:
        if hash_path.exists() or (output_dir / "rephrase_cache.jsonl").exists():
            return PARTIAL, "Rephrasing done, no output files yet", _file_age(hash_path), cost
        return MISSING, "No AGENTS.md files generated", "", ""

    total_bytes = sum(f.stat().st_size for f in agents_files + topic_files)
    newest = max(agents_files, key=lambda f: f.stat().st_mtime)

    locations = []
    for f in sorted(agents_files):
        rel = f.relative_to(output_dir)
        parent = str(rel.parent)
        if parent == ".":
            locations.append("root")
        else:
            locations.append(parent)

    details = (
        f"{len(agents_files)} AGENTS.md files ({total_bytes / 1024:.0f} KB) | "
        f"{', '.join(locations)}"
    )
    if topic_files:
        details = (
            f"{len(agents_files)} AGENTS.md files, {len(topic_files)} topic docs ({total_bytes / 1024:.0f} KB) | "
            f"{', '.join(locations)}"
        )
    age = _newest_age(newest, hash_path)

    # Stale check
    stale = _is_stale(hash_path, config.organization_path)
    if stale:
        return STALE, details + " (input changed)", age, cost

    return DONE, details, age, cost


# ============================================================================
# Table & CLI
# ============================================================================


def _find_next_stage(results: list[tuple[str, str, str, str, str]]) -> str | None:
    """Find the first incomplete or stale stage."""
    for stage, status, _, _, _ in results:
        if "missing" in status or "partial" in status or "stale" in status:
            return stage
    return None


def build_status_table(config: RepoConfig) -> tuple[Table, list[tuple[str, str, str, str, str]]]:
    """Build the status table for all pipeline stages.

    Returns (table, results) where results is a list of (stage, status, details, age, cost) tuples.
    """
    checks = [
        ("download", _check_download),
        ("extract", _check_extract),
        ("synthesize", _check_synthesize),
        ("dedupe", _check_dedupe),
        ("place", _check_place),
        ("group", _check_group),
        ("generate", _check_generate),
    ]

    # Check if any stage has cost data
    raw_results: list[tuple[str, str, str, str, str]] = []
    for stage_name, check_fn in checks:
        status, details, age, cost = check_fn(config)
        raw_results.append((stage_name, status, details, age, cost))

    has_any_cost = any(cost for _, _, _, _, cost in raw_results)

    table = Table(show_header=True, expand=True, pad_edge=False)
    table.add_column("Stage", style="bold", width=12)
    table.add_column("Status", width=9)
    table.add_column("Details", ratio=1)
    table.add_column("Updated", width=9, style="dim")
    if has_any_cost:
        table.add_column("Cost", width=9, style="dim cyan", justify="right")

    for stage_name, status, details, age, cost in raw_results:
        if has_any_cost:
            table.add_row(stage_name, status, details, age, cost)
        else:
            table.add_row(stage_name, status, details, age)

    return table, raw_results


def get_total_cost(config: RepoConfig) -> Decimal:
    """Sum up persisted costs across all pipeline stages."""
    return sum(
        (
            load_stage_cost(config.stage_dir(d) / f)
            for d, f in [
                ("2-extract", "cost.json"),
                ("3-synthesize", "cost.json"),
                ("4-dedupe", "cost.json"),
                ("5-place", "cost.json"),
                ("6-group", "cost.json"),
                ("7-generate", "cost.json"),
            ]
        ),
        Decimal(0),
    )


def _run(config: RepoConfig) -> None:
    """Show pipeline status."""
    console.print(f"\n[bold]Pipeline status for [cyan]{config.repo}[/cyan][/bold]")
    console.print(f"[dim]Data: {config.data_dir}[/dim]\n")

    table, results = build_status_table(config)
    console.print(table)

    # Show total cost if any stages have cost data
    total_cost = get_total_cost(config)
    if total_cost > 0:
        console.print(f"\n[dim]Total cost: [cyan]{format_cost(total_cost)}[/cyan][/dim]")

    # Suggest next action
    next_stage = _find_next_stage(results)
    if next_stage:
        # Determine what kind of action is needed
        for stage, status_str, _, _, _ in results:
            if stage == next_stage:
                if "stale" in status_str:
                    console.print(
                        f"\n[bold]Next:[/bold] re-run [cyan]braindump --repo {config.repo} run --from {next_stage}[/cyan] (input changed)"
                    )
                elif "partial" in status_str:
                    console.print(
                        f"\n[bold]Next:[/bold] resume [cyan]braindump --repo {config.repo} {next_stage}[/cyan]"
                    )
                else:
                    console.print(
                        f"\n[bold]Next:[/bold] run [cyan]braindump --repo {config.repo} {next_stage}[/cyan]"
                    )
                    console.print(
                        f"  or: [cyan]braindump --repo {config.repo} run --from {next_stage}[/cyan]"
                    )
                break
    else:
        console.print("\n[bold green]All stages complete![/bold green]")

    # Check for overrides
    overrides_path = config.overrides_path
    if overrides_path.exists():
        override_count = _count_lines(overrides_path)
        if override_count:
            console.print(f"\n[dim]Rule overrides: {override_count} in {overrides_path}[/dim]")

    # Show generated files
    if config.output_dir.exists():
        gen_files = sorted(config.output_dir.rglob("*.md"))
        if gen_files:
            console.print("\n[bold]Generated files:[/bold]")
            for f in gen_files:
                try:
                    rel = f.relative_to(Path.cwd())
                except ValueError:
                    rel = f
                console.print(f"  {rel}")

    console.print()


def status(
    ctx: typer.Context,
) -> None:
    """Show pipeline status and data summary."""
    config: RepoConfig = ctx.obj["config"]
    _run(config)
