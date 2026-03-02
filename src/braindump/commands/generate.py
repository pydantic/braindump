"""Stage 5: Generate AGENTS.md files and topic docs from organized rules."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from textwrap import dedent

import logfire
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rich.progress import Progress, TaskID

from braindump.config import RepoConfig, compute_file_hash
from braindump.progress import StageStats, console, format_cost, get_result_cost, stage_progress

# ============================================================================
# Models
# ============================================================================


class RephrasedRule(BaseModel):
    rule_text: str = Field(description="The rephrased rule text, concise and actionable")
    rationale: str = Field(description="Brief explanation of WHY this rule matters (1 sentence)")


# ============================================================================
# Lazy Agent
# ============================================================================

_rephrase_agent: Agent[None, RephrasedRule] | None = None


def get_rephrase_agent() -> Agent[None, RephrasedRule]:
    global _rephrase_agent
    if _rephrase_agent is None:
        from braindump.config import make_model

        _rephrase_agent = Agent(
            make_model(),
            output_type=RephrasedRule,
            system_prompt=dedent("""
                You are rephrasing coding rules for an AGENTS.md file that AI coding assistants will read.

                Transform the raw rule into a concise, actionable guideline that:
                1. Is clear and direct (imperative mood)
                2. Includes the WHY - why does this matter? (backward compat, prevents bugs, consistency, etc.)
                3. Is NOT deontological ("do X because rules say so")
                4. Fits in a bullet point (ideally under 100 chars for the rule, rationale can be longer)
                5. Preserves project-specific references — class names, file paths, directory conventions,
                   config patterns. These are the most valuable parts of a rule for an AGENTS.md.
                   Don't strip them for brevity.

                Use backticks for all code identifiers, file paths, class names, function names,
                parameter names, etc. — this is markdown and should be formatted accordingly.

                Good examples:
                - "Use keyword-only args for public APIs — prevents breakage when adding parameters"
                - "Wrap optional imports in `try`/`except` — keeps the package installable without all deps"
                - "Match naming patterns across providers — users expect consistent APIs"

                Bad examples (too long, no rationale, deontological):
                - "When adding parameters to functions, always use keyword-only arguments"
                - "You must use try/except for imports"
                - "Follow the established naming convention"
            """).strip(),
        )
    return _rephrase_agent


# ============================================================================
# Data Loading
# ============================================================================


def load_data(
    organization_path: Path,
    rules_path: Path,
    placements_path: Path,
) -> tuple[dict, dict[int, dict], dict[int, dict]]:
    with open(organization_path) as f:
        organization = json.load(f)
    rules = {}
    with open(rules_path) as f:
        for line in f:
            if line.strip():
                rule = json.loads(line)
                rules[rule["rule_id"]] = rule
    placements = {}
    with open(placements_path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                placements[p["rule_id"]] = p
    return organization, rules, placements


# ============================================================================
# Rule Rephrasing
# ============================================================================


@logfire.instrument("rephrase rule")
async def rephrase_rule(rule: dict, placement: dict) -> tuple[RephrasedRule, Decimal]:
    context = f"""
Rule text: {rule["text"]}

Category: {rule.get("category", "unknown")}
Scope: {rule.get("scope", "unknown")}

Placement rationale: {placement["placement"].get("rationale", "N/A")[:500]}

Example of bad code: {rule.get("example_bad", "N/A")}
Example of good code: {rule.get("example_good", "N/A")}
    """.strip()
    result = await get_rephrase_agent().run(context)
    return result.output, get_result_cost(result)


def _load_rephrase_cache(cache_path: Path) -> dict[int, RephrasedRule]:
    """Load persistent rephrase cache from disk."""
    cache: dict[int, RephrasedRule] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cache[entry["rule_id"]] = RephrasedRule(
                    rule_text=entry["rule_text"],
                    rationale=entry["rationale"],
                )
    return cache


def _save_rephrase_entry(cache_path: Path, rule_id: int, rephrased: RephrasedRule) -> None:
    """Append one rephrase entry to the persistent cache."""
    with open(cache_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "rule_id": rule_id,
                    "rule_text": rephrased.rule_text,
                    "rationale": rephrased.rationale,
                }
            )
            + "\n"
        )


async def rephrase_rules_batch(
    rule_ids: list[int],
    rules: dict[int, dict],
    placements: dict[int, dict],
    cache: dict[int, RephrasedRule],
    concurrency: int = 10,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    cache_path: Path | None = None,
    is_pipeline: bool = False,
    stats: StageStats | None = None,
) -> dict[int, RephrasedRule]:
    to_rephrase = [rid for rid in rule_ids if rid not in cache]
    if not to_rephrase:
        return {rid: cache[rid] for rid in rule_ids}

    if not is_pipeline:
        console.print(
            f"Rephrasing {len(to_rephrase)} rules (cached: {len(rule_ids) - len(to_rephrase)})"
        )

    if progress and task_id is not None:
        progress.update(
            task_id, total=len(to_rephrase), completed=0, description="Rephrasing rules..."
        )

    semaphore = asyncio.Semaphore(concurrency)

    async def rephrase_one(rid: int) -> tuple[int, RephrasedRule] | None:
        async with semaphore:
            try:
                rule = rules.get(rid)
                if rule is None:
                    return None
                placement = placements.get(rid, {"placement": {}})
                result, cost = await rephrase_rule(rule, placement)
                if stats:
                    stats.add_cost(cost)
                if cache_path:
                    _save_rephrase_entry(cache_path, rid, result)
                return rid, result
            finally:
                if progress and task_id is not None:
                    progress.advance(task_id)
                    if stats:
                        progress.update(task_id, cost=format_cost(stats.total_cost))

    tasks = [rephrase_one(rid) for rid in to_rephrase]
    results = await asyncio.gather(*tasks)
    for entry in results:
        if entry is not None:
            rid, rephrased = entry
            cache[rid] = rephrased
    return {rid: cache[rid] for rid in rule_ids}


# ============================================================================
# Markdown Generation
# ============================================================================


def sort_by_signal(rule_ids: list[int], placements: dict[int, dict]) -> list[int]:
    return sorted(
        rule_ids,
        key=lambda rid: placements.get(rid, {}).get(
            "rule_score", placements.get(rid, {}).get("signal_score", 0)
        ),
        reverse=True,
    )


def format_rule_line(rule_id: int, rephrased: RephrasedRule) -> str:
    return f"<!-- rule:{rule_id} -->\n- {rephrased.rule_text} — {rephrased.rationale}"


CATEGORY_NAMES = {
    "api_design": "API Design",
    "code_style": "Code Style",
    "documentation": "Documentation",
    "error_handling": "Error Handling",
    "testing": "Testing",
    "typing": "Type System",
    "imports": "Imports",
    "naming": "Naming",
    "exports": "Exports",
    "deprecation": "Deprecation",
}


def _render_rules_by_category(
    rule_ids: list[int],
    rules: dict[int, dict],
    rephrased: dict[int, RephrasedRule],
    placements: dict[int, dict],
) -> list[str]:
    lines = []
    by_category: dict[str, list[int]] = {}
    for rid in rule_ids:
        cat = rules.get(rid, {}).get("category", "general")
        by_category.setdefault(cat, []).append(rid)

    general_rids = []
    real_categories: dict[str, list[int]] = {}
    for cat, rids in by_category.items():
        if len(rids) == 1:
            general_rids.extend(rids)
        else:
            real_categories[cat] = rids

    def cat_signal(cat: str) -> float:
        return sum(
            placements.get(rid, {}).get(
                "rule_score", placements.get(rid, {}).get("signal_score", 0)
            )
            for rid in real_categories.get(cat, [])
        )

    sorted_cats = sorted(real_categories.keys(), key=cat_signal, reverse=True)

    for cat in sorted_cats:
        rids = real_categories[cat]
        heading = CATEGORY_NAMES.get(cat, cat.replace("_", " ").title())
        lines.append(f"## {heading}")
        lines.append("")
        for rid in sort_by_signal(rids, placements):
            if rid in rephrased:
                lines.append(format_rule_line(rid, rephrased[rid]))
        lines.append("")

    if general_rids:
        lines.append("## General")
        lines.append("")
        for rid in sort_by_signal(general_rids, placements):
            if rid in rephrased:
                lines.append(format_rule_line(rid, rephrased[rid]))
        lines.append("")

    return lines


def generate_agents_md(
    location: str,
    loc_data: dict,
    rephrased: dict[int, RephrasedRule],
    rules: dict[int, dict],
    placements: dict[int, dict],
    subdirectories: list[str] | None = None,
) -> str:
    lines = []
    lines.append("<!-- braindump: rules extracted from PR review patterns -->")
    lines.append("")
    if location == "root":
        lines.append("# Coding Guidelines")
    else:
        lines.append(f"# {location}/ Guidelines")
    lines.append("")

    if location == "root" and subdirectories:
        lines.append("Also see directory-specific guidelines:")
        lines.append("")
        for subdir in sorted(subdirectories):
            lines.append(f"- [{subdir}/AGENTS.md]({subdir}/AGENTS.md)")
        lines.append("")

    inline_rules = loc_data.get("inline_rules", [])
    if inline_rules:
        lines.extend(_render_rules_by_category(inline_rules, rules, rephrased, placements))

    topic_refs = loc_data.get("topic_references", [])
    if topic_refs:
        lines.append("## Topic Guides")
        lines.append("")
        lines.append("Check these when working in specific areas:")
        lines.append("")
        for topic in topic_refs:
            trigger = topic.get("trigger_description", "When relevant")
            topic_id = topic["topic_id"]
            lines.append(f"- **[{topic['topic_name']}](agent_docs/{topic_id}.md)**: {trigger}")
        lines.append("")

    file_rules = loc_data.get("file_specific_rules", {})
    if file_rules:
        lines.append("## File-Specific Rules")
        lines.append("")
        for file_path, rule_ids in sorted(file_rules.items()):
            lines.append(f"### `{file_path}`")
            lines.append("")
            for rid in sort_by_signal(rule_ids, placements):
                if rid in rephrased:
                    lines.append(format_rule_line(rid, rephrased[rid]))
            lines.append("")

    lines.append("<!-- /braindump -->")
    return "\n".join(lines)


def generate_topic_doc(
    topic: dict,
    rephrased: dict[int, RephrasedRule],
    placements: dict[int, dict],
) -> str:
    lines = []
    lines.append(f"# {topic['topic_name']}")
    lines.append("")
    lines.append(f"> {topic['topic_description']}")
    lines.append("")
    lines.append(f"**When to check**: {topic['trigger_description']}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid in sort_by_signal(topic["rule_ids"], placements):
        if rid in rephrased:
            lines.append(format_rule_line(rid, rephrased[rid]))
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main Generation
# ============================================================================


async def generate_all_files(
    organization: dict,
    rules: dict[int, dict],
    placements: dict[int, dict],
    output_dir: Path,
    concurrency: int = 10,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats: StageStats | None = None,
    is_pipeline: bool = False,
) -> dict[str, str]:
    all_rule_ids = set()
    for loc_data in organization["locations"]:
        all_rule_ids.update(loc_data.get("inline_rules", []))
        for topic in loc_data.get("topic_references", []):
            all_rule_ids.update(topic.get("rule_ids", []))

    logfire.info(
        "Rules to rephrase", total=len(all_rule_ids), locations=len(organization["locations"])
    )
    if stats:
        stats.set("rules_to_rephrase", len(all_rule_ids))

    rephrase_cache_path = output_dir / "rephrase_cache.jsonl"
    cache: dict[int, RephrasedRule] = _load_rephrase_cache(rephrase_cache_path)
    with logfire.span("rephrase rules", total=len(all_rule_ids)):
        rephrased = await rephrase_rules_batch(
            list(all_rule_ids),
            rules,
            placements,
            cache,
            concurrency,
            progress=progress,
            task_id=task_id,
            cache_path=rephrase_cache_path,
            is_pipeline=is_pipeline,
            stats=stats,
        )

    file_rules_by_dir: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for loc_data in organization["locations"]:
        location = loc_data["location"]
        if location.startswith("file:"):
            file_path = location.replace("file:", "")
            dir_path = str(Path(file_path).parent)
            if dir_path == ".":
                dir_path = "root"
            rule_ids = loc_data.get("inline_rules", [])
            if rule_ids:
                file_rules_by_dir[dir_path][file_path] = rule_ids

    all_dir_locations = [
        loc_data["location"]
        for loc_data in organization["locations"]
        if not loc_data["location"].startswith("file:")
        and loc_data["location"] != "root"
        and (loc_data.get("inline_rules") or loc_data.get("topic_references"))
    ]

    # Phase 2: Write files
    if progress and task_id is not None:
        progress.update(task_id, description="Writing files...", completed=0)

    files: dict[str, str] = {}
    for loc_data in organization["locations"]:
        location = loc_data["location"]
        if location.startswith("file:"):
            continue
        loc_data_with_files = dict(loc_data)
        if location in file_rules_by_dir:
            loc_data_with_files["file_specific_rules"] = file_rules_by_dir[location]
        if location == "root":
            base_path = output_dir
            subdirs = all_dir_locations
        else:
            base_path = output_dir / location
            subdirs = None
        has_content = (
            loc_data.get("inline_rules")
            or loc_data.get("topic_references")
            or loc_data_with_files.get("file_specific_rules")
        )
        if has_content:
            content = generate_agents_md(
                location, loc_data_with_files, rephrased, rules, placements, subdirectories=subdirs
            )
            files[str(base_path / "AGENTS.md")] = content
            for topic in loc_data.get("topic_references", []):
                topic_content = generate_topic_doc(topic, rephrased, placements)
                topic_path = base_path / "agent_docs" / f"{topic['topic_id']}.md"
                files[str(topic_path)] = topic_content

    if progress and task_id is not None:
        progress.update(task_id, total=len(files), completed=0, description="Writing files...")

    logfire.info("Files to write", total=len(files))
    if stats:
        stats.set("files", len(files))

    return files


def write_files(
    files: dict[str, str],
    dry_run: bool = False,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
):
    for path_str, content in sorted(files.items()):
        path = Path(path_str)
        if dry_run:
            console.print(f"[dim]Would write: {path} ({len(content)} bytes)[/dim]")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        if progress and task_id is not None:
            progress.advance(task_id)


# ============================================================================
# CLI Command
# ============================================================================


async def _async_run(
    config: RepoConfig,
    is_pipeline: bool = False,
    **kwargs,
) -> StageStats | None:
    concurrency = kwargs.get("concurrency", 10)
    dry_run = kwargs.get("dry_run", False)
    fresh = kwargs.get("fresh", False)

    output_dir = config.output_dir

    # Input hash: detect if organized_rules.json changed
    input_hash = compute_file_hash(config.organization_path)
    hash_path = output_dir / "input_hash.txt"

    if fresh:
        if output_dir.exists():
            shutil.rmtree(output_dir)
    elif output_dir.exists() and hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        if stored_hash != input_hash:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(input_hash)

    if not is_pipeline:
        console.print("Loading data...")
    organization, rules, placements = load_data(
        config.organization_path, config.rules_deduped_path, config.placements_path
    )
    if not is_pipeline:
        console.print(f"Loaded {len(organization['locations'])} locations, {len(rules)} rules")

    with stage_progress(
        "Generate", is_pipeline=is_pipeline, cost_path=output_dir / "cost.json"
    ) as (progress, stats):
        task = progress.add_task("Rephrasing rules...", total=0)

        files = await generate_all_files(
            organization,
            rules,
            placements,
            output_dir,
            concurrency,
            progress=progress,
            task_id=task,
            stats=stats,
            is_pipeline=is_pipeline,
        )

        # Remove generated .md files before writing to avoid stale files,
        # but preserve cache files (rephrase_cache.jsonl, input_hash.txt)
        if not dry_run:
            for md_file in output_dir.rglob("*.md"):
                md_file.unlink()

        write_files(files, dry_run, progress=progress, task_id=task)

        if dry_run:
            stats.set("mode", "dry-run")

        num_files = int(stats.get("files", 0))
        stats.detail = f"{num_files} files generated"

    # Print generated file paths after progress bar is done (not in pipeline mode)
    if files and not is_pipeline:
        console.print()
        console.print("[bold]Generated files:[/bold]")
        for path_str in sorted(files.keys()):
            try:
                rel = Path(path_str).relative_to(Path.cwd())
            except ValueError:
                rel = Path(path_str)
            console.print(f"  {rel}")

    return stats


def _run(config: RepoConfig, is_pipeline: bool = False, **kwargs) -> StageStats | None:
    return asyncio.run(_async_run(config, is_pipeline=is_pipeline, **kwargs))


def generate(
    ctx: typer.Context,
    concurrency: int = typer.Option(10, help="Number of parallel LLM requests"),
    dry_run: bool = typer.Option(False, help="Don't write files, just show what would be written"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Generate AGENTS.md files."""
    config: RepoConfig = ctx.obj["config"]
    _run(config, concurrency=concurrency, dry_run=dry_run, fresh=fresh)
