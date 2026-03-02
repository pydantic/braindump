"""Stage 4: Organize rules by topic and signal level for progressive disclosure."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import logfire
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rich.progress import Progress, TaskID

from braindump.config import RepoConfig, compute_file_hash
from braindump.progress import StageStats, console, format_cost, get_result_cost, stage_progress

# ============================================================================
# Configuration
# ============================================================================

MIN_RULES_FOR_TOPICS = 10
MIN_CLUSTER_SIZE = 10
MIN_RULES_FOR_DIR_FILE = 3


# ============================================================================
# Models
# ============================================================================


class TopicCluster(BaseModel):
    topic_id: str
    topic_name: str
    topic_description: str
    trigger_description: str
    rule_ids: list[int]


class TopicClusteringResult(BaseModel):
    clusters: list[TopicCluster]
    unclustered_rule_ids: list[int]


class OrganizedLocation(BaseModel):
    location: str
    inline_rules: list[int] = Field(default_factory=list)
    topic_references: list[TopicCluster] = Field(default_factory=list)


class Stage4Output(BaseModel):
    locations: list[OrganizedLocation]
    rules_by_id: dict[int, dict]
    stats: dict


# ============================================================================
# System Prompt
# ============================================================================

TOPIC_CLUSTERING_PROMPT = """You are organizing coding rules into topic clusters for progressive disclosure.

Given a set of rules, group them into coherent topics that a developer would naturally look up together.

## Good Topics
- "API Design" - rules about designing public interfaces
- "Type System" - rules about typing, generics, type hints
- "Error Handling" - rules about exceptions, validation, error messages
- "Testing Patterns" - rules about test structure and assertions
- "Documentation" - rules about docstrings, comments, docs
- "Async Patterns" - rules about async/await, concurrency
- "Module Conventions" - rules specific to module implementation patterns

## Topic Requirements
- Each topic should have 3+ rules (otherwise keep rules unclustered)
- Topics should be actionable: "When doing X, check this topic"
- Don't over-cluster: if rules are truly miscellaneous, leave them unclustered

## Output
- topic_id: lowercase-with-dashes identifier
- topic_name: Human readable title
- topic_description: What the topic covers
- trigger_description: "When [doing something specific]" - when to check this topic
- rule_ids: Which rules belong here

Rules that don't fit any topic go in unclustered_rule_ids.
"""


# ============================================================================
# Lazy Agent
# ============================================================================

_clustering_agent: Agent[None, TopicClusteringResult] | None = None


def get_clustering_agent() -> Agent[None, TopicClusteringResult]:
    global _clustering_agent
    if _clustering_agent is None:
        from braindump.config import make_model

        _clustering_agent = Agent(
            make_model(),
            output_type=TopicClusteringResult,
            system_prompt=TOPIC_CLUSTERING_PROMPT,
        )
    return _clustering_agent


# ============================================================================
# Data Loading
# ============================================================================


def load_placements(placements_path: Path) -> list[dict]:
    by_id: dict[int, dict] = {}
    with open(placements_path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                rid = p["rule_id"]
                # Keep highest-scored placement per rule_id
                if rid not in by_id or p.get("rule_score", 0) > by_id[rid].get("rule_score", 0):
                    by_id[rid] = p
    return list(by_id.values())


def load_rules(rules_path: Path) -> dict[int, dict]:
    rules = {}
    with open(rules_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rules[r["rule_id"]] = r
    return rules


# ============================================================================
# Topic Clustering
# ============================================================================


def build_clustering_prompt(rules_to_cluster: list[dict]) -> str:
    rules_text = []
    for p in rules_to_cluster:
        rules_text.append(f"""
Rule {p["rule_id"]} (score={p.get("rule_score", p.get("signal_score", 0)):.2f}):
  {p["rule_text"]}
  Category: {p["rule_category"]}
""")
    return f"""## Rules to Cluster

The following {len(rules_to_cluster)} rules need to be organized into topics.
Group related rules together based on what task/activity they apply to.

{"".join(rules_text)}

---

Create topic clusters. Rules that don't fit any topic should go in unclustered_rule_ids.
"""


@logfire.instrument("cluster rules by topic")
async def cluster_rules_by_topic(
    rules_to_cluster: list[dict], min_cluster_size: int = 3
) -> tuple[TopicClusteringResult, Decimal]:
    if len(rules_to_cluster) < min_cluster_size:
        return TopicClusteringResult(
            clusters=[], unclustered_rule_ids=[r["rule_id"] for r in rules_to_cluster]
        ), Decimal(0)
    prompt = build_clustering_prompt(rules_to_cluster)
    result = await get_clustering_agent().run(prompt)
    output = result.output
    logfire.info(
        "Topic clustering result",
        num_rules=len(rules_to_cluster),
        topics=len(output.clusters),
        unclustered=len(output.unclustered_rule_ids),
    )
    return output, get_result_cost(result)


# ============================================================================
# Organization Logic
# ============================================================================


def get_location_key(placement: dict) -> str:
    p = placement["placement"]
    ptype = p["placement_type"]
    if ptype == "agents_md_root":
        return "root"
    elif ptype == "agents_md_dir":
        return p.get("directory", "unknown")
    elif ptype == "file":
        return f"file:{p.get('file_path', 'unknown')}"
    elif ptype == "cross_file":
        return "root"
    return "unknown"


@logfire.instrument("organize location: {location}")
async def organize_location(
    location: str,
    placements: list[dict],
    rules: dict[int, dict],
) -> tuple[OrganizedLocation, Decimal]:
    valid_rules = placements
    valid_ids = {p["rule_id"] for p in valid_rules}

    if len(valid_rules) < MIN_RULES_FOR_TOPICS:
        topic_clusters = []
        unclustered_ids = [p["rule_id"] for p in valid_rules]
        cost = Decimal(0)
    else:
        clustering_result, cost = await cluster_rules_by_topic(valid_rules)
        topic_clusters = []
        # Filter out hallucinated rule IDs from LLM output
        unclustered_ids = [
            rid for rid in clustering_result.unclustered_rule_ids if rid in valid_ids
        ]
        for cluster in clustering_result.clusters:
            cluster.rule_ids = [rid for rid in cluster.rule_ids if rid in valid_ids]
            if len(cluster.rule_ids) >= MIN_CLUSTER_SIZE:
                topic_clusters.append(cluster)
            else:
                unclustered_ids.extend(cluster.rule_ids)

    inline_rule_ids = list(dict.fromkeys(unclustered_ids))  # dedupe, preserve order

    return OrganizedLocation(
        location=location,
        inline_rules=inline_rule_ids,
        topic_references=topic_clusters,
    ), cost


async def organize_all_rules(
    placements: list[dict],
    rules: dict[int, dict],
    concurrency: int = 5,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats: StageStats | None = None,
) -> list[OrganizedLocation]:
    by_location: dict[str, list[dict]] = defaultdict(list)
    seen_per_location: dict[str, set[int]] = defaultdict(set)
    for p in placements:
        loc = get_location_key(p)
        rid = p["rule_id"]
        if rid not in seen_per_location[loc]:
            seen_per_location[loc].add(rid)
            by_location[loc].append(p)

    if progress and task_id is not None:
        progress.update(task_id, total=len(by_location), description="Organizing locations...")

    semaphore = asyncio.Semaphore(concurrency)

    async def process_location(location: str, loc_placements: list[dict]) -> OrganizedLocation:
        async with semaphore:
            result, cost = await organize_location(location, loc_placements, rules)
            if stats:
                stats.add_cost(cost)
            if progress and task_id is not None:
                progress.advance(task_id)
                if stats:
                    progress.update(task_id, cost=format_cost(stats.total_cost))
            return result

    tasks = [process_location(loc, pls) for loc, pls in sorted(by_location.items())]
    results = await asyncio.gather(*tasks)

    locations_by_key = {r.location: r for r in results}

    def total_rule_count(org: OrganizedLocation) -> int:
        return len(org.inline_rules) + sum(len(t.rule_ids) for t in org.topic_references)

    too_small = {
        org.location
        for org in results
        if not org.location.startswith("file:")
        and org.location != "root"
        and total_rule_count(org) <= MIN_RULES_FOR_DIR_FILE
    }

    def find_rollup_target(location: str) -> str:
        parts = location.rstrip("/").split("/")
        while len(parts) > 1:
            parts = parts[:-1]
            parent = "/".join(parts)
            if parent in locations_by_key and parent not in too_small:
                return parent
        return "root"

    final_results = []
    for org in results:
        if org.location.startswith("file:"):
            final_results.append(org)
            continue
        if org.location in too_small:
            target_key = find_rollup_target(org.location)
            target = locations_by_key.get(target_key)
            if target:
                existing = set(target.inline_rules)
                target.inline_rules.extend(rid for rid in org.inline_rules if rid not in existing)
        else:
            final_results.append(org)

    total_inline = sum(len(r.inline_rules) for r in final_results)
    total_topic_rules = sum(sum(len(t.rule_ids) for t in r.topic_references) for r in final_results)
    logfire.info(
        "Organization complete",
        locations=len(final_results),
        rolled_up=len(too_small),
        inline_rules=total_inline,
        topic_rules=total_topic_rules,
    )

    if stats:
        stats.set("locations", len(final_results))

    return final_results


# ============================================================================
# Preview
# ============================================================================


def preview_thresholds(placements: list[dict]) -> None:
    """Show rule counts and marginal examples at different score thresholds."""
    from rich.table import Table

    thresholds = [round(0.3 + i * 0.1, 1) for i in range(7)]  # 0.3 to 0.9

    sorted_placements = sorted(
        placements,
        key=lambda p: p.get("rule_score", p.get("signal_score", 0)),
        reverse=True,
    )

    table = Table(title="Rule counts by score threshold", expand=True, show_lines=True)
    table.add_column("Threshold", justify="right")
    table.add_column("Rules", justify="right")
    table.add_column("Marginal rules (lowest-scoring included)")

    for t in thresholds:
        passing = [
            p for p in sorted_placements if p.get("rule_score", p.get("signal_score", 0)) >= t
        ]
        count = len(passing)
        # Marginal = the 3 lowest-scoring that still pass
        marginal = passing[-3:] if len(passing) >= 3 else passing
        marginal_strs = []
        for p in marginal:
            score = p.get("rule_score", p.get("signal_score", 0))
            text = p.get("rule_text", "")
            rid = p.get("rule_id", "?")
            marginal_strs.append(f"#{rid} ({score:.2f}) {text}")
        table.add_row(
            f">= {t:.1f}",
            str(count),
            "\n".join(marginal_strs) if marginal_strs else "(none)",
        )

    console.print(table)
    console.print(f"\nTotal placements: {len(placements)}")
    console.print("Tip: use [bold]lookup <id>[/bold] to trace a rule to its source PRs")


# ============================================================================
# CLI Command
# ============================================================================


async def _async_run(
    config: RepoConfig,
    is_pipeline: bool = False,
    **kwargs,
) -> StageStats | None:
    import shutil

    output_dir = config.stage_dir("6-group")
    concurrency = kwargs.get("concurrency", 5)
    min_score = kwargs.get("min_score", 0.5)
    max_rules: int | None = kwargs.get("max_rules")
    preview = kwargs.get("preview", False)
    fresh = kwargs.get("fresh", False)

    # Input hash: detect if placements.jsonl changed
    input_hash = compute_file_hash(config.placements_path)
    hash_path = output_dir / "input_hash.txt"

    resuming = False
    if fresh:
        if output_dir.exists():
            shutil.rmtree(output_dir)
    elif output_dir.exists() and hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        if stored_hash != input_hash:
            shutil.rmtree(output_dir)
        else:
            resuming = True

    if preview:
        console.print("Loading data...")
        all_placements = load_placements(config.placements_path)
        console.print(f"Loaded {len(all_placements)} placements")
        preview_thresholds(all_placements)
        return None

    # Skip-if-done: if resuming and output exists, skip entirely
    if resuming and config.organization_path.exists():
        console.print("Group stage already complete, skipping")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(input_hash)

    if not is_pipeline:
        console.print("Loading data...")
    all_placements = load_placements(config.placements_path)
    rules = load_rules(config.rules_path)
    if not is_pipeline:
        console.print(f"Loaded {len(all_placements)} placements, {len(rules)} rules")

    placements = [
        p for p in all_placements if p.get("rule_score", p.get("signal_score", 0)) >= min_score
    ]
    if not is_pipeline and len(placements) < len(all_placements):
        console.print(f"Filtered to {len(placements)} placements (score >= {min_score})")

    if max_rules is not None and len(placements) > max_rules:
        placements.sort(key=lambda p: p.get("rule_score", p.get("signal_score", 0)), reverse=True)
        cutoff_score = placements[max_rules - 1].get(
            "rule_score", placements[max_rules - 1].get("signal_score", 0)
        )
        placements = placements[:max_rules]
        console.print(f"Capped to {max_rules} rules (score >= {cutoff_score:.2f})")

    with stage_progress("Group", is_pipeline=is_pipeline, cost_path=output_dir / "cost.json") as (
        progress,
        stats,
    ):
        stats.set("placements", len(placements))
        task = progress.add_task("Organizing...", total=len(placements))

        organized = await organize_all_rules(
            placements,
            rules,
            concurrency=concurrency,
            progress=progress,
            task_id=task,
            stats=stats,
        )

        rules_by_id = {}
        for p in placements:
            rules_by_id[p["rule_id"]] = {
                "rule_id": p["rule_id"],
                "rule_text": p["rule_text"],
                "rule_category": p["rule_category"],
                "rule_score": p.get("rule_score", p.get("signal_score", 0)),
                "placement": p["placement"],
                "unique_pr_count": p["unique_pr_count"],
            }

        total_inline = 0
        total_topics = 0
        for org in organized:
            if not org.location.startswith("file:"):
                total_inline += len(org.inline_rules)
                total_topics += sum(len(t.rule_ids) for t in org.topic_references)

        stats.set("inline_rules", total_inline)
        stats.set("topic_rules", total_topics)

        file_stats = {
            "total_placements": len(all_placements),
            "filtered_placements": len(placements),
            "min_score": min_score,
            "locations": len(organized),
            "by_location": {},
        }
        for org in organized:
            if not org.location.startswith("file:"):
                file_stats["by_location"][org.location] = {
                    "inline_rules": len(org.inline_rules),
                    "topic_clusters": len(org.topic_references),
                    "topic_rules": sum(len(t.rule_ids) for t in org.topic_references),
                }
        file_stats["totals"] = {"inline_rules": total_inline, "topic_rules": total_topics}

        output = Stage4Output(locations=organized, rules_by_id=rules_by_id, stats=file_stats)
        output_path = output_dir / "organized_rules.json"
        with open(output_path, "w") as f:
            json.dump(output.model_dump(), f, indent=2)

        num_placements = int(stats.get("placements", 0))
        num_locations = int(stats.get("locations", 0))
        stats.detail = f"{num_placements} rules \u2192 {num_locations} locations | {total_inline} inline, {total_topics} in topics"
        return stats


def _run(config: RepoConfig, is_pipeline: bool = False, **kwargs) -> StageStats | None:
    return asyncio.run(_async_run(config, is_pipeline=is_pipeline, **kwargs))


def group(
    ctx: typer.Context,
    min_score: float = typer.Option(0.5, help="Minimum rule score to include (0-1)"),
    max_rules: int | None = typer.Option(
        None, "--max-rules", help="Maximum number of rules to include (top-scored)"
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Show rule counts and examples at different score thresholds, then exit",
    ),
    concurrency: int = typer.Option(5, help="Number of parallel LLM requests"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Organize rules by topic."""
    config: RepoConfig = ctx.obj["config"]
    _run(
        config,
        min_score=min_score,
        max_rules=max_rules,
        preview=preview,
        concurrency=concurrency,
        fresh=fresh,
    )
