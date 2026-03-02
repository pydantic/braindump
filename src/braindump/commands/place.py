"""Stage 3: Determine optimal placement for extracted rules."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

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

PlacementType = Literal["agents_md_root", "agents_md_dir", "file", "cross_file"]


class RulePlacement(BaseModel):
    placement_type: PlacementType
    directory: str | None = None
    file_path: str | None = None
    target_element: str | None = None
    source_files: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class PlacementOutput(BaseModel):
    rule_id: int
    rule_text: str
    rule_category: str
    original_scope: str
    rule_score: float
    unique_pr_count: int
    placement: RulePlacement
    source_file_patterns: list[str]
    mentioned_entities: list[str]
    evidence_summary: str


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are deciding where coding rules should be documented so that AI coding assistants (like Claude Code) will find and apply them.

## Placement Options

1. **agents_md_root** - `AGENTS.md` at repository root
   - For truly global rules that apply everywhere

2. **agents_md_dir** - `<directory>/AGENTS.md`
   - For rules specific to a directory/module
   - Specify the directory path (e.g., "src/models", "tests", "docs")

3. **file** - Rule about a specific file or element within it
   - Specify `file_path` for the file
   - Optionally specify `target_element` for a specific class, function, or constant

4. **cross_file** - Relationship between files
   - For rules about keeping files in sync
   - Specify source_files (what triggers) and target_files (what must update)

## Decision Criteria

1. **Scope breadth**: Multiple directories → global; Single directory → directory-scoped
2. **Entity patterns**: Specific entities → file-level; General patterns → higher-level
3. **Rule nature**: Behavioral → global; Relationship → cross_file; Module-specific → agents_md_dir
4. **Practical findability**: Where would an AI assistant look for this rule?

## Important Guidelines

- Prefer broader placement when uncertain (agents_md_root over agents_md_dir)
- Use cross_file ONLY when there's clear evidence of file coordination
- Directory paths should be relative to repo root
- For file patterns, use glob-style patterns
"""


# ============================================================================
# Lazy Agent
# ============================================================================

_agent: Agent[None, RulePlacement] | None = None


def get_agent() -> Agent[None, RulePlacement]:
    global _agent
    if _agent is None:
        from braindump.config import make_model

        _agent = Agent(
            make_model(),
            output_type=RulePlacement,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


# ============================================================================
# Data Loading
# ============================================================================


def load_rules(rules_path: Path) -> list[dict]:
    rules = []
    with open(rules_path) as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))
    return rules


def load_extractions(extractions_path: Path) -> dict[int, dict]:
    extractions = {}
    with open(extractions_path) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                extractions[c["comment_id"]] = c
    return extractions


def gather_evidence(rule: dict, extractions: dict[int, dict]) -> dict:
    source_files = []
    mentioned_entities = []
    context_entities = []
    related_files = []
    keywords = []
    themes = []

    for cid in rule.get("source_comments", []):
        if cid not in extractions:
            continue
        c = extractions[cid]
        path = c.get("source", {}).get("path", "")
        if path:
            source_files.append(path)
        if "themes" in c:
            themes.extend(c.get("themes", []))
        if "keywords" in c:
            keywords.extend(c.get("keywords", []))
        if "mentioned_entities" in c:
            mentioned_entities.extend(c.get("mentioned_entities", []))
        if "context_entities" in c:
            context_entities.extend(c.get("context_entities", []))
        if "related_files" in c:
            related_files.extend(c.get("related_files", []))
        for change in c.get("changes", []):
            change_path = change.get("file_path")
            if change_path:
                source_files.append(change_path)
        for pr_rule in c.get("potential_rules", []):
            scope = pr_rule.get("scope")
            if scope and scope not in ("global", None):
                related_files.append(scope)

    source_files = list(dict.fromkeys(source_files))
    mentioned_entities = list(dict.fromkeys(mentioned_entities))
    context_entities = list(dict.fromkeys(context_entities))
    related_files = list(dict.fromkeys(related_files))

    dir_counts: dict[str, int] = {}
    for f in source_files:
        parts = f.split("/")
        dir_key = parts[0] if parts else "root"
        dir_counts[dir_key] = dir_counts.get(dir_key, 0) + 1

    return {
        "source_files": source_files,
        "mentioned_entities": mentioned_entities,
        "context_entities": context_entities,
        "related_files": related_files,
        "keywords": list(dict.fromkeys(keywords)),
        "themes": list(dict.fromkeys(themes)),
        "dir_counts": dir_counts,
        "num_source_comments": len(rule.get("source_comments", [])),
    }


def build_placement_prompt(rule: dict, evidence: dict) -> str:
    dir_summary = ", ".join(
        f"{d}: {c}" for d, c in sorted(evidence["dir_counts"].items(), key=lambda x: -x[1])
    )
    sample_files = evidence["source_files"][:10]
    return f"""## Rule to Place

**Rule text**: {rule["text"]}

**Category**: {rule["category"]}
**Original scope**: {rule["scope"]}
**Score**: {rule.get("rule_score", rule.get("adjusted_confidence", 0)):.2f}
**Validated across**: {rule["unique_pr_count"]} PRs, {evidence["num_source_comments"]} comments

**Reason**: {rule.get("reason", "N/A")}

## Evidence from Source Comments

**Directory distribution**: {dir_summary}

**Sample source files** (where this rule was applied):
{chr(10).join(f"- {f}" for f in sample_files)}

**Mentioned entities**:
{chr(10).join(f"- {e}" for e in evidence["mentioned_entities"][:10]) or "(none)"}

**Context entities** (inferred from diff):
{chr(10).join(f"- {e}" for e in evidence["context_entities"][:10]) or "(none)"}

**Related files mentioned** (files that needed coordinated updates):
{chr(10).join(f"- {f}" for f in evidence["related_files"][:5]) or "(none)"}

**Keywords**: {", ".join(evidence["keywords"][:15])}

**Themes**: {", ".join(evidence["themes"][:5])}

---

Based on this evidence, determine the optimal placement for this rule.
"""


# ============================================================================
# Processing
# ============================================================================


@logfire.instrument("place rule")
async def place_rule(rule: dict, evidence: dict) -> tuple[RulePlacement, Decimal]:
    prompt = build_placement_prompt(rule, evidence)
    result = await get_agent().run(prompt)
    return result.output, get_result_cost(result)


async def place_all_rules(
    rules: list[dict],
    extractions: dict[int, dict],
    limit: int | None = None,
    concurrency: int = 10,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats: StageStats | None = None,
    done_rule_ids: set[int] | None = None,
    output_path: Path | None = None,
    is_pipeline: bool = False,
) -> list[PlacementOutput]:
    if limit:
        rules = rules[:limit]

    if done_rule_ids is None:
        done_rule_ids = set()

    all_rules = rules
    rules_to_process = [r for r in rules if r["rule_id"] not in done_rule_ids]

    if not is_pipeline:
        console.print(
            f"Placing {len(rules_to_process)} rules (already placed: {len(done_rule_ids)})"
        )

    if progress and task_id is not None:
        progress.update(
            task_id,
            total=len(all_rules),
            completed=len(done_rule_ids),
            description="Placing rules...",
        )

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def place_one(rule: dict) -> PlacementOutput | None:
        evidence = gather_evidence(rule, extractions)
        evidence_summary = (
            f"{len(evidence['source_files'])} files across {len(evidence['dir_counts'])} dirs"
        )
        async with semaphore:
            try:
                placement, cost = await place_rule(rule, evidence)
                if stats:
                    stats.add_cost(cost)
                logfire.info(
                    "Rule placed",
                    rule_id=rule["rule_id"],
                    placement_type=placement.placement_type,
                    directory=placement.directory,
                    confidence=placement.confidence,
                )
                output = PlacementOutput(
                    rule_id=rule["rule_id"],
                    rule_text=rule["text"],
                    rule_category=rule["category"],
                    original_scope=rule["scope"],
                    rule_score=rule.get("rule_score", rule.get("adjusted_confidence", 0)),
                    unique_pr_count=rule["unique_pr_count"],
                    placement=placement,
                    source_file_patterns=evidence["source_files"][:5],
                    mentioned_entities=evidence["mentioned_entities"][:5],
                    evidence_summary=evidence_summary,
                )
                if output_path:
                    async with write_lock:
                        with open(output_path, "a") as f:
                            f.write(output.model_dump_json() + "\n")
                if stats:
                    stats.inc(placement.placement_type)
                return output
            except Exception as e:
                logfire.error("Rule placement failed", rule_id=rule["rule_id"], error=str(e))
                if stats:
                    stats.inc("errors")
                return None
            finally:
                if progress and task_id is not None:
                    progress.advance(task_id)
                    if stats:
                        progress.update(task_id, cost=format_cost(stats.total_cost))

    tasks = [place_one(r) for r in rules_to_process]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ============================================================================
# CLI Command
# ============================================================================


def _load_existing_placements(path: Path) -> tuple[list[PlacementOutput], set[int]]:
    """Load existing placements from a previous run. Returns (placements, done_rule_ids)."""
    placements: list[PlacementOutput] = []
    done_ids: set[int] = set()
    if not path.exists():
        return placements, done_ids
    with open(path) as f:
        for line in f:
            if line.strip():
                p = PlacementOutput.model_validate_json(line)
                placements.append(p)
                done_ids.add(p.rule_id)
    return placements, done_ids


async def _async_run(
    config: RepoConfig,
    is_pipeline: bool = False,
    **kwargs,
) -> StageStats | None:
    import shutil

    output_dir = config.stage_dir("5-place")
    rules_path = config.rules_deduped_path
    extractions_path = config.extractions_path
    limit = kwargs.get("limit", 0)
    min_score = kwargs.get("min_score", 0.5)
    concurrency = kwargs.get("concurrency", 10)
    fresh = kwargs.get("fresh", False)

    # Input hash: detect if deduped rules changed
    input_hash = compute_file_hash(rules_path)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(input_hash)

    if not is_pipeline:
        console.print("Loading data...")
    rules = load_rules(rules_path)
    extractions = load_extractions(extractions_path)
    if not is_pipeline:
        console.print(f"Loaded {len(rules)} rules, {len(extractions)} extractions")

    filtered = [
        r for r in rules if r.get("rule_score", r.get("adjusted_confidence", 0)) >= min_score
    ]
    filtered.sort(key=lambda r: r.get("rule_score", r.get("adjusted_confidence", 0)), reverse=True)
    if not is_pipeline:
        console.print(f"Filtered to {len(filtered)} rules (score >= {min_score})")

    # Load existing placements for resume
    output_path = output_dir / "placements.jsonl"
    existing_placements: list[PlacementOutput] = []
    done_rule_ids: set[int] = set()
    if resuming:
        existing_placements, done_rule_ids = _load_existing_placements(output_path)

    with stage_progress("Place", is_pipeline=is_pipeline, cost_path=output_dir / "cost.json") as (
        progress,
        stats,
    ):
        stats.set("rules", len(filtered))
        task = progress.add_task("Placing rules...", total=len(filtered))
        new_outputs = await place_all_rules(
            filtered,
            extractions,
            limit=limit or None,
            concurrency=concurrency,
            progress=progress,
            task_id=task,
            stats=stats,
            done_rule_ids=done_rule_ids,
            output_path=output_path,
            is_pipeline=is_pipeline,
        )
        all_outputs = existing_placements + new_outputs
        stats.set("placed", len(all_outputs))

        # Save stats with filter params
        by_type: dict[str, int] = {}
        for o in all_outputs:
            ptype = o.placement.placement_type
            by_type[ptype] = by_type.get(ptype, 0) + 1
        stats_path = output_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "input_rules": len(rules),
                    "filtered_rules": len(filtered),
                    "placed": len(all_outputs),
                    "min_score": min_score,
                    "by_type": by_type,
                },
                f,
                indent=2,
            )

        type_str = ", ".join(f"{t}: {n}" for t, n in sorted(by_type.items(), key=lambda x: -x[1]))
        stats.detail = f"{len(filtered)} \u2192 {len(all_outputs)} placed (score \u2265 {min_score}) | {type_str}"
        return stats


def _run(config: RepoConfig, is_pipeline: bool = False, **kwargs) -> StageStats | None:
    return asyncio.run(_async_run(config, is_pipeline=is_pipeline, **kwargs))


def place(
    ctx: typer.Context,
    limit: int = typer.Option(0, help="Limit number of rules to process (0 for all)"),
    min_score: float = typer.Option(
        0.5,
        help="Minimum rule score floor for placement (0-1). Use group --min-score to filter output.",
    ),
    concurrency: int = typer.Option(10, help="Number of parallel LLM requests"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Determine rule placement locations."""
    config: RepoConfig = ctx.obj["config"]
    _run(config, limit=limit, min_score=min_score, concurrency=concurrency, fresh=fresh)
