"""Stage 1: Extract changes and potential rules from PR review comments."""

from __future__ import annotations

import asyncio
import json
import random
from decimal import Decimal
from pathlib import Path

import logfire
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rich.progress import Progress, TaskID

from braindump.config import RepoConfig, compute_multi_file_hash
from braindump.progress import StageStats, console, format_cost, get_result_cost, stage_progress

# ============================================================================
# Models
# ============================================================================


class ExtractedChange(BaseModel):
    change_description: str = Field(
        description="What specific change was requested - factual, concrete"
    )
    file_path: str | None = Field(
        default=None, description="The file path where this change applies"
    )
    code_context: str | None = Field(
        default=None, description="Relevant code snippet showing the issue"
    )


class PotentialRule(BaseModel):
    generalization: str = Field(description="The rule text - actionable, clear guidance")
    motivation: str = Field(description="WHY this rule matters - the underlying principle")
    scope: str | None = Field(
        default=None,
        description="Where this rule applies: 'global', a directory path, a file pattern, or None if unclear",
    )
    source_change_index: int = Field(
        description="Index into the changes list that this rule generalizes from"
    )


class RejectionReason(BaseModel):
    reason: str = Field(
        description="One of: 'question', 'acknowledgment', 'pr_workflow', 'unclear'"
    )
    explanation: str = Field(description="Brief explanation of why this comment is not actionable")


class CommentExtraction(BaseModel):
    is_actionable: bool = Field(description="Whether comment contains actionable guidance")
    rejection: RejectionReason | None = Field(default=None, description="If not actionable, why")
    changes: list[ExtractedChange] = Field(default_factory=list)
    potential_rules: list[PotentialRule] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    comment_id: int
    pr_number: int
    author: str
    is_actionable: bool
    rejection_reason: str | None = None
    rejection_explanation: str | None = None
    changes: list[dict] = Field(default_factory=list)
    potential_rules: list[dict] = Field(default_factory=list)
    source: dict


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are extracting coding guidance from PR review comments.

## Your Goal

For each comment, extract:
1. **Changes**: What specific changes were requested?
2. **Potential Rules**: What generalizable rules can be derived?

## Classification

**Actionable comments contain:**
- Suggestions for how code should be written
- Naming conventions
- Documentation requirements
- API design preferences
- Testing patterns
- Error handling guidance

**Non-actionable (reject with reason):**
- `question`: "Why?", "What about X?", "Could we also...?"
- `acknowledgment`: "Done", "Thanks", "LGTM", "Fixed", "Good point"
- `pr_workflow`: "Split into separate PR", "Let's discuss offline", "Will do in follow-up"
- `unclear`: References without enough context to understand

## Extracting Changes

For each distinct change requested:
- Describe WHAT was asked for (not why)
- Note the file path if clear from context
- Include relevant code snippet if available

A comment may request multiple distinct changes.

## Generating Potential Rules

For each change, generate rules at these DISTINCT levels:

### Level 1: THE PATTERN (Required)
Generalize the specific *instance* but preserve the project *pattern*. Strip entity names (e.g., "OpenAI" → "provider") but keep project-specific conventions, directory structures, and file patterns.

- BAD: "OpenAI provider docs should list supported features" (entity-specific instance)
- BAD: "Document provider-specific support in feature documentation" (too generic — lost project context)
- GOOD: "Each provider doc in `docs/providers/` should list which features (streaming, tools, etc.) are supported"

- BAD: "Put the Anthropic profile in profiles/anthropic.py" (specific instance)
- BAD: "Organize files by domain" (too generic)
- GOOD: "Place model profiles in `profiles/{provider}.py` with a standard interface"

- BAD: "Test organization tests should go in test_organization.py" (entity-specific)
- GOOD: "Add tests to existing `test_{module}.py` files rather than creating new scattered files"

### Level 2: SPECIFIC INSTANCE (Only if truly entity-specific)
Only include if this rule ONLY applies to this specific entity and would NOT apply elsewhere.
Skip this level if the pattern applies generally.

### CRITICAL: What NOT to Generate

1. **REPHRASING**: Don't generate "use X", "prefer X", "always X" as separate rules. Pick one.

2. **COMPLEMENTARY HALVES**: Don't split "do X when Y, don't do X when Z" into two rules.
   - BAD: "Extract helpers for duplication" + "Avoid single-use helpers"
   - GOOD: "Create helpers only when there's actual reuse (2+ call sites)"

3. **ENTITY-SPECIFIC VERSIONS** of general patterns:
   - BAD: "Tests for agents should...", "Tests for models should...", "Tests for tools should..."
   - GOOD: "Consolidate tests for related functionality"

4. **CONTRADICTORY GENERALIZATIONS**: If the feedback is context-dependent, note the context.
   - BAD: "Prefer isinstance()" + "Prefer discriminator properties"
   - GOOD: "Use isinstance() for type narrowing in static analysis; use discriminators for runtime polymorphism"

5. **INVERSES**: Don't generate both "do X" and "avoid not-X"
   - BAD: "Use integration tests" + "Avoid unit tests when integration covers it"
   - GOOD: Pick one framing

### Example

For "use isinstance(part, TextPart) instead of part.part_kind == 'text'":

Rule 1 (THE PATTERN - required):
- generalization: "Prefer isinstance() checks over string comparison on discriminator fields for type verification"
- motivation: "isinstance() provides type safety, enables proper type narrowing for static analysis, and is more refactor-safe"
- scope: "global"

That's it. ONE rule. The entity-specific version ("When filtering message parts...") is just a restatement. The abstract principle ("Use type-safe constructs...") is too vague to be actionable. One good rule is better than three overlapping ones.

## Important Notes

- A single comment may produce multiple changes and multiple rules per change
- Rules at different abstraction levels from the same change should reference the same source_change_index
- Not every comment needs rules at all levels - use what's natural
- If the comment just says "same here" or references another comment, look at thread context
"""


# ============================================================================
# Lazy Agent
# ============================================================================

_agent: Agent[None, CommentExtraction] | None = None


def get_agent() -> Agent[None, CommentExtraction]:
    global _agent
    if _agent is None:
        from braindump.config import make_model

        _agent = Agent(
            make_model(),
            output_type=CommentExtraction,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


# ============================================================================
# Comment Processing
# ============================================================================


def load_comments(review_dir: Path) -> list[dict]:
    all_comments = []
    for filepath in sorted(review_dir.glob("*.json")):
        with open(filepath) as f:
            comments = json.load(f)
            pr_number = int(filepath.stem)
            for comment in comments:
                comment["_pr_number"] = pr_number
                all_comments.append(comment)
    return all_comments


def filter_bots(comments: list[dict]) -> list[dict]:
    """Exclude comments from bot accounts."""
    return [
        c
        for c in comments
        if c.get("user", {}).get("type") != "Bot"
        and not c.get("user", {}).get("login", "").endswith("[bot]")
    ]


def filter_by_author(comments: list[dict], authors: set[str] | None) -> list[dict]:
    if authors is None:
        return comments
    return [c for c in comments if c.get("user", {}).get("login") in authors]


def build_thread_map(comments: list[dict]) -> dict[int, list[dict]]:
    by_id: dict[int, dict] = {c["id"]: c for c in comments}
    thread_map: dict[int, list[dict]] = {}
    for comment in comments:
        comment_id = comment["id"]
        chain = []
        current = comment
        while "in_reply_to_id" in current and current.get("in_reply_to_id"):
            parent_id = current["in_reply_to_id"]
            if parent_id in by_id:
                parent = by_id[parent_id]
                chain.insert(0, parent)
                current = parent
            else:
                break
        thread_map[comment_id] = chain
    return thread_map


def format_thread_context(thread: list[dict]) -> str:
    if not thread:
        return "No prior thread context."
    parts = []
    for i, comment in enumerate(thread):
        user = comment.get("user", {}).get("login", "unknown")
        body = comment.get("body", "")
        parts.append(f"[{i + 1}] @{user}: {body}")
    return "\n".join(parts)


def build_comment_prompt(comment: dict, thread_context: str) -> str:
    body = comment.get("body", "")
    path = comment.get("path", "unknown")
    diff_hunk = comment.get("diff_hunk", "")
    return f"""## Comment to Extract From

**File**: `{path}`

**Comment Body**:
{body}

**Diff Context**:
```
{diff_hunk}
```

**Thread Context**:
{thread_context}

---

Extract changes and potential rules from this comment. If not actionable, explain why.
"""


# ============================================================================
# Checkpointing
# ============================================================================


def load_checkpoint(checkpoint_path: Path) -> set[int]:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
            return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(checkpoint_path: Path, processed_ids: set[int]):
    with open(checkpoint_path, "w") as f:
        json.dump({"processed_ids": sorted(processed_ids)}, f, indent=2)


def append_result(output_path: Path, result: ExtractionOutput):
    with open(output_path, "a") as f:
        f.write(result.model_dump_json() + "\n")


def update_stats(stats_path: Path, result: ExtractionOutput):
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {
            "total_processed": 0,
            "actionable": 0,
            "non_actionable": 0,
            "by_rejection_reason": {},
            "total_changes": 0,
            "total_potential_rules": 0,
            "rules_per_comment": [],
        }
    stats["total_processed"] += 1
    if result.is_actionable:
        stats["actionable"] += 1
        stats["total_changes"] += len(result.changes)
        stats["total_potential_rules"] += len(result.potential_rules)
        stats["rules_per_comment"].append(len(result.potential_rules))
    else:
        stats["non_actionable"] += 1
        reason = result.rejection_reason or "unknown"
        stats["by_rejection_reason"][reason] = stats["by_rejection_reason"].get(reason, 0) + 1
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


# ============================================================================
# Processing
# ============================================================================


@logfire.instrument("process comment", extract_args={"comment": ["id", "_pr_number"]})
async def process_comment(
    comment: dict, thread_map: dict[int, list[dict]]
) -> tuple[CommentExtraction, Decimal]:
    comment_id = comment["id"]
    thread = thread_map.get(comment_id, [])
    thread_context = format_thread_context(thread)
    prompt = build_comment_prompt(comment, thread_context)
    result = await get_agent().run(prompt)
    return result.output, get_result_cost(result)


async def process_all(
    comments: list[dict],
    thread_map: dict[int, list[dict]],
    output_dir: Path,
    limit: int | None = None,
    concurrency: int = 10,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats: StageStats | None = None,
    is_pipeline: bool = False,
):
    output_path = output_dir / "extractions.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    stats_path = output_dir / "stats.json"

    processed_ids = load_checkpoint(checkpoint_path)
    to_process = [c for c in comments if c["id"] not in processed_ids]
    if limit:
        to_process = to_process[:limit]

    if not is_pipeline:
        console.print(
            f"Processing {len(to_process)} comments (already processed: {len(processed_ids)})"
        )
    logfire.info(
        "Starting comment processing",
        total=len(to_process),
        already_processed=len(processed_ids),
        concurrency=concurrency,
    )

    if progress and task_id is not None:
        progress.update(task_id, total=len(to_process), description="Extracting...")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def process_one(
        comment: dict,
    ) -> tuple[dict, ExtractionOutput | None, str | None, Decimal]:
        comment_id = comment["id"]
        pr_number = comment.get("_pr_number", 0)
        async with semaphore:
            try:
                result, cost = await process_comment(comment, thread_map)
                logfire.info(
                    "Comment extracted",
                    comment_id=comment_id,
                    pr_number=pr_number,
                    is_actionable=result.is_actionable,
                    num_changes=len(result.changes) if result.is_actionable else 0,
                    num_rules=len(result.potential_rules) if result.is_actionable else 0,
                    rejection_reason=result.rejection.reason if result.rejection else None,
                )
                author = comment.get("user", {}).get("login", "unknown")
                output = ExtractionOutput(
                    comment_id=comment_id,
                    pr_number=pr_number,
                    author=author,
                    is_actionable=result.is_actionable,
                    rejection_reason=result.rejection.reason if result.rejection else None,
                    rejection_explanation=result.rejection.explanation
                    if result.rejection
                    else None,
                    changes=[c.model_dump() for c in result.changes],
                    potential_rules=[r.model_dump() for r in result.potential_rules],
                    source={
                        "body": comment.get("body", ""),
                        "path": comment.get("path", ""),
                        "diff_hunk": comment.get("diff_hunk", ""),
                        "thread_context": format_thread_context(thread_map.get(comment_id, [])),
                    },
                )
                return comment, output, None, cost
            except Exception as e:
                logfire.error(
                    "Comment extraction failed",
                    comment_id=comment_id,
                    pr_number=pr_number,
                    error=str(e),
                )
                return comment, None, str(e), Decimal(0)

    batch_size = concurrency * 2
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        with logfire.span(
            "extract batch {batch_num}/{total_batches}",
            batch_num=batch_num,
            total_batches=total_batches,
            batch_size=len(batch),
        ):
            tasks = [process_one(c) for c in batch]
            results = await asyncio.gather(*tasks)
            async with write_lock:
                for comment, output, error, cost in results:
                    comment_id = comment["id"]
                    if stats:
                        stats.add_cost(cost)
                    if error:
                        if stats:
                            stats.inc("errors")
                        continue
                    if output:
                        append_result(output_path, output)
                        processed_ids.add(comment_id)
                        update_stats(stats_path, output)
                        if stats:
                            if output.is_actionable:
                                stats.inc("actionable")
                                stats.inc("rules", len(output.potential_rules))
                            else:
                                stats.inc("rejected")
                    if progress and task_id is not None:
                        progress.advance(task_id)
                        if stats:
                            progress.update(task_id, cost=format_cost(stats.total_cost))
                save_checkpoint(checkpoint_path, processed_ids)


# ============================================================================
# CLI Command
# ============================================================================


def _run(
    config: RepoConfig,
    authors: str = "all",
    limit: int | None = None,
    random_sample: bool = False,
    seed: int = 42,
    prs: str | None = None,
    concurrency: int = 10,
    is_pipeline: bool = False,
    fresh: bool = False,
) -> StageStats | None:
    """Programmatic entry point for extract stage."""
    import shutil

    review_dir = config.review_comments_dir
    output_dir = config.stage_dir("2-extract")

    # Input hash: detect if download data changed
    input_files = sorted(review_dir.glob("*.json"))
    if input_files:
        input_hash = compute_multi_file_hash(input_files)
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
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not is_pipeline:
        console.print("Loading comments...")
    all_comments = load_comments(review_dir)
    if not is_pipeline:
        console.print(f"Loaded {len(all_comments)} total comments")

    all_comments = filter_bots(all_comments)
    if not is_pipeline:
        console.print(f"After excluding bots: {len(all_comments)} comments")

    if prs:
        pr_numbers = set(int(p.strip()) for p in prs.split(","))
        all_comments = [c for c in all_comments if c.get("_pr_number") in pr_numbers]
        if not is_pipeline:
            console.print(
                f"Filtered to {len(all_comments)} comments from PRs: {sorted(pr_numbers)}"
            )

    author_set = None if authors == "all" else {a.strip() for a in authors.split(",")}
    if not is_pipeline and author_set:
        console.print(f"Filtering to {', '.join(sorted(author_set))} comments...")
    filtered_comments = filter_by_author(all_comments, author_set)
    if not is_pipeline:
        console.print(f"Found {len(filtered_comments)} comments to process")

    if random_sample and limit:
        random.seed(seed)
        if limit < len(filtered_comments):
            filtered_comments = random.sample(filtered_comments, limit)
            if not is_pipeline:
                console.print(f"Randomly sampled {len(filtered_comments)} comments (seed={seed})")

    thread_map = build_thread_map(all_comments)

    with stage_progress("Extract", is_pipeline=is_pipeline, cost_path=output_dir / "cost.json") as (
        progress,
        stats,
    ):
        stats.set("comments", len(filtered_comments))
        task = progress.add_task("Extracting...", total=len(filtered_comments))
        asyncio.run(
            process_all(
                filtered_comments,
                thread_map,
                output_dir,
                limit=limit if not random_sample else None,
                concurrency=concurrency,
                progress=progress,
                task_id=task,
                stats=stats,
                is_pipeline=is_pipeline,
            )
        )

        comments = int(stats.get("comments", 0))
        actionable = int(stats.get("actionable", 0))
        rejected = int(stats.get("rejected", 0))
        rules = int(stats.get("rules", 0))
        stats.detail = f"{comments} comments \u2192 {actionable} actionable, {rejected} rejected \u2192 {rules} generalizations"
        return stats


def extract(
    ctx: typer.Context,
    authors: str = typer.Option(
        "all", help="Filter to specific author(s), comma-separated, or 'all'"
    ),
    limit: int | None = typer.Option(None, help="Limit number of comments to process"),
    random_sample: bool = typer.Option(False, "--random", help="Randomly sample comments"),
    seed: int = typer.Option(42, help="Random seed for reproducible sampling"),
    prs: str | None = typer.Option(None, help="Comma-separated list of PR numbers"),
    concurrency: int = typer.Option(10, help="Number of parallel LLM requests"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Extract potential rules from review comments."""
    config: RepoConfig = ctx.obj["config"]

    if authors == "all":
        prompted = typer.prompt("Filter to author login (or 'all' for everyone)", default="all")
        authors = prompted

    _run(
        config,
        authors=authors,
        limit=limit,
        random_sample=random_sample,
        seed=seed,
        prs=prs,
        concurrency=concurrency,
        fresh=fresh,
    )
