"""Stage 2: Cluster generalizations and extract validated rules."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Literal

import logfire
import numpy as np
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Embedder
from rich.progress import Progress, TaskID

from braindump.config import RepoConfig, compute_file_hash
from braindump.progress import StageStats, console, format_cost, get_result_cost, stage_progress

# ============================================================================
# Models
# ============================================================================

Category = Literal[
    "naming",
    "documentation",
    "api_design",
    "typing",
    "testing",
    "error_handling",
    "deprecation",
    "code_style",
    "imports",
    "exports",
    "config",
    "dependencies",
]

Scope = Literal["global", "cli", "tests", "docs", "examples"]


class ValidatedRule(BaseModel):
    text: str = Field(description="The generalized rule statement - actionable, clear")
    reason: str = Field(description="WHY this rule exists - the underlying principle")
    confidence: float = Field(ge=0.0, le=1.0)
    category: Category
    scope: Scope | str
    example_bad: str | None = None
    example_good: str | None = None
    related_entities: list[str] = Field(default_factory=list)
    source_comment_ids: list[int] = Field(
        default_factory=list,
        description="IDs of the source comments (shown as 'comment #ID' in the input) that this rule is derived from",
    )


class ClusterAnalysis(BaseModel):
    cluster_coherence: float = Field(ge=0.0, le=1.0)
    common_pattern: str
    rules: list[ValidatedRule] = Field(default_factory=list)
    rejection_reason: str | None = None


class RuleOutput(BaseModel):
    rule_id: int
    cluster_id: int
    text: str
    reason: str
    confidence: float
    rule_score: float
    category: str
    scope: str
    example_bad: str | None = None
    example_good: str | None = None
    related_entities: list[str] = Field(default_factory=list)
    source_comments: list[int]
    source_prs: list[int]
    unique_pr_count: int
    cluster_coherence: float
    common_pattern: str


# Multiplier for how many unique PRs the rule's evidence spans
PR_FACTORS: dict[int, float] = {1: 0.6, 2: 0.85, 3: 0.95}
PR_FACTOR_DEFAULT: float = 1.0  # 4+ PRs


def calculate_rule_score(confidence: float, unique_prs: int) -> float:
    pr_factor = PR_FACTORS.get(unique_prs, PR_FACTOR_DEFAULT)
    return min(1.0, confidence * pr_factor)


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are extracting validated coding rules from clusters of related GENERALIZATIONS.

## Context

These generalizations were extracted from PR review comments and clustered by semantic similarity.
Each generalization came from a specific comment and has:
- The rule text (what was generalized)
- The motivation (why it matters)
- An optional scope suggestion
- The source comment context

## Your Task

1. Assess if the generalizations truly share a common pattern (cluster_coherence score)
2. Identify what that pattern is (common_pattern)
3. Extract ONE well-formed rule (rarely more) that captures the pattern
4. Detect and handle convergence, complementary halves, and contradictions

## Detecting and Handling Patterns

### Convergence (MOST COMMON)
When a cluster contains multiple rules that are rephrasing or entity-specific versions of one principle:
→ Output ONE rule that captures the general principle.

### Complementary Halves
When a cluster contains rules that are complementary conditions:
→ Merge into ONE rule.

### Inverses
When a cluster contains positive and negative framings:
→ Pick ONE framing (usually positive).

### Contradictions
When a cluster contains rules that genuinely contradict:
→ Either output one rule noting the context, or set rules=[] with rejection_reason.

### Entity-Specific Patterns
If the cluster discusses specific entities (project classes, directory conventions, config patterns):
→ Keep the project-specific framing — these are the most valuable rules for an AGENTS.md.
→ Only generalize away entities that are truly incidental (e.g., two unrelated variable names).

## Output Guidelines

**Prefer ONE rule per cluster.** Multiple rules only when genuinely distinct sub-patterns exist.

**Source attribution**: For each rule, include `source_comment_ids` with the comment IDs (from "comment #ID" in the input) of the generalizations that directly support that specific rule. A comment can appear in multiple rules if it supports both.

## Confidence Levels

- **High (0.8+)**: Generalizations clearly converge on the same pattern
- **Medium (0.5-0.8)**: Related but abstraction is uncertain
- **Low (<0.5)**: Cluster may be coincidental, rule is speculative

## Categories (ONLY these)

naming, documentation, api_design, typing, testing, error_handling, deprecation, code_style, imports, exports, config, dependencies

## When NOT to Extract a Rule

Set rules=[] and provide rejection_reason if:
- Generalizations are only superficially related
- The pattern is too specific to one PR/feature
- Generalizations genuinely contradict each other
- Not enough signal to determine the right abstraction level
"""


# ============================================================================
# Lazy Agent / Embedder
# ============================================================================

_agent: Agent[None, ClusterAnalysis] | None = None
_embedder: Embedder | None = None


def get_agent() -> Agent[None, ClusterAnalysis]:
    global _agent
    if _agent is None:
        from braindump.config import make_model

        _agent = Agent(
            make_model(),
            output_type=ClusterAnalysis,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        from braindump.config import make_embedding_model

        _embedder = Embedder(make_embedding_model())
    return _embedder


# ============================================================================
# Data Structures
# ============================================================================


class GeneralizationItem:
    def __init__(
        self,
        generalization: str,
        motivation: str,
        scope: str | None,
        comment_id: int,
        pr_number: int,
        source_change_index: int,
        source_context: dict,
    ):
        self.generalization = generalization
        self.motivation = motivation
        self.scope = scope
        self.comment_id = comment_id
        self.pr_number = pr_number
        self.source_change_index = source_change_index
        self.source_context = source_context
        self.embedding: list[float] | None = None


# ============================================================================
# Data Loading
# ============================================================================

BOT_AUTHORS = {"Copilot"}


def is_bot_author(author: str) -> bool:
    """Check if an author login belongs to a bot account."""
    return author.endswith("[bot]") or author in BOT_AUTHORS


def load_extractions(input_path: Path) -> list[dict]:
    extractions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                ext = json.loads(line)
                if is_bot_author(ext.get("author", "")):
                    continue
                extractions.append(ext)
    return extractions


def extract_generalizations(extractions: list[dict]) -> list[GeneralizationItem]:
    items = []
    for ext in extractions:
        if not ext.get("is_actionable", False):
            continue
        comment_id = ext["comment_id"]
        pr_number = ext["pr_number"]
        source_context = ext.get("source", {})
        for rule in ext.get("potential_rules", []):
            gen = rule.get("generalization", "")
            if not gen:
                continue
            item = GeneralizationItem(
                generalization=gen,
                motivation=rule.get("motivation", ""),
                scope=rule.get("scope"),
                comment_id=comment_id,
                pr_number=pr_number,
                source_change_index=rule.get("source_change_index", 0),
                source_context=source_context,
            )
            items.append(item)
    return items


# ============================================================================
# Embedding Cache
# ============================================================================


def _save_embeddings_cache(items: list[GeneralizationItem], output_dir: Path) -> None:
    """Save embeddings and their keys to disk for resume."""
    keys = [item.generalization for item in items]
    embeddings = np.array([item.embedding for item in items if item.embedding is not None])
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=embeddings)
    with open(output_dir / "embeddings_keys.json", "w") as f:
        json.dump(keys, f)


def _load_embeddings_cache(items: list[GeneralizationItem], output_dir: Path) -> bool:
    """Load cached embeddings if keys match. Returns True if cache hit."""
    npz_path = output_dir / "embeddings.npz"
    keys_path = output_dir / "embeddings_keys.json"
    if not npz_path.exists() or not keys_path.exists():
        return False
    with open(keys_path) as f:
        cached_keys = json.load(f)
    current_keys = [item.generalization for item in items]
    if cached_keys != current_keys:
        return False
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    if len(embeddings) != len(items):
        return False
    for item, emb in zip(items, embeddings, strict=True):
        item.embedding = list(emb)
    return True


# ============================================================================
# Embedding & Clustering
# ============================================================================


@logfire.instrument("build embeddings")
async def build_embeddings(
    items: list[GeneralizationItem],
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> None:
    texts = [item.generalization for item in items]
    logfire.info("Generating embeddings", total=len(texts))
    all_embeddings = []
    batch_size = 100
    if progress and task_id is not None:
        progress.update(task_id, total=len(texts), description="Embedding generalizations...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with logfire.span(
            "embed batch: {batch_num}", batch_num=i // batch_size + 1, batch_size=len(batch)
        ):
            result = await get_embedder().embed_documents(batch)
            all_embeddings.extend(result.embeddings)
        if progress and task_id is not None:
            progress.update(task_id, completed=min(i + batch_size, len(texts)))
    for item, emb in zip(items, all_embeddings, strict=True):
        item.embedding = list(emb)
    logfire.info("Embeddings complete")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def _build_similarity_matrix(items: list[GeneralizationItem]) -> tuple[np.ndarray, list[int]]:
    """Build pairwise cosine similarity matrix for items with embeddings."""
    valid_indices = [i for i, item in enumerate(items) if item.embedding is not None]
    n = len(valid_indices)
    if n == 0:
        return np.array([]), valid_indices

    # Stack embeddings into a matrix and normalize rows
    embeddings = np.array([items[i].embedding for i in valid_indices])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = embeddings / norms

    # Cosine similarity = dot product of normalized vectors
    sim_matrix = normalized @ normalized.T
    return sim_matrix, valid_indices


@logfire.instrument("cluster generalizations")
def cluster_generalizations(
    items: list[GeneralizationItem],
    similarity_threshold: float = 0.65,
    min_cluster_size: int = 2,
) -> list[list[GeneralizationItem]]:
    """Agglomerative clustering with average-linkage.

    Builds a full similarity matrix, then iteratively merges the two most
    similar clusters until no pair exceeds the threshold. Uses a heap with
    lazy deletion for efficiency on large inputs.
    """
    import heapq

    logfire.info(
        "Clustering generalizations",
        num_items=len(items),
        threshold=similarity_threshold,
        min_size=min_cluster_size,
    )

    sim_matrix, valid_indices = _build_similarity_matrix(items)
    n = len(valid_indices)
    if n == 0:
        return []

    # Initialize: each valid item is its own cluster
    cluster_size: dict[int, int] = {i: 1 for i in range(n)}
    cluster_members: dict[int, list[int]] = {i: [i] for i in range(n)}

    # Cache of sum-of-similarities between cluster pairs
    # Average-linkage = sim_sums[key] / (size_a * size_b)
    sim_sums: dict[tuple[int, int], float] = {}

    # Max-heap using negative similarity (heapq is a min-heap)
    # Entries: (-avg_sim, ca, cb, size_a_when_pushed, size_b_when_pushed)
    # Stale entries are detected by checking if sizes still match
    heap: list[tuple[float, int, int, int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            sim_sums[(i, j)] = s
            if s >= similarity_threshold:
                heapq.heappush(heap, (-s, i, j, 1, 1))

    while heap:
        neg_sim, ca, cb, size_a_snap, size_b_snap = heapq.heappop(heap)

        # Skip stale entries (cluster was merged or sizes changed)
        if ca not in cluster_size or cb not in cluster_size:
            continue
        if cluster_size[ca] != size_a_snap or cluster_size[cb] != size_b_snap:
            continue

        avg_sim = -neg_sim
        if avg_sim < similarity_threshold:
            break

        # Merge cb into ca
        new_size = cluster_size[ca] + cluster_size[cb]

        # Update sim_sums and push new heap entries for all other clusters
        for cc in list(cluster_size):
            if cc in (ca, cb):
                continue
            key_a = (min(ca, cc), max(ca, cc))
            key_b = (min(cb, cc), max(cb, cc))
            new_sum = sim_sums.pop(key_a, 0.0) + sim_sums.pop(key_b, 0.0)
            new_key = (min(ca, cc), max(ca, cc))
            sim_sums[new_key] = new_sum
            new_avg = new_sum / (new_size * cluster_size[cc])
            if new_avg >= similarity_threshold:
                if ca < cc:
                    heapq.heappush(heap, (-new_avg, ca, cc, new_size, cluster_size[cc]))
                else:
                    heapq.heappush(heap, (-new_avg, cc, ca, cluster_size[cc], new_size))

        # Clean up the merged pair's sim_sums entry
        sim_sums.pop((ca, cb) if ca < cb else (cb, ca), None)

        cluster_members[ca].extend(cluster_members.pop(cb))
        cluster_size[ca] = new_size
        del cluster_size[cb]

    # Convert to output format, filtering by min size
    clusters: list[list[GeneralizationItem]] = []
    assigned: set[int] = set()
    for members in cluster_members.values():
        if len(members) >= min_cluster_size:
            cluster = [items[valid_indices[i]] for i in members]
            clusters.append(cluster)
            assigned.update(valid_indices[i] for i in members)

    logfire.info(
        "Clustering complete",
        clusters=len(clusters),
        clustered_items=len(assigned),
        unclustered=len(items) - len(assigned),
    )
    return clusters


# ============================================================================
# Rule Extraction
# ============================================================================


def format_cluster_for_prompt(cluster: list[GeneralizationItem]) -> str:
    parts = []
    by_pr: dict[int, list[GeneralizationItem]] = defaultdict(list)
    for item in cluster:
        by_pr[item.pr_number].append(item)
    pr_count = len(by_pr)
    parts.append("**Cluster Statistics**:")
    parts.append(f"- Total generalizations: {len(cluster)}")
    parts.append(f"- From {pr_count} unique PR(s)")
    parts.append(
        f"- PR distribution: {', '.join(f'PR#{pr}: {len(items)}' for pr, items in sorted(by_pr.items()))}"
    )
    parts.append("")
    parts.append("**Generalizations in this cluster:**")
    for i, item in enumerate(cluster[:15]):
        scope_str = f" [scope: {item.scope}]" if item.scope else ""
        parts.append(f"""
### Generalization {i + 1} (PR#{item.pr_number}, comment #{item.comment_id})
**Rule text**: {item.generalization}
**Motivation**: {item.motivation}{scope_str}
**Source context**: {item.source_context.get("body", "")[:200]}...
""")
    if len(cluster) > 15:
        parts.append(f"\n... and {len(cluster) - 15} more generalizations")
    return "\n".join(parts)


@logfire.instrument("analyze cluster: {cluster_id}")
async def analyze_cluster(
    cluster_id: int, cluster: list[GeneralizationItem]
) -> tuple[ClusterAnalysis, Decimal]:
    formatted = format_cluster_for_prompt(cluster)
    unique_prs = set(item.pr_number for item in cluster)
    prompt = f"""## Cluster Analysis

**Cluster ID**: {cluster_id}

{formatted}

---

Analyze this cluster of related generalizations. Assess coherence, identify the common pattern, and extract validated rule(s) if appropriate.

**Important**: This cluster spans {len(unique_prs)} unique PR(s). Prefer the most general rule that still has multi-PR support. If distinct sub-patterns exist, extract multiple rules.
"""
    result = await get_agent().run(prompt)
    return result.output, get_result_cost(result)


def _load_existing_rules(rules_path: Path) -> tuple[list[RuleOutput], set[int]]:
    """Load existing rules from a previous run. Returns (rules, done_cluster_ids)."""
    rules: list[RuleOutput] = []
    done_ids: set[int] = set()
    if not rules_path.exists():
        return rules, done_ids
    with open(rules_path) as f:
        for line in f:
            if line.strip():
                rule = RuleOutput.model_validate_json(line)
                rules.append(rule)
                done_ids.add(rule.cluster_id)
    return rules, done_ids


def _load_existing_cluster_info(clusters_path: Path) -> dict[int, dict]:
    """Load existing cluster info keyed by cluster_id."""
    if not clusters_path.exists():
        return {}
    with open(clusters_path) as f:
        records = json.load(f)
    return {r["cluster_id"]: r for r in records}


def _write_clusters_json(cluster_info: list[dict], clusters_path: Path) -> None:
    with open(clusters_path, "w") as f:
        json.dump(cluster_info, f, indent=2)


def _write_stats_json(
    rules: list[RuleOutput],
    cluster_info: list[dict],
    stats_path: Path,
    total_generalizations: int,
    total_clusters: int,
    unclustered_count: int,
    similarity_threshold: float,
    min_cluster_size: int,
) -> None:
    file_stats = {
        "total_generalizations": total_generalizations,
        "total_clusters": total_clusters,
        "rules_extracted": len(rules),
        "avg_coherence": sum(c.get("coherence", 0) for c in cluster_info) / len(cluster_info)
        if cluster_info
        else 0,
        "unclustered": unclustered_count,
        "similarity_threshold": similarity_threshold,
        "min_cluster_size": min_cluster_size,
        "by_category": {},
        "by_scope": {},
    }
    for rule in rules:
        file_stats["by_category"][rule.category] = (
            file_stats["by_category"].get(rule.category, 0) + 1
        )
        scope = rule.scope if isinstance(rule.scope, str) else rule.scope
        file_stats["by_scope"][scope] = file_stats["by_scope"].get(scope, 0) + 1
    with open(stats_path, "w") as f:
        json.dump(file_stats, f, indent=2)


async def process_all_clusters(
    clusters: list[list[GeneralizationItem]],
    output_dir: Path,
    limit: int | None = None,
    concurrency: int = 10,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats: StageStats | None = None,
    unclustered_count: int = 0,
    total_generalizations: int = 0,
    similarity_threshold: float = 0.65,
    min_cluster_size: int = 3,
    resuming: bool = False,
    is_pipeline: bool = False,
) -> list[RuleOutput]:
    rules_path = output_dir / "rules.jsonl"
    clusters_path = output_dir / "clusters.json"
    stats_path = output_dir / "stats.json"

    if limit:
        clusters = clusters[:limit]

    # Load existing progress if resuming
    existing_rules: list[RuleOutput] = []
    done_cluster_ids: set[int] = set()
    existing_cluster_info: dict[int, dict] = {}
    if resuming:
        existing_rules, done_cluster_ids = _load_existing_rules(rules_path)
        existing_cluster_info = _load_existing_cluster_info(clusters_path)

    # Build initial cluster_info with membership data for all clusters
    cluster_info_by_id: dict[int, dict] = {}
    for i, cluster in enumerate(clusters):
        comment_ids = list(set(item.comment_id for item in cluster))
        pr_numbers = sorted(set(item.pr_number for item in cluster))
        if i in existing_cluster_info:
            cluster_info_by_id[i] = existing_cluster_info[i]
        else:
            cluster_info_by_id[i] = {
                "cluster_id": i,
                "num_generalizations": len(cluster),
                "comment_ids": comment_ids,
                "pr_numbers": pr_numbers,
                "unique_pr_count": len(pr_numbers),
            }

    # Write initial clusters.json with membership info
    _write_clusters_json(list(cluster_info_by_id.values()), clusters_path)

    # Carry forward existing rules, assign next rule_id
    rules = list(existing_rules)
    rule_id = max((r.rule_id for r in rules), default=-1) + 1

    # Count how many clusters need analysis
    to_analyze = [i for i in range(len(clusters)) if i not in done_cluster_ids]

    if not is_pipeline:
        console.print(
            f"Analyzing {len(to_analyze)} clusters (already analyzed: {len(done_cluster_ids)})"
        )

    if progress and task_id is not None:
        progress.update(
            task_id,
            total=len(clusters),
            completed=len(done_cluster_ids),
            description="Analyzing clusters...",
        )

    if stats:
        # Initialize stats with cached counts
        for _rule in existing_rules:
            stats.inc("rules")
        cached_rejected = sum(
            1
            for cid in done_cluster_ids
            if cid in existing_cluster_info
            and existing_cluster_info[cid].get("num_rules", -1) == 0
            and existing_cluster_info[cid].get("rejection_reason")
        )
        for _ in range(cached_rejected):
            stats.inc("rejected_clusters")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def analyze_one(i: int) -> None:
        nonlocal rule_id
        cluster = clusters[i]
        comment_ids = list(set(item.comment_id for item in cluster))
        pr_numbers = sorted(set(item.pr_number for item in cluster))
        unique_pr_count = len(pr_numbers)
        async with semaphore:
            try:
                analysis, cost = await analyze_cluster(i, cluster)
            except Exception as e:
                logfire.error(
                    "Cluster analysis failed",
                    cluster_id=i,
                    num_generalizations=len(cluster),
                    error=str(e),
                )
                async with write_lock:
                    cluster_info_by_id[i] = {
                        "cluster_id": i,
                        "num_generalizations": len(cluster),
                        "comment_ids": comment_ids,
                        "error": str(e),
                    }
                    if stats:
                        stats.inc("errors")
                    _write_clusters_json(list(cluster_info_by_id.values()), clusters_path)
                    if progress and task_id is not None:
                        progress.advance(task_id)
                return

        # Build comment_id -> pr_number mapping from cluster items
        comment_to_pr: dict[int, int] = {item.comment_id: item.pr_number for item in cluster}
        cluster_comment_ids_set = set(comment_ids)

        async with write_lock:
            if stats:
                stats.add_cost(cost)
            cluster_record = {
                "cluster_id": i,
                "num_generalizations": len(cluster),
                "comment_ids": comment_ids,
                "pr_numbers": pr_numbers,
                "unique_pr_count": unique_pr_count,
                "coherence": analysis.cluster_coherence,
                "common_pattern": analysis.common_pattern,
                "num_rules": len(analysis.rules),
                "rejection_reason": analysis.rejection_reason,
            }
            cluster_info_by_id[i] = cluster_record
            if analysis.rules:
                for rule in analysis.rules:
                    # Per-rule source attribution from LLM-provided source_comment_ids
                    matched_comment_ids = [
                        cid for cid in rule.source_comment_ids if cid in cluster_comment_ids_set
                    ]
                    if matched_comment_ids:
                        rule_comment_ids = matched_comment_ids
                        rule_pr_numbers = sorted(
                            set(
                                comment_to_pr[cid]
                                for cid in rule_comment_ids
                                if cid in comment_to_pr
                            )
                        )
                        rule_unique_pr_count = (
                            len(rule_pr_numbers) if rule_pr_numbers else unique_pr_count
                        )
                        # Fallback if pr lookup yields nothing
                        if not rule_pr_numbers:
                            rule_pr_numbers = pr_numbers
                    else:
                        # Fallback: use full cluster attribution
                        rule_comment_ids = comment_ids
                        rule_pr_numbers = pr_numbers
                        rule_unique_pr_count = unique_pr_count

                    score = calculate_rule_score(rule.confidence, rule_unique_pr_count)
                    rule_output = RuleOutput(
                        rule_id=rule_id,
                        cluster_id=i,
                        text=rule.text,
                        reason=rule.reason,
                        confidence=rule.confidence,
                        rule_score=score,
                        category=rule.category,
                        scope=rule.scope if isinstance(rule.scope, str) else rule.scope,
                        example_bad=rule.example_bad,
                        example_good=rule.example_good,
                        related_entities=rule.related_entities,
                        source_comments=rule_comment_ids,
                        source_prs=rule_pr_numbers,
                        unique_pr_count=rule_unique_pr_count,
                        cluster_coherence=analysis.cluster_coherence,
                        common_pattern=analysis.common_pattern,
                    )
                    rules.append(rule_output)
                    with open(rules_path, "a") as f:
                        f.write(rule_output.model_dump_json() + "\n")
                    if stats:
                        stats.inc("rules")
                    rule_id += 1
            else:
                if stats:
                    stats.inc("rejected_clusters")
            _write_clusters_json(list(cluster_info_by_id.values()), clusters_path)
            if progress and task_id is not None:
                progress.advance(task_id)
                if stats:
                    progress.update(task_id, cost=format_cost(stats.total_cost))

    # Process in batches for roughly ordered rule_ids
    batch_size = concurrency * 2
    for batch_start in range(0, len(to_analyze), batch_size):
        batch = to_analyze[batch_start : batch_start + batch_size]
        await asyncio.gather(*[analyze_one(i) for i in batch])

    # Final writes
    _write_clusters_json(list(cluster_info_by_id.values()), clusters_path)
    _write_stats_json(
        rules,
        list(cluster_info_by_id.values()),
        stats_path,
        total_generalizations,
        len(clusters),
        unclustered_count,
        similarity_threshold,
        min_cluster_size,
    )
    return rules


# ============================================================================
# CLI Command
# ============================================================================


async def _async_run(
    config: RepoConfig,
    is_pipeline: bool = False,
    **kwargs,
) -> StageStats | None:
    input_dir = config.stage_dir("2-extract")
    output_dir = config.stage_dir("3-synthesize")
    similarity_threshold = kwargs.get("similarity_threshold", 0.65)
    min_cluster_size = kwargs.get("min_cluster_size", 2)
    limit = kwargs.get("limit")
    fresh = kwargs.get("fresh", False)
    concurrency = kwargs.get("concurrency", 10)

    input_path = input_dir / "extractions.jsonl"
    input_hash = compute_file_hash(input_path)
    hash_path = output_dir / "input_hash.txt"

    # Decide whether to wipe: --fresh flag or input changed
    resuming = False
    if fresh:
        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)
    elif output_dir.exists() and hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        if stored_hash != input_hash:
            import shutil

            shutil.rmtree(output_dir)
        else:
            resuming = True

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(input_hash)

    if not is_pipeline:
        console.print(f"Loading extractions from {input_path}...")
    extractions = load_extractions(input_path)
    if not is_pipeline:
        console.print(f"Loaded {len(extractions)} extractions")

    items = extract_generalizations(extractions)
    if not is_pipeline:
        actionable_count = sum(1 for e in extractions if e.get("is_actionable", False))
        console.print(
            f"Found {len(items)} generalizations from {actionable_count} actionable comments"
        )

    with stage_progress(
        "Synthesize", is_pipeline=is_pipeline, cost_path=output_dir / "cost.json"
    ) as (progress, stats):
        stats.set("generalizations", len(items))

        # Phase 1: Embeddings (with disk cache)
        task = progress.add_task("Embedding generalizations...", total=len(items))
        if resuming and _load_embeddings_cache(items, output_dir):
            progress.update(task, completed=len(items))
        else:
            await build_embeddings(items, progress=progress, task_id=task)
            _save_embeddings_cache(items, output_dir)

        # Phase 2: Clustering
        progress.update(task, description="Clustering...", completed=0, total=len(items))
        clusters = cluster_generalizations(
            items, similarity_threshold=similarity_threshold, min_cluster_size=min_cluster_size
        )
        progress.update(task, completed=len(items))
        stats.set("clusters", len(clusters))

        total_clustered = sum(len(c) for c in clusters)
        unclustered = len(items) - total_clustered
        stats.set("unclustered", unclustered)

        unclustered_items = [item for item in items if not any(item in c for c in clusters)]
        unclustered_path = output_dir / "unclustered.json"
        with open(unclustered_path, "w") as f:
            json.dump(
                [
                    {
                        "generalization": item.generalization,
                        "motivation": item.motivation,
                        "scope": item.scope,
                        "comment_id": item.comment_id,
                        "pr_number": item.pr_number,
                    }
                    for item in unclustered_items[:200]
                ],
                f,
                indent=2,
            )

        # Phase 3: Analyze clusters
        progress.reset(task, total=len(clusters))
        progress.update(task, description="Analyzing clusters...")
        rules = await process_all_clusters(
            clusters,
            output_dir,
            limit=limit,
            concurrency=concurrency,
            progress=progress,
            task_id=task,
            stats=stats,
            unclustered_count=unclustered,
            total_generalizations=len(items),
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            resuming=resuming,
            is_pipeline=is_pipeline,
        )
        stats.set("rules", len(rules))

        generalizations = int(stats.get("generalizations", 0))
        clustered = generalizations - unclustered
        num_clusters = int(stats.get("clusters", 0))
        num_rules = len(rules)
        stats.detail = f"{generalizations} generalizations \u2192 {clustered} in {num_clusters} clusters, {unclustered} unclustered \u2192 {num_rules} rules"
        return stats


def _run(config: RepoConfig, is_pipeline: bool = False, **kwargs) -> StageStats | None:
    return asyncio.run(_async_run(config, is_pipeline=is_pipeline, **kwargs))


def synthesize(
    ctx: typer.Context,
    min_cluster_size: int = typer.Option(2, help="Minimum generalizations per cluster"),
    limit: int | None = typer.Option(None, help="Limit number of clusters to process"),
    similarity_threshold: float = typer.Option(
        0.65, help="Similarity threshold for clustering (0-1)"
    ),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
    concurrency: int = typer.Option(10, help="Number of parallel LLM requests"),
) -> None:
    """Cluster and synthesize validated rules."""
    config: RepoConfig = ctx.obj["config"]
    _run(
        config,
        min_cluster_size=min_cluster_size,
        limit=limit,
        similarity_threshold=similarity_threshold,
        fresh=fresh,
        concurrency=concurrency,
    )
