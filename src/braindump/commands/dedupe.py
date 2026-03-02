"""Stage 4: Deduplicate semantically similar rules using cluster-based consolidation."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import logfire
import numpy as np
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Embedder
from rich.progress import Progress, TaskID

from braindump.commands.synthesize import calculate_rule_score
from braindump.config import RepoConfig, compute_file_hash
from braindump.progress import StageStats, console, format_cost, get_result_cost, stage_progress

# ============================================================================
# Models
# ============================================================================


class ConsolidatedRule(BaseModel):
    text: str = Field(description="The deduplicated rule text")
    reason: str = Field(description="The motivation/reason for this rule")
    confidence: float = Field(ge=0.0, le=1.0)
    source_rule_ids: list[int] = Field(description="Which input rule IDs this output rule covers")
    category: str
    scope: str


class ClusterConsolidationResult(BaseModel):
    rules: list[ConsolidatedRule]
    rationale: str


class DuplicateGroup(BaseModel):
    rule_ids: list[int] = Field(min_length=2)
    reason: str


class CategoryDuplicatesResult(BaseModel):
    duplicate_groups: list[DuplicateGroup]
    rationale: str


# ============================================================================
# System Prompts
# ============================================================================

CONSOLIDATE_SYSTEM_PROMPT = """You are consolidating a cluster of semantically similar coding rules that were independently extracted from PR review comments.

## Your Task

Given N similar rules (connected by embedding similarity), produce the **minimal set of distinct rules** that covers all the guidance. Merge true duplicates and complementary framings, but keep genuinely distinct rules separate.

## When to MERGE rules

Merge rules that are:
1. **Rephrases** of the same guidance ("use X" vs "prefer X" vs "always X")
2. **Complementary framings** ("do X when A" + "don't do X when B" → single rule with clear condition)
3. **Positive/negative framings** of the same idea ("use X" + "avoid not-X")
4. **Different specificity levels** of the same concept — keep the more specific version when it references project conventions
5. **Overlapping subsets** where one rule subsumes another

## When to KEEP rules SEPARATE

Keep rules separate when they:
1. Address **genuinely different concerns** even if related
2. Apply to **different contexts** with different guidance
3. Would lose important nuance if merged

## Output Requirements

- Each output rule must list which `source_rule_ids` it covers (by rule_id)
- Every input rule must be covered by exactly one output rule
- Use the most specific phrasing that still covers all source rules — preserve project-specific references (class names, directory conventions, file patterns)
- Pick the most appropriate category and scope for each output rule
"""

CATEGORY_REVIEW_SYSTEM_PROMPT = """You are reviewing a group of coding rules that share the same category, looking for duplicates that were missed by embedding-based similarity detection.

## Your Task

Given all rules in a category, identify groups of rules that are **semantically duplicates** despite having different wording.

**Only return groups of 2+ rules that should be merged.** Most rules are unique — you only need to flag the duplicates.

## When rules ARE duplicates

- Same guidance, different wording
- Facets of one rule
- Same concept from different angles

## When rules are NOT duplicates

- Related but different guidance
- Both about same topic but different actions
"""


# ============================================================================
# Lazy Agents
# ============================================================================

_consolidate_agent: Agent[None, ClusterConsolidationResult] | None = None
_category_review_agent: Agent[None, CategoryDuplicatesResult] | None = None
_embedder: Embedder | None = None


def get_consolidate_agent() -> Agent[None, ClusterConsolidationResult]:
    global _consolidate_agent
    if _consolidate_agent is None:
        from braindump.config import make_model

        _consolidate_agent = Agent(
            make_model(),
            output_type=ClusterConsolidationResult,
            system_prompt=CONSOLIDATE_SYSTEM_PROMPT,
            retries=3,
        )
    return _consolidate_agent


def get_category_review_agent() -> Agent[None, CategoryDuplicatesResult]:
    global _category_review_agent
    if _category_review_agent is None:
        from braindump.config import make_model

        _category_review_agent = Agent(
            make_model(),
            output_type=CategoryDuplicatesResult,
            system_prompt=CATEGORY_REVIEW_SYSTEM_PROMPT,
            retries=3,
        )
    return _category_review_agent


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        from braindump.config import make_embedding_model

        _embedder = Embedder(make_embedding_model())
    return _embedder


# ============================================================================
# Utility Functions
# ============================================================================


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


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


def load_overrides(overrides_path: Path) -> list[dict]:
    if not overrides_path.exists():
        return []
    overrides = []
    with open(overrides_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                override = json.loads(line)
                override_rule = {
                    "rule_id": -(i + 1),
                    "cluster_id": -1,
                    "text": override["text"],
                    "reason": override.get("reason", "Manual override"),
                    "confidence": 0.95,
                    "rule_score": calculate_rule_score(0.95, 5),
                    "category": override.get("category", "general"),
                    "scope": override.get("scope", "global"),
                    "_explicit_category": "category" in override,
                    "_explicit_scope": "scope" in override,
                    "example_bad": override.get("example_bad"),
                    "example_good": override.get("example_good"),
                    "related_entities": [],
                    "source_comments": [],
                    "source_prs": [],
                    "unique_pr_count": 5,
                    "cluster_coherence": 1.0,
                    "common_pattern": "",
                    "_is_override": True,
                }
                overrides.append(override_rule)
    return overrides


# ============================================================================
# Clustering Logic
# ============================================================================


@logfire.instrument("build rule embeddings")
async def build_rule_embeddings(
    rules: list[dict],
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> list[list[float]]:
    texts = [rule["text"] for rule in rules]
    logfire.info("Generating rule embeddings", total=len(texts))
    all_embeddings = []
    batch_size = 100
    if progress and task_id is not None:
        progress.update(task_id, total=len(texts), description="Embedding rules...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with logfire.span(
            "embed batch: {batch_num}", batch_num=i // batch_size + 1, batch_size=len(batch)
        ):
            result = await get_embedder().embed_documents(batch)
            all_embeddings.extend([list(e) for e in result.embeddings])
        if progress and task_id is not None:
            progress.update(task_id, completed=min(i + batch_size, len(texts)))
    logfire.info("Rule embeddings complete")
    return all_embeddings


@logfire.instrument("find similar clusters")
def find_similar_clusters(
    rules: list[dict],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.75,
) -> list[list[int]]:
    n = len(rules)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                union(i, j)

    clusters_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)
    clusters = list(clusters_map.values())
    multi_rule_clusters = [c for c in clusters if len(c) > 1]
    logfire.info(
        "Clustering complete",
        total_rules=n,
        clusters=len(clusters),
        multi_rule_clusters=len(multi_rule_clusters),
        largest_cluster=max(len(c) for c in clusters) if clusters else 0,
    )
    return clusters


# ============================================================================
# Cluster Consolidation
# ============================================================================


def _build_output_rules(
    consolidation: ClusterConsolidationResult, input_rules: list[dict]
) -> list[dict]:
    rule_by_id = {r["rule_id"]: r for r in input_rules}
    output_rules = []
    used_ids: set[int] = set()
    for consolidated in consolidation.rules:
        source_ids = consolidated.source_rule_ids
        source_rules = [rule_by_id[rid] for rid in source_ids if rid in rule_by_id]
        if not source_rules:
            continue
        combined_comments = list({c for r in source_rules for c in r["source_comments"]})
        combined_prs = sorted({p for r in source_rules for p in r["source_prs"]})
        unique_pr_count = len(combined_prs)
        # Pick the lowest unused source rule ID to avoid duplicates
        # when the LLM references overlapping sources across output rules
        available_ids = sorted(r["rule_id"] for r in source_rules)
        lowest_rule_id = next(
            (rid for rid in available_ids if rid not in used_ids), available_ids[0]
        )
        used_ids.add(lowest_rule_id)
        example_bad = None
        example_good = None
        for r in source_rules:
            if not example_bad and r.get("example_bad"):
                example_bad = r["example_bad"]
            if not example_good and r.get("example_good"):
                example_good = r["example_good"]
        related_entities = list({e for r in source_rules for e in r.get("related_entities", [])})
        best_source = max(
            source_rules, key=lambda r: r.get("rule_score", r.get("adjusted_confidence", 0))
        )
        output_rule = {
            "rule_id": lowest_rule_id,
            "cluster_id": best_source["cluster_id"],
            "text": consolidated.text,
            "reason": consolidated.reason,
            "confidence": consolidated.confidence,
            "rule_score": calculate_rule_score(consolidated.confidence, unique_pr_count),
            "category": consolidated.category,
            "scope": consolidated.scope,
            "example_bad": example_bad,
            "example_good": example_good,
            "related_entities": related_entities,
            "source_comments": combined_comments,
            "source_prs": combined_prs,
            "unique_pr_count": unique_pr_count,
            "cluster_coherence": max(r.get("cluster_coherence", 0) for r in source_rules),
            "common_pattern": best_source.get("common_pattern", ""),
            "_merged_from": sorted(r["rule_id"] for r in source_rules),
        }
        output_rules.append(output_rule)
    return output_rules


@logfire.instrument("consolidate cluster")
async def consolidate_cluster(cluster_rules: list[dict]) -> tuple[list[dict], Decimal]:
    if len(cluster_rules) == 1:
        return [cluster_rules[0].copy()], Decimal(0)

    overrides = [r for r in cluster_rules if r.get("_is_override")]
    if overrides:
        override = overrides[0]
        extracted = [r for r in cluster_rules if not r.get("_is_override")]
        combined_comments = list({c for r in extracted for c in r.get("source_comments", [])})
        combined_prs = sorted({p for r in extracted for p in r.get("source_prs", [])})
        unique_pr_count = max(len(combined_prs), override["unique_pr_count"])
        example_bad = override.get("example_bad")
        example_good = override.get("example_good")
        for r in extracted:
            if not example_bad and r.get("example_bad"):
                example_bad = r["example_bad"]
            if not example_good and r.get("example_good"):
                example_good = r["example_good"]
        extracted_ids = [r["rule_id"] for r in extracted if r["rule_id"] > 0]
        rule_id = min(extracted_ids) if extracted_ids else override["rule_id"]
        best_extracted = max(extracted, key=lambda r: r.get("rule_score", 0)) if extracted else None
        category = (
            override["category"]
            if override.get("_explicit_category")
            else (best_extracted["category"] if best_extracted else override["category"])
        )
        scope = (
            override["scope"]
            if override.get("_explicit_scope")
            else (best_extracted["scope"] if best_extracted else override["scope"])
        )
        output_rule = {
            "rule_id": rule_id,
            "cluster_id": extracted[0]["cluster_id"] if extracted else override["cluster_id"],
            "text": override["text"],
            "reason": override["reason"],
            "confidence": override["confidence"],
            "rule_score": calculate_rule_score(override["confidence"], unique_pr_count),
            "category": category,
            "scope": scope,
            "example_bad": example_bad,
            "example_good": example_good,
            "related_entities": list({e for r in extracted for e in r.get("related_entities", [])}),
            "source_comments": combined_comments,
            "source_prs": combined_prs,
            "unique_pr_count": unique_pr_count,
            "cluster_coherence": max(
                (r.get("cluster_coherence", 0) for r in extracted), default=1.0
            ),
            "common_pattern": "",
            "_merged_from": sorted(r["rule_id"] for r in cluster_rules),
            "_override_applied": True,
        }
        return [output_rule], Decimal(0)

    rule_descriptions = []
    for rule in cluster_rules:
        rule_descriptions.append(
            f"- **Rule ID {rule['rule_id']}**\n"
            f"  - Text: {rule['text']}\n"
            f"  - Reason: {rule['reason']}\n"
            f"  - Category: {rule['category']}\n"
            f"  - Scope: {rule['scope']}\n"
            f"  - Confidence: {rule['confidence']:.2f}\n"
            f"  - Source PRs: {rule['source_prs']} ({rule['unique_pr_count']} unique)"
        )
    prompt = f"""## Consolidate Cluster of {len(cluster_rules)} Similar Rules

{chr(10).join(rule_descriptions)}

---

Produce the minimal set of distinct rules that covers all the guidance above.
"""
    result = await get_consolidate_agent().run(prompt)
    return _build_output_rules(result.output, cluster_rules), get_result_cost(result)


@logfire.instrument("category review: {category}")
async def category_review(
    category: str,
    category_rules: list[dict],
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[dict], list[dict], Decimal]:
    if len(category_rules) <= 1:
        return category_rules, [], Decimal(0)

    rule_descriptions = []
    for rule in category_rules:
        desc = f"- **Rule ID {rule['rule_id']}**: {rule['text']}"
        if rule.get("reason"):
            desc += f"\n  - Reason: {rule['reason']}"
        if rule.get("example_bad"):
            desc += f"\n  - Bad example: `{rule['example_bad']}`"
        if rule.get("example_good"):
            desc += f"\n  - Good example: `{rule['example_good']}`"
        rule_descriptions.append(desc)
    prompt = f"""## Review Category "{category}" ({len(category_rules)} rules)

These rules all share the same category. Identify any groups of rules that are semantically duplicates and should be merged.

{chr(10).join(rule_descriptions)}

---

Return only groups of 2+ duplicate rules. Rules not in any group are unique and will be kept as-is.
"""
    if semaphore:
        async with semaphore:
            result = await get_category_review_agent().run(prompt)
    else:
        result = await get_category_review_agent().run(prompt)
    total_cost = get_result_cost(result)
    duplicate_groups = result.output.duplicate_groups

    if not duplicate_groups:
        return category_rules, [], total_cost

    rule_by_id = {r["rule_id"]: r for r in category_rules}
    merged_rule_ids: set[int] = set()
    output_rules = []
    merge_log_entries = []

    # Transitively merge overlapping groups via union-find:
    # If the LLM identifies {A, B} and {B, C} as separate duplicate groups,
    # merge them into {A, B, C} since B bridges the two groups.
    all_group_ids: list[list[int]] = [
        [rid for rid in group.rule_ids if rid in rule_by_id] for group in duplicate_groups
    ]
    all_group_ids = [g for g in all_group_ids if len(g) >= 2]

    if all_group_ids:
        # Collect all rule IDs mentioned in any group
        all_ids = sorted({rid for g in all_group_ids for rid in g})
        id_to_idx = {rid: i for i, rid in enumerate(all_ids)}
        parent = list(range(len(all_ids)))
        rank = [0] * len(all_ids)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for group_ids in all_group_ids:
            for rid in group_ids[1:]:
                union(id_to_idx[group_ids[0]], id_to_idx[rid])

        clusters: dict[int, list[int]] = {}
        for rid in all_ids:
            root = find(id_to_idx[rid])
            clusters.setdefault(root, []).append(rid)

        valid_groups = [
            [rule_by_id[rid] for rid in rids] for rids in clusters.values() if len(rids) >= 2
        ]
    else:
        valid_groups = []

    async def consolidate_one(group_rules: list[dict]) -> tuple[list[dict], dict, Decimal]:
        if semaphore:
            async with semaphore:
                consolidated, cost = await consolidate_cluster(group_rules)
        else:
            consolidated, cost = await consolidate_cluster(group_rules)
        log_entry = {
            "pass": "category_review",
            "category": category,
            "input_rule_ids": sorted(r["rule_id"] for r in group_rules),
            "output_rules": [
                {
                    "rule_id": r["rule_id"],
                    "source_rule_ids": r.get("_merged_from", []),
                    "text": r["text"],
                }
                for r in consolidated
            ],
        }
        return consolidated, log_entry, cost

    results = await asyncio.gather(*[consolidate_one(gr) for gr in valid_groups])
    for consolidated, log_entry, cost in results:
        total_cost += cost
        output_rules.extend(consolidated)
        merged_rule_ids.update(rid for rid in log_entry["input_rule_ids"])
        merge_log_entries.append(log_entry)

    for rule in category_rules:
        if rule["rule_id"] not in merged_rule_ids:
            output_rules.append(rule)

    return output_rules, merge_log_entries, total_cost


# ============================================================================
# Main Deduplication Pipeline
# ============================================================================


async def deduplicate_rules(
    rules: list[dict],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.75,
    concurrency: int = 10,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats: StageStats | None = None,
) -> tuple[list[dict], list[dict]]:
    clusters = find_similar_clusters(rules, embeddings, similarity_threshold)
    multi_clusters = [c for c in clusters if len(c) > 1]

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    # Pass 1: embedding-based clusters
    if progress and task_id is not None:
        progress.update(
            task_id,
            total=len(multi_clusters),
            completed=0,
            description="Pass 1: Consolidating clusters...",
        )

    pass1_output: list[dict] = []
    merge_log: list[dict] = []

    with logfire.span(
        "pass 1: embedding clusters",
        total_clusters=len(clusters),
        multi_clusters=len(multi_clusters),
    ):
        # Single-item clusters bypass LLM — collect inline
        for cluster_indices in clusters:
            if len(cluster_indices) == 1:
                pass1_output.append(rules[cluster_indices[0]].copy())

        # Multi-item clusters get parallelized
        async def consolidate_pass1(cluster_indices: list[int]) -> None:
            cluster_rules = [rules[i] for i in cluster_indices]
            async with semaphore:
                output_rules, cost = await consolidate_cluster(cluster_rules)
            async with write_lock:
                if stats:
                    stats.add_cost(cost)
                pass1_output.extend(output_rules)
                merge_log.append(
                    {
                        "pass": "embedding_cluster",
                        "input_rule_ids": sorted(r["rule_id"] for r in cluster_rules),
                        "output_rules": [
                            {
                                "rule_id": r["rule_id"],
                                "source_rule_ids": r.get("_merged_from", []),
                                "text": r["text"],
                            }
                            for r in output_rules
                        ],
                    }
                )
                if progress and task_id is not None:
                    progress.advance(task_id)
                    if stats:
                        progress.update(task_id, cost=format_cost(stats.total_cost))

        await asyncio.gather(*[consolidate_pass1(ci) for ci in clusters if len(ci) > 1])

    logfire.info(
        "Pass 1 complete",
        input_rules=len(rules),
        output_rules=len(pass1_output),
        merged=len(rules) - len(pass1_output),
        multi_clusters=len(multi_clusters),
    )
    if stats:
        stats.set("after_pass1", len(pass1_output))

    # Pass 2: category-based review
    by_category: dict[str, list[dict]] = {}
    for rule in pass1_output:
        by_category.setdefault(rule["category"], []).append(rule)

    cats_to_review = {
        cat: cat_rules for cat, cat_rules in by_category.items() if len(cat_rules) >= 2
    }

    if progress and task_id is not None:
        progress.update(
            task_id,
            total=len(cats_to_review),
            completed=0,
            description="Pass 2: Category review...",
        )

    pass2_output: list[dict] = []
    for cat, cat_rules in by_category.items():
        if cat not in cats_to_review:
            pass2_output.extend(cat_rules)

    with logfire.span("pass 2: category review", categories=len(cats_to_review)):

        async def review_one_category(cat: str) -> None:
            cat_rules = cats_to_review[cat]
            output_rules, log_entries, cost = await category_review(
                cat, cat_rules, semaphore=semaphore
            )
            async with write_lock:
                if stats:
                    stats.add_cost(cost)
                if log_entries:
                    merge_log.extend(log_entries)
                pass2_output.extend(output_rules)
                if progress and task_id is not None:
                    progress.advance(task_id)
                    if stats:
                        progress.update(task_id, cost=format_cost(stats.total_cost))

        await asyncio.gather(*[review_one_category(cat) for cat in sorted(cats_to_review)])

    logfire.info(
        "Pass 2 complete",
        input_rules=len(pass1_output),
        output_rules=len(pass2_output),
        merged=len(pass1_output) - len(pass2_output),
        categories_reviewed=len(cats_to_review),
    )
    if stats:
        stats.set("after_pass2", len(pass2_output))

    # Pass 3: post-consolidation review — re-check categories where Pass 2
    # merged rules, since consolidated outputs from separate merge groups
    # may themselves be redundant (the LLM in Pass 2 identifies groups
    # independently, so cross-group overlaps aren't caught).
    pass2_by_category: dict[str, list[dict]] = {}
    for rule in pass2_output:
        pass2_by_category.setdefault(rule["category"], []).append(rule)

    # Only re-review categories that had merges in Pass 2 and still have 3+ rules
    pass2_merged_cats = {entry["category"] for entry in merge_log if entry["pass"] == "category_review"}
    cats_to_recheck = {
        cat: cat_rules
        for cat, cat_rules in pass2_by_category.items()
        if cat in pass2_merged_cats and len(cat_rules) >= 3
    }

    if cats_to_recheck:
        if progress and task_id is not None:
            progress.update(
                task_id,
                total=len(cats_to_recheck),
                completed=0,
                description="Pass 3: Post-consolidation review...",
            )

        pass3_output: list[dict] = []
        for cat, cat_rules in pass2_by_category.items():
            if cat not in cats_to_recheck:
                pass3_output.extend(cat_rules)

        with logfire.span("pass 3: post-consolidation review", categories=len(cats_to_recheck)):

            async def recheck_one_category(cat: str) -> None:
                cat_rules = cats_to_recheck[cat]
                output_rules, log_entries, cost = await category_review(
                    cat, cat_rules, semaphore=semaphore
                )
                async with write_lock:
                    if stats:
                        stats.add_cost(cost)
                    if log_entries:
                        for entry in log_entries:
                            entry["pass"] = "post_consolidation_review"
                        merge_log.extend(log_entries)
                    pass3_output.extend(output_rules)
                    if progress and task_id is not None:
                        progress.advance(task_id)
                        if stats:
                            progress.update(task_id, cost=format_cost(stats.total_cost))

            await asyncio.gather(
                *[recheck_one_category(cat) for cat in sorted(cats_to_recheck)]
            )

        logfire.info(
            "Pass 3 complete",
            input_rules=len(pass2_output),
            output_rules=len(pass3_output),
            merged=len(pass2_output) - len(pass3_output),
            categories_rechecked=len(cats_to_recheck),
        )
        if stats:
            stats.set("after_pass3", len(pass3_output))

        return pass3_output, merge_log

    return pass2_output, merge_log


# ============================================================================
# CLI Command
# ============================================================================

DEDUPE_FILES = [
    "rules.jsonl",
    "merge_log.json",
    "input_hash.txt",
    "embeddings.npz",
    "embeddings_keys.json",
]


def _save_embeddings(
    rules: list[dict], embeddings: list[list[float]], output_dir: Path
) -> None:
    """Save embeddings and keys to disk for resume."""
    keys = [rule["text"] for rule in rules]
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=np.array(embeddings))
    with open(output_dir / "embeddings_keys.json", "w") as f:
        json.dump(keys, f)


def _load_embeddings(rules: list[dict], output_dir: Path) -> list[list[float]] | None:
    """Load cached embeddings if keys match. Returns embeddings or None."""
    npz_path = output_dir / "embeddings.npz"
    keys_path = output_dir / "embeddings_keys.json"
    if not npz_path.exists() or not keys_path.exists():
        return None
    with open(keys_path) as f:
        cached_keys = json.load(f)
    current_keys = [rule["text"] for rule in rules]
    if cached_keys != current_keys:
        return None
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    if len(embeddings) != len(rules):
        return None
    return [list(e) for e in embeddings]


async def _async_run(
    config: RepoConfig,
    is_pipeline: bool = False,
    **kwargs,
) -> StageStats | None:
    output_dir = config.stage_dir("4-dedupe")
    output_dir.mkdir(parents=True, exist_ok=True)

    rules_path = kwargs.get("rules_path") or config.rules_path
    overrides_path = kwargs.get("overrides_path") or config.overrides_path
    similarity_threshold = kwargs.get("similarity_threshold", 0.75)
    concurrency = kwargs.get("concurrency", 10)
    fresh = kwargs.get("fresh", False)

    # Input hash: detect if rules.jsonl or overrides changed
    hash_files = [p for p in [rules_path, overrides_path] if p.exists()]
    if len(hash_files) == 1:
        input_hash = compute_file_hash(hash_files[0])
    elif len(hash_files) > 1:
        from braindump.config import compute_multi_file_hash

        input_hash = compute_multi_file_hash(hash_files)
    else:
        input_hash = ""

    hash_path = output_dir / "input_hash.txt"
    deduped_path = output_dir / "rules.jsonl"

    def _wipe_output_files():
        for fname in DEDUPE_FILES:
            fpath = output_dir / fname
            if fpath.exists():
                fpath.unlink()

    resuming = False
    if fresh:
        _wipe_output_files()
    elif hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        if stored_hash != input_hash:
            _wipe_output_files()
        else:
            resuming = True

    hash_path.write_text(input_hash)

    # Skip-if-done: if resuming and output exists, skip entirely
    if resuming and deduped_path.exists():
        deduped = load_rules(deduped_path)
        console.print(f"Dedupe already complete ({len(deduped)} rules), skipping")
        return None

    if not is_pipeline:
        console.print(f"Loading rules from {rules_path}...")
    rules = load_rules(rules_path)
    if not is_pipeline:
        console.print(f"Loaded {len(rules)} extracted rules")

    overrides = load_overrides(overrides_path)
    if overrides:
        if not is_pipeline:
            console.print(f"Loaded {len(overrides)} override(s)")
        rules = overrides + rules
    elif not is_pipeline:
        console.print("No overrides found")

    if len(rules) == 0:
        console.print("No rules to deduplicate")
        return None

    with stage_progress(
        "Dedupe", is_pipeline=is_pipeline, cost_path=output_dir / "cost.json"
    ) as (progress, stats):
        stats.set("input_rules", len(rules))

        # Phase 1: Embeddings (with disk cache)
        task = progress.add_task("Embedding rules...", total=len(rules))
        cached_embeddings = _load_embeddings(rules, output_dir) if resuming else None
        if cached_embeddings is not None:
            embeddings = cached_embeddings
            progress.update(task, completed=len(rules))
        else:
            embeddings = await build_rule_embeddings(rules, progress=progress, task_id=task)
            _save_embeddings(rules, embeddings, output_dir)

        # Phase 2 & 3: Deduplication (pass 1 + pass 2)
        progress.reset(task)
        deduped, merge_log = await deduplicate_rules(
            rules,
            embeddings,
            similarity_threshold=similarity_threshold,
            concurrency=concurrency,
            progress=progress,
            task_id=task,
            stats=stats,
        )
        stats.set("output_rules", len(deduped))
        stats.set("merged", len(rules) - len(deduped))

        # Deduplicate by rule_id — LLM consolidation can produce collisions
        seen_ids: set[int] = set()
        unique_deduped = []
        for rule in deduped:
            if rule["rule_id"] not in seen_ids:
                seen_ids.add(rule["rule_id"])
                unique_deduped.append(rule)
        if len(unique_deduped) < len(deduped):
            logfire.warn(
                "Removed duplicate rule_ids from output", removed=len(deduped) - len(unique_deduped)
            )
            deduped = unique_deduped
            stats.set("output_rules", len(deduped))

        with open(deduped_path, "w") as f:
            for rule in deduped:
                f.write(json.dumps(rule) + "\n")

        if merge_log:
            merge_log_path = output_dir / "merge_log.json"
            with open(merge_log_path, "w") as f:
                json.dump(merge_log, f, indent=2)

        input_count = int(stats.get("input_rules", 0))
        output_count = int(stats.get("output_rules", 0))
        merged_count = int(stats.get("merged", 0))
        stats.detail = f"{input_count} \u2192 {output_count} rules ({merged_count} merged)"
        return stats


def _run(config: RepoConfig, is_pipeline: bool = False, **kwargs) -> StageStats | None:
    return asyncio.run(_async_run(config, is_pipeline=is_pipeline, **kwargs))


def dedupe(
    ctx: typer.Context,
    similarity_threshold: float = typer.Option(
        0.75, help="Similarity threshold for clustering (0-1)"
    ),
    concurrency: int = typer.Option(10, help="Number of parallel LLM requests"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Deduplicate similar rules."""
    config: RepoConfig = ctx.obj["config"]
    _run(config, similarity_threshold=similarity_threshold, concurrency=concurrency, fresh=fresh)
