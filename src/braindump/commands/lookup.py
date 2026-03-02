"""Lookup tool for rule provenance.

Traces a rule ID back through all pipeline stages to the original comments.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from braindump.config import RepoConfig

# ============================================================================
# Data Loading
# ============================================================================


def _load_comments(data_dir: Path) -> dict[int, dict]:
    """Load review comments from raw PR data, indexed by comment ID."""
    comments = {}
    review_dir = data_dir / "1-download" / "review_comments"
    if not review_dir.exists():
        return comments
    for path in review_dir.glob("*.json"):
        pr_number = path.stem
        with open(path) as f:
            for c in json.load(f):
                c["pr_number"] = int(pr_number)
                comments[c["id"]] = c
    return comments


def _load_rules(data_dir: Path) -> dict[int, dict]:
    rules = {}
    path = data_dir / "4-dedupe" / "rules.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rules[r["rule_id"]] = r
    return rules


def _load_placements(data_dir: Path) -> dict[int, dict]:
    placements = {}
    path = data_dir / "5-place" / "placements.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    placements[p["rule_id"]] = p
    return placements


def _load_organization(data_dir: Path) -> dict:
    path = data_dir / "6-group" / "organized_rules.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"locations": []}


# ============================================================================
# Lookup Functions
# ============================================================================


def _find_rule_location(rule_id: int, organization: dict) -> dict | None:
    for loc_data in organization["locations"]:
        location = loc_data["location"]
        if rule_id in loc_data.get("inline_rules", []):
            return {"location": location, "type": "inline"}
        for topic in loc_data.get("topic_references", []):
            if rule_id in topic.get("rule_ids", []):
                return {
                    "location": location,
                    "type": "topic",
                    "topic_id": topic["topic_id"],
                    "topic_name": topic["topic_name"],
                }
    return None


def _lookup_rule(rule_id: int, data_dir: Path, verbose: bool = False) -> None:
    rules = _load_rules(data_dir)
    placements = _load_placements(data_dir)
    comments = _load_comments(data_dir)
    organization = _load_organization(data_dir)

    if rule_id not in rules:
        print(f"Rule {rule_id} not found")
        return

    rule = rules[rule_id]
    placement = placements.get(rule_id, {})
    org_location = _find_rule_location(rule_id, organization)

    print(f"\n{'=' * 70}")
    print(f"RULE {rule_id}")
    print(f"{'=' * 70}")
    print(f"\n{rule['text'][:200]}{'...' if len(rule['text']) > 200 else ''}")

    print("\n--- Metadata ---")
    print(f"Category: {rule.get('category', 'unknown')}")
    print(f"Scope: {rule.get('scope', 'unknown')}")
    print(f"Confidence: {rule.get('confidence', 0):.2f}")

    if placement:
        p = placement.get("placement", {})
        print("\n--- Placement (Stage 3) ---")
        print(f"Type: {p.get('placement_type', 'unknown')}")
        print(f"Timing: {p.get('timing', 'unknown')}")
        print(f"Abstraction: {p.get('abstraction', 'unknown')}")
        if p.get("directory"):
            print(f"Directory: {p['directory']}")
        if p.get("file_path"):
            print(f"File: {p['file_path']}")
        if p.get("target_element"):
            print(f"Element: {p['target_element']}")
        print(f"Rule Score: {placement.get('rule_score', placement.get('signal_score', 0)):.2f}")
        print(f"PRs: {placement.get('unique_pr_count', 0)}")

        if verbose and p.get("rationale"):
            print(f"\nRationale: {p['rationale'][:500]}...")

    if org_location:
        print("\n--- Final Location (Stage 4) ---")
        print(f"Location: {org_location['location']}")
        print(f"Type: {org_location['type']}")
        if org_location.get("topic_id"):
            print(f"Topic: {org_location['topic_name']} ({org_location['topic_id']})")

    source_comment_ids = rule.get("source_comments", [])
    source_prs = rule.get("source_prs", [])

    print("\n--- Source Evidence ---")
    print(f"PRs: {len(source_prs)} - {source_prs[:10]}{'...' if len(source_prs) > 10 else ''}")
    print(f"Comments: {len(source_comment_ids)}")

    if verbose and comments:
        print("\n--- Source Comments ---")
        for i, cid in enumerate(source_comment_ids[:5]):
            if cid in comments:
                c = comments[cid]
                user = c.get("user", {}).get("login", "?")
                print(f"\n[{i + 1}] PR #{c.get('pr_number', '?')} by @{user}")
                path = c.get("path", "")
                if path:
                    print(f"    {path}")
                diff_hunk = c.get("diff_hunk", "")
                if diff_hunk:
                    # Show last few lines of the diff hunk for context
                    lines = diff_hunk.strip().splitlines()
                    context_lines = lines[-8:] if len(lines) > 8 else lines
                    if len(lines) > 8:
                        context_lines = ["..."] + context_lines
                    indented = "\n    ".join(context_lines)
                    print(f"    {indented}")
                body = c.get("body", "")
                if body:
                    truncated = body[:300] + ("..." if len(body) > 300 else "")
                    indented = truncated.replace("\n", "\n    > ")
                    print(f"\n    > {indented}")
        if len(source_comment_ids) > 5:
            print(f"\n    ... and {len(source_comment_ids) - 5} more comments")

    print()


def _search_rules(query: str, data_dir: Path) -> None:
    rules = _load_rules(data_dir)
    placements = _load_placements(data_dir)

    query_lower = query.lower()
    matches = [
        (rid, rule) for rid, rule in rules.items() if query_lower in rule.get("text", "").lower()
    ]

    if not matches:
        print(f"No rules found matching '{query}'")
        return

    print(f"\nFound {len(matches)} rules matching '{query}':\n")
    for rid, rule in matches[:20]:
        p = placements.get(rid, {})
        score = p.get("rule_score", p.get("signal_score", 0))
        prs = p.get("unique_pr_count", 0)
        text = rule["text"][:80]
        print(f"  [{rid:3}] (score={score:.2f}, prs={prs}) {text}...")
    if len(matches) > 20:
        print(f"\n  ... and {len(matches) - 20} more")


def _list_rules_by_location(location: str, data_dir: Path) -> None:
    organization = _load_organization(data_dir)
    rules = _load_rules(data_dir)
    placements = _load_placements(data_dir)

    for loc_data in organization["locations"]:
        if loc_data["location"] == location:
            print(f"\n{'=' * 70}")
            print(f"RULES AT: {location}")
            print(f"{'=' * 70}")

            all_rule_ids = list(loc_data.get("inline_rules", []))
            for topic in loc_data.get("topic_references", []):
                all_rule_ids.extend(topic.get("rule_ids", []))

            for rid in sorted(set(all_rule_ids)):
                if rid in rules:
                    rule = rules[rid]
                    p = placements.get(rid, {})
                    score = p.get("rule_score", p.get("signal_score", 0))
                    text = rule["text"][:60]
                    print(f"  [{rid:3}] (score={score:.2f}) {text}...")
            return

    print(f"Location '{location}' not found")


# ============================================================================
# CLI Command
# ============================================================================


def lookup(
    ctx: typer.Context,
    rule_id: int | None = typer.Argument(None, help="Rule ID to look up"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full details"),
    search: str | None = typer.Option(None, "--search", "-s", help="Search rules by text"),
    location: str | None = typer.Option(None, "--location", "-l", help="List rules at a location"),
) -> None:
    """Trace rules back to source PRs."""
    config: RepoConfig = ctx.obj["config"]
    data_dir = config.data_dir

    if search:
        _search_rules(search, data_dir)
    elif location:
        _list_rules_by_location(location, data_dir)
    elif rule_id is not None:
        _lookup_rule(rule_id, data_dir, verbose)
    else:
        typer.echo("Provide a rule_id, --search, or --location")
        raise typer.Exit(1)


def _run(config: RepoConfig, **kwargs) -> None:
    """Programmatic entry point (not used by pipeline, but kept for consistency)."""
    pass
