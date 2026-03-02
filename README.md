# Braindump

Extract coding rules from PR review comments and generate AGENTS.md files for any GitHub repository.

## How it works

```
GitHub PR reviews → download → extract → synthesize → dedupe → place → group → generate → AGENTS.md
```

1. **Download** — fetch PR data (reviews, comments, diffs) via `gh` CLI
2. **Extract** — use Claude to identify actionable changes and generalizable rules from each comment (bot comments are filtered out automatically)
3. **Synthesize** — embed generalizations, cluster by similarity, extract validated rules
4. **Dedupe** — three-pass deduplication (embedding clusters + category review + post-consolidation review)
5. **Place** — determine where each rule belongs (root, directory, file, cross-file), filtering by `min_score` floor (default 0.5)
6. **Group** — filter by `min_score` threshold (default 0.5) and organize by topic for progressive disclosure
7. **Generate** — rephrase rules and write final `AGENTS.md` files

Each stage is resumable — if interrupted, it picks up from where it left off. Pass `--fresh` to any stage or `run` to wipe previous outputs and start clean.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python package management
- [GitHub CLI](https://cli.github.com/) (`gh`) authenticated for repo access
- A [Pydantic AI Gateway](https://ai.pydantic.dev/gateway/) API token (or direct provider API keys — see [Model configuration](#model-configuration))

## Setup

```bash
# Install dependencies
uv sync

# Add your Pydantic AI Gateway token to .env
echo "PYDANTIC_AI_GATEWAY_API_KEY=your-token" > .env

# Authenticate GitHub CLI (if not already)
gh auth login
```

## Quick start

Run the full pipeline:

```bash
uv run braindump --repo pydantic/pydantic-ai run --since 2025-10-01
```

Or set the repo once via environment variable:

```bash
export BRAINDUMP_REPO=pydantic/pydantic-ai
uv run braindump run --since 2025-10-01
```

## Commands

All stages support `--concurrency N` to control parallel LLM/API requests and `--fresh` to wipe previous outputs before running.

### `run` — Full pipeline

```bash
uv run braindump --repo owner/repo run [--from STAGE] [--skip STAGE ...] [--since DATE] [--authors USER] [--min-score 0.5] [--max-rules N] [--fresh]
```

- `--from`: Start from a specific stage (e.g. `--from synthesize`)
- `--skip`: Skip stages (repeatable, e.g. `--skip download --skip extract`)
- `--since`: Date filter for download (YYYY-MM-DD)
- `--authors`: Author filter for extract (default: `all`)
- `--min-score`: Override rule score threshold (default: 0.5)
- `--max-rules N`: Cap the number of rules in group stage (top-scored)
- `--fresh`: Wipe all stage outputs and start from scratch

### `download` — Fetch PR data

```bash
uv run braindump --repo owner/repo download [--since YYYY-MM-DD] [--concurrency 5]
```

### `extract` — Extract rules from comments

```bash
uv run braindump --repo owner/repo extract [--authors USER] [--limit N] [--random] [--prs 1,2,3] [--concurrency 10]
```

- `--prs`: Filter to specific PR numbers (comma-separated)
- `--limit N`: Limit number of comments to process
- `--random`: Randomly sample comments (with `--seed` for reproducibility)

### `synthesize` — Cluster and validate rules

```bash
uv run braindump --repo owner/repo synthesize [--similarity-threshold 0.65] [--min-cluster-size 3] [--concurrency 10]
```

### `dedupe` — Deduplicate similar rules

```bash
uv run braindump --repo owner/repo dedupe [--similarity-threshold 0.75] [--concurrency 10]
```

### `place` — Determine rule placement

```bash
uv run braindump --repo owner/repo place [--min-score 0.5] [--concurrency 10]
```

### `group` — Organize by topic

```bash
uv run braindump --repo owner/repo group [--min-score 0.5] [--max-rules N] [--preview]
```

- `--min-score`: Minimum rule score to include (default: 0.5)
- `--max-rules N`: Cap the number of rules included (top-scored). Applied after `--min-score` filtering.
- `--preview`: Show a table of rule counts and marginal examples at different score thresholds, then exit without running the LLM grouping. Useful for picking an appropriate `--min-score` or `--max-rules`.

Re-run from group with a different threshold to adjust how many rules end up in the output — no need to re-run place:

```bash
uv run braindump --repo owner/repo run --from group --min-score 0.6
uv run braindump --repo owner/repo run --from group --max-rules 80
```

### `generate` — Write AGENTS.md files

```bash
uv run braindump --repo owner/repo generate [--dry-run] [--concurrency 10]
```

### `status` — Pipeline status

```bash
uv run braindump --repo owner/repo status
```

Shows what data exists per stage, key metrics (including cost), and suggests the next stage to run. Stages whose inputs have changed since they last ran are marked as stale.

### `lookup` — Trace rules to source comments

```bash
uv run braindump --repo owner/repo lookup 42
uv run braindump --repo owner/repo lookup --search isinstance
uv run braindump --repo owner/repo lookup --location root
```

## Data directory structure

All per-repo data lives under `data/<owner>/<repo>/`:

```
data/
  pydantic/
    pydantic-ai/
      1-download/         # Raw GitHub API data (PRs, diffs, review comments)
      2-extract/          # extractions.jsonl, checkpoint.json
      3-synthesize/       # rules.jsonl, embeddings, clusters
      4-dedupe/           # rules.jsonl (deduped), embeddings, merge_log
      5-place/            # placements.jsonl
      6-group/            # organized_rules.json
      7-generate/         # Final AGENTS.md files
      rule_overrides.jsonl
```

## Rule overrides

Place manual rule corrections in `data/<owner>/<repo>/rule_overrides.jsonl`:

```jsonl
{"text": "Use tuple syntax for isinstance() checks, not | union"}
{"text": "Always use explicit re-exports in __init__.py", "reason": "Enforced by our linter", "category": "code_style", "scope": "global"}
```

Only `text` is required. Optional fields: `reason` (default: `"Manual override"`), `category` (default: `"general"`), `scope` (default: `"global"`), `example_bad`, `example_good`.

Overrides cluster with similar extracted rules and replace them during the `dedupe` stage. After adding or changing overrides, re-run from dedupe:

```bash
uv run braindump --repo owner/repo run --from dedupe --fresh
```

## Model configuration

By default, braindump uses the [Pydantic AI Gateway](https://ai.pydantic.dev/gateway/) with `anthropic:claude-sonnet-4-5` for LLM tasks and `openai:text-embedding-3-small` for embeddings. Both models are configurable via flags or environment variables:

```bash
# Use flags
uv run braindump --model anthropic:claude-sonnet-4-5 --embedding-model openai:text-embedding-3-small --repo owner/repo run

# Or use environment variables
export BRAINDUMP_MODEL=anthropic:claude-sonnet-4-5
export BRAINDUMP_EMBEDDING_MODEL=openai:text-embedding-3-small
```

Model strings follow the [pydantic-ai format](https://ai.pydantic.dev/models/): `provider:model-name`. Examples:

- `gateway/anthropic:claude-sonnet-4-5` (default LLM, uses Pydantic AI Gateway)
- `gateway/openai:text-embedding-3-small` (default embeddings, uses Pydantic AI Gateway)
- `anthropic:claude-sonnet-4-5` (direct Anthropic)
- `openai:text-embedding-3-small` (direct OpenAI)

When using `gateway/...` models, set `PYDANTIC_AI_GATEWAY_API_KEY`. For direct providers, set the provider-specific key (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Observability

Braindump is automatically instrumented with [Pydantic Logfire](https://pydantic.dev/logfire) for tracing and observability. All LLM calls, HTTP requests, and pipeline stages are traced.

To enable, authenticate and select a project:

```bash
uv run logfire auth
uv run logfire projects use
```

Traces are sent only when a Logfire token is present — if you skip this step, everything still works, just without tracing.
