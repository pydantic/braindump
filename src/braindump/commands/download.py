"""Download PR data from GitHub via gh CLI."""

from __future__ import annotations

import asyncio
import json
import subprocess

import logfire
import typer

from braindump.config import RepoConfig
from braindump.progress import StageStats, console, stage_progress


def _gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gh CLI command (sync, for auth check and PR list)."""
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


async def _gh_async(*args: str) -> tuple[int, str, str]:
    """Run a gh CLI command asynchronously. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "gh",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(), stderr.decode()


@logfire.instrument("download", extract_args={"config": ["repo"]})
def _run(
    config: RepoConfig,
    since: str | None = None,
    concurrency: int = 5,
    is_pipeline: bool = False,
    fresh: bool = False,
) -> StageStats | None:
    """Download PR reviews for a repo."""
    import shutil

    repo = config.repo
    output_dir = config.pr_data_dir

    # Check gh auth
    result = _gh("auth", "status", check=False)
    if result.returncode != 0:
        console.print("[red]Error: gh CLI is not authenticated. Run 'gh auth login' first.[/red]")
        raise typer.Exit(1)

    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)

    # Create directories
    for subdir in ["prs", "reviews", "review_comments", "diffs", "issue_comments"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Build search filter
    search_filter = f"updated:>={since}" if since else ""

    if not is_pipeline:
        if since:
            console.print(f"Fetching PRs from {repo} since {since}...")
        else:
            console.print(f"Fetching all PRs from {repo}...")

    # Get PR list (sync — runs once)
    gh_args = [
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "all",
        "--limit",
        "1000",
        "--json",
        "number,title,author,state,createdAt,updatedAt,mergedAt,closedAt,baseRefName,headRefName,url",
    ]
    if search_filter:
        gh_args.extend(["--search", search_filter])

    with logfire.span("fetch PR list"):
        result = _gh(*gh_args)
        pr_list_path = output_dir / "pr_list.json"
        pr_list_path.write_text(result.stdout)

    prs = json.loads(result.stdout)
    pr_numbers = [pr["number"] for pr in prs]
    logfire.info("Found PRs to download", total=len(pr_numbers), since=since)

    return asyncio.run(_async_download(repo, output_dir, pr_numbers, concurrency, is_pipeline))


async def _async_download(
    repo: str,
    output_dir,
    pr_numbers: list[int],
    concurrency: int,
    is_pipeline: bool,
) -> StageStats | None:
    """Download all PR data in parallel using asyncio."""
    semaphore = asyncio.Semaphore(concurrency)

    def _pr_already_downloaded(pr_num: int) -> bool:
        """Check if all 5 expected files exist for a PR."""
        return all(
            [
                (output_dir / "prs" / f"{pr_num}.json").exists(),
                (output_dir / "reviews" / f"{pr_num}.json").exists(),
                (output_dir / "review_comments" / f"{pr_num}.json").exists(),
                (output_dir / "diffs" / f"{pr_num}.diff").exists(),
                (output_dir / "issue_comments" / f"{pr_num}.json").exists(),
            ]
        )

    already_done = [pr for pr in pr_numbers if _pr_already_downloaded(pr)]
    remaining = [pr for pr in pr_numbers if not _pr_already_downloaded(pr)]

    async def download_pr(pr_num: int) -> None:
        """Download all 5 resources for a single PR."""
        async with semaphore:
            with logfire.span("download PR #{pr_num}", pr_num=pr_num):
                # PR details
                with logfire.span("fetch PR details: #{pr_num}", pr_num=pr_num):
                    rc, stdout, stderr = await _gh_async("api", f"repos/{repo}/pulls/{pr_num}")
                    if rc == 0:
                        (output_dir / "prs" / f"{pr_num}.json").write_text(stdout)
                    elif "403" in stderr or "429" in stderr:
                        logfire.warn(
                            "Rate limited fetching PR details", pr_num=pr_num, stderr=stderr[:200]
                        )

                # Reviews (with --paginate)
                with logfire.span("fetch reviews: #{pr_num}", pr_num=pr_num):
                    rc, stdout, stderr = await _gh_async(
                        "api", f"repos/{repo}/pulls/{pr_num}/reviews", "--paginate"
                    )
                    if rc == 0:
                        (output_dir / "reviews" / f"{pr_num}.json").write_text(stdout)
                    elif "403" in stderr or "429" in stderr:
                        logfire.warn(
                            "Rate limited fetching reviews", pr_num=pr_num, stderr=stderr[:200]
                        )

                # Review comments
                with logfire.span("fetch review comments: #{pr_num}", pr_num=pr_num):
                    rc, stdout, stderr = await _gh_async(
                        "api", f"repos/{repo}/pulls/{pr_num}/comments", "--paginate"
                    )
                    if rc == 0:
                        (output_dir / "review_comments" / f"{pr_num}.json").write_text(stdout)
                    elif "403" in stderr or "429" in stderr:
                        logfire.warn(
                            "Rate limited fetching review comments",
                            pr_num=pr_num,
                            stderr=stderr[:200],
                        )

                # Diff
                with logfire.span("fetch diff: #{pr_num}", pr_num=pr_num):
                    rc, stdout, stderr = await _gh_async(
                        "api",
                        f"repos/{repo}/pulls/{pr_num}",
                        "-H",
                        "Accept: application/vnd.github.v3.diff",
                    )
                    if rc == 0:
                        (output_dir / "diffs" / f"{pr_num}.diff").write_text(stdout)
                    elif "403" in stderr or "429" in stderr:
                        logfire.warn(
                            "Rate limited fetching diff", pr_num=pr_num, stderr=stderr[:200]
                        )

                # Issue comments
                with logfire.span("fetch issue comments: #{pr_num}", pr_num=pr_num):
                    rc, stdout, stderr = await _gh_async(
                        "api",
                        f"repos/{repo}/issues/{pr_num}/comments",
                        "--paginate",
                    )
                    if rc == 0:
                        (output_dir / "issue_comments" / f"{pr_num}.json").write_text(stdout)
                    elif "403" in stderr or "429" in stderr:
                        logfire.warn(
                            "Rate limited fetching issue comments",
                            pr_num=pr_num,
                            stderr=stderr[:200],
                        )

    with stage_progress("Download", is_pipeline=is_pipeline) as (progress, stats):
        stats.set("PRs", len(pr_numbers))
        if not is_pipeline:
            console.print(
                f"Downloading {len(remaining)} PRs (already downloaded: {len(already_done)})"
            )

        task = progress.add_task(
            "Downloading PR data...", total=len(pr_numbers), completed=len(already_done)
        )

        async def download_with_progress(pr_num: int) -> None:
            await download_pr(pr_num)
            progress.advance(task)

        await asyncio.gather(*[download_with_progress(pr_num) for pr_num in remaining])

        stats.detail = f"{len(pr_numbers)} PRs"
        return stats


def download(
    ctx: typer.Context,
    since: str | None = typer.Option(
        None, help="Only fetch PRs updated since this date (YYYY-MM-DD)"
    ),
    concurrency: int = typer.Option(5, help="Number of parallel PR downloads"),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Download PR data from GitHub via gh CLI."""
    config: RepoConfig = ctx.obj["config"]
    _run(config, since=since, concurrency=concurrency, fresh=fresh)
