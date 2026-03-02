"""Shared Rich progress utilities for braindump pipeline stages."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

console = Console()


def format_cost(cost: Decimal) -> str:
    """Format a cost value for display."""
    if cost <= 0:
        return ""
    if cost < Decimal("0.01"):
        # Show enough precision for sub-cent values
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def get_result_cost(result: Any) -> Decimal:
    """Extract total price from a PydanticAI agent run result."""
    try:
        cost_info = result.response.cost()
        if cost_info and cost_info.total_price is not None:
            return cost_info.total_price
    except Exception:
        pass
    return Decimal(0)


def save_stage_cost(cost_path: Path, cost: Decimal) -> None:
    """Persist stage cost to disk as JSON."""
    if cost > 0:
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cost_path, "w") as f:
            json.dump({"cost": str(cost)}, f)


def load_stage_cost(cost_path: Path) -> Decimal:
    """Load persisted stage cost from disk."""
    if not cost_path.exists():
        return Decimal(0)
    try:
        with open(cost_path) as f:
            data = json.load(f)
        return Decimal(data["cost"])
    except Exception:
        return Decimal(0)


class CostColumn(ProgressColumn):
    """Renders accumulated stage cost from task fields."""

    def render(self, task: Task) -> Text:
        cost = task.fields.get("cost", "")
        if not cost:
            return Text("")
        return Text(cost, style="dim cyan")


@dataclass
class StageStats:
    """Accumulate metrics during a pipeline stage."""

    _data: dict[str, int | float | str] = field(default_factory=dict)
    detail: str | None = None
    _cost: Decimal = field(default_factory=lambda: Decimal(0))

    def set(self, key: str, value: int | float | str) -> None:
        self._data[key] = value

    def inc(self, key: str, amount: int = 1) -> None:
        self._data[key] = self._data.get(key, 0) + amount  # type: ignore[operator]

    def get(self, key: str, default: int | float | str = 0) -> int | float | str:
        return self._data.get(key, default)

    def summary(self) -> str:
        return " | ".join(f"{k}: {v}" for k, v in self._data.items())

    def as_dict(self) -> dict[str, int | float | str]:
        return dict(self._data)

    def add_cost(self, cost: Decimal) -> None:
        self._cost += cost

    @property
    def total_cost(self) -> Decimal:
        return self._cost


def make_progress() -> Progress:
    """Factory for a Progress bar with consistent columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        CostColumn(),
        console=console,
    )


@contextmanager
def stage_progress(
    stage_name: str,
    *,
    is_pipeline: bool = False,
    cost_path: Path | None = None,
) -> Generator[tuple[Progress, StageStats], None, None]:
    """Context manager that yields (Progress, StageStats).

    Prints a stage header on entry and a compact stats summary on exit.
    If cost_path is provided, persists accumulated cost to disk on exit.
    """
    if not is_pipeline:
        console.print(f"\n[bold]{stage_name}[/bold]")
    # When is_pipeline=True, run.py prints the header before calling _run()

    stats = StageStats()
    progress = make_progress()
    with progress:
        yield progress, stats

    # Persist cost to disk for status command
    if cost_path:
        save_stage_cost(cost_path, stats.total_cost)

    detail = stats.detail or (stats.summary() if stats._data else None)
    if detail:
        cost_str = format_cost(stats.total_cost)
        if cost_str:
            detail = f"{detail} | {cost_str}"
        console.print(f"  [dim]{detail}[/dim]")


def print_pipeline_header(stages: list[str], repo: str) -> None:
    """Print a styled panel at pipeline start."""
    arrow = " \u2192 "
    content = f"[bold]Pipeline:[/bold] {arrow.join(stages)}\n[bold]Repo:[/bold] {repo}"
    console.print(Panel(content, title="[bold]braindump[/bold]", expand=True))
    console.print()
