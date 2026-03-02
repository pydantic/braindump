"""Shared configuration for braindump CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gateway/anthropic:claude-sonnet-4-5"
DEFAULT_EMBEDDING_MODEL = "gateway/openai:text-embedding-3-small"

_model_name: str = DEFAULT_MODEL
_embedding_model_name: str = DEFAULT_EMBEDDING_MODEL


def configure_models(model: str | None = None, embedding_model: str | None = None) -> None:
    """Set the model names used by make_model/make_embedding_model."""
    global _model_name, _embedding_model_name
    if model:
        _model_name = model
    if embedding_model:
        _embedding_model_name = embedding_model


def init_env() -> None:
    """Load .env and configure logfire. Must be called before importing pydantic_ai."""
    from dotenv import load_dotenv

    load_dotenv()

    import logfire

    logfire.configure(send_to_logfire="if-token-present", console=False)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


def make_retry_client():
    """Create an httpx AsyncClient with retry transport for rate limits."""
    from httpx import AsyncClient, HTTPStatusError
    from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
    from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

    return AsyncClient(
        transport=AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
                wait=wait_retry_after(fallback_strategy=wait_exponential(multiplier=1, max=60)),
                stop=stop_after_attempt(5),
                reraise=True,
            ),
            validate_response=lambda r: r.raise_for_status(),
        ),
    )


def _retry_provider_factory(provider: str) -> Any:
    """Create a provider with retry HTTP client, similar to pydantic_ai's infer_provider."""
    http_client = make_retry_client()
    if provider.startswith("gateway/"):
        from pydantic_ai.providers.gateway import gateway_provider

        return gateway_provider(provider.removeprefix("gateway/"), http_client=http_client)  # type: ignore[arg-type]
    elif provider in ("google-vertex", "google-gla"):
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleProvider(vertexai=provider == "google-vertex", http_client=http_client)
    else:
        from pydantic_ai.providers import infer_provider_class

        return infer_provider_class(provider)(http_client=http_client)  # type: ignore[call-arg]


def make_model():
    """Create a pydantic_ai Model from the configured model name, with retry HTTP client."""
    from pydantic_ai.models import infer_model

    return infer_model(_model_name, provider_factory=_retry_provider_factory)


def make_embedding_model():
    """Create a pydantic_ai EmbeddingModel from the configured name, with retry HTTP client."""
    from pydantic_ai.embeddings import infer_embedding_model

    return infer_embedding_model(_embedding_model_name, provider_factory=_retry_provider_factory)


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_multi_file_hash(paths: list[Path]) -> str:
    """Compute SHA-256 hash over multiple files (sorted by name for stability)."""
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return h.hexdigest()


class RepoConfig:
    """Resolves all data paths from a repo string (owner/repo)."""

    def __init__(self, repo: str) -> None:
        self.repo = repo
        # Find project root (where pyproject.toml lives)
        self._root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self._root / "data" / repo

    def stage_dir(self, stage: str) -> Path:
        return self.data_dir / stage

    @property
    def pr_data_dir(self) -> Path:
        return self.stage_dir("1-download")

    @property
    def review_comments_dir(self) -> Path:
        return self.pr_data_dir / "review_comments"

    @property
    def extractions_path(self) -> Path:
        return self.stage_dir("2-extract") / "extractions.jsonl"

    @property
    def rules_path(self) -> Path:
        return self.stage_dir("3-synthesize") / "rules.jsonl"

    @property
    def rules_deduped_path(self) -> Path:
        return self.stage_dir("4-dedupe") / "rules.jsonl"

    @property
    def overrides_path(self) -> Path:
        return self.data_dir / "rule_overrides.jsonl"

    @property
    def placements_path(self) -> Path:
        return self.stage_dir("5-place") / "placements.jsonl"

    @property
    def organization_path(self) -> Path:
        return self.stage_dir("6-group") / "organized_rules.json"

    @property
    def output_dir(self) -> Path:
        return self.stage_dir("7-generate")
