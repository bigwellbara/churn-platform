"""Runtime configuration from environment variables (12-factor style).

The same Docker image runs locally, in CI and in the cloud; only env vars change.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from churn.config import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(protected_namespaces=(), extra="ignore")

    # Folder holding model.skops + model_meta.json (see churn.bundle).
    model_dir: Path = PROJECT_ROOT / "models" / "champion"
    # e.g. postgresql+psycopg://user:pass@host:5432/db. Unset = prediction logging disabled.
    database_url: SecretStr | None = None
    # If set, /predict endpoints require the header `X-API-Key: <value>`.
    api_key: SecretStr | None = None
    log_level: str = "INFO"
    max_batch_size: int = 1_000
    # Threads per prediction. 1 is fastest for request-at-a-time serving: concurrency
    # comes from handling many requests, not from parallelising each one.
    model_threads: int = 1
    # Worker processes that run the model (see churn.api.inference). 0 = run it in a
    # thread of the API process (simplest; fine for tests and light load).
    inference_processes: int = 0
    # Combine concurrent /predict requests into one model call (see churn.api.batching).
    micro_batching: bool = True
    max_micro_batch: int = 64
    # Prediction-log rows are buffered and written in bulk this often (seconds).
    log_flush_interval: float = 0.5
    # Commit the image was built from (Docker build arg). GET / reports it so a deploy can
    # verify the running revision is the one it just built.
    git_sha: str = "unknown"
