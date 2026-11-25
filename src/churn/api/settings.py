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
