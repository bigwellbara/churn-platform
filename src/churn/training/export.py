"""Export a pinned model version from the MLflow registry into a serving bundle.

Usage:
    uv run churn-export                       # whatever @champion points to
    uv run churn-export --version 3           # an exact version (pinning / rollback)
    uv run churn-export --out some/dir

The bundle (see churn.bundle) is what the Docker image ships, so the API never needs
to reach an MLflow server at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from churn.bundle import write_bundle
from churn.config import (
    CHAMPION_ALIAS,
    PROJECT_ROOT,
    RAW_FEATURES,
    REGISTERED_MODEL_NAME,
    default_tracking_uri,
)

logger = logging.getLogger("churn.export")
DEFAULT_OUT = PROJECT_ROOT / "models" / "champion"


def export_model(
    out_dir: Path = DEFAULT_OUT,
    version: str | None = None,
    alias: str = CHAMPION_ALIAS,
    tracking_uri: str | None = None,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())
    client = MlflowClient()
    mv = (
        client.get_model_version(REGISTERED_MODEL_NAME, str(version))
        if version
        else client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    )
    pipeline = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{mv.version}")
    run_metrics = client.get_run(mv.run_id).data.metrics

    meta = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(mv.version),
        "exported_from": f"version {version}" if version else f"alias @{alias}",
        "model_type": mv.tags.get("model_type", "unknown"),
        "decision_threshold": float(mv.tags["decision_threshold"]),
        "run_id": mv.run_id,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": RAW_FEATURES,
        "test_metrics": {
            k.removeprefix("test_"): v
            for k, v in sorted(run_metrics.items())
            if k.startswith("test_")
        },
    }
    write_bundle(pipeline, meta, Path(out_dir))
    logger.info("Exported %s v%s to %s", REGISTERED_MODEL_NAME, mv.version, out_dir)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a registered model for serving.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--version", help="Exact registry version to export")
    group.add_argument("--alias", default=CHAMPION_ALIAS, help="Registry alias to export")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = export_model(args.out, version=args.version, alias=args.alias)
    print(json.dumps({k: meta[k] for k in ("model_version", "model_type")}, indent=2))


if __name__ == "__main__":
    main()
