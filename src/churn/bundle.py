"""A model bundle: the hand-off format between training and serving.

    <dir>/model.skops        the fitted sklearn Pipeline (skops format, not pickle)
    <dir>/model_meta.json    version, decision threshold, metrics, trusted types

Only depends on skops, so the API can load a model without installing MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import skops.io as skops_io
from sklearn.pipeline import Pipeline

MODEL_FILE = "model.skops"
META_FILE = "model_meta.json"

# Modules whose types we accept inside a saved model.
TRUSTED_MODULE_PREFIXES = ("churn.", "numpy.", "sklearn.", "xgboost.", "scipy.")


def trusted_types_for(pipeline: Pipeline) -> list[str]:
    """Types skops must be told to trust so the saved pipeline can be loaded again.

    skops refuses to load any type not on an explicit allowlist, because unpickling a
    tampered file can execute arbitrary code. We only vouch for types from libraries we
    actually use; anything else is a red flag and stops the save.
    """
    untrusted = skops_io.get_untrusted_types(data=skops_io.dumps(pipeline))
    unexpected = [t for t in untrusted if not t.startswith(TRUSTED_MODULE_PREFIXES)]
    if unexpected:
        raise RuntimeError(f"Refusing to trust unexpected types in model: {unexpected}")
    return untrusted


def write_bundle(pipeline: Pipeline, meta: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "trusted_types": trusted_types_for(pipeline)}
    skops_io.dump(pipeline, out_dir / MODEL_FILE)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_bundle(model_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    model_dir = Path(model_dir)
    meta_path, model_path = model_dir / META_FILE, model_dir / MODEL_FILE
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No model bundle in {model_dir}. Train and export one: "
            "`uv run churn-train` then `uv run churn-export`."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("model_version", "decision_threshold", "trusted_types"):
        if key not in meta:
            raise ValueError(f"{meta_path} is missing required key '{key}'")
    pipeline = skops_io.load(model_path, trusted=meta["trusted_types"])
    return pipeline, meta
