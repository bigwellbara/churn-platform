"""Micro-benchmark of the model alone (no HTTP), broken down by pipeline stage.

Run inside the API image so it measures the production environment:
    docker run --rm -v "$PWD/loadtest:/mnt/locust:ro" churn-api:local \
        python /mnt/locust/bench_model.py
"""

import os
import statistics
import time

import pandas as pd

from churn.api.model import ModelService
from churn.api.schemas import EXAMPLE_CUSTOMER, CustomerFeatures
from churn.config import RAW_FEATURES


def timed(fn, repeats: int) -> tuple[float, float]:
    for _ in range(10):  # warm-up
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return statistics.median(samples), samples[int(0.95 * len(samples)) - 1]


service = ModelService.from_dir(os.environ.get("MODEL_DIR", "models/champion"))
xgb = service.pipeline.named_steps["model"]
preprocess = service.pipeline[:-1]
one = [CustomerFeatures(**EXAMPLE_CUSTOMER)]
fifty = one * 50
frame = pd.DataFrame([c.features() for c in one], columns=RAW_FEATURES)
transformed = preprocess.transform(frame)

print(f"CPUs visible: {os.cpu_count()}")
print(f"{'setting':<22}{'stage':<26}{'median ms':>10}{'p95 ms':>9}")
for threads in (xgb.get_params()["n_jobs"], 1):
    xgb.set_params(n_jobs=threads)
    label = f"xgboost n_jobs={threads}"
    rows = [
        (
            "build DataFrame",
            lambda: pd.DataFrame([c.features() for c in one], columns=RAW_FEATURES),
        ),
        ("features + preprocess", lambda: preprocess.transform(frame)),
        ("xgboost predict_proba", lambda: xgb.predict_proba(transformed)),
        ("ModelService.predict (1)", lambda: service.predict(one)),
        ("ModelService.predict (50)", lambda: service.predict(fifty)),
    ]
    for stage, fn in rows:
        median, p95 = timed(fn, 200)
        print(f"{label:<22}{stage:<26}{median:>10.2f}{p95:>9.2f}")
