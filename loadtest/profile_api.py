"""Where does the CPU go per request? Times each stage of the request path.

Runs inside the API image against the real Postgres:
    docker compose run --rm --no-deps -v ./loadtest:/mnt/locust:ro api \
        python /mnt/locust/profile_api.py
"""

import os
import statistics
import time

from locust_payloads import random_customer

from churn.api.db import PredictionStore
from churn.api.model import ModelService
from churn.api.schemas import BatchRequest, CustomerFeatures, PredictionResponse


def timed(fn, repeats: int) -> float:
    for _ in range(5):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


model = ModelService.from_dir(os.environ["MODEL_DIR"], threads=1)
store = PredictionStore(os.environ["DATABASE_URL"])

one_json = random_customer()
batch_json = {"customers": [random_customer() for _ in range(50)]}
one = [CustomerFeatures(**one_json)]
batch = BatchRequest(**batch_json).customers


def rows(customers, predictions):
    return [
        {
            "request_id": "profile",
            "customer_id": c.customer_id,
            "model_version": model.version,
            "features": c.features(),
            "churn_probability": p.churn_probability,
            "churn_prediction": p.churn_prediction,
            "threshold": model.threshold,
        }
        for c, p in zip(customers, predictions, strict=True)
    ]


one_rows, batch_rows = rows(one, model.predict(one)), rows(batch, model.predict(batch))
stages = [
    ("validate 1 (pydantic)", lambda: CustomerFeatures(**one_json), 300),
    ("predict 1", lambda: model.predict(one), 300),
    (
        "build response 1",
        lambda: PredictionResponse(
            **model.predict(one)[0].model_dump(), model_version="1", threshold=0.5
        ).model_dump_json(),
        300,
    ),
    ("db insert 1 row", lambda: store.save(one_rows), 100),
    ("validate 50", lambda: BatchRequest(**batch_json), 100),
    ("predict 50", lambda: model.predict(batch), 100),
    ("db insert 50 rows", lambda: store.save(batch_rows), 50),
]
print(f"{'stage':<24}{'median ms':>10}")
for name, fn, n in stages:
    print(f"{name:<24}{timed(fn, n):>10.2f}")
