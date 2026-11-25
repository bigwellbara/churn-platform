"""Locust load test for the prediction API.

Each simulated user sends about one request per second (constant_throughput(1)),
so --users is roughly the target requests/second. Default traffic mix: 90% single
predictions, 10% batches of 50 (set BATCH_WEIGHT=0 for single predictions only).
Customers are random but always valid, so the test needs no dataset.

Run (from the repo root, with the compose stack up):
    uv run --group load locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless --users 50 --spawn-rate 10 --run-time 90s \
        --csv reports/loadtest/baseline --only-summary
"""

import os

from locust import HttpUser, constant_throughput
from locust_payloads import random_customer

BATCH_WEIGHT = int(os.environ.get("BATCH_WEIGHT", "1"))


def predict_one(user):
    user.client.post("/predict", json=random_customer(), name="/predict")


def predict_batch(user):
    customers = [random_customer() for _ in range(50)]
    user.client.post("/predict/batch", json={"customers": customers}, name="/predict/batch (50)")


class ScoringUser(HttpUser):
    wait_time = constant_throughput(1)
    tasks = {predict_one: 9, **({predict_batch: BATCH_WEIGHT} if BATCH_WEIGHT > 0 else {})}
