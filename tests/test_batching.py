import asyncio

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from churn.api.batching import MicroBatcher
from churn.api.main import create_app
from churn.api.model import ModelService
from churn.api.schemas import EXAMPLE_CUSTOMER, CustomerFeatures
from churn.api.settings import Settings


def customers(n: int) -> list[CustomerFeatures]:
    return [
        CustomerFeatures(**{**EXAMPLE_CUSTOMER, "customer_id": f"c{i}", "tenure": i % 72})
        for i in range(n)
    ]


def test_concurrent_requests_each_get_their_own_result(model_bundle_dir):
    service = ModelService.from_dir(model_bundle_dir)
    batch = customers(40)
    expected = {p.customer_id: p.churn_probability for p in service.predict(batch)}
    before = REGISTRY.get_sample_value("churn_micro_batch_size_count") or 0

    async def predict(batch_):
        await asyncio.sleep(0.01)  # like a real model call: lets other requests queue up
        return service.predict(batch_)

    async def scenario():
        batcher = MicroBatcher(predict, max_batch_size=16)
        batcher.start()
        try:
            return await asyncio.gather(*(batcher.predict(c) for c in batch))
        finally:
            await batcher.stop()

    results = asyncio.run(scenario())
    assert [r.customer_id for r in results] == [c.customer_id for c in batch]
    for r in results:
        assert r.churn_probability == pytest.approx(expected[r.customer_id])
    calls = REGISTRY.get_sample_value("churn_micro_batch_size_count") - before
    assert calls < len(batch)  # requests were combined into fewer model calls


@pytest.mark.parametrize("max_in_flight", [1, 2])
def test_in_flight_batches_are_capped(model_bundle_dir, max_in_flight):
    service = ModelService.from_dir(model_bundle_dir)
    state = {"running": 0, "peak": 0}

    async def slow_predict(batch_):
        state["running"] += 1
        state["peak"] = max(state["peak"], state["running"])
        await asyncio.sleep(0.02)
        state["running"] -= 1
        return service.predict(batch_)

    async def scenario():
        batcher = MicroBatcher(slow_predict, max_batch_size=4, max_in_flight=max_in_flight)
        batcher.start()
        try:
            return await asyncio.gather(*(batcher.predict(c) for c in customers(20)))
        finally:
            await batcher.stop()

    results = asyncio.run(scenario())
    assert len(results) == 20
    assert state["peak"] == max_in_flight


def test_model_failure_fails_the_batch_but_not_the_worker(model_bundle_dir):
    service = ModelService.from_dir(model_bundle_dir)
    calls = {"n": 0}

    async def flaky(batch):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return service.predict(batch)

    async def scenario():
        batcher = MicroBatcher(flaky)
        batcher.start()
        try:
            with pytest.raises(RuntimeError, match="transient"):
                await batcher.predict(customers(1)[0])
            return await batcher.predict(customers(1)[0])  # worker is still alive
        finally:
            await batcher.stop()

    assert asyncio.run(scenario()).customer_id == "c0"


def test_model_in_worker_process_matches_in_process(model_bundle_dir, customer):
    """INFERENCE_PROCESSES=1: the model runs in a spawned worker process. Results must be
    identical to scoring in-process, and shutdown must stop the worker cleanly."""
    expected = ModelService.from_dir(model_bundle_dir).predict([CustomerFeatures(**customer)])[0]
    settings = Settings(model_dir=model_bundle_dir, database_url=None, inference_processes=1)
    with TestClient(create_app(settings)) as client:
        assert client.app.state.inference is not None
        single = client.post("/predict", json=customer).json()
        batch = client.post("/predict/batch", json={"customers": [customer] * 3}).json()
    assert single["churn_probability"] == pytest.approx(expected.churn_probability)
    assert {p["churn_probability"] for p in batch["predictions"]} == {single["churn_probability"]}


@pytest.mark.parametrize("micro_batching", [True, False])
def test_predict_endpoint_with_and_without_batching(model_bundle_dir, customer, micro_batching):
    settings = Settings(
        model_dir=model_bundle_dir, database_url=None, micro_batching=micro_batching
    )
    with TestClient(create_app(settings)) as client:
        assert (client.app.state.batcher is not None) is micro_batching
        r = client.post("/predict", json=customer)
        assert r.status_code == 200
        assert 0 <= r.json()["churn_probability"] <= 1
