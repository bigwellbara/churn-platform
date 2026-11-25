"""Run the model in a separate process, so it never competes with the web server for
Python's GIL.

Measured under load (docs/LOAD_TEST.md): with the model in a thread of the API
process, model calls that take ~11 ms in isolation stretched to 76 ms (p50) and 237 ms
(p95) while the CPU sat ~75% idle. The event-loop thread (all network I/O) and the model
thread kept handing the GIL back and forth (the "GIL convoy" effect). A worker process
has its own interpreter and its own GIL, so neither waits for the other. Shipping a
batch of feature dicts to the worker and floats back costs about a millisecond.
"""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from churn.api.model import ModelService

_worker_model: ModelService | None = None


def _init_worker(model_dir: str, threads: int) -> None:
    """Runs once in each worker process: load the model there."""
    global _worker_model
    _worker_model = ModelService.from_dir(Path(model_dir), threads=threads)


def _score_in_worker(features: list[dict[str, Any]]) -> list[float]:
    assert _worker_model is not None, "worker not initialised"
    return _worker_model.score(features)


class ProcessInference:
    def __init__(self, model_dir: Path, threads: int = 1, processes: int = 1):
        # "spawn" starts a clean interpreter. Forking a process that already runs threads
        # (event loop, log writer) can deadlock on locks held at fork time.
        self._pool = ProcessPoolExecutor(
            max_workers=processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(model_dir), threads),
        )

    def warm_up(self, features: list[dict[str, Any]], timeout: float = 120) -> None:
        """Block until a worker has loaded the model and scored once, so the first real
        request is not slow and a broken model fails startup (fail fast)."""
        self._pool.submit(_score_in_worker, features).result(timeout=timeout)

    async def score(self, features: list[dict[str, Any]]) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, _score_in_worker, features)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
