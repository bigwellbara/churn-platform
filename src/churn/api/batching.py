"""Dynamic micro-batching: concurrent single predictions share one model call.

Profiling showed that scoring 50 customers costs about the same as scoring one
(~11 ms; the time is fixed per-call overhead in the sklearn pipeline). So instead of
one model call per request, each request puts its customer on a queue and awaits a
future. A single worker takes *everything* waiting on the queue, scores it in one call,
and resolves each future with its own result.

Batches form naturally: while the model is busy, new requests queue up and become the
next batch. Light load means batches of ~1 (no added latency); heavy load means bigger
batches and much higher throughput. Production servers such as Triton and TF Serving
use the same technique.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from churn.api import metrics
from churn.api.schemas import CustomerFeatures, Prediction

PredictFn = Callable[[list[CustomerFeatures]], Awaitable[list[Prediction]]]


class MicroBatcher:
    """`max_in_flight`: how many batches may be scored at once. Set it to the number of
    model worker processes, so a request that arrives while one worker is busy is sent to
    an idle worker immediately instead of waiting for the running batch to finish."""

    def __init__(self, predict: PredictFn, max_batch_size: int = 64, max_in_flight: int = 1):
        self._predict = predict
        self.max_batch_size = max_batch_size
        self._slots = asyncio.Semaphore(max_in_flight)
        self._queue: asyncio.Queue[tuple[CustomerFeatures, asyncio.Future]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._in_flight: set[asyncio.Task] = set()

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        if self._in_flight:  # let batches already being scored finish
            await asyncio.gather(*self._in_flight, return_exceptions=True)

    async def predict(self, customer: CustomerFeatures) -> Prediction:
        future: asyncio.Future[Prediction] = asyncio.get_running_loop().create_future()
        await self._queue.put((customer, future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()  # wait for at least one request
            await self._slots.acquire()  # ...and for a free model worker
            # Collect *after* a worker is free: everything that queued meanwhile joins.
            batch = [first]
            while len(batch) < self.max_batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            metrics.MICRO_BATCH_SIZE.observe(len(batch))
            task = asyncio.create_task(self._score(batch))
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    async def _score(self, batch: list[tuple[CustomerFeatures, asyncio.Future]]) -> None:
        try:
            predictions = await self._predict([customer for customer, _ in batch])
        except Exception as exc:  # fail every request in the batch, not the batcher
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        finally:
            self._slots.release()
        for (_, future), prediction in zip(batch, predictions, strict=True):
            if not future.done():  # the client may have disconnected
                future.set_result(prediction)
