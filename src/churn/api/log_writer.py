"""Write-behind buffer for the prediction log.

Profiling under load showed per-request database writes were the bottleneck: each
request ran its own INSERT transaction in a background task, tying up a thread-pool
thread and a pooled connection, and (with HTTP keep-alive) delaying that client's next
request until the write finished.

Instead, requests put rows on an in-memory queue (microseconds, never blocks) and one
writer thread flushes them every `flush_interval` seconds in a single bulk INSERT.

Trade-offs, chosen deliberately:
  * up to `flush_interval` seconds of rows can be lost if the process crashes
    (a clean shutdown flushes everything);
  * the queue is bounded: during a long database outage new rows are dropped and
    counted (churn_prediction_log_dropped_total) rather than exhausting memory.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from churn.api import metrics
from churn.api.db import PredictionStore

logger = logging.getLogger("churn.api.log_writer")


class PredictionLogWriter:
    def __init__(
        self,
        store: PredictionStore,
        flush_interval: float = 0.5,
        max_rows_per_insert: int = 1_000,
        max_queue: int = 20_000,
    ):
        self.store = store
        self.flush_interval = flush_interval
        self.max_rows_per_insert = max_rows_per_insert
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._flush_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="prediction-log-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, rows: list[dict[str, Any]]) -> None:
        """Called on the request path: enqueue only, never touch the database."""
        for row in rows:
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                metrics.LOG_ROWS_DROPPED.inc()

    def flush(self) -> int:
        """Write everything queued so far. Returns the number of rows written."""
        written = 0
        with self._flush_lock:
            while True:
                rows = self._take(self.max_rows_per_insert)
                if not rows:
                    break
                try:
                    self.store.save(rows)
                    written += len(rows)
                except Exception:
                    metrics.DB_ERRORS.inc()
                    metrics.LOG_ROWS_DROPPED.inc(len(rows))
                    logger.exception(
                        "Failed to write prediction log", extra={"fields": {"rows": len(rows)}}
                    )
                    break  # back off until the next interval instead of hammering the DB
        metrics.LOG_QUEUE_DEPTH.set(self._queue.qsize())
        return written

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread.is_alive():  # safe even if start() was never reached
            self._thread.join(timeout)
        self.flush()  # anything submitted after the thread's last pass

    def _take(self, limit: int) -> list[dict[str, Any]]:
        rows = []
        while len(rows) < limit:
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return rows

    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval):
            self.flush()
