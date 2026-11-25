"""Structured JSON logging: one JSON object per line.

Log platforms (Cloud Logging, Loki, Datadog) parse these fields, so you can filter
on e.g. `latency_ms > 100` or a single `request_id` instead of grepping free text.
Pass extra fields with: logger.info("msg", extra={"fields": {...}}).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the `churn` logger tree only, never the root logger."""
    logger = logging.getLogger("churn")
    logger.setLevel(level.upper())
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
