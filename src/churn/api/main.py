"""FastAPI application.

Run locally:
    uv run uvicorn churn.api.main:create_app --factory --reload
Then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from churn import __version__
from churn.api import metrics
from churn.api.db import PredictionStore
from churn.api.log_config import configure_logging
from churn.api.model import ModelService
from churn.api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictionResponse,
)
from churn.api.settings import Settings

logger = logging.getLogger("churn.api")

# Polled constantly by Prometheus / health checks: log at DEBUG to keep logs readable.
QUIET_PATHS = {"/metrics", "/health"}

DESCRIPTION = """
Predicts the probability that a telecom customer will churn.

* `POST /predict`: score one customer
* `POST /predict/batch`: score up to `MAX_BATCH_SIZE` customers in one call
* `GET /model`: which model version is serving, its threshold and test metrics

When the server has an API key configured, send it in the `X-API-Key` header.
"""

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _connect_store(settings: Settings) -> PredictionStore | None:
    """Prediction logging is best-effort: a database outage must not stop predictions."""
    if settings.database_url is None:
        logger.warning("DATABASE_URL not set; prediction logging disabled")
        return None
    try:
        return PredictionStore(settings.database_url.get_secret_value())
    except Exception:
        logger.exception("Database unavailable at startup; prediction logging disabled")
        return None


def _save_rows(store: PredictionStore, rows: list[dict]) -> None:
    try:
        store.save(rows)
    except Exception:
        metrics.DB_ERRORS.inc()
        logger.exception("Failed to log predictions", extra={"fields": {"rows": len(rows)}})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: without a model the service is useless. A crash at startup makes the
        # platform keep routing traffic to the previous healthy version instead.
        model = ModelService.from_dir(settings.model_dir)
        app.state.model = model
        app.state.store = _connect_store(settings)
        metrics.MODEL_INFO.clear()
        metrics.MODEL_INFO.labels(model.version, model.model_type).set(1)
        logger.info(
            "Startup complete",
            extra={
                "fields": {
                    "model_version": model.version,
                    "model_type": model.model_type,
                    "threshold": model.threshold,
                    "prediction_logging": app.state.store is not None,
                }
            },
        )
        yield
        if app.state.store is not None:
            app.state.store.dispose()

    app = FastAPI(
        title="Churn Prediction API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )

    # ---------------------------------------------------------------- middleware
    @app.middleware("http")
    async def observe_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            # Route template (e.g. "/predict"), never the raw URL: keeps metric labels bounded.
            route = getattr(request.scope.get("route"), "path", "unmatched")
            metrics.REQUESTS.labels(request.method, route, str(status)).inc()
            metrics.LATENCY.labels(request.method, route).observe(elapsed)
            logger.log(
                logging.DEBUG if route in QUIET_PATHS else logging.INFO,
                "request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 2),
                    }
                },
            )

    # ---------------------------------------------------------------- errors
    def _error(request: Request, status: int, error: str, detail) -> JSONResponse:
        body = {"error": error, "detail": detail, "request_id": request.state.request_id}
        return JSONResponse(status_code=status, content=body)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        detail = [
            {
                "field": ".".join(str(p) for p in err["loc"] if p != "body") or "body",
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
        return _error(request, 422, "validation_error", detail)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        response = _error(request, exc.status_code, "http_error", exc.detail)
        response.headers.update(exc.headers or {})
        return response

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        logger.exception(
            "Unhandled error", extra={"fields": {"request_id": request.state.request_id}}
        )
        return _error(request, 500, "internal_error", "Internal server error")

    # ---------------------------------------------------------------- auth
    def require_api_key(key: str | None = Security(api_key_header)) -> None:
        if settings.api_key is None:
            return
        expected = settings.api_key.get_secret_value()
        # Constant-time comparison: `==` leaks how many leading characters matched.
        if key is None or not secrets.compare_digest(key.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    # ---------------------------------------------------------------- helpers
    def record(
        request: Request,
        background: BackgroundTasks,
        customers: list[CustomerFeatures],
        predictions: list[Prediction],
    ) -> None:
        model: ModelService = request.app.state.model
        for p in predictions:
            metrics.PREDICTIONS.labels("churn" if p.churn_prediction else "stay").inc()
            metrics.PROBABILITY.observe(p.churn_probability)
        store: PredictionStore | None = request.app.state.store
        if store is None:
            return
        rows = [
            {
                "request_id": request.state.request_id,
                "customer_id": c.customer_id,
                "model_version": model.version,
                "features": c.features(),
                "churn_probability": p.churn_probability,
                "churn_prediction": p.churn_prediction,
                "threshold": model.threshold,
            }
            for c, p in zip(customers, predictions, strict=True)
        ]
        # Runs after the response is sent, so logging adds no user-facing latency.
        background.add_task(_save_rows, store, rows)

    # ---------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "churn-prediction-api", "version": __version__, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request):
        store: PredictionStore | None = request.app.state.store
        if settings.database_url is None:
            database = "disabled"
        elif store is not None and store.is_healthy():
            database = "connected"
        else:
            database = "unavailable"
        return HealthResponse(
            status="ok", model_version=request.app.state.model.version, database=database
        )

    @app.get("/model", response_model=ModelInfoResponse, tags=["model"])
    def model_info(request: Request):
        meta = request.app.state.model.meta
        return ModelInfoResponse(
            model_name=meta.get("model_name", "churn-model"),
            model_version=str(meta["model_version"]),
            model_type=meta.get("model_type", "unknown"),
            decision_threshold=float(meta["decision_threshold"]),
            exported_at=meta.get("exported_at"),
            features=meta.get("features", []),
            test_metrics=meta.get("test_metrics", {}),
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["predictions"],
        dependencies=[Depends(require_api_key)],
    )
    def predict(customer: CustomerFeatures, request: Request, background: BackgroundTasks):
        model: ModelService = request.app.state.model
        [prediction] = model.predict([customer])
        record(request, background, [customer], [prediction])
        return PredictionResponse(
            **prediction.model_dump(), model_version=model.version, threshold=model.threshold
        )

    @app.post(
        "/predict/batch",
        response_model=BatchResponse,
        tags=["predictions"],
        dependencies=[Depends(require_api_key)],
    )
    def predict_batch(body: BatchRequest, request: Request, background: BackgroundTasks):
        count = len(body.customers)
        if count > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch of {count} customers exceeds the limit of {settings.max_batch_size}",
            )
        model: ModelService = request.app.state.model
        predictions = model.predict(body.customers)
        metrics.BATCH_SIZE.observe(count)
        record(request, background, body.customers, predictions)
        return BatchResponse(
            model_version=model.version,
            threshold=model.threshold,
            count=count,
            predictions=predictions,
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
