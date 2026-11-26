# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder
# Has uv and build tooling; produces /app/.venv. Nothing from here ships except the venv.
FROM python:3.11.17-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependencies only. This layer is cached until pyproject.toml / uv.lock change,
#    so editing source code does not trigger a full reinstall.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group api --no-install-project

# 2) The project itself, installed as a regular (non-editable) package.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group api --no-editable

# ---------------------------------------------------------------- runtime
FROM python:3.11.17-slim-bookworm AS runtime

# libgomp: OpenMP runtime that XGBoost needs for multi-threaded prediction.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user: a compromised process is not root in the container.
RUN useradd --create-home --uid 10001 app

COPY --from=builder /app/.venv /app/.venv
COPY models/champion /app/models/champion

# INFERENCE_PROCESSES=2: the model runs in two worker processes, so it never competes with
# the web server for the GIL and two batches can be scored at once
# (see src/churn/api/inference.py and docs/LOAD_TEST.md).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/champion \
    INFERENCE_PROCESSES=2 \
    PORT=8000

USER app
WORKDIR /app
EXPOSE 8000

# The commit this image was built from; GET / reports it. Declared last so that a new
# commit only rebuilds this tiny layer, not the dependency layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"

# exec: uvicorn becomes PID 1 and receives SIGTERM directly, so shutdowns are graceful.
# Our middleware already writes a JSON log line per request, so uvicorn's access log is off.
CMD ["sh", "-c", "exec uvicorn churn.api.main:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]
