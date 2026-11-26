"""Smoke test for a running API: does this deployment actually work?

Used in three places, against the same checks:
  * CI, against the freshly built Docker image (`.github/workflows/ci.yml`)
  * the deploy pipeline, against a new Cloud Run revision *before* it gets any traffic,
    and again after it is promoted (`.github/workflows/deploy.yml`)
  * by hand: `python src/churn/smoke.py https://<your-service>.run.app`

Standard library only, so the deploy job can run it with the runner's own Python without
installing the project (pandas, scikit-learn, ...). Keep it that way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# A copy of churn.api.schemas.EXAMPLE_CUSTOMER (a test asserts they stay identical; this
# module cannot import the package).
CUSTOMER: dict[str, Any] = {
    "customer_id": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class SmokeTestError(AssertionError):
    pass


def _request(
    base_url: str,
    path: str,
    body: Any = None,
    api_key: str | None = None,
    timeout: float = 30,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json", "User-Agent": "churn-smoke-test"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:  # 4xx/5xx still carry a JSON body
        payload = err.read()
        try:
            return err.code, json.loads(payload or b"null")
        except json.JSONDecodeError:
            return err.code, payload.decode(errors="replace")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestError(message)


def wait_until_up(
    base_url: str, timeout: float = 120, interval: float = 3, expect_sha: str | None = None
) -> dict:
    """Poll until /health answers 200 and, if `expect_sha` is given, until GET / reports
    that commit. Absorbs start-up, cold starts, and hosts that keep serving the previous
    version while the new one boots (Render does this)."""
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = _request(base_url, "/health", timeout=30)
            if status != 200:
                last_error = f"HTTP {status}: {body}"
            elif not expect_sha:
                return body
            else:
                _, root = _request(base_url, "/", timeout=30)
                serving = root.get("git_sha") if isinstance(root, dict) else None
                if serving == expect_sha:
                    return body
                last_error = f"serving commit {serving!r}, expected {expect_sha!r}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as err:
            last_error = repr(err)
        time.sleep(interval)
    raise SmokeTestError(f"{base_url} not ready after {timeout:.0f}s ({last_error})")


def run_checks(
    base_url: str,
    expect_sha: str | None = None,
    api_key: str | None = None,
    startup_timeout: float = 120,
) -> list[str]:
    """Run every check; raise SmokeTestError on the first failure. Returns a log."""
    log: list[str] = []

    health = wait_until_up(base_url, timeout=startup_timeout, expect_sha=expect_sha)
    _check(health.get("status") == "ok", f"/health status is {health.get('status')!r}")
    version = health.get("model_version")
    _check(bool(version), "/health reports no model version")
    log.append(f"health ok (model v{version}, database {health.get('database')})")

    status, root = _request(base_url, "/")
    _check(status == 200, f"GET / returned {status}")
    if expect_sha:
        # Proves the revision answering is the commit we just built, not an older one.
        _check(
            root.get("git_sha") == expect_sha,
            f"serving commit {root.get('git_sha')!r}, expected {expect_sha!r}",
        )
        log.append(f"serving the expected commit {expect_sha[:12]}")

    status, model = _request(base_url, "/model")
    _check(status == 200, f"GET /model returned {status}")
    _check(0 < model.get("decision_threshold", -1) < 1, "decision threshold not in (0, 1)")
    log.append(f"model {model.get('model_type')} threshold {model['decision_threshold']}")

    status, pred = _request(base_url, "/predict", CUSTOMER, api_key=api_key)
    _check(status == 200, f"POST /predict returned {status}: {pred}")
    p = pred.get("churn_probability")
    _check(isinstance(p, float) and 0.0 <= p <= 1.0, f"bad probability {p!r}")
    _check(isinstance(pred.get("churn_prediction"), bool), "churn_prediction is not a boolean")
    _check(pred.get("model_version") == version, "prediction model version != /health")
    _check(pred.get("customer_id") == CUSTOMER["customer_id"], "customer_id not echoed")
    log.append(f"predict ok (p={p:.3f}, risk {pred.get('risk_level')})")

    status, batch = _request(base_url, "/predict/batch", {"customers": [CUSTOMER] * 3}, api_key)
    _check(status == 200, f"POST /predict/batch returned {status}: {batch}")
    _check(len(batch.get("predictions", [])) == 3, "batch did not return 3 predictions")
    _check(
        all(abs(b["churn_probability"] - p) < 1e-9 for b in batch["predictions"]),
        "batch and single predictions disagree for the same customer",
    )
    log.append("batch ok (matches single prediction)")

    bad = {**CUSTOMER, "tenure": -5}
    status, _ = _request(base_url, "/predict", bad, api_key=api_key)
    _check(status == 422, f"invalid input returned {status}, expected 422")
    log.append("input validation ok (bad request rejected with 422)")
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="e.g. http://localhost:8000")
    parser.add_argument("--expect-sha", help="fail unless GET / reports this git commit")
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args(argv)
    api_key = os.environ.get("SMOKE_API_KEY") or None

    print(f"Smoke testing {args.base_url}")
    try:
        for line in run_checks(args.base_url, args.expect_sha, api_key, args.startup_timeout):
            print(f"  PASS  {line}")
    except SmokeTestError as err:
        print(f"  FAIL  {err}")
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
