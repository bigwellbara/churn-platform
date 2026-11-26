"""Release checks: the model quality gate and the deployment smoke test."""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time

import pytest
import uvicorn

from churn import gate, smoke
from churn.api.main import create_app
from churn.api.schemas import EXAMPLE_CUSTOMER
from churn.api.settings import Settings
from churn.bundle import META_FILE, REFERENCE_FILE

# The test bundle (conftest.model_bundle_dir) records roc_auc 0.8 and pr_auc 0.6.
TEST_FLOORS = {"roc_auc": 0.75, "pr_auc": 0.5}


@pytest.fixture
def bundle_copy(model_bundle_dir, tmp_path):
    """A writable copy of the test bundle, for breaking it in specific ways."""
    return shutil.copytree(model_bundle_dir, tmp_path / "bundle")


def _set_metrics(bundle, **metrics):
    meta_path = bundle / META_FILE
    meta = json.loads(meta_path.read_text())
    meta["test_metrics"] = metrics
    meta_path.write_text(json.dumps(meta))


# ------------------------------------------------------------------ model gate


def test_gate_passes_a_good_bundle(model_bundle_dir):
    result = gate.run_gate(model_bundle_dir, TEST_FLOORS)
    assert result.ok, result.failed
    assert any("loyal two-year customer" in line for line in result.passed)


def test_gate_fails_when_a_metric_is_below_its_floor(bundle_copy):
    _set_metrics(bundle_copy, roc_auc=0.9, pr_auc=0.31)
    result = gate.run_gate(bundle_copy, TEST_FLOORS)
    assert [line for line in result.failed if "pr_auc 0.310" in line]
    assert not [line for line in result.failed if "roc_auc" in line]


def test_gate_fails_when_a_metric_is_missing(bundle_copy):
    _set_metrics(bundle_copy, roc_auc=0.9)
    result = gate.run_gate(bundle_copy, TEST_FLOORS)
    assert "test pr_auc missing from model metadata" in result.failed


def test_gate_fails_without_drift_reference(bundle_copy):
    (bundle_copy / REFERENCE_FILE).unlink()
    result = gate.run_gate(bundle_copy, TEST_FLOORS)
    assert not result.ok
    assert "No reference profile" in result.failed[0]


def test_gate_fails_cleanly_when_bundle_is_missing(tmp_path):
    result = gate.run_gate(tmp_path / "nothing-here", TEST_FLOORS)
    assert len(result.failed) == 1
    assert result.failed[0].startswith("bundle does not load")


def test_gate_cli_exit_codes(bundle_copy, capsys):
    _set_metrics(bundle_copy, roc_auc=0.85, pr_auc=0.65, recall=0.70)
    assert gate.main(["--model-dir", str(bundle_copy)]) == 0
    assert "Model gate passed." in capsys.readouterr().out

    assert gate.main(["--model-dir", str(bundle_copy), "--min-recall", "0.9"]) == 1
    assert "FAIL  test recall 0.700 (floor 0.90)" in capsys.readouterr().out


def test_gate_passes_the_shipped_champion():
    """The committed model must always clear the default floors."""
    result = gate.run_gate(gate.PROJECT_ROOT / "models" / "champion")
    assert result.ok, result.failed


# ------------------------------------------------------------------ smoke test


def test_smoke_payload_matches_the_api_example():
    # smoke.py cannot import the package (it runs on a bare Python), so it keeps a copy.
    assert smoke.CUSTOMER == EXAMPLE_CUSTOMER


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api(model_bundle_dir):
    """A real HTTP server (uvicorn in a thread), so the smoke test makes real requests."""
    settings = Settings(model_dir=model_bundle_dir, git_sha="abc1234", database_url=None)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "test server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_smoke_checks_pass_against_a_live_api(live_api):
    log = smoke.run_checks(live_api, expect_sha="abc1234", startup_timeout=10)
    assert "serving the expected commit abc1234" in log
    assert any(line.startswith("batch ok") for line in log)
    assert any(line.startswith("explain ok (top factor") for line in log)
    assert log[-1].startswith("input validation ok")


def _patch_explain(monkeypatch, respond):
    """Route /predict/explain through `respond(real_status, real_body)`; everything else
    goes to the live server unchanged."""
    real = smoke._request

    def fake(base_url, path, *args, **kwargs):
        status, body = real(base_url, path, *args, **kwargs)
        return respond(status, body) if path == "/predict/explain" else (status, body)

    monkeypatch.setattr(smoke, "_request", fake)


def test_smoke_skips_explanations_on_versions_without_them(live_api, monkeypatch):
    # A rollback target from before /predict/explain existed must still pass.
    _patch_explain(monkeypatch, lambda status, body: (404, {"detail": "Not Found"}))
    log = smoke.run_checks(live_api, startup_timeout=10)
    assert "explain skipped (HTTP 404: not available in this version)" in log


def test_smoke_rejects_explanations_that_do_not_add_up(live_api, monkeypatch):
    _patch_explain(
        monkeypatch, lambda status, body: (status, {**body, "base_value": body["base_value"] + 1})
    )
    with pytest.raises(smoke.SmokeTestError, match="not the model's log-odds"):
        smoke.run_checks(live_api, startup_timeout=10)


def test_smoke_detects_the_wrong_commit(live_api):
    # Keeps polling (a new version may still be starting), then gives up.
    with pytest.raises(smoke.SmokeTestError, match="serving commit 'abc1234', expected 'def5678'"):
        smoke.run_checks(live_api, expect_sha="def5678", startup_timeout=1)


def test_smoke_cli(live_api, capsys):
    assert smoke.main([live_api, "--expect-sha", "abc1234"]) == 0
    assert "All smoke checks passed." in capsys.readouterr().out


def test_smoke_reports_an_api_that_never_comes_up(capsys):
    url = f"http://127.0.0.1:{_free_port()}"  # nothing listening
    assert smoke.main([url, "--startup-timeout", "1"]) == 1
    assert "not ready after 1s" in capsys.readouterr().out


def test_smoke_waits_for_the_new_version_to_go_live():
    """Render keeps serving the old version while the new one boots: the smoke test must
    wait for the new commit rather than fail on (or, worse, pass against) the old one."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = {"root": 0}

    class Host(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                calls["root"] += 1
                sha = "old1234" if calls["root"] <= 2 else "new5678"
                body = {"git_sha": sha}
            else:
                body = {"status": "ok", "model_version": "1"}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Host)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        health = smoke.wait_until_up(url, timeout=10, interval=0.05, expect_sha="new5678")
        assert health["status"] == "ok"
        assert calls["root"] == 3  # polled through two "old" answers
    finally:
        server.shutdown()
