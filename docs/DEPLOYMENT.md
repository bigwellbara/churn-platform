# Deployment: CI/CD to Google Cloud Run

Every merge to `main` that passes CI is deployed to Cloud Run in Johannesburg
(`africa-south1`) automatically, with no stored cloud credentials, and only after the new
version has proved it works on a private URL.

## The pipeline

```
 pull request ──▶ CI ──────────────────────────────────────────────────────────────────┐
                  ├─ lint-and-test: ruff, black, 93 tests (coverage ≥ 80%), model gate │
                  └─ docker: build the production image, run it at 1 vCPU / 1 GiB,     │
                             smoke test it                                              │
 merge to main ─▶ CI again (on the merged code) ──green──▶ Deploy                      │
                                                          ├─ build + push image         │
                                                          ├─ deploy revision, 0% traffic│
                                                          ├─ smoke test its tag URL     │
                                                          ├─ move 100% traffic to it    │
                                                          └─ smoke test the public URL  │
```

Files: [`ci.yml`](../.github/workflows/ci.yml), [`deploy.yml`](../.github/workflows/deploy.yml),
[`infra/gcp/bootstrap.sh`](../infra/gcp/bootstrap.sh) (one-time setup),
[`src/churn/gate.py`](../src/churn/gate.py), [`src/churn/smoke.py`](../src/churn/smoke.py).

### Quality gates, in order

| Gate | Catches | Where |
|---|---|---|
| Lint, format, unit tests | code bugs | CI |
| **Model gate** (`churn-gate`) | a worse or broken model: test ROC-AUC ≥ 0.80, PR-AUC ≥ 0.60, recall ≥ 0.65; probabilities valid and deterministic; a loyal two-year customer scores below a new month-to-month one; drift reference present | CI |
| **Container smoke test** | an image that builds but doesn't serve: missing system library, bad entrypoint, out of memory at Cloud Run's 1 GiB | CI |
| **Candidate smoke test** | anything specific to the cloud: config, permissions, start-up | Deploy, before any traffic |
| **Production smoke test** | a bad traffic switch | Deploy, after promotion |

The smoke test checks `/health`, `/model`, a real prediction, that a batch of the same
customer agrees with the single prediction, that invalid input is rejected with 422, and
that `GET /` reports the **exact commit** that was built (`--expect-sha`). The last check
proves the revision answering is the new one, not a cached older one.

## Zero-downtime releases without a rollback step

Cloud Run keeps every deployed version as an immutable **revision** and lets you choose
how traffic is split between them. The pipeline uses that as a blue/green deploy:

1. `gcloud run deploy --no-traffic --tag sha-abc1234` starts the new revision alongside
   the old one. Users get 0% of traffic to it, but it gets its own URL:
   `https://sha-abc1234---churn-api-<hash>.a.run.app`.
2. The smoke test runs against that URL.
3. Only if it passes: `update-traffic --to-revisions <new>=100`.

If step 2 fails, the job stops and users never saw the new version, so nothing needs to
be rolled back. The image is deployed **by digest** (`@sha256:...`), not by tag, so the
revision runs exactly the bytes CI built; a re-pushed tag cannot change it.

**Manual rollback** (e.g. a bug the smoke test can't see), instant because the old
revision still exists:

```bash
gcloud run revisions list --service churn-api --region africa-south1
gcloud run services update-traffic churn-api --region africa-south1 --to-revisions <old-revision>=100
```

## Keyless authentication (Workload Identity Federation)

The usual shortcut is to download a JSON key for a service account and paste it into
GitHub Secrets. That key never expires, works from anywhere, and is a common source of
cloud breaches. This project has no keys:

1. GitHub Actions issues each job a short-lived signed token (OIDC) stating facts like
   `repository: bigwellbara/churn-platform`, `ref: refs/heads/main`.
2. Google verifies the signature against GitHub's public keys and checks the **attribute
   condition** set by `bootstrap.sh`: only this repository's `main` branch is accepted.
   A fork, another repo, or a pull-request branch is refused.
3. Google returns short-lived credentials for the `github-deployer` service account,
   usable only for the rest of that job.

The repository variables (`GCP_PROJECT_ID`, `GCP_WIF_PROVIDER`, ...) are not secrets: they
say *who* may deploy, not *how*.

### Least privilege

| Identity | Can | Cannot |
|---|---|---|
| `github-deployer` (CI) | push to the `churn` image repo, deploy Cloud Run, start revisions as `churn-api-runtime` | read data, change IAM, touch other services |
| `churn-api-runtime` (the running API) | nothing | anything: the API needs no Google APIs, so a compromised container gets no cloud access |

`run.admin` is broader than strictly needed: it is required to make the service public
(set its invoker policy). A tighter setup would grant `run.developer` and make the
service public once by hand.

## Service configuration and why

| Setting | Value | Reason |
|---|---|---|
| Region | `africa-south1` (Johannesburg) | lowest latency for South African users. It is a Tier 2 (pricier) region; at demo traffic the free tier still covers it |
| CPU / memory | 1 vCPU, 1 GiB | measured: ~310 MiB in use after start-up |
| Min / max instances | 0 / 2 | scale to zero when idle (no cost); 2 caps the bill if the public URL is abused |
| Concurrency | 40 requests per instance | micro-batching works best with many concurrent requests in one process |
| `INFERENCE_PROCESSES` | 0 (model in the API process) | see below |
| `--cpu-boost` | on | extra CPU during start-up shortens cold starts |
| Execution environment | gen2 | full Linux compatibility (multiprocessing, `/dev/shm`) |

**Why not the worker processes that won the load test?** Measured at 1 vCPU, a model
worker process adds about 4 s to every cold start (start to healthy: **6 s** in-process,
**10 s** with one worker), because it re-imports pandas, scikit-learn and XGBoost. With
scale-to-zero, cold starts are exactly what a visitor to a demo hits, while 50 req/s
sustained is not. With one vCPU there is also little parallelism for a second process to
use. The local high-throughput setup (`compose.yaml`) keeps `INFERENCE_PROCESSES=2`. Same
image, different environment variables: that is the 12-factor idea in practice.

## Logs and metrics in the cloud

- **Logs:** the API writes one JSON object per line with a `severity` field, which Cloud
  Logging parses, so you can filter e.g. `jsonPayload.latency_ms > 100` or a single
  `jsonPayload.request_id` in the Logs Explorer.
- **Metrics:** Cloud Run records request count, latency percentiles, instance count and
  cold starts with no setup. The app's own Prometheus metrics stay available at
  `/metrics`.

### Not deployed (yet)

The Postgres prediction log, drift monitor, Prometheus and Grafana run in
`docker compose` locally but not in the cloud: the API runs with `DATABASE_URL` unset,
which turns prediction logging off (it is optional by design). Adding them would mean a
managed Postgres (Cloud SQL, roughly US$10/month for the smallest instance, or a free-tier
serverless Postgres), `DATABASE_URL` stored in Secret Manager, the drift monitor as a
Cloud Run job on a schedule, and Google Managed Service for Prometheus.

## Cost

At demo traffic this should cost nothing:

- Cloud Run bills only while handling requests (min instances 0), and the monthly free
  tier (millions of requests, many vCPU-hours) far exceeds demo use. See the
  [Cloud Run pricing page](https://cloud.google.com/run/pricing).
- Artifact Registry: the first 0.5 GB of storage is free. The cleanup policy keeps the 10
  newest images and deletes older ones after 30 days; images share most layers, so this
  stays small.
- `bootstrap.sh` can add a monthly **budget alert** (`BUDGET=100ZAR`), which emails at
  50%, 90% and 100%.

## One-time setup

1. A Google Cloud project with billing enabled, and the
   [gcloud CLI](https://cloud.google.com/sdk/docs/install) logged in (`gcloud auth login`).
2. Run:

   ```bash
   PROJECT_ID=<your-project> BUDGET=100ZAR bash infra/gcp/bootstrap.sh
   ```

   It enables the APIs, creates the image repository, both service accounts and the
   identity federation, and (with the GitHub CLI) sets the repository variables.
3. Push to `main`, or run the **Deploy** workflow from the Actions tab.
