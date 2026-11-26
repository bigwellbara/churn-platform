# Deployment: CI/CD

Every merge to `main` that passes CI is built once into a container image and deployed
automatically, only after the new version has proved it works.

| Target | Status | Cost |
|---|---|---|
| **Render** (free web service, Frankfurt): the API | live | free |
| **Vercel** (Hobby): the dashboard, [churn-platform-dusky.vercel.app](https://churn-platform-dusky.vercel.app) | live; redeploys on every push to `main`, preview URL per pull request | free |
| **Google Cloud Run** (Johannesburg) | pipeline ready, runs once a GCP project is configured | free tier, but GCP needs billing set up |

## The pipeline

```
 pull request ──▶ CI
                  ├─ lint-and-test: ruff, black, tests (coverage ≥ 80%), model gate
                  └─ docker: build the production image, run it at 512 MB, smoke test it

 merge to main ─▶ CI again (on the merged code) ──green──▶ Deploy
                                                           ├─ publish: image ─▶ ghcr.io
                                                           ├─ render: deploy by digest,
                                                           │   wait for new commit, smoke
                                                           │   test, auto-rollback on failure
                                                           └─ cloud-run: (if configured)

 every 10 min ──▶ Uptime: smoke test production (opt-in, also keeps the free instance awake)
```

Files: [`ci.yml`](../.github/workflows/ci.yml), [`deploy.yml`](../.github/workflows/deploy.yml),
[`uptime.yml`](../.github/workflows/uptime.yml), [`src/churn/gate.py`](../src/churn/gate.py),
[`src/churn/smoke.py`](../src/churn/smoke.py), [`infra/gcp/bootstrap.sh`](../infra/gcp/bootstrap.sh).

### Quality gates, in order

| Gate | Catches | Where |
|---|---|---|
| Lint, format, unit tests | code bugs | CI |
| **Model gate** (`churn-gate`) | a worse or broken model: test ROC-AUC ≥ 0.80, PR-AUC ≥ 0.60, recall ≥ 0.65; probabilities valid and deterministic; a loyal two-year customer scores below a new month-to-month one; drift reference present | CI |
| **Container smoke test** | an image that builds but doesn't serve: missing system library, bad entrypoint, out of memory at 512 MB | CI |
| **Production smoke test** | anything specific to the host: configuration, start-up, networking | Deploy |
| **Uptime check** | a deployment that breaks later, while nobody is using it | every 10 min (opt-in) |

The smoke test checks `/health`, `/model`, a real prediction, that a batch of the same
customer agrees with the single prediction, that invalid input is rejected with 422, and
that `GET /` reports the **exact commit** that was built (`--expect-sha`). The last check
proves the version answering is the new one, not the previous one still running.

## Build once, deploy the same bytes

The `publish` job builds the image once and pushes it to GitHub Container Registry as
`ghcr.io/bigwellbara/churn-api:<commit-sha>` (and `:latest`). Deploys reference it **by
digest** (`@sha256:...`), so what runs is exactly what was built from the commit CI
tested; re-pushing a tag cannot change it. GHCR is free for public repositories, and the
job authenticates with its own short-lived `GITHUB_TOKEN`, so there is no registry
password to manage.

## Render: deploy, verify, roll back

1. The workflow records which commit is live now (`GET /` reports it).
2. It calls the service's **deploy hook** with `imgURL=ghcr.io/...@sha256:<digest>`.
3. Render starts the new version and only switches traffic once its `/health` check
   passes; until then the old version keeps serving.
4. The smoke test polls until `GET /` reports the **new** commit, then runs every check.
5. **If anything fails, the workflow redeploys the previous commit's image** through the
   same hook, waits for it, and smoke tests it, then marks the run failed.

The deploy hook URL contains a key, so it is stored as a GitHub **secret**
(`RENDER_DEPLOY_HOOK_URL`). The service URL is a plain variable (`RENDER_SERVICE_URL`).

### Sizing for a free instance (measured)

Render's free instance has **0.1 CPU and 512 MB**. Running the image under exactly those
limits (`docker run --cpus 0.1 --memory 512m`):

| | 0.1 CPU (Render free) | 0.5 CPU | 1 CPU |
|---|---|---|---|
| Start to healthy | ~95 s | ~12 s | ~6 s |
| Memory in use | 169 MB | 169 MB | ~170 MB |
| `/predict` once warm | 150–380 ms | 15–80 ms | ~20 ms |

Start-up is almost entirely importing pandas, scikit-learn and XGBoost and loading the
model, which is CPU-bound, so it scales with the CPU share. The model is fast; the box
is small.

**Why the image now defaults to `INFERENCE_PROCESSES=0`.** The load test's winning
setup ran the model in worker processes (docs/LOAD_TEST.md). On a tiny instance that is
the wrong trade: a worker re-imports the whole ML stack, which measured **199 s** to start
at 0.1 CPU (vs 96 s) and **308 MB** of memory (vs 169 MB), uncomfortably close to the
512 MB limit. Free instances serve demo traffic, not 50 req/s, so the default is the
small, fast-starting configuration, and `compose.yaml` sets `INFERENCE_PROCESSES=2` for
the high-throughput local stack. Same image, different environment variable.

### Cold starts and keeping it awake

Free instances sleep after 15 minutes without traffic. Waking means Render's own spin-up
plus the ~95 s start-up above, so the first visitor after a quiet period waits roughly
two minutes. The **Uptime** workflow fixes that as a side effect of being a monitor: it
smoke tests production every 10 minutes. One always-awake service uses ~744 of the 750
free instance hours Render gives each workspace per month, so it is opt-in (repository
variable `KEEP_WARM=true`); leave it off if other free services share the workspace.

The real fix for cold starts is a lighter serving stack: exporting the pipeline to ONNX
would drop pandas and scikit-learn from the image (see docs/LOAD_TEST.md, next steps).

### Render setup (one time)

1. **Merge to `main` once** so the `publish` job pushes the first image. The `render` and
   `cloud-run` jobs skip themselves until configured.
2. **Make the image public:** GitHub → your profile → Packages → `churn-api` →
   Package settings → Change visibility → Public. (Personal-account packages start
   private; a public image needs no registry credentials on Render.)
3. **Create the service** at [render.com](https://render.com) (sign in with GitHub):
   New → Web Service → **Existing image** → `ghcr.io/bigwellbara/churn-api:latest`.
   Name `churn-api`, region **Frankfurt** (closest to South Africa), instance type
   **Free**. Under Advanced, set **Health Check Path** to `/health`. No environment
   variables are needed; Render sets `PORT` and the image reads it.
4. **Connect the pipeline:** in the service's Settings copy the **Deploy Hook**, then:

   ```bash
   gh secret set RENDER_DEPLOY_HOOK_URL          # paste the hook URL when prompted
   gh variable set RENDER_SERVICE_URL --body https://churn-api-xxxx.onrender.com
   gh variable set KEEP_WARM --body true          # optional, see above
   ```

5. Actions → **Deploy** → Run workflow, to watch the first gated deploy.

## Google Cloud Run (ready, needs billing)

The `cloud-run` job runs automatically once `infra/gcp/bootstrap.sh` has been run
against a project with billing. It is a stricter blue/green deploy than Render's:

1. `gcloud run deploy --no-traffic --tag sha-abc1234` starts the new revision beside the
   old one with **0% of traffic**, reachable on its own URL
   (`https://sha-abc1234---churn-api-<hash>.a.run.app`).
2. The smoke test runs against that private URL.
3. Only if it passes: `update-traffic --to-revisions <new>=100`, then the public URL is
   smoke tested.

A failed candidate never receives a single user request, so nothing needs rolling back.
Manual rollback is instant, because old revisions are kept:
`gcloud run services update-traffic churn-api --region africa-south1 --to-revisions <old>=100`.

**Keyless authentication (Workload Identity Federation).** No service-account key is ever
created. GitHub Actions gives each job a short-lived signed OIDC token stating its
repository and branch; Google verifies it and checks an attribute condition that only
accepts this repository's `main` branch (forks and pull-request branches are refused),
then issues short-lived credentials for a `github-deployer` service account.

| Identity | Can | Cannot |
|---|---|---|
| `github-deployer` (CI) | push to the `churn` image repo, deploy Cloud Run, start revisions as `churn-api-runtime` | read data, change IAM, touch other services |
| `churn-api-runtime` (the running API) | nothing | anything: the API needs no Google APIs |

Service settings: 1 vCPU / 1 GiB, scale to zero, at most 2 instances (a cost ceiling),
40 concurrent requests per instance (micro-batching benefits from concurrency),
start-up CPU boost, gen2 execution environment. Johannesburg is a Tier 2 (pricier)
region, but demo traffic stays within the free tier
([Cloud Run pricing](https://cloud.google.com/run/pricing)). `bootstrap.sh` also sets an
image cleanup policy and, with `BUDGET=100ZAR`, a budget alert.

Setup: a project with billing, `gcloud auth login`, then
`PROJECT_ID=<project> BUDGET=100ZAR bash infra/gcp/bootstrap.sh`.

## Logs and metrics

- **Logs:** one JSON object per line with a `severity` field (the key Google Cloud
  Logging reads), so log platforms can filter on `latency_ms` or a single `request_id`.
- **Metrics:** the app's Prometheus metrics are at `/metrics`; Render and Cloud Run also
  chart request rates, latency and memory without setup.

### Not deployed (yet)

The Postgres prediction log, drift monitor, Prometheus and Grafana run in
`docker compose` locally but not in the cloud: the API runs with `DATABASE_URL` unset,
which turns prediction logging off (it is optional by design). Adding them needs a hosted
Postgres (Render's free Postgres is time-limited; a free-tier serverless Postgres or Cloud
SQL are alternatives), with `DATABASE_URL` stored as a secret.
