# Load test and performance optimisation

**Result:** at 50 requests/second, `/predict` latency went from **p50 1,400 ms / p95
1,700 ms** (and an API that could not keep up) to **p50 130 ms / p95 280 ms**, with
0 errors. That is **11× faster at p50 and 6× at p95**, measured the same way before and
after. The 100 ms p95 target is **not yet met** at 50 req/s on this machine; why, and what
would close the gap, is below.

## Setup

- **Load:** [Locust](https://locust.io), `loadtest/locustfile.py`. 50 users × ~1 request/s each,
  90 s. Mix: 90% single predictions, 10% batches of 50 customers (random, valid
  customers; `BATCH_WEIGHT=0` gives singles only).
- **System under test:** the full `docker compose` stack (API, Postgres, drift monitor,
  Prometheus) on a 16-vCPU WSL2 VM on a Windows laptop. The API container has no CPU
  limit.
- **Two ways of running Locust:** from Windows (crosses the Windows→WSL network relay,
  same method as the baseline) and from inside WSL (no relay, cleaner).
- **Variance is high on this machine:** the same configuration measured p95 260 ms and
  440 ms in two runs. Final numbers are therefore given as ranges over 3 repeats.
- Raw CSVs go to `reports/loadtest/` (git-ignored). Summaries: `loadtest/summarize.py`.

```bash
uv run --group load locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless --users 50 --spawn-rate 10 --run-time 90s --csv reports/loadtest/run --only-summary
uv run python loadtest/summarize.py reports/loadtest/run
```

## Results, step by step

50 users, 90 s, standard mix, Locust on Windows (same method throughout):

| Step | Change | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|
| 0 | Baseline | 35.6 req/s (saturated) | 1,400 | 1,700 | 2,000 |
| 1 | XGBoost `n_jobs=1` for serving | 38.6 | 1,300 | 1,500 | 1,600 |
| 2 | Dynamic micro-batching + bulk INSERT | 49.1 | 290 | 460 | 630 |
| 3 | Write-behind prediction log | 49.2 | 190 | 330 | 390 |
| 4 | Model in a separate worker process | 49.3 | 150 | 260 | 300 |
| 5 | 2 worker processes, 2 batches in flight | 49.3 | **130** | **280** | 350 |

Latencies in ms. Step 5 from inside WSL, 3 repeats: p50 **89–110 ms**, p95 **200–290 ms**
(mixed traffic); singles only: p50 79–120 ms, p95 190–260 ms. A single request with no
concurrent load: **p50 23 ms, p95 37 ms**.

## What each step fixed, and the evidence

**Step 0, measure first.** Profiling the model alone (`loadtest/bench_model.py`) showed one
prediction took ~32 ms, with XGBoost spending 17 ms at p95 for work that takes 1 ms.

**Step 1, thread oversubscription.** The model was trained with `n_jobs=-1` (all 16 cores).
Right for training; wrong for serving, where every single-row prediction woke 16 threads.
`n_jobs=1` cut a prediction from 32 to 11 ms in isolation. Under load it barely helped,
because the API was limited elsewhere: one process at ~100% CPU (the GIL).

**Step 2, dynamic micro-batching** (`src/churn/api/batching.py`). Profiling showed scoring
**50 customers costs the same as scoring 1** (~11 ms of fixed sklearn overhead). Concurrent
`/predict` requests now join a queue, and one model call scores everything waiting.
Average batch under load: 12 to 16 requests. Turning batching off (A/B) took p50 from 190 to
650 ms. Same idea as Triton / TF Serving dynamic batching.

**Step 3, write-behind logging** (`src/churn/api/log_writer.py`). A sampling profiler
(`loadtest/sampling_profiler.py`) showed **49% of busy-thread time** in per-request
database writes: 50 INSERT transactions/s, each holding a thread-pool thread and a pooled
connection, and (because of HTTP keep-alive) delaying that client's next request. Rows now
go into an in-memory queue and are bulk-inserted every 0.5 s. Trade-offs: up to 0.5 s of
log rows lost on a crash (a clean shutdown flushes), and a bounded queue that drops and
counts rows during long DB outages instead of exhausting memory.

**Step 4, the GIL convoy** (`src/churn/api/inference.py`). New metrics showed model calls
taking **76 ms p50 / 237 ms p95 under load versus 11 ms in isolation, while the CPU was 75%
idle**: slow but not computing. The event-loop thread (all network I/O) and the model
thread kept handing Python's GIL back and forth. The model now runs in a separate process
with its own GIL; the round trip costs ~1 ms (measured). Event-loop lag p95 fell from
52 ms to 8 ms.

**Step 5, parallel model workers.** With one worker, single predictions could queue behind
50-customer batch requests. Two workers let batches and singles proceed side by side
(in-WSL p50 120 → 85 ms). Allowing two micro-batches in flight added no clear further gain.

## Hypotheses tested and rejected (with data)

| Hypothesis | Test | Result |
|---|---|---|
| Windows→WSL network relay is the bottleneck | Locust inside WSL vs from Windows | relay adds ~30 ms at p50; not the main cost |
| Python's 5 ms GIL switch interval | set it to 0.5 ms | **worse**: p95 980 ms |
| One web-server process is the limit | `WEB_CONCURRENCY=2` | marginal (p95 290 → 250 ms) |
| Other containers on the machine | CPU survey | VM 93% idle |
| Idle-CPU wake-up delays in the VM | benchmark with 0–150 ms pauses | no difference |
| TCP Nagle / delayed-ACK stall | time single requests | p50 23 ms: no stall |

## Why p95 is still above 100 ms, and what would close the gap

- **The model has a floor of ~17 ms p50 / ~30 ms p95 per call**, almost all fixed
  per-call overhead in the scikit-learn preprocessing (pandas DataFrame, ColumnTransformer,
  one-hot encoding). With micro-batching, a request can wait for the batch in progress
  plus its own, so ~2× that floor, plus tail effects under concurrency.
- **Next steps, in order of expected impact:**
  1. Make each model call cheaper: compile the pipeline to ONNX (`skl2onnx` +
     `onnxruntime`) or a numpy-only preprocessing path, verified against the sklearn
     pipeline by a test. Typically a 5–10× speed-up for small batches.
  2. Scale horizontally (more containers behind a load balancer): Week 4 on Cloud Run.
  3. Re-measure on cloud infrastructure: a laptop VM with high run-to-run variance is
     not a production benchmark.

## Lessons from the testing environment

- **WSL shuts its VM down** when no Windows-side `wsl` process is attached, even with
  Docker running inside. The first baseline run started while WSL was cold-booting and
  recorded 24% connection failures; it was discarded and re-run.
- **Verify the target before sending load.** One A/B run hit a port already used by
  another local service; the A/B script now refuses to start if the port is taken and
  checks it is talking to this API before any load is sent.
- **Verify the build before measuring.** One "final" run measured the previous image
  because the rebuild had silently failed on a network timeout; it was discarded.
