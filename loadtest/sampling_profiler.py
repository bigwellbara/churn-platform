"""Minimal sampling profiler for the running API (no extra dependencies).

Starts the API with uvicorn, and every 2 ms records what each busy thread is executing.
After --seconds it prints the functions seen most often: where the time goes.

    docker compose run --rm --no-deps -p 18002:8000 -v ./loadtest:/mnt/locust:ro api \
        python /mnt/locust/sampling_profiler.py --seconds 40
    (then drive load at http://localhost:18002 with Locust)
"""

import argparse
import collections
import os
import sys
import threading
import time

import uvicorn

# Frames that mean "this thread is idle, waiting for work", not using CPU.
IDLE = {"select", "wait", "_wait_for_tstate_lock", "get", "poll", "sleep", "accept", "run_forever"}


def frame_name(frame) -> str:
    code = frame.f_code
    path = code.co_filename
    marker = "site-packages/"
    short = path.split(marker, 1)[1] if marker in path else os.path.basename(path)
    return f"{short}:{code.co_name}"


def sample(stop: threading.Event, inclusive: collections.Counter, totals: list[int]) -> None:
    me = threading.get_ident()
    while not stop.is_set():
        for tid, frame in sys._current_frames().items():
            if tid == me or frame.f_code.co_name in IDLE:
                continue
            totals[0] += 1
            seen = set()
            while frame is not None:
                name = frame_name(frame)
                if name not in seen:
                    inclusive[name] += 1
                    seen.add(name)
                frame = frame.f_back
        time.sleep(0.002)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    inclusive: collections.Counter = collections.Counter()
    totals = [0]
    stop = threading.Event()
    config = uvicorn.Config(
        "churn.api.main:create_app", factory=True, host="0.0.0.0", port=8000, access_log=False
    )
    server = uvicorn.Server(config)
    threading.Thread(target=sample, args=(stop, inclusive, totals), daemon=True).start()
    threading.Timer(args.seconds, lambda: setattr(server, "should_exit", True)).start()
    server.run()
    stop.set()

    print(f"\n{totals[0]} busy-thread samples. Inclusive share of samples per function:")
    for name, count in inclusive.most_common(args.top):
        print(f"{100 * count / max(totals[0], 1):6.1f}%  {name}")


if __name__ == "__main__":
    main()
