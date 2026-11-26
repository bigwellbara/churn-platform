"""Print a compact summary of one or more Locust CSV runs.

uv run python loadtest/summarize.py reports/loadtest/baseline reports/loadtest/optimized
"""

import csv
import sys
from pathlib import Path

print(
    f"{'run':<12}{'endpoint':<22}{'reqs':>6}{'fail%':>7}{'req/s':>7}{'p50':>6}{'p95':>6}{'p99':>6}"
)
for prefix in sys.argv[1:]:
    path = Path(f"{prefix}_stats.csv")
    for row in csv.DictReader(path.open(encoding="utf-8")):
        reqs = int(row["Request Count"])
        fail_pct = 100 * int(row["Failure Count"]) / max(reqs, 1)
        print(
            f"{Path(prefix).name:<12}{row['Name']:<22}{reqs:>6}{fail_pct:>6.1f}%"
            f"{float(row['Requests/s']):>7.1f}{row['50%']:>6}{row['95%']:>6}{row['99%']:>6}"
        )
print("latencies in ms")
