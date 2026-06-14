#!/usr/bin/env python3
"""Run the multi-backend fractional benchmark N times, collect per-rep
wall-clock measurements, and emit a long-format CSV plus a summary
table with median and inter-quartile range (IQR) per backend.

The underlying benchmark script (benchmark_fractional_heat_3d_all_backends.py)
already does mean-of-5 inside each backend's timing function. We wrap
it with an outer loop of N_OUTER repetitions to get a distribution of
wall-clock numbers; the median + IQR of that distribution is what
Section 6.5 of the manuscript reports as 'absolute timings.'

Outputs:
  benchmark_repeated_long.csv     : one row per (rep, backend, method)
  benchmark_repeated_summary.csv  : one row per (backend, method),
                                    columns: median_ms, p25_ms, p75_ms,
                                    iqr_ms, min_ms, max_ms, n_reps,
                                    speedup_vs_baseline_median

Usage:
  python benchmark_repeated.py [n_outer]   # default n_outer=10
"""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(REPRO_ROOT, "data")
# Honor REGEN_DIR so summary CSVs land in regen/ for diff against
# the archive instead of overwriting the archive in data/.
REGEN = os.environ.get("REGEN_DIR", DATA)
os.makedirs(REGEN, exist_ok=True)
INNER = os.path.join(HERE, "benchmark_fractional_heat_3d_all_backends.py")
INNER_CSV = os.path.join(HERE, "benchmark_fractional_heat_3d_all_backends.csv")
OUT_LONG = os.path.join(REGEN, "benchmark_repeated_long.csv")
OUT_SUMMARY = os.path.join(REGEN, "benchmark_repeated_summary.csv")


def parse_inner_csv() -> list:
    """Read the CSV the inner benchmark wrote. Each row is
    (backend, method, wall_ms, ...)."""
    with open(INNER_CSV) as f:
        rows = list(csv.DictReader(f))
    return [(r["backend"], r["method"], float(r["wall_ms"])) for r in rows]


def run_once(rep_idx: int) -> list:
    """Run the inner benchmark and return the resulting per-backend rows.

    The subprocess environment disables JAX's default 75-percent-VRAM
    preallocator (XLA_PYTHON_CLIENT_PREALLOCATE=false) so PyTorch CUDA,
    CuPy, and Julia-CUDA can coexist on a single 8 GB consumer GPU
    without OOM. JAX falls back to on-demand allocation, which adds a
    small per-launch overhead but is essential for fair multi-backend
    timing on shared hardware.
    """
    print(f"  rep {rep_idx + 1}: running benchmark...", flush=True)
    env = dict(os.environ)
    env.update({
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
        # Keep PyTorch/CuPy/JAX from fighting over the default stream.
        "CUPY_ACCELERATORS": env.get("CUPY_ACCELERATORS", "cub,cutensor"),
    })
    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, INNER],
        capture_output=True, text=True, env=env,
    )
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print("inner script failed:")
        print(res.stderr[-2000:])
        sys.exit(2)
    print(f"  rep {rep_idx + 1}: done in {elapsed:.1f}s")
    return parse_inner_csv()


def summarize(long_rows: list) -> list:
    """Group by (backend, method), compute median, IQR, min, max."""
    grouped: dict = {}
    for rep, backend, method, wall_ms in long_rows:
        grouped.setdefault((backend, method), []).append(wall_ms)

    summary = []
    # First pass: collect medians
    medians: dict = {}
    for key, vals in grouped.items():
        medians[key] = statistics.median(vals)

    for (backend, method), vals in sorted(grouped.items()):
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        median_ms = medians[(backend, method)]
        # statistics.quantiles requires n>=2; fall back to min/max otherwise
        if n >= 2:
            q25, q75 = statistics.quantiles(vals_sorted, n=4)[0], \
                       statistics.quantiles(vals_sorted, n=4)[2]
        else:
            q25 = q75 = median_ms
        iqr_ms = q75 - q25

        baseline_med = medians.get((backend, "ftcs_l1"), None)
        scalpel_med = medians.get((backend, "scalpel"), None)
        if method == "scalpel" and baseline_med is not None and scalpel_med:
            speedup = baseline_med / scalpel_med
        else:
            speedup = ""

        summary.append({
            "backend": backend,
            "method": method,
            "n_reps": n,
            "median_ms": median_ms,
            "p25_ms": q25,
            "p75_ms": q75,
            "iqr_ms": iqr_ms,
            "min_ms": vals_sorted[0],
            "max_ms": vals_sorted[-1],
            "speedup_vs_baseline_median": speedup,
        })
    return summary


def main():
    # 15 reps matches the manuscript's reported median+IQR campaign
    # (Table 2). CLI override is honored if a different count is needed.
    n_outer = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print(f"Running multi-backend benchmark {n_outer} times...")
    long_rows = []
    for rep in range(n_outer):
        for backend, method, wall_ms in run_once(rep):
            long_rows.append((rep, backend, method, wall_ms))

    # Long-format CSV
    with open(OUT_LONG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep", "backend", "method", "wall_ms"])
        w.writerows(long_rows)
    print(f"\nLong-format: {OUT_LONG} ({len(long_rows)} rows)")

    # Summary CSV
    summary = summarize(long_rows)
    with open(OUT_SUMMARY, "w", newline="") as f:
        cols = ["backend", "method", "n_reps", "median_ms", "p25_ms",
                "p75_ms", "iqr_ms", "min_ms", "max_ms",
                "speedup_vs_baseline_median"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"Summary:     {OUT_SUMMARY} ({len(summary)} rows)")

    # Human-readable summary
    print(f"\n{'Backend':<14}{'Method':<10}{'median':>10}{'IQR':>10}{'speedup':>10}")
    print("-" * 60)
    for row in summary:
        speedup_str = f"{row['speedup_vs_baseline_median']:.1f}x" if row["speedup_vs_baseline_median"] != "" else ""
        print(f"{row['backend']:<14}{row['method']:<10}"
              f"{row['median_ms']:>9.1f}ms"
              f"{row['iqr_ms']:>9.1f}ms"
              f"{speedup_str:>10}")


if __name__ == "__main__":
    main()
