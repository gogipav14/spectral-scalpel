#!/usr/bin/env python3
"""
Compare two CSV directories row-by-row within a relative tolerance.

Used by paper_toms/reproducibility/reproduce.sh to verify a fresh
reproduction matches the archived CSVs in paper_toms/data/.

Exit code 0 = all rows within tolerance; non-zero = drift detected.

Usage:
    python csv_diff.py --archive paper_toms/data --regen reports/repro_run --tolerance 0.05
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, help="Directory of archived CSVs (truth)")
    p.add_argument("--regen", required=True, help="Directory of regenerated CSVs")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="Relative tolerance per numeric cell (default 0.05)")
    p.add_argument("--skip-missing", action="store_true",
                   help="Treat missing regenerated copies as 'skipped', not 'failed' "
                        "(useful for CPU-only CI where the GPU-only CSVs aren't regenerated)")
    p.add_argument("--only", nargs="+", default=None,
                   help="If given, restrict the diff to these CSV basenames (allow-list)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def read_csv(path: str) -> tuple:
    """Return (header, rows) for a CSV; rows are list-of-strings."""
    with open(path, "r") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def cell_diff(archived: str, regen: str, tol: float) -> tuple:
    """Compare two cell values. Returns (ok: bool, rel_err: float)."""
    if archived == regen:
        return True, 0.0
    try:
        a = float(archived)
        r = float(regen)
    except ValueError:
        return False, float("nan")
    if a == 0 and r == 0:
        return True, 0.0
    denom = max(abs(a), abs(r), 1e-300)
    rel = abs(a - r) / denom
    return rel <= tol, rel


def diff_file(archive_path: str, regen_path: str, tol: float,
              verbose: bool = False, skip_missing: bool = False) -> tuple:
    """Diff two CSV files. Returns (rows_compared, rows_failing, issues, skipped).

    If ``skip_missing`` is True and the regenerated copy is absent, the
    function returns rows_failing=0 and adds a SKIPPED note instead of
    a failure; the caller decides how to surface skipped files.
    """
    if not os.path.exists(regen_path):
        if skip_missing:
            return 0, 0, [f"{os.path.basename(archive_path)}: SKIPPED (regen copy missing)"], True
        return 0, 0, [f"{os.path.basename(archive_path)}: regenerated copy missing"], False

    archived_header, archived_rows = read_csv(archive_path)
    regen_header, regen_rows = read_csv(regen_path)

    issues = []
    if archived_header != regen_header:
        issues.append(
            f"{os.path.basename(archive_path)}: header mismatch "
            f"({archived_header} vs {regen_header})"
        )
        return 0, 1, issues

    skipped = False
    n_rows = min(len(archived_rows), len(regen_rows))
    if len(archived_rows) != len(regen_rows):
        issues.append(
            f"{os.path.basename(archive_path)}: row count "
            f"{len(archived_rows)} (archived) vs {len(regen_rows)} (regen)"
        )

    failing = 0
    for ri in range(n_rows):
        a_row = archived_rows[ri]
        r_row = regen_rows[ri]
        if len(a_row) != len(r_row):
            issues.append(f"row {ri}: column count mismatch")
            failing += 1
            continue
        for ci, (a, r) in enumerate(zip(a_row, r_row)):
            ok, rel = cell_diff(a, r, tol)
            if not ok:
                issues.append(
                    f"row {ri} col {ci} ({archived_header[ci] if ci < len(archived_header) else '?'}): "
                    f"archived={a} regen={r} rel_err={rel:.3g}"
                )
                failing += 1
                break  # one failure per row is enough to report

    if verbose and not issues:
        print(f"OK: {os.path.basename(archive_path)} ({n_rows} rows)")
    return n_rows, failing, issues, skipped


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.archive):
        print(f"archive dir not found: {args.archive}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.regen):
        print(f"regen dir not found: {args.regen}", file=sys.stderr)
        return 2

    archived_files = sorted(f for f in os.listdir(args.archive) if f.endswith(".csv"))
    if args.only:
        archived_files = [f for f in archived_files if f in set(args.only)]
    if not archived_files:
        print(f"no .csv files in archive dir {args.archive}", file=sys.stderr)
        return 2

    total_rows = 0
    total_fail = 0
    total_skipped = 0
    all_issues = []
    for name in archived_files:
        n, f, issues, skipped = diff_file(
            os.path.join(args.archive, name),
            os.path.join(args.regen, name),
            args.tolerance,
            verbose=args.verbose,
            skip_missing=args.skip_missing,
        )
        total_rows += n
        total_fail += f
        if skipped:
            total_skipped += 1
        all_issues.extend(issues)

    print(f"\nCSV diff: {len(archived_files)} files inspected, "
          f"{total_rows} rows compared, "
          f"{total_fail} row(s) out of tolerance, "
          f"{total_skipped} file(s) skipped.")
    if all_issues:
        print("\nIssues / notes:")
        for line in all_issues:
            print(f"  - {line}")
    if total_fail:
        return 1
    # Treat "missing regenerated copy" as failure only when not skipping.
    if not args.skip_missing and any(
        "missing" in i or "mismatch" in i for i in all_issues
    ):
        return 1
    print("All compared cells within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
