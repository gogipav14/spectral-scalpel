"""
Parameter-sweep harness for batched scalpel runs.

Lets users (and reviewers) sweep one or more PDE parameters over a grid
of values and collect the results in a single call. The sweep is the
outer loop over parameters; the inner batched FFT-NILT primitive is
unchanged. Wall time scales linearly with the sweep length, but the
per-call setup cost (JIT compile, contour build) is amortized.

Example
-------
>>> from scalpel.sweep import parameter_sweep
>>> results = parameter_sweep(
...     forward_callable=propagate_maxwell,
...     varying={'depth': [0.1, 0.2, 0.5, 1.0, 2.0]},
...     fixed={'source_xy': source, 'material': material,
...            'grid': grid, 'nilt': nilt},
...     reduce='centerline',           # keep only the centerline trace
...     progress=True,
... )
>>> results.to_csv('depth_sweep.csv')
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SweepResult:
    """One row per parameter combination."""

    columns: list                # parameter names + reduction name(s)
    rows: list                   # list of tuples matching columns
    elapsed_seconds: float

    def to_csv(self, path: str) -> None:
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            for row in self.rows:
                w.writerow(row)

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": [list(r) for r in self.rows],
            "elapsed_seconds": self.elapsed_seconds,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return (
            f"SweepResult({len(self.rows)} rows, "
            f"{self.elapsed_seconds:.2f}s, columns={self.columns})"
        )


def _reduce_field(field, reduction: str) -> tuple:
    """Reduce a (Nt, Nx, Ny) field to a scalar or short-vector summary."""
    arr = np.asarray(field)
    if reduction == "centerline":
        nt = arr.shape[0] if arr.ndim == 3 else 1
        cx, cy = arr.shape[1] // 2, arr.shape[2] // 2
        return (float(arr[nt // 2, cx, cy]),)
    if reduction == "rms":
        return (float(np.sqrt(np.mean(arr ** 2))),)
    if reduction == "peak":
        return (float(np.max(np.abs(arr))),)
    if reduction == "rms_and_peak":
        return (
            float(np.sqrt(np.mean(arr ** 2))),
            float(np.max(np.abs(arr))),
        )
    if reduction == "raw":
        return (arr,)
    raise ValueError(
        f"unknown reduction {reduction!r}; choose from "
        f"'centerline' | 'rms' | 'peak' | 'rms_and_peak' | 'raw'"
    )


def _reduction_columns(reduction: str) -> list:
    if reduction == "rms_and_peak":
        return ["rms", "peak"]
    return [reduction]


def parameter_sweep(
    forward_callable: Callable,
    varying: dict,
    fixed: dict,
    reduce: str = "rms",
    progress: bool = False,
) -> SweepResult:
    """Sweep ``varying`` (Cartesian product) over a fixed call.

    Parameters
    ----------
    forward_callable : Callable
        A scalpel forward function (e.g., ``propagate_maxwell``).
    varying : dict
        Each key is a parameter name; each value is an iterable of values
        to sweep. The sweep is the Cartesian product of all keys.
    fixed : dict
        Other keyword arguments passed unchanged to every call.
    reduce : str, default 'rms'
        How to reduce each full output to a row of the result table.
        Use 'raw' to keep the full array (memory-heavy).
    progress : bool, default False
        Print a progress line per call.

    Returns
    -------
    SweepResult
    """
    keys = list(varying.keys())
    grids = [list(v) for v in varying.values()]
    combos = list(itertools.product(*grids))
    rows = []
    t0 = time.perf_counter()

    for idx, combo in enumerate(combos):
        call_kwargs = dict(fixed)
        for k, v in zip(keys, combo):
            call_kwargs[k] = v
        result = forward_callable(**call_kwargs)
        # forward callables typically return (field, t); reduce the field.
        if isinstance(result, tuple):
            field_out = result[0]
        else:
            field_out = result
        reduced = _reduce_field(field_out, reduce)
        rows.append(tuple(combo) + reduced)
        if progress:
            print(f"[sweep] {idx + 1}/{len(combos)} {dict(zip(keys, combo))} -> {reduced}")

    elapsed = time.perf_counter() - t0
    return SweepResult(
        columns=keys + _reduction_columns(reduce),
        rows=rows,
        elapsed_seconds=elapsed,
    )
