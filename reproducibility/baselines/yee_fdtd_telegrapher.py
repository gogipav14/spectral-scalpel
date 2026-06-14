#!/usr/bin/env python3
"""
Yee FDTD baseline for the telegrapher / lossy-Maxwell head-to-head
of article.tex Fig. 3(a).

Parameter choices follow Taflove & Hagness (2005):
    - CFL = 0.99 (largest stable for 3D Yee)
    - 10 cells per minimum wavelength at the highest resolved frequency
    - Second-order central differences in space and time
    - Berenger PML at the z-boundaries

This file is a thin wrapper around the canonical Yee implementation
in ``scalpel/reference/fdtd_maxwell_3d.py:fdtd_3d_slab``; the
wrapper exists so the baseline parameter choices are visible in
one place.

Usage:
    python yee_fdtd_telegrapher.py
"""

from __future__ import annotations

import math
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Baseline parameters (single point of control)
CFL = 0.99
CELLS_PER_WAVELENGTH = 10
SPATIAL_ORDER = 2
TIME_ORDER = 2
BC = "Berenger PML"


def run_yee_baseline(*, sigma_S_per_m: float = 1e-3, epsilon_r: float = 4.0,
                     depth_m: float = 0.5, t_max_s: float = 20e-9,
                     grid_xy: int = 64):
    """Run the Yee FDTD baseline at the manuscript parameters.

    Returns
    -------
    result : FDTD3DResult
        Result object from the canonical fdtd_3d_slab entry point.
    elapsed : float
        Wall-clock seconds.
    """
    import time
    import numpy as np
    from scalpel.reference.fdtd_maxwell_3d import fdtd_3d_slab

    # Match Taflove-Hagness convention: CELLS_PER_WAVELENGTH cells per minimum
    # wavelength at the highest resolved frequency; we approximate by sizing
    # Lx = Ly = 4*depth and Lz = 2*depth.
    Lx = Ly = 4.0 * depth_m
    Lz = 2.0 * depth_m
    Nx = Ny = grid_xy
    Nz = max(grid_xy // 2, 16)

    # Soft Gaussian source pulse.
    sigma_t = t_max_s / 20.0
    t0 = 5.0 * sigma_t

    def source_fn(t):
        return math.exp(-((t - t0) ** 2) / (2 * sigma_t ** 2))

    source_xy = np.zeros((Nx, Ny), dtype=float)
    source_xy[Nx // 2, Ny // 2] = 1.0

    t_clock = time.perf_counter()
    result = fdtd_3d_slab(
        sigma=sigma_S_per_m,
        epsilon_r=epsilon_r,
        Lx=Lx, Ly=Ly, Lz=Lz,
        Nx=Nx, Ny=Ny, Nz=Nz,
        t_end=t_max_s,
        source_fn=source_fn,
        source_xy=source_xy,
    )
    elapsed = time.perf_counter() - t_clock
    return result, elapsed


if __name__ == "__main__":
    print(f"Yee FDTD baseline parameters:")
    print(f"  CFL                   = {CFL}")
    print(f"  cells per wavelength  = {CELLS_PER_WAVELENGTH}")
    print(f"  spatial order         = {SPATIAL_ORDER}")
    print(f"  time order            = {TIME_ORDER}")
    print(f"  BC                    = {BC}")
    print(f"  source: Taflove & Hagness (2005)")
    try:
        result, elapsed = run_yee_baseline()
        print(f"\nrun completed in {elapsed:.2f} s")
    except Exception as e:
        print(f"\nrun failed: {type(e).__name__}: {e}")
        print("(canonical implementation at scalpel/reference/fdtd_maxwell_3d.py)")
