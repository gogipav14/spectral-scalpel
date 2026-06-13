"""
Accuracy benchmarks: convergence studies and per-mode analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..backends import get_backend
from ..core.engine import SpectralEngine, GridParams, NILTParams
from ..core.nilt import nilt_scalar


@dataclass
class ConvergenceResult:
    """Result from an N_NILT convergence study."""
    N_values: list
    rel_l2_peak: list
    rel_l2_mean: list
    wall_ms: list
    backend_name: str


def convergence_study(
    dispersion_fn,
    analytical_fn,
    grid: GridParams,
    source_np: np.ndarray,
    depth: float,
    a: float,
    T: float,
    N_values: list[int],
    t_end: float,
    backend=None,
) -> ConvergenceResult:
    """Sweep N_NILT and measure error vs analytical solution.

    Parameters
    ----------
    dispersion_fn : callable
        Dispersion relation.
    analytical_fn : callable
        analytical_fn(t, kx_grid, ky_grid, source_fft) -> field_exact(Nx,Ny)
        Returns exact field at a single time.
    grid : GridParams
        Spatial grid.
    source_np : ndarray
        Source field (numpy).
    depth : float
        Propagation depth.
    a, T : float
        Bromwich shift and half-period.
    N_values : list of int
        N_NILT values to sweep.
    t_end : float
        Observation window.
    backend : Backend, optional

    Returns
    -------
    ConvergenceResult
    """
    import time
    b = backend or get_backend()
    engine = SpectralEngine(dispersion_fn, b)
    source = b.array(source_np, dtype=complex)

    rel_peaks = []
    rel_means = []
    wall_ms = []

    for N_nilt in N_values:
        nilt_p = NILTParams(a=a, T=T, N=N_nilt)

        # Warmup
        _ = engine.forward(source, depth, grid, nilt_p)
        if b.name == "jax":
            _[0].block_until_ready()

        t0 = time.perf_counter()
        field, t_arr = engine.forward(source, depth, grid, nilt_p)
        if b.name == "jax":
            field.block_until_ready()
        wall = (time.perf_counter() - t0) * 1e3

        field_np = b.to_numpy(field)
        t_np = b.to_numpy(t_arr)

        # Compute error per time step
        kx = np.fft.fftfreq(grid.Nx, grid.dx) * 2 * np.pi
        ky = np.fft.fftfreq(grid.Ny, grid.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        S_hat = np.fft.fft2(source_np)

        valid = (t_np > 0.005) & (t_np <= t_end)
        l2_errs = []
        l2_refs = []
        for i in np.where(valid)[0]:
            exact = analytical_fn(t_np[i], KX, KY, S_hat)
            l2_errs.append(np.sqrt(np.mean((field_np[:, :, i] - exact)**2)))
            l2_refs.append(np.sqrt(np.mean(exact**2)))

        l2_errs = np.array(l2_errs)
        l2_refs = np.array(l2_refs)
        sig = l2_refs > 0.01 * l2_refs.max()
        rel = l2_errs[sig] / l2_refs[sig]

        rel_peaks.append(float(np.min(rel)) if len(rel) > 0 else np.nan)
        rel_means.append(float(np.mean(rel)) if len(rel) > 0 else np.nan)
        wall_ms.append(wall)

    return ConvergenceResult(
        N_values=N_values, rel_l2_peak=rel_peaks,
        rel_l2_mean=rel_means, wall_ms=wall_ms,
        backend_name=b.name,
    )
