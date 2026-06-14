"""
Performance benchmarks: spectral engine timing vs grid size.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass

from ..backends import get_backend
from ..core.engine import SpectralEngine, GridParams, NILTParams


@dataclass
class TimingResult:
    """Result from a timing benchmark."""
    grid_sizes: list
    n_modes: list
    wall_ms: list
    n_nilt: int
    backend_name: str


def benchmark_spectral_engine(
    dispersion_fn,
    grid_configs: list[tuple[int, float]],
    nilt_params: NILTParams,
    depth: float = 0.005,
    source_width: float = 0.008,
    n_repeats: int = 5,
    backend=None,
) -> TimingResult:
    """Benchmark spectral engine across grid sizes.

    Parameters
    ----------
    dispersion_fn : callable
        Dispersion relation function.
    grid_configs : list of (Nx, dx) tuples
        Grid configurations to benchmark.
    nilt_params : NILTParams
        NILT parameters.
    depth : float
        Propagation depth.
    source_width : float
        Gaussian source width.
    n_repeats : int
        Number of timed repetitions (median reported).
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    TimingResult
    """
    b = backend or get_backend()
    engine = SpectralEngine(dispersion_fn, b)

    grid_sizes = []
    n_modes = []
    wall_ms = []

    for Nx, dx in grid_configs:
        grid = GridParams(Nx=Nx, Ny=Nx, dx=dx, dy=dx)

        x = (np.arange(Nx) - Nx // 2) * dx
        X, Y = np.meshgrid(x, x, indexing="ij")
        source_np = np.exp(-(X**2 + Y**2) / (2 * source_width**2))
        source = b.array(source_np, dtype=complex)

        # Warmup
        field, _ = engine.forward(source, depth, grid, nilt_params)
        if b.name == "jax":
            field.block_until_ready()

        # Timed
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            field, _ = engine.forward(source, depth, grid, nilt_params)
            if b.name == "jax":
                field.block_until_ready()
            elif b.name == "torch":
                import torch
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        grid_sizes.append(Nx)
        n_modes.append(Nx * Nx)
        wall_ms.append(float(np.median(times)))

    return TimingResult(
        grid_sizes=grid_sizes, n_modes=n_modes, wall_ms=wall_ms,
        n_nilt=nilt_params.N, backend_name=b.name,
    )
