"""Regression test for Theorem 1-vec: slope-2 convergence of the
matrix-exponential vector cascade on 2-species reaction-diffusion.

Tests both FitzHugh-Nagumo and Gray-Scott at a coarse K ladder and
asserts the final-doubling slope is within 0.05 of 2.0.
"""
from __future__ import annotations

import numpy as np
import pytest

from scalpel.nonlinear.vector_cascade import (
    spec_fitzhugh_nagumo, spec_gray_scott, run_vector_cascade, reference_2d,
)


def _slope(err_k, err_2k):
    return float(np.log2(err_k / err_2k))


def _rms_rel_err(u_est, ref):
    rms = np.sqrt(np.mean(ref ** 2))
    return float(np.sqrt(np.mean((u_est - ref) ** 2)) / rms)


def test_vector_cascade_fhn_slope2():
    Nx = Ny = 16
    Lx = Ly = 10.0
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u0 = 0.1 * np.cos(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)
    v0 = 0.05 * np.sin(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)
    u0_vec = np.stack([u0, v0])

    spec = spec_fitzhugh_nagumo(Du=1.0, Dv=40.0, epsilon=0.01, gamma=1.0)
    T = 0.5

    ref = reference_2d(spec, u0_vec, x, y, T)
    errs = []
    for K in (16, 32):
        u_est = run_vector_cascade(u0_vec, spec, x, y, T, K, n_picard=2)
        errs.append(_rms_rel_err(u_est, ref))

    slope = _slope(errs[0], errs[1])
    assert 1.95 <= slope <= 2.05, f"FHN slope {slope} outside [1.95, 2.05]"


def test_vector_cascade_gray_scott_slope2():
    Nx = Ny = 16
    Lx = Ly = 1.0
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)

    rng = np.random.default_rng(0)
    u0 = np.ones((Nx, Ny)) + 0.01 * rng.standard_normal((Nx, Ny))
    v0 = 0.01 * rng.standard_normal((Nx, Ny))
    u0_vec = np.stack([u0, v0])

    spec = spec_gray_scott(Du=2e-5, Dv=1e-5, F=0.04, k=0.06)
    T = 10.0

    ref = reference_2d(spec, u0_vec, x, y, T)
    errs = []
    for K in (16, 32):
        u_est = run_vector_cascade(u0_vec, spec, x, y, T, K, n_picard=2)
        errs.append(_rms_rel_err(u_est, ref))

    slope = _slope(errs[0], errs[1])
    assert 1.95 <= slope <= 2.05, f"GS slope {slope} outside [1.95, 2.05]"


if __name__ == "__main__":
    test_vector_cascade_fhn_slope2()
    test_vector_cascade_gray_scott_slope2()
    print("OK")
