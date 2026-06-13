"""Empirical validation of Theorem 1-vec: matrix-exponential cascade.

Tests the prediction that the vector-valued cascade with 2-iter Picard
converges as O(T^3/K^2) for FitzHugh-Nagumo (simpler) and Gray-Scott (Turing).

Small grid to keep the per-window 2x2 matrix-exp loop tractable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalpel.nonlinear.vector_cascade import (
    spec_fitzhugh_nagumo, spec_gray_scott, run_vector_cascade, reference_2d,
)


def slopes(Ks, errs):
    out = []
    for i in range(1, len(Ks)):
        if Ks[i] == 2 * Ks[i - 1] and errs[i - 1] > 0 and errs[i] > 0:
            out.append(np.log2(errs[i - 1] / errs[i]))
        else:
            out.append(float("nan"))
    return out


def validate_fhn():
    print("=" * 80)
    print("VALIDATION: FitzHugh-Nagumo 2D, matrix-exponential cascade")
    print("=" * 80)
    Nx, Ny = 16, 16
    Lx, Ly = 10.0, 10.0
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Small perturbation IC, close to a stable state
    u0 = 0.1 * np.cos(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)
    v0 = 0.05 * np.sin(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)
    u0_vec = np.stack([u0, v0])

    spec = spec_fitzhugh_nagumo(Du=1.0, Dv=40.0, epsilon=0.01, gamma=1.0)
    T = 0.5

    ref = reference_2d(spec, u0_vec, x, y, T)
    rms = np.sqrt(np.mean(ref ** 2))
    print(f"  Ref shape {ref.shape}, rms={rms:.3e}, range=[{ref.min():.3f}, {ref.max():.3f}]")

    Ks = [2, 4, 8, 16, 32]
    errs = []
    for K in Ks:
        u_est = run_vector_cascade(u0_vec, spec, x, y, T, K, n_picard=2)
        err = np.sqrt(np.mean((u_est - ref) ** 2)) / rms
        errs.append(err)
    ss = slopes(Ks, errs)
    print(f"  {'K':>4s}  {'err':>12s}  {'slope':>8s}")
    for i, K in enumerate(Ks):
        s = f"{ss[i-1]:.3f}" if i > 0 else "-"
        print(f"  {K:>4d}  {errs[i]:>12.3e}  {s:>8s}")


def validate_gray_scott():
    print("\n" + "=" * 80)
    print("VALIDATION: Gray-Scott 2D, matrix-exponential cascade")
    print("=" * 80)
    Nx, Ny = 16, 16
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # GS near homogeneous state (u, v) ~ (1, 0) with small perturbation
    rng = np.random.default_rng(0)
    u0 = np.ones((Nx, Ny)) + 0.01 * rng.standard_normal((Nx, Ny))
    v0 = 0.01 * rng.standard_normal((Nx, Ny))
    u0_vec = np.stack([u0, v0])

    spec = spec_gray_scott(Du=2e-5, Dv=1e-5, F=0.04, k=0.06)
    T = 10.0  # short enough to keep u, v bounded and avoid pattern formation interfering with convergence tests

    ref = reference_2d(spec, u0_vec, x, y, T)
    rms = np.sqrt(np.mean(ref ** 2))
    print(f"  Ref range u: [{ref[0].min():.3f}, {ref[0].max():.3f}], v: [{ref[1].min():.3f}, {ref[1].max():.3f}]")

    Ks = [2, 4, 8, 16, 32]
    errs = []
    for K in Ks:
        u_est = run_vector_cascade(u0_vec, spec, x, y, T, K, n_picard=2)
        err = np.sqrt(np.mean((u_est - ref) ** 2)) / rms
        errs.append(err)
    ss = slopes(Ks, errs)
    print(f"  {'K':>4s}  {'err':>12s}  {'slope':>8s}")
    for i, K in enumerate(Ks):
        s = f"{ss[i-1]:.3f}" if i > 0 else "-"
        print(f"  {K:>4d}  {errs[i]:>12.3e}  {s:>8s}")


if __name__ == "__main__":
    validate_fhn()
    validate_gray_scott()
