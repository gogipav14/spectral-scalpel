"""Task 3: 2D Fisher-KPP cascade validation.

Tests the prediction that Theorem 1's cubic balance (slope 2 in K) extends
to 2D spatial dimensions. The unified cascade is dimension-agnostic; we need
to confirm empirically.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalpel.nonlinear.unified_cascade_2d import (
    fkpp_2d_cascade, fkpp_2d_reference,
)


def slope_analysis(Ks, errs):
    slopes = []
    for i in range(1, len(Ks)):
        if Ks[i] == 2 * Ks[i - 1] and errs[i - 1] > 0 and errs[i] > 0:
            slopes.append(np.log2(errs[i - 1] / errs[i]))
        else:
            slopes.append(float("nan"))
    return slopes


def validate_2d_fkpp():
    print("=" * 100)
    print("2D FISHER-KPP — logit gauge, 2-iter Picard")
    print("=" * 100)

    Nx, Ny = 32, 32  # coarse to keep runtime reasonable
    Lx, Ly = 4.0, 4.0
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)

    X, Y = np.meshgrid(x, y, indexing="ij")
    # Smooth IC in 2D, bounded in (0.1, 0.9) for regular logit
    u0 = 0.5 + 0.2 * np.cos(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)

    T = 0.3
    D = 0.1

    for r in [0.5, 2.0, 5.0]:
        print(f"\n  D={D}, r={r}")
        ref = fkpp_2d_reference(u0, x, y, D, r, T)
        rms_ref = np.sqrt(np.mean(ref ** 2))
        print(f"  IC range: [{u0.min():.3f}, {u0.max():.3f}]")
        print(f"  Ref range: [{ref.min():.3f}, {ref.max():.3f}]")
        print(f"  {'K':>4s}  {'L2':>12s}  {'slope':>8s}")

        errs, Ks = [], []
        for K in [2, 4, 8, 16, 32]:
            u_est = fkpp_2d_cascade(u0, x, y, D, r, T, K, n_picard=2)
            if np.any(np.isnan(u_est)):
                print(f"  {K:>4d}       NaN")
                continue
            err = np.sqrt(np.mean((u_est - ref) ** 2)) / rms_ref
            errs.append(err)
            Ks.append(K)
        slopes = slope_analysis(Ks, errs)
        for i, K in enumerate(Ks):
            s = f"{slopes[i-1]:>8.2f}" if i > 0 else "  -   "
            print(f"  {K:>4d}  {errs[i]:>12.3e}  {s}")


if __name__ == "__main__":
    validate_2d_fkpp()
    print("\n" + "=" * 100)
    print("EXPECTED: slope -> 2.00 (cubic balance preserved in 2D)")
    print("=" * 100)
