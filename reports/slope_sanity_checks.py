"""Four independent sanity checks on the suspicious slope=2.00 result.

Check 1: push K further (K=128, 256) to see if slope stays 2 or we hit a floor.
Check 2: vary reference tolerance. If slope=2 survives RK45 at rtol=1e-6, we're
         seeing the cascade's intrinsic error. If it's a reference artifact, slope changes.
Check 3: compare to an analytic solution (Fisher-KPP traveling wave on steady state).
Check 4: use a non-smooth initial condition (tanh-front + small noise) where
         theorem should predict slope=2 but preconstant should grow.

If all four agree — the slope=2 result is real.
If any disagrees — we know exactly where the artifact lives.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp

from scalpel.nonlinear.unified_cascade import (
    spec_fisher_kpp, unified_cascade, spectral_diff,
)


def make_grid(L=4.0, Nx=128):
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    return x, dx, k


def fkpp_rhs(t, u, D, r, k):
    return D * spectral_diff(u, k, 2) + r * u * (1 - u)


def solve_ref(u0, T, k, D, r, rtol, atol):
    sol = solve_ivp(fkpp_rhs, [0, T], u0, args=(D, r, k),
                    method="RK45", rtol=rtol, atol=atol, t_eval=[T])
    return sol.y[:, -1]


def slopes(Ks, errs):
    out = []
    for i in range(1, len(Ks)):
        if Ks[i] == 2 * Ks[i - 1] and errs[i - 1] > 0 and errs[i] > 0:
            out.append(np.log2(errs[i - 1] / errs[i]))
        else:
            out.append(float("nan"))
    return out


# ============================================================================
# CHECK 1: push K further
# ============================================================================

def check1_extended_K():
    print("=" * 90)
    print("CHECK 1 — push K to 128, 256. Does slope hold or do we hit a floor?")
    print("=" * 90)
    x, dx, k = make_grid()
    u0 = 0.5 + 0.2 * np.cos(2 * np.pi * x / 4.0)
    T, D, r = 0.5, 0.1, 2.0
    ref = solve_ref(u0, T, k, D, r, rtol=1e-13, atol=1e-15)  # ultra-tight
    rms = np.sqrt(np.mean(ref ** 2))
    spec = spec_fisher_kpp(D=D, r=r)

    Ks = [8, 16, 32, 64, 128, 256]
    errs = []
    for K in Ks:
        u_est = unified_cascade(u0, x, spec, T, K, n_picard=2)
        e = np.sqrt(np.mean((u_est - ref) ** 2)) / rms
        errs.append(e)
    ss = slopes(Ks, errs)
    print(f"  {'K':>5s}  {'err':>12s}  {'slope':>8s}")
    for i, K in enumerate(Ks):
        s = f"{ss[i-1]:.3f}" if i > 0 else "-"
        print(f"  {K:>5d}  {errs[i]:>12.3e}  {s:>8s}")
    print(f"\n  Interpretation: if slope stays at 2.00 for K=256, cubic balance is real.")
    print(f"  If slope drops at K=256, we hit either NILT floor or reference noise floor.")


# ============================================================================
# CHECK 2: vary reference tolerance
# ============================================================================

def check2_ref_tol():
    print("\n" + "=" * 90)
    print("CHECK 2 — vary reference solver tolerance")
    print("=" * 90)
    x, dx, k = make_grid()
    u0 = 0.5 + 0.2 * np.cos(2 * np.pi * x / 4.0)
    T, D, r = 0.5, 0.1, 2.0
    spec = spec_fisher_kpp(D=D, r=r)
    Ks = [4, 8, 16, 32, 64]

    for rtol in [1e-6, 1e-8, 1e-11, 1e-13]:
        ref = solve_ref(u0, T, k, D, r, rtol=rtol, atol=rtol * 1e-2)
        rms = np.sqrt(np.mean(ref ** 2))
        errs = []
        for K in Ks:
            u_est = unified_cascade(u0, x, spec, T, K, n_picard=2)
            e = np.sqrt(np.mean((u_est - ref) ** 2)) / rms
            errs.append(e)
        ss = slopes(Ks, errs)
        print(f"\n  rtol={rtol}:")
        for i, K in enumerate(Ks):
            s = f"{ss[i-1]:.3f}" if i > 0 else "-"
            print(f"    K={K:>3d}  err={errs[i]:>11.3e}  slope={s:>7s}")
    print(f"\n  Interpretation: if slope=2 only with rtol=1e-11 but fails with rtol=1e-6,")
    print(f"  the tight reference is masking a lower-order cascade truncation error.")


# ============================================================================
# CHECK 3: analytic solution — Fisher-KPP stationary solution u(x) = 0 (trivial)
# is not useful. Instead, compare TWO cascade runs at different K (Richardson-like).
# Self-consistency: if E(K) = c/K^2 exactly, E(K) - 4·E(2K) ≈ 0.
# ============================================================================

def check3_richardson():
    print("\n" + "=" * 90)
    print("CHECK 3 — Richardson self-consistency (no reference needed)")
    print("=" * 90)
    print("  If err = c/K^2 exactly, then err(K) - 4*err(2K) -> 0 cleanly.")
    print("  If err = c/K^alpha with alpha != 2, Richardson fails.")
    print()
    x, dx, k = make_grid()
    u0 = 0.5 + 0.2 * np.cos(2 * np.pi * x / 4.0)
    T, D, r = 0.5, 0.1, 2.0
    spec = spec_fisher_kpp(D=D, r=r)

    solns = {}
    for K in [8, 16, 32, 64, 128]:
        solns[K] = unified_cascade(u0, x, spec, T, K, n_picard=2)

    def l2(u):
        return np.sqrt(np.mean(u ** 2))

    print(f"  {'K':>5s}  {'K2':>5s}  {'||u_K - u_{2K}||':>20s}  {'ratio(pred 4)':>15s}")
    for i, K in enumerate([8, 16, 32, 64]):
        d1 = l2(solns[K] - solns[2 * K])
        d2 = l2(solns[2 * K] - solns[4 * K]) if 4 * K in solns else None
        ratio = (d1 / d2) if (d2 is not None and d2 > 0) else None
        r_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        print(f"  {K:>5d}  {2*K:>5d}  {d1:>20.3e}  {r_str:>15s}")
    print(f"\n  Expected ratio = 4.00 if slope=2 exactly, independent of reference.")


# ============================================================================
# CHECK 4: perturbed IC (small noise) to see if preconstant is sensitive
# ============================================================================

def check4_perturbed_ic():
    print("\n" + "=" * 90)
    print("CHECK 4 — noisy IC. Does slope=2 hold for non-cosine smooth profiles?")
    print("=" * 90)
    x, dx, k = make_grid()
    np.random.seed(42)
    T, D, r = 0.5, 0.1, 2.0
    spec = spec_fisher_kpp(D=D, r=r)
    Ks = [8, 16, 32, 64]

    # Three different ICs: cosine (canonical), high-freq cosine, sum of cosines
    ics = {
        "cos(2pi x/L)":   0.5 + 0.2 * np.cos(2 * np.pi * x / 4.0),
        "cos(6pi x/L)":   0.5 + 0.2 * np.cos(6 * np.pi * x / 4.0),
        "2-mode mix":     0.5 + 0.1 * np.cos(2 * np.pi * x / 4.0) + 0.1 * np.sin(6 * np.pi * x / 4.0),
        "narrow bump":    0.5 + 0.3 * np.exp(-x ** 2 / 0.1) - 0.2,
    }

    for name, u0 in ics.items():
        if np.any(u0 <= 0) or np.any(u0 >= 1):
            print(f"\n  {name}: IC out of (0,1), skip.")
            continue
        ref = solve_ref(u0, T, k, D, r, rtol=1e-12, atol=1e-14)
        rms = np.sqrt(np.mean(ref ** 2))
        errs = []
        for K in Ks:
            u_est = unified_cascade(u0, x, spec, T, K, n_picard=2)
            e = np.sqrt(np.mean((u_est - ref) ** 2)) / rms
            errs.append(e)
        ss = slopes(Ks, errs)
        print(f"\n  IC = {name}  (range [{u0.min():.3f}, {u0.max():.3f}])")
        for i, K in enumerate(Ks):
            s = f"{ss[i-1]:.3f}" if i > 0 else "-"
            print(f"    K={K:>3d}  err={errs[i]:>11.3e}  slope={s:>7s}")


if __name__ == "__main__":
    check1_extended_K()
    check2_ref_tol()
    check3_richardson()
    check4_perturbed_ic()
    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)
