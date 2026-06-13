"""Unified validation for Theorem 1 across Burgers, FKPP, AC, KS.

Tests the same quantitative claim — cubic balance slope=2 with 2-iter Picard —
across all four PDEs using a single harness.

Reference: RK45 in u-space (original PDE) with tight tolerances.
Cascade: unified gauge-transformed cascade with 2-iter Picard.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp

from scalpel.nonlinear.unified_cascade import (
    spec_fisher_kpp, spec_allen_cahn, unified_cascade,
    burgers_cascade_with_transport, spectral_diff,
)


def make_grid(L, Nx):
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    return x, dx, k


# Reference RHS functions (all in u-space)
def burgers_rhs(t, u, nu, k):
    return -u * spectral_diff(u, k, 1) + nu * spectral_diff(u, k, 2)


def fkpp_rhs(t, u, D, r, k):
    return D * spectral_diff(u, k, 2) + r * u * (1 - u)


def ac_rhs(t, u, eps2, k):
    return eps2 * spectral_diff(u, k, 2) + u - u ** 3


def ks_rhs(t, u, k):
    return (-u * spectral_diff(u, k, 1)
            - spectral_diff(u, k, 2)
            - spectral_diff(u, k, 4))


def solve_reference(rhs, u0, T, args, rtol=1e-11, atol=1e-13):
    sol = solve_ivp(rhs, [0, T], u0, args=args,
                    method="RK45", rtol=rtol, atol=atol, t_eval=[T])
    return sol.y[:, -1]


# ============================================================================
# Slope/convergence analysis helper
# ============================================================================

def slope_analysis(Ks, errs):
    """Return log2 slopes of error reduction per K-doubling."""
    slopes = []
    for i in range(1, len(Ks)):
        if Ks[i] == 2 * Ks[i - 1] and errs[i - 1] > 0 and errs[i] > 0:
            slopes.append(np.log2(errs[i - 1] / errs[i]))
        else:
            slopes.append(float("nan"))
    return slopes


# ============================================================================
# Per-PDE validators
# ============================================================================

def validate_burgers():
    print("=" * 100)
    print("BURGERS — identity gauge, 2-iter Picard")
    print("=" * 100)
    x, dx, k = make_grid(L=16.0, Nx=256)
    u0 = 1.0 * np.exp(-x ** 2 / (2 * 0.5 ** 2))
    T = 2.0

    for nu in [0.5, 0.1, 0.02]:
        ref = solve_reference(burgers_rhs, u0, T, (nu, k))
        rms_ref = np.sqrt(np.mean(ref ** 2))
        print(f"\n  nu={nu} (Re={1/nu:.0f})")
        print(f"  {'K':>4s}  {'L2':>12s}  {'slope':>8s}")
        errs, Ks = [], []
        for K in [2, 4, 8, 16, 32, 64]:
            u_est = burgers_cascade_with_transport(u0, x, nu, T, K, n_picard=2)
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


def validate_fkpp():
    print("\n" + "=" * 100)
    print("FISHER-KPP — logit gauge, 2-iter Picard")
    print("=" * 100)
    x, dx, k = make_grid(L=4.0, Nx=128)
    u0 = 0.5 + 0.2 * np.cos(2 * np.pi * x / 4.0)
    T = 0.5
    D = 0.1

    for r in [0.5, 2.0, 5.0]:
        ref = solve_reference(fkpp_rhs, u0, T, (D, r, k))
        rms_ref = np.sqrt(np.mean(ref ** 2))
        spec = spec_fisher_kpp(D=D, r=r)
        print(f"\n  D={D}, r={r}")
        print(f"  {'K':>4s}  {'L2':>12s}  {'slope':>8s}")
        errs, Ks = [], []
        for K in [2, 4, 8, 16, 32, 64]:
            u_est = unified_cascade(u0, x, spec, T, K, n_picard=2)
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


def validate_allen_cahn():
    print("\n" + "=" * 100)
    print("ALLEN-CAHN — arctanh gauge, 2-iter Picard")
    print("=" * 100)
    x, dx, k = make_grid(L=4.0, Nx=128)
    u0 = 0.5 * np.cos(2 * np.pi * x / 4.0) + 0.2 * np.sin(4 * np.pi * x / 4.0)
    T = 0.3

    for eps2 in [0.05, 0.02]:
        ref = solve_reference(ac_rhs, u0, T, (eps2, k))
        rms_ref = np.sqrt(np.mean(ref ** 2))
        spec = spec_allen_cahn(eps2=eps2)
        print(f"\n  eps2={eps2}")
        print(f"  {'K':>4s}  {'L2':>12s}  {'slope':>8s}")
        errs, Ks = [], []
        for K in [2, 4, 8, 16, 32, 64]:
            u_est = unified_cascade(u0, x, spec, T, K, n_picard=2)
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


def validate_ks():
    print("\n" + "=" * 100)
    print("KURAMOTO-SIVASHINSKY — identity gauge, 2-iter Picard")
    print("=" * 100)
    x, dx, k = make_grid(L=32.0, Nx=256)
    u0 = 0.3 * np.exp(-x ** 2 / (2 * 4.0 ** 2))
    T = 1.0

    ref = solve_reference(ks_rhs, u0, T, (k,))
    rms_ref = np.sqrt(np.mean(ref ** 2))

    # For KS, use the burgers-cascade harness with u_star absorbed into
    # transport, but add -u_xx - u_xxxx to the linear part. Custom driver:
    def ks_cascade(u0, x, T, K, n_picard=2, omega=1.0):
        dx = x[1] - x[0]
        Nx = len(x)
        k_local = np.fft.fftfreq(Nx, dx) * 2 * np.pi
        u = u0.copy()
        dt = T / K
        for _ in range(K):
            u_star = float(np.max(np.abs(u)))
            sigma_k = (-1j * k_local * u_star
                       + k_local ** 2 - k_local ** 4)
            u_hat = np.fft.fft(u)
            v_hat = u_hat * np.exp(sigma_k * dt)
            v = np.real(np.fft.ifft(v_hat))

            if abs(omega) > 1e-10:
                v_current = v.copy()
                for _pic in range(n_picard):
                    def picard_rhs(t, h, k=k_local, u_star=u_star,
                                    u_init=u, v_end=v_current):
                        alpha_t = t / dt
                        v_t = (1.0 - alpha_t) * u_init + alpha_t * v_end
                        v_x = spectral_diff(v_t, k, 1)
                        rho = -(v_t - u_star) * v_x
                        # linear part: -u_star h_x - h_xx - h_xxxx
                        lin = (-u_star * spectral_diff(h, k, 1)
                               - spectral_diff(h, k, 2)
                               - spectral_diff(h, k, 4))
                        return lin + rho

                    sol = solve_ivp(picard_rhs, [0, dt], np.zeros_like(u),
                                     method="RK45", rtol=1e-9, atol=1e-11,
                                     t_eval=[dt])
                    h_end = sol.y[:, -1]
                    v_current = v + omega * h_end
                u = v_current
            else:
                u = v
        return u

    print(f"\n  KS on single-sign bump")
    print(f"  {'K':>4s}  {'L2':>12s}  {'slope':>8s}")
    errs, Ks = [], []
    for K in [2, 4, 8, 16, 32]:
        u_est = ks_cascade(u0, x, T, K, n_picard=2)
        if np.any(np.isnan(u_est)) or np.max(np.abs(u_est)) > 100:
            print(f"  {K:>4d}       unstable")
            continue
        err = np.sqrt(np.mean((u_est - ref) ** 2)) / rms_ref
        errs.append(err)
        Ks.append(K)
    slopes = slope_analysis(Ks, errs)
    for i, K in enumerate(Ks):
        s = f"{slopes[i-1]:>8.2f}" if i > 0 else "  -   "
        print(f"  {K:>4d}  {errs[i]:>12.3e}  {s}")


if __name__ == "__main__":
    validate_burgers()
    validate_fkpp()
    validate_allen_cahn()
    validate_ks()
    print("\n" + "=" * 100)
    print("EXPECTED: slope -> 2.00 (cubic balance) for all four PDEs")
    print("=" * 100)
