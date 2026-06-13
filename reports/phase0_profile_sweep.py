"""Phase 0 profile sweep: does algorithm2_tune (cubic-balance) generalize?

Sweeps three IC families:
  (a) Gaussian: u0 = A * exp(-x^2 / (2 sigma^2))
  (b) tanh (step-like): u0 = A * (1 - tanh(x / w)^2) or step
  (c) Sine (oscillatory): u0 = A * sin(k0 * x)

For each, at fixed nu, compare:
  - algorithm2_tune's K_cubic vs brute-force K_opt
  - algorithm2_tune's omega_cubic vs brute-force omega_opt

Phase 0 deliverable: does the cubic-balance law hold across profile shapes,
or is it specific to smooth Gaussian ICs?
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalpel.nonlinear.tuner import tune_burgers
from reports.phase0_tune_compare import (
    picard_cascade, cole_hopf_exact, brute_force_optimum,
)


def run_profile_sweep():
    L_domain = 16.0
    t_total = 2.0
    Nx = 256
    x = np.linspace(-L_domain / 2, L_domain / 2, Nx)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi

    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    profiles = []

    # Gaussian, 3 widths
    for w in [0.25, 0.5, 1.0]:
        def u0_func(xv, _w=w):
            return 1.0 * np.exp(-xv ** 2 / (2 * _w ** 2))
        u0 = np.array([u0_func(xv) for xv in x])
        profiles.append(("gauss", f"A=1.0, sigma={w}", u0, u0_func))

    # tanh step
    def tanh_step(xv):
        return 0.5 * (1.0 - np.tanh(xv / 0.3))
    u0_tanh = np.array([tanh_step(xv) for xv in x])
    profiles.append(("tanh", "tanh width=0.3", u0_tanh, tanh_step))

    # Sine wave (periodic on a compact window, approximation)
    def sine_wave(xv):
        return np.sin(xv) * np.exp(-xv ** 2 / (2 * 3.0 ** 2))
    u0_sin = np.array([sine_wave(xv) for xv in x])
    profiles.append(("sine", "sin*gauss", u0_sin, sine_wave))

    nu_cases = [0.2, 0.05, 0.02]

    print("=" * 130)
    print("PHASE 0 — algorithm2_tune across profile families")
    print("=" * 130)
    print(
        f"\n{'profile':<18s}  {'nu':>5s}  {'K_cubic':>7s}  {'om_cub':>6s}  "
        f"{'L2_cub':>7s}  {'K_opt':>5s}  {'om_opt':>6s}  {'L2_opt':>7s}  "
        f"{'K_ratio':>7s}  {'L2_ratio':>8s}"
    )
    print("-" * 130)

    all_rows = []
    for name, label, u0, u0_func in profiles:
        for nu in nu_cases:
            try:
                ref = cole_hopf_exact(x, t_total, nu, u0_func)
            except Exception as e:
                print(f"  skip {name} nu={nu}: {e}")
                continue
            rms_ref = np.sqrt(np.mean(ref ** 2))
            if rms_ref < 1e-6:
                continue

            res = tune_burgers(u0, x, nu, t_total)
            u_cubic = picard_cascade(u0, x, k, nu, t_total, res.K, res.omega)
            err_cubic = np.sqrt(np.mean((u_cubic - ref) ** 2)) / rms_ref

            K_opt, om_opt, err_opt = brute_force_optimum(
                u0, x, k, nu, t_total, ref, K_grid, omega_grid
            )

            K_ratio = res.K / max(K_opt, 1)
            L2_ratio = err_cubic / max(err_opt, 1e-30)

            print(
                f"  {name:<5s} {label:<11s}  {nu:>5.3f}  {res.K:>7d}  "
                f"{res.omega:>6.3f}  {err_cubic:>7.1%}  {K_opt:>5d}  "
                f"{om_opt:>6.2f}  {err_opt:>7.1%}  {K_ratio:>7.2f}  "
                f"{L2_ratio:>8.2f}"
            )
            all_rows.append(dict(
                profile=name, label=label, nu=nu,
                K_cubic=res.K, om_cubic=res.omega, err_cubic=err_cubic,
                K_opt=K_opt, om_opt=om_opt, err_opt=err_opt,
                K_ratio=K_ratio, L2_ratio=L2_ratio,
            ))

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY — K_ratio (cubic/opt) and L2_ratio (cubic/opt)")
    print("=" * 130)
    K_ratios = [r["K_ratio"] for r in all_rows]
    L2_ratios = [r["L2_ratio"] for r in all_rows]
    print(f"  K_ratio: min={min(K_ratios):.2f}  median={np.median(K_ratios):.2f}  max={max(K_ratios):.2f}")
    print(f"  L2_ratio: min={min(L2_ratios):.2f}  median={np.median(L2_ratios):.2f}  max={max(L2_ratios):.2f}")
    print(f"  Perfect tuner would have both ratios = 1.0.")

    # Per-profile breakdown
    print("\n  Per-profile K_ratio medians:")
    for name in ["gauss", "tanh", "sine"]:
        ratios = [r["K_ratio"] for r in all_rows if r["profile"] == name]
        if ratios:
            print(f"    {name:<6s}: median={np.median(ratios):.2f}, n={len(ratios)}")

    return all_rows


if __name__ == "__main__":
    run_profile_sweep()
