"""Phase 1a validation: does algorithm2_tune_v2 fix the gaps Phase 0 found?

Reruns the same 15 cases from phase0_profile_sweep.py and phase0_tune_compare.py
with the new tuner, and reports K_ratio and L2_ratio per case.

Target from phase0_tuner_validation.md: K_ratio ∈ [0.8, 1.2] on all cases.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalpel.nonlinear.tuner import tune_burgers, tune_burgers_v2
from reports.phase0_tune_compare import (
    brute_force_optimum,
    cole_hopf_exact,
    picard_cascade,
)


def run_burgers_re_scan():
    """Scan Re for a fixed Gaussian — the core Phase 0 comparison."""
    A_amp = 1.0
    w_sigma = 0.5
    L_domain = 16.0
    t_total = 2.0
    Nx = 256
    x = np.linspace(-L_domain / 2, L_domain / 2, Nx)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    u0 = np.array([A_amp * np.exp(-xv ** 2 / (2 * w_sigma ** 2)) for xv in x])

    nu_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("=" * 130)
    print("PHASE 1a — tune_burgers_v2 vs brute-force (Gaussian Re scan)")
    print("=" * 130)
    print(
        f"\n{'nu':>5s}  {'Re':>4s}  "
        f"{'K_v1':>4s}  {'om_v1':>5s}  {'L2_v1':>7s}  "
        f"{'K_v2':>4s}  {'om_v2':>5s}  {'L2_v2':>7s}  "
        f"{'K_opt':>5s}  {'om_opt':>6s}  {'L2_opt':>7s}  "
        f"{'Kr_v1':>5s}  {'Kr_v2':>5s}"
    )
    print("-" * 130)

    rows = []
    for nu in nu_values:
        ref = cole_hopf_exact(
            x, t_total, nu, lambda xv: A_amp * np.exp(-xv ** 2 / (2 * w_sigma ** 2))
        )
        rms_ref = np.sqrt(np.mean(ref ** 2))

        r_v1 = tune_burgers(u0, x, nu, t_total)
        u_v1 = picard_cascade(u0, x, k, nu, t_total, r_v1.K, r_v1.omega)
        err_v1 = np.sqrt(np.mean((u_v1 - ref) ** 2)) / rms_ref

        r_v2 = tune_burgers_v2(u0, x, nu, t_total)
        u_v2 = picard_cascade(u0, x, k, nu, t_total, r_v2.K, r_v2.omega)
        err_v2 = np.sqrt(np.mean((u_v2 - ref) ** 2)) / rms_ref

        K_opt, om_opt, err_opt = brute_force_optimum(
            u0, x, k, nu, t_total, ref, K_grid, omega_grid
        )

        Kr_v1 = r_v1.K / max(K_opt, 1)
        Kr_v2 = r_v2.K / max(K_opt, 1)

        print(
            f"{nu:>5.3f}  {A_amp/nu:>4.0f}  "
            f"{r_v1.K:>4d}  {r_v1.omega:>5.2f}  {err_v1:>7.1%}  "
            f"{r_v2.K:>4d}  {r_v2.omega:>5.2f}  {err_v2:>7.1%}  "
            f"{K_opt:>5d}  {om_opt:>6.2f}  {err_opt:>7.1%}  "
            f"{Kr_v1:>5.2f}  {Kr_v2:>5.2f}"
        )
        rows.append(
            dict(nu=nu, K_v1=r_v1.K, K_v2=r_v2.K, K_opt=K_opt,
                 om_v1=r_v1.omega, om_v2=r_v2.omega, om_opt=om_opt,
                 err_v1=err_v1, err_v2=err_v2, err_opt=err_opt,
                 Kr_v1=Kr_v1, Kr_v2=Kr_v2)
        )
    return rows


def run_profile_sweep():
    """Same profile sweep as Phase 0: Gaussian, tanh, sine."""
    L_domain = 16.0
    t_total = 2.0
    Nx = 256
    x = np.linspace(-L_domain / 2, L_domain / 2, Nx)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi

    profiles = []
    for w in [0.25, 0.5, 1.0]:
        def u0_func(xv, _w=w):
            return 1.0 * np.exp(-xv ** 2 / (2 * _w ** 2))
        u0 = np.array([u0_func(xv) for xv in x])
        profiles.append(("gauss", f"sigma={w}", u0, u0_func))

    def tanh_step(xv):
        return 0.5 * (1.0 - np.tanh(xv / 0.3))
    u0_tanh = np.array([tanh_step(xv) for xv in x])
    profiles.append(("tanh", "width=0.3", u0_tanh, tanh_step))

    def sine_wave(xv):
        return np.sin(xv) * np.exp(-xv ** 2 / (2 * 3.0 ** 2))
    u0_sin = np.array([sine_wave(xv) for xv in x])
    profiles.append(("sine", "sin*gauss", u0_sin, sine_wave))

    nu_cases = [0.2, 0.05, 0.02]
    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n" + "=" * 130)
    print("PHASE 1a — tune_burgers_v2 across profile families")
    print("=" * 130)
    print(
        f"\n{'profile':<20s}  {'nu':>5s}  "
        f"{'K_v1':>4s}  {'Kr_v1':>5s}  {'L2r_v1':>6s}  "
        f"{'K_v2':>4s}  {'Kr_v2':>5s}  {'L2r_v2':>6s}  "
        f"{'K_opt':>5s}  {'L2_opt':>7s}"
    )
    print("-" * 130)

    all_rows = []
    for name, label, u0, u0_func in profiles:
        for nu in nu_cases:
            try:
                ref = cole_hopf_exact(x, t_total, nu, u0_func)
            except Exception:
                continue
            rms_ref = np.sqrt(np.mean(ref ** 2))
            if rms_ref < 1e-6:
                continue

            r_v1 = tune_burgers(u0, x, nu, t_total)
            u_v1 = picard_cascade(u0, x, k, nu, t_total, r_v1.K, r_v1.omega)
            err_v1 = np.sqrt(np.mean((u_v1 - ref) ** 2)) / rms_ref

            r_v2 = tune_burgers_v2(u0, x, nu, t_total)
            u_v2 = picard_cascade(u0, x, k, nu, t_total, r_v2.K, r_v2.omega)
            err_v2 = np.sqrt(np.mean((u_v2 - ref) ** 2)) / rms_ref

            K_opt, om_opt, err_opt = brute_force_optimum(
                u0, x, k, nu, t_total, ref, K_grid, omega_grid
            )

            Kr_v1 = r_v1.K / max(K_opt, 1)
            Kr_v2 = r_v2.K / max(K_opt, 1)
            L2r_v1 = err_v1 / max(err_opt, 1e-30)
            L2r_v2 = err_v2 / max(err_opt, 1e-30)

            print(
                f"  {name:<5s} {label:<13s}  {nu:>5.3f}  "
                f"{r_v1.K:>4d}  {Kr_v1:>5.2f}  {L2r_v1:>6.2f}  "
                f"{r_v2.K:>4d}  {Kr_v2:>5.2f}  {L2r_v2:>6.2f}  "
                f"{K_opt:>5d}  {err_opt:>7.1%}"
            )
            all_rows.append(
                dict(profile=name, nu=nu, Kr_v1=Kr_v1, Kr_v2=Kr_v2,
                     L2r_v1=L2r_v1, L2r_v2=L2r_v2)
            )
    return all_rows


def summarize(rows_scan, rows_profile):
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    Kr1_scan = np.array([r["Kr_v1"] for r in rows_scan])
    Kr2_scan = np.array([r["Kr_v2"] for r in rows_scan])
    print(f"\n  Re-scan (Gaussian, n={len(rows_scan)}):")
    print(f"    v1: Kr median={np.median(Kr1_scan):.2f}, range=[{Kr1_scan.min():.2f}, {Kr1_scan.max():.2f}]")
    print(f"    v2: Kr median={np.median(Kr2_scan):.2f}, range=[{Kr2_scan.min():.2f}, {Kr2_scan.max():.2f}]")

    for pname in ["gauss", "tanh", "sine"]:
        r1 = [r["Kr_v1"] for r in rows_profile if r["profile"] == pname]
        r2 = [r["Kr_v2"] for r in rows_profile if r["profile"] == pname]
        if not r1:
            continue
        print(f"\n  {pname} (n={len(r1)}):")
        print(f"    v1: Kr median={np.median(r1):.2f}, range=[{min(r1):.2f}, {max(r1):.2f}]")
        print(f"    v2: Kr median={np.median(r2):.2f}, range=[{min(r2):.2f}, {max(r2):.2f}]")

    # Scaling-law re-fit
    def loglog_slope(x_arr, y_arr):
        valid = (x_arr > 0) & (y_arr > 0)
        lx = np.log(x_arr[valid])
        ly = np.log(y_arr[valid])
        if len(lx) < 2:
            return float("nan"), float("nan")
        slope, intercept = np.polyfit(lx, ly, 1)
        return slope, np.exp(intercept)

    Re_arr = np.array([1.0 / r["nu"] for r in rows_scan])
    Kopt_arr = np.array([r["K_opt"] for r in rows_scan])
    Kv1_arr = np.array([r["K_v1"] for r in rows_scan])
    Kv2_arr = np.array([r["K_v2"] for r in rows_scan])

    a_opt, c_opt = loglog_slope(Re_arr, Kopt_arr)
    a_v1, c_v1 = loglog_slope(Re_arr, Kv1_arr)
    a_v2, c_v2 = loglog_slope(Re_arr, Kv2_arr)

    print(f"\n  Scaling: K ~ Re^alpha")
    print(f"    brute-force: K_opt = {c_opt:.2f} * Re^{a_opt:.3f}")
    print(f"    v1 (cubic): K_v1 = {c_v1:.2f} * Re^{a_v1:.3f}")
    print(f"    v2 (new):   K_v2 = {c_v2:.2f} * Re^{a_v2:.3f}")


if __name__ == "__main__":
    rows_scan = run_burgers_re_scan()
    rows_profile = run_profile_sweep()
    summarize(rows_scan, rows_profile)
