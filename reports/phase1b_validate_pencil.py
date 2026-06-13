"""Phase 1b validation: does the pencil-derived K* match Re^{0.5} scaling?

Tests:
  (a) K_pencil / K_opt across Re scan and three profile families
  (b) Calibration of kappa: try kappa in {pi, pi/2, pi/4, pi/8}
  (c) Re-scaling exponent of K_pencil vs K_opt

Target: K_ratio ∈ [0.8, 1.2] across ALL profiles after kappa calibration.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalpel.nonlinear.tuner import tune_burgers_pencil
from reports.phase0_tune_compare import (
    brute_force_optimum,
    cole_hopf_exact,
    picard_cascade,
)


def re_scan(kappa):
    """Scan Re for fixed Gaussian — the central scaling test."""
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

    rows = []
    for nu in nu_values:
        ref = cole_hopf_exact(
            x, t_total, nu, lambda xv: A_amp * np.exp(-xv ** 2 / (2 * w_sigma ** 2))
        )
        rms_ref = np.sqrt(np.mean(ref ** 2))

        r = tune_burgers_pencil(u0, x, nu, t_total, kappa=kappa)
        u_pred = picard_cascade(u0, x, k, nu, t_total, r.K, r.omega)
        err = np.sqrt(np.mean((u_pred - ref) ** 2)) / rms_ref

        K_opt, om_opt, err_opt = brute_force_optimum(
            u0, x, k, nu, t_total, ref, K_grid, omega_grid
        )

        rows.append(dict(
            nu=nu, Re=A_amp / nu,
            K_p=r.K, om_p=r.omega, err_p=err,
            K_opt=K_opt, om_opt=om_opt, err_opt=err_opt,
            Kr=r.K / max(K_opt, 1),
            L2r=err / max(err_opt, 1e-30),
        ))
    return rows


def profile_sweep(kappa):
    """Three profile families × three viscosities."""
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
    profiles.append(("tanh", "width=0.3",
                      np.array([tanh_step(xv) for xv in x]), tanh_step))

    def sine_wave(xv):
        return np.sin(xv) * np.exp(-xv ** 2 / (2 * 3.0 ** 2))
    profiles.append(("sine", "sin*gauss",
                      np.array([sine_wave(xv) for xv in x]), sine_wave))

    nu_cases = [0.2, 0.05, 0.02]
    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for name, label, u0, u0_func in profiles:
        for nu in nu_cases:
            try:
                ref = cole_hopf_exact(x, t_total, nu, u0_func)
            except Exception:
                continue
            rms_ref = np.sqrt(np.mean(ref ** 2))
            if rms_ref < 1e-6:
                continue

            r = tune_burgers_pencil(u0, x, nu, t_total, kappa=kappa)
            u_pred = picard_cascade(u0, x, k, nu, t_total, r.K, r.omega)
            err = np.sqrt(np.mean((u_pred - ref) ** 2)) / rms_ref

            K_opt, om_opt, err_opt = brute_force_optimum(
                u0, x, k, nu, t_total, ref, K_grid, omega_grid
            )

            rows.append(dict(
                profile=name, nu=nu,
                K_p=r.K, om_p=r.omega, err_p=err,
                K_opt=K_opt, err_opt=err_opt,
                Kr=r.K / max(K_opt, 1),
                L2r=err / max(err_opt, 1e-30),
            ))
    return rows


def loglog_slope(x_arr, y_arr):
    valid = (np.array(x_arr) > 0) & (np.array(y_arr) > 0)
    lx = np.log(np.array(x_arr)[valid])
    ly = np.log(np.array(y_arr)[valid])
    if len(lx) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, np.exp(intercept)


def calibrate_kappa():
    """Sweep kappa to find best calibration."""
    print("=" * 110)
    print("PENCIL TUNER — calibration sweep over kappa")
    print("=" * 110)

    kappas = [np.pi, np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]
    print(f"\n{'kappa':>10s}  {'gauss_Kr':>10s}  {'tanh_Kr':>10s}  {'sine_Kr':>10s}  "
          f"{'L2r_med':>9s}  {'Re_alpha':>9s}")
    print("-" * 75)
    best = None
    for kappa in kappas:
        rows_re = re_scan(kappa)
        rows_p = profile_sweep(kappa)
        all_rows = rows_re + rows_p
        gKr = [r["Kr"] for r in rows_p if r["profile"] == "gauss"]
        tKr = [r["Kr"] for r in rows_p if r["profile"] == "tanh"]
        sKr = [r["Kr"] for r in rows_p if r["profile"] == "sine"]
        L2_med = np.median([r["L2r"] for r in all_rows])
        Re_arr = [r["Re"] for r in rows_re]
        Kp_arr = [r["K_p"] for r in rows_re]
        slope, _ = loglog_slope(Re_arr, Kp_arr)
        print(f"{kappa:>10.4f}  {np.median(gKr):>10.2f}  {np.median(tKr):>10.2f}  "
              f"{np.median(sKr):>10.2f}  {L2_med:>9.2f}  {slope:>9.3f}")
        # Score: penalize deviation from Kr=1 and L2r=1
        score = (
            abs(np.log(np.median(gKr) + 1e-30))
            + abs(np.log(np.median(tKr) + 1e-30))
            + abs(np.log(np.median(sKr) + 1e-30))
            + abs(L2_med - 1)
        )
        if best is None or score < best[0]:
            best = (score, kappa, rows_re, rows_p, slope)

    print(f"\nBest: kappa={best[1]:.4f}  (score={best[0]:.3f}, Re_alpha={best[4]:.3f})")
    return best


def detail_print(rows_re, rows_p, kappa):
    print("\n" + "=" * 110)
    print(f"DETAIL with kappa={kappa:.4f}")
    print("=" * 110)
    print(f"\nRe scan:")
    print(f"{'nu':>5s}  {'Re':>4s}  {'K_p':>4s}  {'om':>5s}  {'L2_p':>7s}  "
          f"{'K_opt':>5s}  {'L2_opt':>7s}  {'Kr':>5s}  {'L2r':>5s}")
    for r in rows_re:
        print(f"{r['nu']:>5.3f}  {r['Re']:>4.0f}  {r['K_p']:>4d}  {r['om_p']:>5.2f}  "
              f"{r['err_p']:>7.1%}  {r['K_opt']:>5d}  {r['err_opt']:>7.1%}  "
              f"{r['Kr']:>5.2f}  {r['L2r']:>5.2f}")
    print(f"\nProfile sweep:")
    print(f"{'profile':<7s}  {'nu':>5s}  {'K_p':>4s}  {'om':>5s}  {'L2_p':>7s}  "
          f"{'K_opt':>5s}  {'L2_opt':>7s}  {'Kr':>5s}  {'L2r':>5s}")
    for r in rows_p:
        print(f"{r['profile']:<7s}  {r['nu']:>5.3f}  {r['K_p']:>4d}  {r['om_p']:>5.2f}  "
              f"{r['err_p']:>7.1%}  {r['K_opt']:>5d}  {r['err_opt']:>7.1%}  "
              f"{r['Kr']:>5.2f}  {r['L2r']:>5.2f}")


if __name__ == "__main__":
    best = calibrate_kappa()
    detail_print(best[2], best[3], best[1])
