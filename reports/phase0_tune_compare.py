"""Phase 0 comparison: production algorithm2_tune (cubic-balance) vs burgers_tuner_v2 (fitted).

Reuses the same brute-force sweep Burgers setup from
nonlinear/scripts/burgers_tuner_v2.py and asks: does the first-principles
cubic-balance law K* = (2*M*||u_t||^2*t^3/eps_N)^{1/3} predict K_opt as well
as the empirically-fitted K* = C_K * Gamma^0.55 law?

This is the Track B Phase 0 Step 2 deliverable: the first direct empirical
check of the theorem the nonlinear paper will stake its claim on.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scalpel.nonlinear.tuner import tune_burgers, profile_diagnostics
# import picard_cascade + cole_hopf_exact from the v2 script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "burgers_tuner_v2",
    os.path.join(os.path.dirname(__file__), "..", "nonlinear", "scripts", "burgers_tuner_v2.py"),
)
# The v2 script executes its main block at import; isolate by reading only defs.
# Simpler: redefine picard_cascade + cole_hopf_exact here (copied verbatim).

from scalpel.core.feasibility import tune_params


def cole_hopf_exact(x, t, nu, u0_func, Nx_quad=4000):
    if t <= 0:
        return np.array([u0_func(xi) for xi in x])
    xi = np.linspace(
        x.min() - 8 * np.sqrt(2 * nu * t), x.max() + 8 * np.sqrt(2 * nu * t), Nx_quad
    )
    dxi = xi[1] - xi[0]
    u0v = np.array([u0_func(s) for s in xi])
    Phi = np.cumsum(u0v) * dxi
    result = np.zeros_like(x)
    for i, xv in enumerate(x):
        G = -1 / (2 * nu) * (Phi + (xv - xi) ** 2 / (2 * t))
        G -= G.max()
        ig = np.exp(G)
        phi = np.sum(ig) * dxi
        dp = np.sum(ig * (-(xv - xi) / (2 * nu * t))) * dxi
        if abs(phi) > 1e-300:
            result[i] = -2 * nu * dp / phi
    return result


def estimate_u0(u, x):
    dudx = np.gradient(u, x)
    w = np.abs(dudx)
    ws = w.sum()
    if ws < 1e-30:
        return 0.0
    return float(np.clip(np.sum(w * u) / ws, 0, 5))


def picard_cascade(u0_x, x, k, nu, t_total, K, omega, N_nilt=2048):
    dt = t_total / K
    u = u0_x.copy()
    for kk in range(K):
        ue = estimate_u0(u, x)
        rho = nu * np.max(k ** 2)
        params = tune_params(
            t_end=dt, alpha_c=0.0, C=1.0, kappa=2.0, eps_tail=1e-6,
            N_init=N_nilt, rho=rho,
        )
        a, T, N = params.a, params.T, params.N
        s_c = a + 1j * np.arange(N) * (np.pi / T)
        u_hat = np.fft.fft(u)
        poles = nu * k ** 2 + 1j * k * ue
        G = u_hat[:, None] / (s_c[None, :] + poles[:, None])
        G[0, :] = u_hat[0] / s_c
        G[:, 0] *= 0.5
        z_raw = N * np.fft.ifft(G, axis=1)
        t_arr = np.arange(N) * 2 * T / N
        corr = np.exp(a * t_arr) / T
        u_hat_t = z_raw * corr[None, :]
        u_xt = np.fft.ifft(u_hat_t, axis=0).real
        idx = np.argmin(np.abs(t_arr - dt))
        dt_act = t_arr[idx]

        if omega > 0.01:
            t_mask = (t_arr >= 0) & (t_arr <= dt * 1.05)
            u_phys = u_xt[:, t_mask]
            t_phys = t_arr[t_mask]
            u_hat_p = np.fft.fft(u_phys, axis=0)
            du_dx = np.fft.ifft(1j * k[:, None] * u_hat_p, axis=0).real
            residual = -(u_phys - ue) * du_dx
            dt_phys = np.diff(t_phys, prepend=0)
            weights = np.exp(-nu * k[1] ** 2 * np.maximum(dt_act - t_phys, 0))
            correction = np.sum(
                residual * np.clip(weights, 0, 1)[None, :] * dt_phys[None, :], axis=1
            )
            u = u_xt[:, idx] + omega * correction
        else:
            u = u_xt[:, idx]
    return u


def brute_force_optimum(u0_x, x, k, nu, t_total, ref, K_grid, omega_grid):
    """Exhaustive search for (K, omega) that minimizes relative L2 error."""
    rms_ref = np.sqrt(np.mean(ref ** 2))
    best_err = np.inf
    best_K = K_grid[0]
    best_omega = omega_grid[0]
    for K_try in K_grid:
        for omega_try in omega_grid:
            u_try = picard_cascade(u0_x, x, k, nu, t_total, K_try, omega_try)
            e = np.sqrt(np.mean((u_try - ref) ** 2)) / rms_ref
            if not np.isnan(e) and e < best_err:
                best_err = e
                best_K = K_try
                best_omega = omega_try
    return best_K, best_omega, best_err


def run_comparison():
    A_amp = 1.0
    w_sigma = 0.5
    L_domain = 16.0
    t_total = 2.0
    Nx = 256
    x = np.linspace(-L_domain / 2, L_domain / 2, Nx)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    u0_x = np.array([A_amp * np.exp(-xv ** 2 / (2 * w_sigma ** 2)) for xv in x])

    print("=" * 120)
    print("PHASE 0 — algorithm2_tune (cubic-balance) vs brute-force optimum")
    print("Burgers equation, Gaussian IC (A=1, sigma=0.5), domain [-8, 8], t=2")
    print("=" * 120)

    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    nu_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    print(
        f"\n{'nu':>6s}  {'Re':>6s}  {'K_cubic':>8s}  {'om_cubic':>9s}  "
        f"{'L2_cubic':>9s}  {'K_opt':>6s}  {'om_opt':>7s}  {'L2_opt':>9s}  "
        f"{'K_ratio':>8s}  {'L2_ratio':>9s}"
    )
    print("-" * 110)

    rows = []
    for nu in nu_values:
        Re = A_amp / nu
        ref = cole_hopf_exact(
            x, t_total, nu, lambda xv: A_amp * np.exp(-xv ** 2 / (2 * w_sigma ** 2))
        )
        rms_ref = np.sqrt(np.mean(ref ** 2))

        # Production tuner (algorithm2_tune via tune_burgers)
        result = tune_burgers(u0_x, x, nu, t_total)
        K_cubic = result.K
        omega_cubic = result.omega

        u_cubic = picard_cascade(u0_x, x, k, nu, t_total, K_cubic, omega_cubic)
        err_cubic = np.sqrt(np.mean((u_cubic - ref) ** 2)) / rms_ref

        # Brute-force optimum
        K_opt, omega_opt, err_opt = brute_force_optimum(
            u0_x, x, k, nu, t_total, ref, K_grid, omega_grid
        )

        K_ratio = K_cubic / K_opt if K_opt > 0 else float("inf")
        L2_ratio = err_cubic / err_opt if err_opt > 0 else float("inf")

        print(
            f"{nu:>6.3f}  {Re:>6.0f}  {K_cubic:>8d}  {omega_cubic:>9.3f}  "
            f"{err_cubic:>9.1%}  {K_opt:>6d}  {omega_opt:>7.2f}  {err_opt:>9.1%}  "
            f"{K_ratio:>8.2f}  {L2_ratio:>9.2f}"
        )
        rows.append(
            {
                "nu": nu, "Re": Re, "K_cubic": K_cubic, "omega_cubic": omega_cubic,
                "err_cubic": err_cubic, "K_opt": K_opt, "omega_opt": omega_opt,
                "err_opt": err_opt, "K_ratio": K_ratio, "L2_ratio": L2_ratio,
                "Gamma_NL": result.Gamma_NL,
                "eps_total_pred": result.eps_total_pred,
            }
        )

    # Scaling-law fit: log-log regression of K_opt vs some nonlinearity measure
    print("\n" + "=" * 120)
    print("SCALING-LAW CHECK")
    print("=" * 120)
    Re_arr = np.array([r["Re"] for r in rows])
    Kopt_arr = np.array([r["K_opt"] for r in rows])
    Kcubic_arr = np.array([r["K_cubic"] for r in rows])
    Gamma_arr = np.array([r["Gamma_NL"] for r in rows])

    # Fit K_opt ~ Re^alpha
    def loglog_slope(x_arr, y_arr):
        valid = (x_arr > 0) & (y_arr > 0)
        lx = np.log(x_arr[valid])
        ly = np.log(y_arr[valid])
        slope, intercept = np.polyfit(lx, ly, 1)
        return slope, np.exp(intercept)

    alpha_Re, c_Re = loglog_slope(Re_arr, Kopt_arr)
    alpha_G, c_G = loglog_slope(Gamma_arr, Kopt_arr)
    alpha_cubic_Re, c_cubic_Re = loglog_slope(Re_arr, Kcubic_arr)

    print(f"\n  Empirical K_opt ~ {c_Re:.2f} * Re^{alpha_Re:.3f}")
    print(f"  Empirical K_opt ~ {c_G:.2f} * Gamma_NL^{alpha_G:.3f}")
    print(f"  Cubic K_cubic ~ {c_cubic_Re:.2f} * Re^{alpha_cubic_Re:.3f}")
    print(f"\n  Cubic-balance theory predicts K ∝ (M·||u_t||^2·t^3/eps_N)^{{1/3}}")
    print(f"  For fixed A, sigma, t, eps_N: M·||u_t||^2 ~ u_x_max^3 · (1/nu)^0")
    print(f"  so with the factors that matter, K_cubic exponent wrt Re is what algorithm2 emits.")

    return rows


if __name__ == "__main__":
    run_comparison()
