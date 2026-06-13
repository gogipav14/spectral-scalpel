"""Phase 1c step 2 — Kuramoto-Sivashinsky validation.

KS has Burgers' nonlinearity -u*u_x (spatial-derivative structure, wave-like),
so it should fit the pencil cascade cleanly.

However, KS naturally evolves to chaotic dynamics with sign changes (violating NL-A).
We test on a SHORT-TIME single-sign regime: take a positive bump IC, run before
it develops complex features.

Predictions from Theorem 1^{KS}:
  K_CFL = ceil(T * sqrt(k^* * M_t / kappa(eta)))
  with k^* = 1 (peak of active band), eta=0.85 so kappa = pi/8.
  Discretization-independent (fixed k*), no Re-analog.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp


def spectral_diff2(u, k):
    return np.real(np.fft.ifft(-(k ** 2) * np.fft.fft(u)))


def spectral_diff4(u, k):
    return np.real(np.fft.ifft((k ** 4) * np.fft.fft(u)))


def ks_rhs(t, u, k):
    # u_t + u u_x + u_xx + u_xxxx = 0
    u_x = np.real(np.fft.ifft(1j * k * np.fft.fft(u)))
    u_xx = spectral_diff2(u, k)
    u_xxxx = spectral_diff4(u, k)
    return -u * u_x - u_xx - u_xxxx


def solve_ks_reference(u0, T, k, rtol=1e-10, atol=1e-12):
    sol = solve_ivp(
        ks_rhs, [0, T], u0, args=(k,),
        method="RK45", rtol=rtol, atol=atol, t_eval=[T],
    )
    return sol.y[:, -1]


def ks_cascade(u0, T, K, omega, k, rtol=1e-8, atol=1e-10):
    dt = T / K
    u = u0.copy()
    for W in range(K):
        u_star = float(np.max(np.abs(u)))
        u_star = np.sign(np.mean(u)) * u_star if np.mean(u) != 0 else u_star

        def linear_rhs(t, v, k=k, us=u_star):
            v_x = np.real(np.fft.ifft(1j * k * np.fft.fft(v)))
            v_xx = spectral_diff2(v, k)
            v_xxxx = spectral_diff4(v, k)
            return -us * v_x - v_xx - v_xxxx

        sol = solve_ivp(
            linear_rhs, [0, dt], u, method="RK45",
            rtol=rtol, atol=atol, dense_output=True, t_eval=[dt],
        )
        v_tW = sol.y[:, -1]

        if abs(omega) > 1e-10:
            v_dense = sol.sol

            def picard_rhs(t, w, k=k, us=u_star, vd=v_dense):
                v_t = vd(t)
                v_x = np.real(np.fft.ifft(1j * k * np.fft.fft(v_t)))
                rho_hat = -(v_t - us) * v_x
                w_x = np.real(np.fft.ifft(1j * k * np.fft.fft(w)))
                w_xx = spectral_diff2(w, k)
                w_xxxx = spectral_diff4(w, k)
                return -us * w_x - w_xx - w_xxxx + rho_hat

            sol2 = solve_ivp(
                picard_rhs, [0, dt], np.zeros_like(u), method="RK45",
                rtol=rtol, atol=atol, t_eval=[dt],
            )
            w_tW = sol2.y[:, -1]
            u = v_tW + omega * w_tW
        else:
            u = v_tW
    return u


def tune_ks_pencil(u0, T, k, eta=0.85, K_min=2, K_max=64, k_star=1.0):
    u_t = ks_rhs(0, u0, k)
    M_t = float(np.max(np.abs(u_t)))
    kappa = np.arccos(np.sqrt(eta))
    K_raw = T * np.sqrt(k_star * M_t / kappa)
    return int(np.clip(int(np.ceil(K_raw)), K_min, K_max)), float(np.cos(min(k_star * M_t * (T/max(K_min, int(np.ceil(K_raw))))**2, np.pi/2))**2)


def brute_force(u0, T, k, ref, K_grid, omega_grid):
    rms_ref = np.sqrt(np.mean(ref ** 2))
    best = (np.inf, 0, 0.0)
    for K_try in K_grid:
        for om_try in omega_grid:
            u_try = ks_cascade(u0, T, K_try, om_try, k)
            err = np.sqrt(np.mean((u_try - ref) ** 2)) / rms_ref
            if not np.isnan(err) and err < best[0]:
                best = (err, K_try, om_try)
    return best[1], best[2], best[0]


def make_grid(L=32.0, Nx=128):
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    return x, k


def test_short_time_ks():
    """KS on single-sign bump ICs, short time."""
    x, k = make_grid(L=32.0, Nx=128)

    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.5, 1.0]

    print("=" * 100)
    print("KS short-time validation (single-sign regime)")
    print("=" * 100)
    print(f"\n{'amp':>5s} {'T':>5s} {'K_t':>4s} {'om_t':>5s} {'L2_t':>10s} "
          f"{'K_opt':>6s} {'om_opt':>6s} {'L2_opt':>10s} {'Kr':>5s} {'L2r':>5s}")
    print("-" * 85)

    rows = []
    for amp in [0.3, 0.5, 1.0]:
        for T in [0.5, 1.0, 2.0]:
            u0 = amp * np.exp(-x ** 2 / (2 * 4.0 ** 2))
            try:
                ref = solve_ks_reference(u0, T, k)
                if np.any(np.isnan(ref)) or np.max(np.abs(ref)) > 100:
                    print(f"{amp:>5.2f} {T:>5.1f}  reference solution blew up")
                    continue
            except Exception as e:
                print(f"{amp:>5.2f} {T:>5.1f}  reference failed: {e}")
                continue

            rms_ref = np.sqrt(np.mean(ref ** 2))
            K_tuner, om_tuner = tune_ks_pencil(u0, T, k)
            u_tuner = ks_cascade(u0, T, K_tuner, om_tuner, k)
            err_t = np.sqrt(np.mean((u_tuner - ref) ** 2)) / rms_ref

            K_opt, om_opt, err_opt = brute_force(u0, T, k, ref, K_grid, omega_grid)

            Kr = K_tuner / max(K_opt, 1)
            L2r = err_t / max(err_opt, 1e-30)

            print(f"{amp:>5.2f} {T:>5.1f} {K_tuner:>4d} {om_tuner:>5.2f} "
                  f"{err_t:>10.3e} {K_opt:>6d} {om_opt:>6.2f} {err_opt:>10.3e} "
                  f"{Kr:>5.2f} {L2r:>5.2f}")
            rows.append(dict(
                amp=amp, T=T, K_t=K_tuner, K_opt=K_opt, Kr=Kr, L2r=L2r,
                err_t=err_t, err_opt=err_opt,
            ))
    return rows


if __name__ == "__main__":
    test_short_time_ks()
