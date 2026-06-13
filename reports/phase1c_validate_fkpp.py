"""Phase 1c step 2 — Fisher-KPP empirical validation.

Tests the three predictions of Theorem 1^F:
  (1) K_CFL = ceil(T * sqrt(2 r M_t / kappa_mag(eta))) ≈ K_opt from brute force
  (2) K scales as sqrt(r), independent of D
  (3) omega* = 1 is optimal (validating §5 correction of phase1c_theorem_generalization.md)

Single-signed IC (Gaussian bump in [0,1]) so (NL-A) holds.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp


def spectral_diff2(u, k):
    return np.real(np.fft.ifft(-(k ** 2) * np.fft.fft(u)))


def fkpp_rhs(t, u, D, r, k):
    return D * spectral_diff2(u, k) + r * u * (1 - u)


def solve_fkpp_reference(u0, D, r, T, k, rtol=1e-10, atol=1e-12):
    sol = solve_ivp(
        fkpp_rhs, [0, T], u0, args=(D, r, k),
        method="RK45", rtol=rtol, atol=atol, t_eval=[T],
    )
    return sol.y[:, -1]


def fkpp_cascade(u0, D, r, T, K, omega, k, rtol=1e-8, atol=1e-10):
    """One window of the Fisher-KPP spectral scalpel cascade."""
    dt = T / K
    u = u0.copy()
    for W in range(K):
        u_star = float(np.max(np.abs(u)))  # L^infty operating state (NL-A-compatible)
        alpha = r * (1 - 2 * u_star)
        beta_source = r * u_star ** 2

        def linear_rhs(t, v, k=k, D=D, alpha=alpha, bs=beta_source):
            return D * spectral_diff2(v, k) + alpha * v + bs

        sol = solve_ivp(
            linear_rhs, [0, dt], u, method="RK45",
            rtol=rtol, atol=atol, dense_output=True, t_eval=[dt],
        )
        v_tW = sol.y[:, -1]

        if abs(omega) > 1e-10:
            v_dense = sol.sol
            def picard_rhs(t, w, k=k, D=D, alpha=alpha, u_star=u_star, r=r, vd=v_dense):
                v_t = vd(t)
                rho_hat = -r * (v_t - u_star) ** 2
                return D * spectral_diff2(w, k) + alpha * w + rho_hat

            sol2 = solve_ivp(
                picard_rhs, [0, dt], np.zeros_like(u), method="RK45",
                rtol=rtol, atol=atol, t_eval=[dt],
            )
            w_tW = sol2.y[:, -1]
            u = v_tW + omega * w_tW
        else:
            u = v_tW

    return u


def tune_fkpp_pencil(u0, D, r, T, k, eta=0.85, K_min=2, K_max=64):
    """Pencil-derived K* from Theorem 1^F."""
    u_t = fkpp_rhs(0, u0, D, r, k)
    M_t = float(np.max(np.abs(u_t)))
    kappa_mag = (1 - eta) / eta
    K_raw = T * np.sqrt(2 * r * M_t / kappa_mag)
    return int(np.clip(int(np.ceil(K_raw)), K_min, K_max)), 1.0


def brute_force(u0, D, r, T, k, ref, K_grid, omega_grid):
    rms_ref = np.sqrt(np.mean(ref ** 2))
    best = (np.inf, 0, 0.0)
    for K_try in K_grid:
        for om_try in omega_grid:
            u_try = fkpp_cascade(u0, D, r, T, K_try, om_try, k)
            err = np.sqrt(np.mean((u_try - ref) ** 2)) / rms_ref
            if not np.isnan(err) and err < best[0]:
                best = (err, K_try, om_try)
    return best[1], best[2], best[0]


def make_grid(L=20.0, Nx=128):
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    return x, k


def gaussian_ic(x, amp=0.3, sigma=1.0):
    return amp * np.exp(-x ** 2 / (2 * sigma ** 2))


def test_r_scaling():
    """Prediction 1+2: K propto sqrt(r), independent of D."""
    x, k = make_grid()
    T = 2.0
    u0 = gaussian_ic(x)

    r_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    D_values = [0.01, 0.05, 0.1]
    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.5, 1.0]

    print("=" * 100)
    print("TEST 1 — K vs r scaling (D-independence)")
    print("=" * 100)
    print(f"\n{'D':>5s} {'r':>5s} {'K_tuner':>8s} {'om_t':>5s} {'L2_tuner':>10s} "
          f"{'K_opt':>6s} {'om_opt':>6s} {'L2_opt':>10s} {'Kr':>5s} {'L2r':>5s}")
    print("-" * 85)

    rows = []
    for D in D_values:
        for r in r_values:
            ref = solve_fkpp_reference(u0, D, r, T, k)
            K_tuner, omega_tuner = tune_fkpp_pencil(u0, D, r, T, k)
            u_tuner = fkpp_cascade(u0, D, r, T, K_tuner, omega_tuner, k)
            rms_ref = np.sqrt(np.mean(ref ** 2))
            err_t = np.sqrt(np.mean((u_tuner - ref) ** 2)) / rms_ref

            K_opt, om_opt, err_opt = brute_force(u0, D, r, T, k, ref, K_grid, omega_grid)

            Kr = K_tuner / max(K_opt, 1)
            L2r = err_t / max(err_opt, 1e-30)
            print(f"{D:>5.2f} {r:>5.1f} {K_tuner:>8d} {omega_tuner:>5.2f} "
                  f"{err_t:>10.3e} {K_opt:>6d} {om_opt:>6.2f} {err_opt:>10.3e} "
                  f"{Kr:>5.2f} {L2r:>5.2f}")
            rows.append(dict(
                D=D, r=r, K_tuner=K_tuner, K_opt=K_opt,
                om_opt=om_opt, Kr=Kr, L2r=L2r,
                err_tuner=err_t, err_opt=err_opt,
            ))

    # Scaling check
    print("\n" + "=" * 100)
    print("SCALING CHECK")
    print("=" * 100)

    def loglog_slope(x_arr, y_arr):
        lx = np.log(np.asarray(x_arr))
        ly = np.log(np.asarray(y_arr))
        m, b = np.polyfit(lx, ly, 1)
        return m, np.exp(b)

    # r-scan at fixed D
    for D in D_values:
        sub = [r for r in rows if abs(r["D"] - D) < 1e-12]
        rs = [s["r"] for s in sub]
        Kts = [s["K_tuner"] for s in sub]
        Kos = [s["K_opt"] for s in sub]
        mt, ct = loglog_slope(rs, Kts)
        mo, co = loglog_slope(rs, Kos)
        print(f"  D={D}: tuner K ~ {ct:.2f} r^{mt:.3f} ; "
              f"opt K ~ {co:.2f} r^{mo:.3f}")

    print("\nTheory predicts: K ~ sqrt(r) (slope 0.5), independent of D.")
    print(f"\nomega_opt values (should all be 1.0): {sorted(set([r['om_opt'] for r in rows]))}")

    return rows


def test_omega_optimality():
    """Prediction 3: omega* = 1 is optimal (reactive full Picard)."""
    x, k = make_grid()
    T = 2.0
    D = 0.05
    r = 2.0
    u0 = gaussian_ic(x)

    ref = solve_fkpp_reference(u0, D, r, T, k)
    rms_ref = np.sqrt(np.mean(ref ** 2))

    K = 8
    print("\n" + "=" * 100)
    print(f"TEST 2 — omega optimality (D={D}, r={r}, K={K})")
    print("=" * 100)
    print(f"\n{'omega':>7s} {'L2_err':>12s}")
    for om in np.linspace(0, 1.2, 13):
        u_try = fkpp_cascade(u0, D, r, T, K, om, k)
        err = np.sqrt(np.mean((u_try - ref) ** 2)) / rms_ref
        star = "  <-- theory predicts optimum" if abs(om - 1.0) < 1e-12 else ""
        print(f"{om:>7.2f} {err:>12.3e}{star}")


if __name__ == "__main__":
    rows = test_r_scaling()
    test_omega_optimality()
