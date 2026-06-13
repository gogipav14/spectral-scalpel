"""Phase 1c step 2 — Allen-Cahn empirical validation.

Tests the two main predictions of Theorem 1^A:
  (1) K_CFL = ceil(T * sqrt(6 |u_0^*| M_t / kappa_mag(eta))) ≈ K_opt from brute force
  (2) K scales with |u_0^*| at the operating state — weak at the wells (|u_0^*|≈1),
      stronger near the unstable equilibrium (|u_0^*|≈0)
  (3) omega* = 1 optimal (same §5 correction as FKPP)

Use transition-layer IC: u0 = tanh(x/width). Unique-signed on [0, inf) and [-inf, 0],
so (NL-A) applies on each half-domain. We test the positive half with u_0^* > 0.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp


def spectral_diff2(u, k):
    return np.real(np.fft.ifft(-(k ** 2) * np.fft.fft(u)))


def ac_rhs(t, u, eps2, k):
    return eps2 * spectral_diff2(u, k) + u - u ** 3


def solve_ac_reference(u0, eps2, T, k, rtol=1e-10, atol=1e-12):
    sol = solve_ivp(
        ac_rhs, [0, T], u0, args=(eps2, k),
        method="RK45", rtol=rtol, atol=atol, t_eval=[T],
    )
    return sol.y[:, -1]


def ac_cascade(u0, eps2, T, K, omega, k, rtol=1e-8, atol=1e-10):
    dt = T / K
    u = u0.copy()
    for W in range(K):
        u_star = float(np.max(np.abs(u)))  # L^inf operating state; same sign as max
        u_star = np.sign(np.mean(u)) * u_star if np.mean(u) != 0 else u_star
        alpha = 1 - 3 * u_star ** 2
        beta_source = u_star - u_star ** 3 - alpha * u_star  # const source

        def linear_rhs(t, v, k=k, eps2=eps2, alpha=alpha, bs=beta_source):
            return eps2 * spectral_diff2(v, k) + alpha * v + bs

        sol = solve_ivp(
            linear_rhs, [0, dt], u, method="RK45",
            rtol=rtol, atol=atol, dense_output=True, t_eval=[dt],
        )
        v_tW = sol.y[:, -1]

        if abs(omega) > 1e-10:
            v_dense = sol.sol

            def picard_rhs(t, w, k=k, eps2=eps2, alpha=alpha, u_star=u_star, vd=v_dense):
                v_t = vd(t)
                # rho_hat = -(v - u*)^2 (v + 2 u*)
                diff = v_t - u_star
                rho_hat = -(diff ** 2) * (v_t + 2 * u_star)
                return eps2 * spectral_diff2(w, k) + alpha * w + rho_hat

            sol2 = solve_ivp(
                picard_rhs, [0, dt], np.zeros_like(u), method="RK45",
                rtol=rtol, atol=atol, t_eval=[dt],
            )
            w_tW = sol2.y[:, -1]
            u = v_tW + omega * w_tW
        else:
            u = v_tW
    return u


def tune_ac_pencil(u0, eps2, T, k, eta=0.85, K_min=2, K_max=64):
    """Pencil-derived K* from Theorem 1^A."""
    u_star = float(np.max(np.abs(u0)))
    u_t = ac_rhs(0, u0, eps2, k)
    M_t = float(np.max(np.abs(u_t)))
    kappa_mag = (1 - eta) / eta
    K_raw = T * np.sqrt(6 * u_star * M_t / kappa_mag)
    return int(np.clip(int(np.ceil(K_raw)), K_min, K_max)), 1.0


def brute_force(u0, eps2, T, k, ref, K_grid, omega_grid):
    rms_ref = np.sqrt(np.mean(ref ** 2))
    best = (np.inf, 0, 0.0)
    for K_try in K_grid:
        for om_try in omega_grid:
            u_try = ac_cascade(u0, eps2, T, K_try, om_try, k)
            err = np.sqrt(np.mean((u_try - ref) ** 2)) / rms_ref
            if not np.isnan(err) and err < best[0]:
                best = (err, K_try, om_try)
    return best[1], best[2], best[0]


def make_grid(L=10.0, Nx=128):
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    return x, k


def transition_ic(x, amp=0.8, width=0.5, center=0.0):
    """Monotone transition layer — single-signed if center = 0 and we're in x > 0.
    Use half-tanh shifted to stay positive on the full domain.
    """
    return amp * 0.5 * (1 + np.tanh((x - center) / width))  # in [0, amp]


def test_operating_state_scaling():
    """Prediction: K depends on |u_0^*|, the amplitude of the operating state."""
    x, k = make_grid()
    T = 1.5
    eps2 = 0.01

    K_grid = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    omega_grid = [0.0, 0.5, 1.0]

    print("=" * 100)
    print("ALLEN-CAHN — K vs operating-state amplitude")
    print("=" * 100)
    print(f"\n{'amp':>5s} {'u_0*':>6s} {'M_t':>7s} {'K_tuner':>8s} {'om_t':>5s} "
          f"{'L2_t':>10s} {'K_opt':>6s} {'om_opt':>6s} {'L2_opt':>10s} "
          f"{'Kr':>5s} {'L2r':>5s}")
    print("-" * 95)

    rows = []
    for amp in [0.2, 0.5, 0.8, 0.95]:
        u0 = transition_ic(x, amp=amp)
        ref = solve_ac_reference(u0, eps2, T, k)
        K_tuner, om_tuner = tune_ac_pencil(u0, eps2, T, k)
        u_tuner = ac_cascade(u0, eps2, T, K_tuner, om_tuner, k)
        rms_ref = np.sqrt(np.mean(ref ** 2))
        err_t = np.sqrt(np.mean((u_tuner - ref) ** 2)) / rms_ref

        K_opt, om_opt, err_opt = brute_force(u0, eps2, T, k, ref, K_grid, omega_grid)

        u_star = float(np.max(np.abs(u0)))
        u_t = ac_rhs(0, u0, eps2, k)
        M_t = float(np.max(np.abs(u_t)))

        Kr = K_tuner / max(K_opt, 1)
        L2r = err_t / max(err_opt, 1e-30)

        print(f"{amp:>5.2f} {u_star:>6.3f} {M_t:>7.3f} {K_tuner:>8d} {om_tuner:>5.2f} "
              f"{err_t:>10.3e} {K_opt:>6d} {om_opt:>6.2f} {err_opt:>10.3e} "
              f"{Kr:>5.2f} {L2r:>5.2f}")

        rows.append(dict(amp=amp, u_star=u_star, M_t=M_t,
                         K_tuner=K_tuner, K_opt=K_opt, om_opt=om_opt,
                         Kr=Kr, L2r=L2r))

    print("\nTheory predicts: K ∝ sqrt(|u_0^*| * M_t)")
    print(f"omega_opt values (should all be 1.0): {sorted(set([r['om_opt'] for r in rows]))}")

    return rows


if __name__ == "__main__":
    test_operating_state_scaling()
