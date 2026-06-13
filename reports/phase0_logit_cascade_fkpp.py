"""Phase 0 for logit-space Fisher-KPP cascade.

Question: does transforming Fisher-KPP to logit coordinates w = log(u/(1-u))
and running the pencil cascade in w-space recover K-convergence?

Logit-space PDE:
    dw/dt = D w_xx + D(1 - 2 sigma(w)) w_x^2 + r

Linearized around scalar w_0^*:
    linear part:     dv/dt = D v_xx + r  (since w_{0,x}^* = 0)
    Picard residual: rho = D(1 - 2 sigma(w)) w_x^2   (no (w-w_0^*) factor)

Test IC: u_0(x) = 0.5 + amp * cos(2 pi x / L).
This stays in (0.5 - amp, 0.5 + amp); for amp <= 0.4, well-inside (0, 1),
logit is regular, reaction term r u(1-u) is always active.

Predictions:
  (a) If the logit cascade converges in K with cubic balance, L2 error
      should drop as ~1/K^2 as K grows (gauge transform succeeds).
  (b) If it plateaus with K like the u-space cascade did, logit doesn't
      help (gauge transform fails).

Reference solver: RK45 on original Fisher-KPP in u-space with tight tolerances.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp


def spectral_diff2(u, k):
    return np.real(np.fft.ifft(-(k ** 2) * np.fft.fft(u)))


def spectral_diff1(u, k):
    return np.real(np.fft.ifft(1j * k * np.fft.fft(u)))


def sigma(w):
    # Sigmoid; clip w to avoid overflow for safety.
    w_c = np.clip(w, -50, 50)
    return 1.0 / (1.0 + np.exp(-w_c))


def logit(u, eps=0.0):
    u_c = np.clip(u, eps, 1 - eps) if eps > 0 else u
    return np.log(u_c / (1.0 - u_c))


def fkpp_rhs_u(t, u, D, r, k):
    return D * spectral_diff2(u, k) + r * u * (1 - u)


def solve_ref(u0, D, r, T, k):
    sol = solve_ivp(
        fkpp_rhs_u, [0, T], u0, args=(D, r, k),
        method="RK45", rtol=1e-11, atol=1e-13, t_eval=[T],
    )
    return sol.y[:, -1]


def fkpp_logit_cascade(u0, D, r, T, K, omega, k, rtol=1e-10, atol=1e-12):
    """Fisher-KPP cascade in logit space with Picard correction."""
    w = logit(u0)
    dt = T / K

    for W in range(K):
        # Linear substep in logit space: dw/dt = D w_xx + r
        # Analytical per-mode propagation (no need for solve_ivp):
        #   DC mode:  w_0(tau) = w_0(0) + r*tau
        #   Non-DC:   w_n(tau) = w_n(0) * exp(-D k_n^2 tau)
        w_hat = np.fft.fft(w)
        decay = np.exp(-D * (k ** 2) * dt)
        v_hat = w_hat * decay
        # Source: r uniformly adds to DC mode at rate r per unit time
        # In Fourier, constant r has coefficient r*Nx only at k=0 (by our FFT norm)
        Nx = len(w)
        v_hat[0] += r * dt * Nx  # constant in x contributes r*Nx to k=0 of FFT
        v_tW = np.real(np.fft.ifft(v_hat))

        if abs(omega) > 1e-10:
            # Dense trajectory of v for Picard integral. Compute at midpoint
            # only for simplicity (midpoint quadrature).
            v_mid_hat = w_hat * np.exp(-D * (k ** 2) * (dt / 2))
            v_mid_hat[0] += r * (dt / 2) * Nx
            v_mid = np.real(np.fft.ifft(v_mid_hat))

            # Picard residual at midpoint:
            #   rho(x, dt/2) = D (1 - 2 sigma(v)) v_x^2
            v_mid_x = spectral_diff1(v_mid, k)
            rho_mid = D * (1.0 - 2.0 * sigma(v_mid)) * v_mid_x ** 2

            # Propagate rho through semigroup from midpoint to end of window
            rho_hat = np.fft.fft(rho_mid)
            decay_half = np.exp(-D * (k ** 2) * (dt / 2))
            h_hat = rho_hat * decay_half * dt  # midpoint-rule integral
            h_tW = np.real(np.fft.ifft(h_hat))

            w = v_tW + omega * h_tW
        else:
            w = v_tW

        # Clip w to stay in representable range
        w = np.clip(w, -50, 50)

    return sigma(w)


def fkpp_u_space_cascade(u0, D, r, T, K, omega, k, rtol=1e-10, atol=1e-12):
    """u-space cascade for comparison (reproduces the Phase 1c failing scheme)."""
    u = u0.copy()
    dt = T / K
    for W in range(K):
        u_star = float(np.max(np.abs(u)))
        alpha = r * (1 - 2 * u_star)
        beta_source = r * u_star ** 2

        def lin_rhs(t, v):
            return D * spectral_diff2(v, k) + alpha * v + beta_source

        sol = solve_ivp(lin_rhs, [0, dt], u, method="RK45",
                        rtol=rtol, atol=atol, dense_output=True, t_eval=[dt])
        v_tW = sol.y[:, -1]

        if abs(omega) > 1e-10:
            vd = sol.sol
            def picard_rhs(t, h, alpha=alpha, u_star=u_star, vd=vd):
                v_t = vd(t)
                rho_hat = -r * (v_t - u_star) ** 2
                return D * spectral_diff2(h, k) + alpha * h + rho_hat
            sol2 = solve_ivp(picard_rhs, [0, dt], np.zeros_like(u),
                             method="RK45", rtol=rtol, atol=atol, t_eval=[dt])
            h_tW = sol2.y[:, -1]
            u = v_tW + omega * h_tW
        else:
            u = v_tW
    return u


def convergence_sweep():
    """Compare K-convergence of logit vs u-space cascade on the same IC."""
    Nx = 128
    L = 4.0  # periodic box
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi

    T = 0.5
    D = 0.1

    # Test 1: near-midpoint IC, u in (0.3, 0.7). Logit regular.
    def ic_midpoint(x):
        return 0.5 + 0.2 * np.cos(2 * np.pi * x / L)

    # Test 2: larger excursion, u in (0.1, 0.9). Logit still OK.
    def ic_stressed(x):
        return 0.5 + 0.4 * np.cos(2 * np.pi * x / L)

    # Test 3: asymmetric, u in (0.2, 0.8).
    def ic_asym(x):
        return 0.5 + 0.3 * np.sin(2 * np.pi * x / L)

    K_values = [2, 4, 8, 16, 32, 64]

    for name, ic_fn in [("midpoint", ic_midpoint),
                         ("stressed", ic_stressed),
                         ("asym", ic_asym)]:
        u0 = ic_fn(x)
        u_min, u_max = u0.min(), u0.max()

        for r in [0.5, 2.0, 5.0]:
            ref = solve_ref(u0, D, r, T, k)
            rms_ref = np.sqrt(np.mean(ref ** 2))

            print(f"\n{'='*100}")
            print(f"IC={name} (u in [{u_min:.2f}, {u_max:.2f}]), D={D}, r={r}, T={T}")
            print(f"{'='*100}")
            print(f"{'K':>4s}  {'L2_logit(om=1)':>15s}  {'L2_logit(om=0)':>15s}  "
                  f"{'L2_u(om=1)':>12s}  {'L2_u(om=0)':>12s}")

            for K in K_values:
                u_logit_picard = fkpp_logit_cascade(u0, D, r, T, K, 1.0, k)
                u_logit_lin = fkpp_logit_cascade(u0, D, r, T, K, 0.0, k)
                u_u_picard = fkpp_u_space_cascade(u0, D, r, T, K, 1.0, k)
                u_u_lin = fkpp_u_space_cascade(u0, D, r, T, K, 0.0, k)

                def err(u_est):
                    if np.any(np.isnan(u_est)) or np.max(np.abs(u_est)) > 10:
                        return float("nan")
                    return np.sqrt(np.mean((u_est - ref) ** 2)) / rms_ref

                print(f"{K:>4d}  {err(u_logit_picard):>15.3e}  "
                      f"{err(u_logit_lin):>15.3e}  "
                      f"{err(u_u_picard):>12.3e}  {err(u_u_lin):>12.3e}")


if __name__ == "__main__":
    convergence_sweep()
