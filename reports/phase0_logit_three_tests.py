"""Three follow-ups to the logit-space Phase 0 finding.

TEST 1 — Simpson-rule / two-iteration Picard for cubic balance.
    Prediction: 2-iteration Picard gives O(Delta t^3) per window -> O(1/K^2)
    aggregate (cubic in K for the K vs eps_N balance).

TEST 2 — Allen-Cahn via arctanh gauge.
    Transform: w = arctanh(u)  <=>  u = tanh(w)
    PDE in w-space:
        w_t = eps^2 w_xx - 2 eps^2 tanh(w) w_x^2 + tanh(w)
    Linearize source tanh(w) around w_0^*: tanh(w_0^*) + sech^2(w_0^*)(w - w_0^*)
    Then linear substep: w_t = eps^2 w_xx + sech^2(w_0^*) w + const
    Residual: -2 eps^2 tanh(w) w_x^2 + [higher order from tanh linearization]

TEST 3 — Regularized logit for bump ICs approaching u=0.
    Transform: w = log((u+eps_reg) / (1-u+eps_reg))
    Need eps_reg small but nonzero to avoid singularity at u->0.
    Target IC: Gaussian bump that propagates as a Fisher front.
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
    w_c = np.clip(w, -50, 50)
    return 1.0 / (1.0 + np.exp(-w_c))


def fkpp_rhs_u(t, u, D, r, k):
    return D * spectral_diff2(u, k) + r * u * (1 - u)


def ac_rhs_u(t, u, eps2, k):
    return eps2 * spectral_diff2(u, k) + u - u ** 3


def solve_ref(rhs_fn, u0, T, k, args, rtol=1e-11, atol=1e-13):
    sol = solve_ivp(
        rhs_fn, [0, T], u0, args=(*args, k),
        method="RK45", rtol=rtol, atol=atol, t_eval=[T],
    )
    return sol.y[:, -1]


# ============================================================================
# TEST 1 — Fisher-KPP in logit space with 1- vs 2-iteration Picard
# ============================================================================

def logit(u, eps=0.0):
    if eps > 0:
        u = np.clip(u, eps, 1.0 - eps)
    return np.log(u / (1.0 - u))


def fkpp_logit_cascade(u0, D, r, T, K, omega, k, n_picard=1, eps_reg=0.0):
    """Fisher-KPP cascade in logit space with n_picard iterations of Picard."""
    w = logit(u0, eps_reg)
    dt = T / K
    Nx = len(w)

    for _ in range(K):
        # Linear substep: dv/dt = D v_xx + r. Exact per mode.
        w_hat = np.fft.fft(w)
        v_hat = w_hat * np.exp(-D * (k ** 2) * dt)
        v_hat[0] += r * dt * Nx
        v = np.real(np.fft.ifft(v_hat))

        # Picard iterations — each one refines using the previous iterate as v
        v_current = v.copy()
        for _pic in range(n_picard):
            if abs(omega) < 1e-10:
                break
            # Compute residual over the window using the current iterate's
            # trajectory. Trajectory of v_current: from w at t=0 to v_current
            # at t=dt. We use solve_ivp to get the ODE integration with
            # time-dependent source.
            def linear_with_resid(t, h, k=k, D=D):
                # Interpolate v_current between w and v (linear in time as a
                # crude approximation) and evaluate rho.
                alpha_t = t / dt
                v_at_t = (1 - alpha_t) * w + alpha_t * v_current
                v_x = spectral_diff1(v_at_t, k)
                rho = D * (1.0 - 2.0 * sigma(v_at_t)) * (v_x ** 2)
                return D * spectral_diff2(h, k) + rho

            sol = solve_ivp(
                linear_with_resid, [0, dt], np.zeros_like(w),
                method="RK45", rtol=1e-9, atol=1e-11, t_eval=[dt],
            )
            h = sol.y[:, -1]
            v_current = v + omega * h

        w = v_current
        w = np.clip(w, -50, 50)

    if eps_reg > 0:
        u_recon = (1.0 + eps_reg) * sigma(w) - eps_reg
        return np.clip(u_recon, 1e-15, 1.0 - 1e-15)
    return sigma(w)


def test1_simpson_picard():
    print("=" * 100)
    print("TEST 1 — does 2-iteration Picard give cubic balance?")
    print("=" * 100)
    Nx = 128
    L = 4.0
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    T = 0.5
    D = 0.1

    u0 = 0.5 + 0.2 * np.cos(2 * np.pi * x / L)

    for r in [0.5, 2.0, 5.0]:
        ref = solve_ref(fkpp_rhs_u, u0, T, k, (D, r))
        rms_ref = np.sqrt(np.mean(ref ** 2))

        print(f"\n  r={r}")
        print(f"  {'K':>4s}  {'L2 (1 Picard)':>14s}  {'L2 (2 Picard)':>14s}  "
              f"{'slope_1':>8s}  {'slope_2':>8s}")
        prev_1 = None
        prev_2 = None
        for K in [2, 4, 8, 16, 32, 64]:
            u1 = fkpp_logit_cascade(u0, D, r, T, K, 1.0, k, n_picard=1)
            u2 = fkpp_logit_cascade(u0, D, r, T, K, 1.0, k, n_picard=2)
            e1 = np.sqrt(np.mean((u1 - ref) ** 2)) / rms_ref
            e2 = np.sqrt(np.mean((u2 - ref) ** 2)) / rms_ref
            s1 = f"{np.log2(prev_1/e1):>8.2f}" if prev_1 else "  -   "
            s2 = f"{np.log2(prev_2/e2):>8.2f}" if prev_2 else "  -   "
            print(f"  {K:>4d}  {e1:>14.3e}  {e2:>14.3e}  {s1}  {s2}")
            prev_1 = e1
            prev_2 = e2
        print("  (slope = log2 of error-reduction per K-doubling; 1=linear, 2=cubic)")


# ============================================================================
# TEST 2 — Allen-Cahn in arctanh space
# ============================================================================


def arctanh_gauge(u, eps=0.0):
    if eps > 0:
        u = np.clip(u, -1.0 + eps, 1.0 - eps)
    return np.arctanh(u)


def ac_arctanh_cascade(u0, eps2, T, K, omega, k, n_picard=2, eps_reg=1e-4):
    """Allen-Cahn cascade in w = arctanh(u) coordinates."""
    w = arctanh_gauge(u0, eps_reg)
    dt = T / K
    Nx = len(w)

    for _ in range(K):
        # Pick scalar w_0^* as spatial mean of current w
        w_0 = float(np.mean(w))
        # Linear substep: dv/dt = eps^2 v_xx + sech^2(w_0) v + const
        # In Fourier, per mode: dv_n/dt = (-eps^2 k_n^2 + sech^2(w_0)) v_n
        # Constant source (tanh(w_0) - sech^2(w_0) w_0) goes to DC mode.
        alpha = 1.0 / np.cosh(w_0) ** 2
        const_source = np.tanh(w_0) - alpha * w_0
        sigma_k = -eps2 * (k ** 2) + alpha
        w_hat = np.fft.fft(w)
        # Exact linear ODE: v_hat_k(dt) = w_hat_k e^{sigma_k dt}
        #                                + (const_source*Nx if k=0) * (e^{sigma_k dt}-1)/sigma_k
        v_hat = w_hat * np.exp(sigma_k * dt)
        # Constant source contribution (only DC mode): int e^{alpha (dt-tau)} * const dtau
        if abs(alpha) > 1e-12:
            v_hat[0] += const_source * Nx * (np.exp(alpha * dt) - 1) / alpha
        else:
            v_hat[0] += const_source * Nx * dt
        v = np.real(np.fft.ifft(v_hat))

        # Picard iterations
        v_current = v.copy()
        for _pic in range(n_picard):
            if abs(omega) < 1e-10:
                break

            def linear_with_resid(t, h, k=k, eps2=eps2, alpha=alpha):
                alpha_t = t / dt
                w_at_t = (1 - alpha_t) * w + alpha_t * v_current
                w_x = spectral_diff1(w_at_t, k)
                # Full nonlinearity N(w) = -2 eps^2 tanh(w) w_x^2 + [tanh(w) - tanh(w_0) - sech^2(w_0)(w - w_0)]
                # The second bracket is the higher-order part of the tanh linearization
                quasilinear = -2.0 * eps2 * np.tanh(w_at_t) * (w_x ** 2)
                src_residual = (
                    np.tanh(w_at_t) - np.tanh(w_0) - alpha * (w_at_t - w_0)
                )
                rho = quasilinear + src_residual
                return eps2 * spectral_diff2(h, k) + alpha * h + rho

            sol = solve_ivp(
                linear_with_resid, [0, dt], np.zeros_like(w),
                method="RK45", rtol=1e-9, atol=1e-11, t_eval=[dt],
            )
            h = sol.y[:, -1]
            v_current = v + omega * h

        w = v_current
        w = np.clip(w, -20, 20)

    if eps_reg > 0:
        return np.clip(np.tanh(w), -1.0 + eps_reg, 1.0 - eps_reg)
    return np.tanh(w)


def ac_u_cascade(u0, eps2, T, K, omega, k):
    """Allen-Cahn u-space cascade for comparison."""
    u = u0.copy()
    dt = T / K
    for _ in range(K):
        u_star = float(np.mean(u))
        alpha = 1 - 3 * u_star ** 2
        beta = u_star - u_star ** 3 - alpha * u_star

        def lin_rhs(t, v, k=k, eps2=eps2, alpha=alpha, beta=beta):
            return eps2 * spectral_diff2(v, k) + alpha * v + beta

        sol = solve_ivp(lin_rhs, [0, dt], u, method="RK45",
                        rtol=1e-9, atol=1e-11, dense_output=True, t_eval=[dt])
        v_tW = sol.y[:, -1]
        if abs(omega) > 1e-10:
            vd = sol.sol

            def picard_rhs(t, h, alpha=alpha, u_star=u_star, vd=vd):
                v_t = vd(t)
                rho = -(v_t - u_star) ** 2 * (v_t + 2 * u_star)
                return eps2 * spectral_diff2(h, k) + alpha * h + rho

            sol2 = solve_ivp(picard_rhs, [0, dt], np.zeros_like(u),
                             method="RK45", rtol=1e-9, atol=1e-11, t_eval=[dt])
            u = v_tW + omega * sol2.y[:, -1]
        else:
            u = v_tW
    return u


def test2_allen_cahn():
    print("\n" + "=" * 100)
    print("TEST 2 — Allen-Cahn via arctanh gauge")
    print("=" * 100)
    Nx = 128
    L = 4.0
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    T = 0.3
    eps2 = 0.05

    # AC with u in (-0.8, 0.8), staying inside the bistable well attractors
    u0 = 0.5 * np.cos(2 * np.pi * x / L) + 0.2 * np.sin(4 * np.pi * x / L)

    ref = solve_ref(ac_rhs_u, u0, T, k, (eps2,))
    rms_ref = np.sqrt(np.mean(ref ** 2))

    print(f"  IC range: [{u0.min():.3f}, {u0.max():.3f}]")
    print(f"  Ref range: [{ref.min():.3f}, {ref.max():.3f}]")
    print(f"\n  {'K':>4s}  {'L2 arctanh(2P)':>15s}  {'L2 u-space(2P)':>15s}  {'Ratio':>6s}")
    prev_at = None
    for K in [2, 4, 8, 16, 32, 64]:
        u_at = ac_arctanh_cascade(u0, eps2, T, K, 1.0, k, n_picard=2)
        u_us = ac_u_cascade(u0, eps2, T, K, 1.0, k)
        e_at = np.sqrt(np.mean((u_at - ref) ** 2)) / rms_ref
        e_us = np.sqrt(np.mean((u_us - ref) ** 2)) / rms_ref
        slope = f"{np.log2(prev_at/e_at):>6.2f}" if prev_at else "  -   "
        print(f"  {K:>4d}  {e_at:>15.3e}  {e_us:>15.3e}  {e_us/max(e_at,1e-30):>6.1f}x  (slope {slope})")
        prev_at = e_at


# ============================================================================
# TEST 3 — Regularized logit for Fisher front propagation
# ============================================================================

def test3_regularized_front():
    print("\n" + "=" * 100)
    print("TEST 3 — Regularized logit for Fisher bump")
    print("=" * 100)
    Nx = 256
    L = 20.0
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    T = 1.0
    D = 0.1
    r = 1.0

    # Gaussian bump on zero-ish background; u in roughly [0, 0.9]
    u0 = 0.9 * np.exp(-x ** 2 / (2 * 1.5 ** 2))

    ref = solve_ref(fkpp_rhs_u, u0, T, k, (D, r))
    rms_ref = np.sqrt(np.mean(ref ** 2))

    print(f"  IC range: [{u0.min():.3e}, {u0.max():.3f}]")
    print(f"  Ref range: [{ref.min():.3e}, {ref.max():.3f}]")

    for eps_reg in [1e-3, 1e-4, 1e-5, 1e-6]:
        print(f"\n  eps_reg = {eps_reg}")
        print(f"  {'K':>4s}  {'L2 logit (2P)':>14s}  {'slope':>8s}")
        prev = None
        for K in [4, 8, 16, 32, 64]:
            u_est = fkpp_logit_cascade(u0, D, r, T, K, 1.0, k, n_picard=2, eps_reg=eps_reg)
            if np.any(np.isnan(u_est)):
                print(f"  {K:>4d}       NaN")
                continue
            e = np.sqrt(np.mean((u_est - ref) ** 2)) / rms_ref
            slope = f"{np.log2(prev/e):>8.2f}" if prev else "  -   "
            print(f"  {K:>4d}  {e:>14.3e}  {slope}")
            prev = e


if __name__ == "__main__":
    test1_simpson_picard()
    test2_allen_cahn()
    test3_regularized_front()
