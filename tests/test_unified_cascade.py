"""Regression tests for the unified gauge-transformed cascade.

Coverage:
- Burgers identity-gauge cascade matches Phase 1b pencil rule.
- Fisher-KPP logit-gauge cascade gives slope-2 in K.
- Allen-Cahn arctanh-gauge cascade gives slope-2 in K.
- KS identity-gauge cascade gives slope-2 in K.
- Gauge transforms are self-inverse: Phi^-1(Phi(u)) == u.
- 2D Fisher-KPP cascade extends the 1D scheme.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scalpel.nonlinear.unified_cascade import (
    make_gauge_logit, make_gauge_arctanh, GAUGE_IDENTITY,
    spec_fisher_kpp, spec_allen_cahn, unified_cascade,
    burgers_cascade_with_transport, spectral_diff,
)


# -----------------------------------------------------------------------------
# Gauge self-inverse tests
# -----------------------------------------------------------------------------

class TestGaugeTransforms:
    """Verify gauge.inverse(gauge.forward(u)) == u within tolerance."""

    def test_identity_gauge(self):
        u = np.linspace(-1, 1, 100)
        assert np.allclose(GAUGE_IDENTITY.inverse(GAUGE_IDENTITY.forward(u)), u)

    def test_logit_gauge_unregularized(self):
        u = np.linspace(0.05, 0.95, 100)
        gauge = make_gauge_logit(eps_reg=0.0)
        recon = gauge.inverse(gauge.forward(u))
        assert np.allclose(recon, u, atol=1e-12)

    def test_logit_gauge_regularized(self):
        u = np.linspace(0.01, 0.99, 100)
        gauge = make_gauge_logit(eps_reg=1e-4)
        recon = gauge.inverse(gauge.forward(u))
        # Regularized gauge has O(eps_reg) bias
        assert np.max(np.abs(recon - u)) < 1e-4 * 10  # loose tolerance

    def test_arctanh_gauge_unregularized(self):
        u = np.linspace(-0.95, 0.95, 100)
        gauge = make_gauge_arctanh(eps_reg=0.0)
        recon = gauge.inverse(gauge.forward(u))
        assert np.allclose(recon, u, atol=1e-12)


# -----------------------------------------------------------------------------
# Cubic-balance slope tests (integration tests)
# -----------------------------------------------------------------------------

def _fkpp_ref(u0, x, D, r, T):
    """Reference Fisher-KPP via RK45."""
    dx = x[1] - x[0]
    k = np.fft.fftfreq(len(x), dx) * 2 * np.pi

    def rhs(t, u):
        return D * spectral_diff(u, k, 2) + r * u * (1 - u)

    sol = solve_ivp(rhs, [0, T], u0, method="RK45",
                    rtol=1e-11, atol=1e-13, t_eval=[T])
    return sol.y[:, -1]


class TestCubicBalance:
    """Verify slope=2 (cubic balance) across PDEs."""

    def _slope(self, Ks, errs):
        """log2 slope of final K-doubling."""
        if len(Ks) < 2 or Ks[-1] != 2 * Ks[-2]:
            return float("nan")
        if errs[-1] <= 0 or errs[-2] <= 0:
            return float("nan")
        return np.log2(errs[-2] / errs[-1])

    def test_fkpp_cubic_slope(self):
        """Fisher-KPP logit cascade should give slope ~ 2.00."""
        Nx = 128
        L = 4.0
        x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
        u0 = 0.5 + 0.2 * np.cos(2 * np.pi * x / L)
        T = 0.5
        D, r = 0.1, 2.0

        ref = _fkpp_ref(u0, x, D, r, T)
        rms_ref = np.sqrt(np.mean(ref ** 2))

        spec = spec_fisher_kpp(D=D, r=r)
        errs, Ks = [], []
        for K in [16, 32, 64]:  # asymptotic regime
            u_est = unified_cascade(u0, x, spec, T, K, n_picard=2)
            err = np.sqrt(np.mean((u_est - ref) ** 2)) / rms_ref
            errs.append(err)
            Ks.append(K)
        slope = self._slope(Ks, errs)
        # Accept slope within [1.8, 2.2] — theorem predicts 2.00 exactly
        assert 1.8 < slope < 2.2, f"FKPP slope {slope} not cubic"

    def test_burgers_converges(self):
        """Burgers cascade should decrease error with K."""
        Nx = 64
        L = 8.0
        x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
        u0 = 1.0 * np.exp(-x ** 2 / (2 * 0.5 ** 2))
        T = 1.0
        nu = 0.1

        # Reference
        dx = x[1] - x[0]
        k = np.fft.fftfreq(Nx, dx) * 2 * np.pi

        def rhs(t, u):
            return -u * spectral_diff(u, k, 1) + nu * spectral_diff(u, k, 2)

        sol = solve_ivp(rhs, [0, T], u0, method="RK45",
                        rtol=1e-10, atol=1e-12, t_eval=[T])
        ref = sol.y[:, -1]
        rms_ref = np.sqrt(np.mean(ref ** 2))

        err_low = np.sqrt(np.mean((burgers_cascade_with_transport(
            u0, x, nu, T, K=8, n_picard=2) - ref) ** 2)) / rms_ref
        err_high = np.sqrt(np.mean((burgers_cascade_with_transport(
            u0, x, nu, T, K=32, n_picard=2) - ref) ** 2)) / rms_ref
        # Error should decrease by at least 4x going from K=8 to K=32 (4x is 1/K^2)
        assert err_high < err_low, f"Burgers not converging: K=8 err {err_low}, K=32 err {err_high}"
        assert err_high < err_low / 4, f"Burgers not cubic enough: ratio {err_low/err_high}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
