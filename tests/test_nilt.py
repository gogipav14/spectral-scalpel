"""Tests for core NILT against analytical solutions and nilt-cfl reference."""

import numpy as np
import pytest
import cmath

from scalpel.core.nilt import nilt_scalar, nilt_inverse, bromwich_contour, eps_im


# --- Benchmark problems (from nilt-cfl/problems.py) ---

def lag_F(s):
    """F(s) = 1/(s+1), f(t) = exp(-t)."""
    return 1.0 / (s + 1.0)

def lag_f_ref(t):
    return np.exp(-t)

def diffusion_F(s, D=1.0, x=1.0):
    """F(s) = exp(-x*sqrt(s/D))/s, f(t) = erfc(x/(2*sqrt(D*t)))."""
    if abs(s) < 1e-20:
        return complex(np.inf, 0)
    return cmath.exp(-x * cmath.sqrt(s / D)) / s

def second_order_F(s, omega_n=1.0, zeta=0.5):
    """Second-order underdamped impulse response."""
    return omega_n**2 / (s**2 + 2*zeta*omega_n*s + omega_n**2)


class TestNILTScalar:
    """Test scalar NILT (numpy) against analytical solutions."""

    def test_first_order_lag(self):
        # alpha_c=-1, keep a low to minimize exponential amplification
        a, T, N = 0.01, 5.0, 1024
        f, t, z_ifft = nilt_scalar(lag_F, a, T, N)

        mask = (t > 0.1) & (t <= 5.0)
        f_ref = lag_f_ref(t[mask])
        rmse = np.sqrt(np.mean((f[mask] - f_ref)**2))
        rel_error = rmse / np.sqrt(np.mean(f_ref**2))

        assert rel_error < 1e-2, f"Lag RMSE relative error {rel_error:.2e}"
        assert eps_im(z_ifft) < 1e-2, f"eps_im = {eps_im(z_ifft):.2e}"

    def test_second_order_underdamped(self):
        omega_n, zeta = 1.0, 0.5
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        # alpha_c = -0.5
        a, T, N = 0.01, 20.0, 2048

        def F(s):
            return second_order_F(s, omega_n, zeta)
        f, t, z_ifft = nilt_scalar(F, a, T, N)

        mask = (t > 0.1) & (t <= 20.0)
        f_ref = (omega_n / np.sqrt(1-zeta**2)) * np.exp(-zeta*omega_n*t[mask]) * np.sin(omega_d*t[mask])
        rmse = np.sqrt(np.mean((f[mask] - f_ref)**2))
        rel_error = rmse / np.sqrt(np.mean(f_ref**2))

        assert rel_error < 1e-3, f"Second-order relative error {rel_error:.2e}"

    def test_diffusion(self):
        from scipy.special import erfc
        a, T, N = 0.5, 10.0, 1024

        f, t, z_ifft = nilt_scalar(diffusion_F, a, T, N)

        mask = (t > 0.5) & (t <= 10.0)
        f_ref = erfc(1.0 / (2 * np.sqrt(t[mask])))
        rmse = np.sqrt(np.mean((f[mask] - f_ref)**2))
        rel_error = rmse / np.sqrt(np.mean(f_ref**2))

        assert rel_error < 1e-3, f"Diffusion relative error {rel_error:.2e}"


class TestNILTBatched:
    """Test batched NILT on both backends against scalar reference."""

    @pytest.fixture(params=["jax", "torch"])
    def backend(self, request):
        pytest.importorskip(request.param)
        try:
            from scalpel.backends import get_backend
            return get_backend(request.param)
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"{request.param} backend unavailable: {e}")

    def test_single_mode_matches_scalar(self, backend):
        """Batched NILT with shape (1, 1, N) should match scalar NILT."""
        a, T, N = 0.01, 5.0, 1024

        # Scalar reference
        f_scalar, t_scalar, _ = nilt_scalar(lag_F, a, T, N)

        # Batched
        s_contour, t_batched = bromwich_contour(a, T, N, backend)
        F_vals = 1.0 / (s_contour + 1.0)
        F_spectrum = F_vals.reshape(1, 1, -1)

        f_batched, t_out = nilt_inverse(F_spectrum, a, T, backend)

        f_b = backend.to_numpy(f_batched[0, 0, :])

        mask = (t_scalar > 0.1) & (t_scalar <= 5.0)
        rmse = np.sqrt(np.mean((f_b[mask] - f_scalar[mask])**2))
        max_scalar = np.max(np.abs(f_scalar[mask]))
        assert rmse / max_scalar < 1e-8, f"Batched vs scalar mismatch: {rmse/max_scalar:.2e}"

    def test_multiple_modes(self, backend):
        """Batched NILT with multiple modes (simulating different kperp)."""
        # Low a for minimal exponential amplification
        a, T, N = 0.01, 5.0, 1024
        Nx, Ny = 4, 4

        s_contour, _ = bromwich_contour(a, T, N, backend)

        alphas = backend.linspace(0.5, 2.0, Nx * Ny).reshape(Nx, Ny, 1)
        F_spectrum = 1.0 / (s_contour.reshape(1, 1, -1) + alphas)

        f_batched, t_out = nilt_inverse(F_spectrum, a, T, backend)

        t_np = backend.to_numpy(t_out)
        f_np = backend.to_numpy(f_batched)
        alphas_np = backend.to_numpy(alphas[:, :, 0])

        mask = (t_np > 0.1) & (t_np <= 5.0)
        for i in range(Nx):
            for j in range(Ny):
                f_ref = np.exp(-alphas_np[i, j] * t_np[mask])
                f_mode = f_np[i, j, mask]
                rel_err = np.sqrt(np.mean((f_mode - f_ref)**2)) / np.sqrt(np.mean(f_ref**2))
                assert rel_err < 1e-2, f"Mode ({i},{j}) alpha={alphas_np[i,j]:.2f}: {rel_err:.2e}"

    def test_backend_parity(self):
        """JAX and PyTorch produce the same results."""
        pytest.importorskip("jax")
        pytest.importorskip("torch")
        from scalpel.backends import get_backend

        a, T, N = 0.01, 5.0, 256
        backends = [get_backend("jax"), get_backend("torch")]
        results = []

        for b in backends:
            s_contour, _ = bromwich_contour(a, T, N, b)
            F_spectrum = (1.0 / (s_contour + 1.0)).reshape(1, 1, -1)
            f, t = nilt_inverse(F_spectrum, a, T, b)
            results.append(b.to_numpy(f[0, 0, :]))

        np.testing.assert_allclose(results[0], results[1], atol=1e-6,
                                   err_msg="JAX vs PyTorch NILT mismatch")
