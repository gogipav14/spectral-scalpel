"""Tests for the spectral factorization engine."""

import numpy as np
import pytest
import math

from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import maxwell_lossy, MU_0, EPS_0
from scalpel.core.nilt import nilt_scalar


@pytest.fixture(params=["jax", "torch"])
def backend(request):
    from scalpel.backends import get_backend
    return get_backend(request.param)


class TestEngine1D:
    """Test engine with kx=ky=0 (1D case) against scalar NILT."""

    def test_maxwell_1d_point_source(self, backend):
        """A delta source at center with 1x1 grid is effectively 1D.
        Should match scalar NILT of exp(-gamma(s)*d)."""
        sigma = 0.01
        epsilon_r = 4.0
        epsilon = EPS_0 * epsilon_r
        depth = 0.5  # meters

        # Small grid — effectively 1D (only DC mode matters)
        grid = GridParams(Nx=1, Ny=1, dx=1.0, dy=1.0)
        nilt_params = NILTParams(a=1.0, T=5e-8, N=512)

        def dispersion_fn(s, KX, KY, b):
            return maxwell_lossy(s, KX, KY, sigma, epsilon_r, b)

        engine = SpectralEngine(dispersion_fn, backend)
        source = backend.array([[1.0]])  # unit point source
        field, t = engine.forward(source, depth, grid, nilt_params)

        field_np = backend.to_numpy(field[0, 0, :])
        t_np = backend.to_numpy(t)

        # Scalar NILT reference: F(s) = exp(-gamma(s) * d)
        def F_scalar(s):
            import cmath
            gamma_sq = MU_0 * (sigma * s + epsilon * s**2)
            gamma = cmath.sqrt(gamma_sq)
            if gamma.real < 0:
                gamma = -gamma
            return cmath.exp(-gamma * depth)

        f_ref, t_ref, _ = nilt_scalar(F_scalar, nilt_params.a, nilt_params.T, nilt_params.N)

        # Compare on valid time range (skip t=0 and late aliasing)
        mask = (t_ref > 1e-9) & (t_ref <= 5e-8)
        max_val = np.max(np.abs(f_ref[mask]))
        if max_val > 1e-20:
            rel_err = np.max(np.abs(field_np[mask] - f_ref[mask])) / max_val
            assert rel_err < 1e-6, f"Engine 1D vs scalar NILT: {rel_err:.2e}"

    def test_backend_parity_engine(self):
        """JAX and PyTorch engines produce identical fields."""
        from scalpel.backends import get_backend

        sigma = 0.01
        epsilon_r = 4.0
        depth = 0.3
        grid = GridParams(Nx=8, Ny=8, dx=0.1, dy=0.1)
        nilt_params = NILTParams(a=1.0, T=5e-8, N=256)

        results = {}
        for name in ["jax", "torch"]:
            b = get_backend(name)

            def dispersion_fn(s, KX, KY, _b):
                return maxwell_lossy(s, KX, KY, sigma, epsilon_r, _b)

            engine = SpectralEngine(dispersion_fn, b)
            source = b.zeros((8, 8), dtype=float)
            # Can't do in-place assignment generically, create with numpy
            src_np = np.zeros((8, 8))
            src_np[4, 4] = 1.0
            source = b.array(src_np, dtype=complex)

            field, t = engine.forward(source, depth, grid, nilt_params)
            results[name] = b.to_numpy(field)

        np.testing.assert_allclose(
            results["jax"], results["torch"], atol=1e-6, rtol=1e-6,
            err_msg="JAX vs PyTorch engine mismatch"
        )


class TestEnginePhysics:
    """Physics sanity checks."""

    def test_field_decays_with_depth(self, backend):
        """Field amplitude should decrease with propagation depth."""
        sigma = 0.1  # lossy
        epsilon_r = 4.0
        grid = GridParams(Nx=16, Ny=16, dx=0.05, dy=0.05)
        nilt_params = NILTParams(a=2.0, T=1e-7, N=256)

        def dispersion_fn(s, KX, KY, b):
            return maxwell_lossy(s, KX, KY, sigma, epsilon_r, b)

        engine = SpectralEngine(dispersion_fn, backend)

        src_np = np.zeros((16, 16))
        src_np[8, 8] = 1.0
        source = backend.array(src_np, dtype=complex)

        field_near, _ = engine.forward(source, 0.1, grid, nilt_params)
        field_far, _ = engine.forward(source, 1.0, grid, nilt_params)

        energy_near = float(backend.mean(backend.abs(field_near)**2))
        energy_far = float(backend.mean(backend.abs(field_far)**2))

        assert energy_far < energy_near, \
            f"Field should decay: near={energy_near:.4e}, far={energy_far:.4e}"
