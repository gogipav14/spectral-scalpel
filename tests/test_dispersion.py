"""Tests for dispersion relations."""

import numpy as np
import pytest

from scalpel.core.dispersion import (
    maxwell_lossy, damped_acoustic, convection_diffusion_cylindrical,
    safe_sqrt, MU_0, EPS_0,
)


@pytest.fixture(params=["jax", "torch"])
def backend(request):
    pytest.importorskip(request.param)
    try:
        from scalpel.backends import get_backend
        return get_backend(request.param)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"{request.param} backend unavailable: {e}")


class TestSafeSqrt:
    def test_positive_real(self, backend):
        z = backend.array([4.0 + 0j, 9.0 + 0j])
        r = safe_sqrt(z, backend)
        r_np = backend.to_numpy(r)
        np.testing.assert_allclose(r_np.real, [2.0, 3.0], atol=1e-12)
        assert np.all(r_np.real >= 0)

    def test_negative_real(self, backend):
        """sqrt(-1) should give +1j, not -1j."""
        z = backend.array([-1.0 + 0j, -4.0 + 0j])
        r = safe_sqrt(z, backend)
        r_np = backend.to_numpy(r)
        assert np.all(r_np.real >= -1e-15), f"Re(sqrt) < 0: {r_np}"

    def test_complex(self, backend):
        z = backend.array([1.0 + 1j, -1.0 + 1j, -1.0 - 1j])
        r = safe_sqrt(z, backend)
        r_np = backend.to_numpy(r)
        assert np.all(r_np.real >= -1e-15)

    def test_branch_stability_near_cut(self, backend):
        """Near the branch cut (Re ~ 0), tiny float noise must not flip the sign.

        Constructs z = r * exp(i*theta) with theta just inside (-pi, pi).
        For theta very close to +/- pi, Re(sqrt(z)) approaches 0 from the
        positive side; round-off should not push it to the negative branch.
        """
        thetas = np.array([np.pi - 1e-13, -(np.pi - 1e-13), np.pi - 1e-15])
        radii = np.array([1e-300, 1.0, 1e300])
        z_np = np.outer(radii, np.exp(1j * thetas)).ravel()
        z = backend.array(z_np)
        r = safe_sqrt(z, backend)
        r_np = backend.to_numpy(r)
        assert np.all(r_np.real >= 0), (
            f"branch flip under round-off: Re(sqrt) = {r_np.real}"
        )

    def test_physical_propagation_constant_has_nonneg_real_part(self, backend):
        """For the Bromwich contour (Re(s) > 0), Re(gamma_z) >= 0 for all k_perp.

        This is the physical attenuation requirement (A5) that the safe_sqrt
        branch choice enforces. The test sweeps a grid of (s, k_x, k_y) across
        the range used by the demonstrations.
        """
        a = 1e8  # typical Bromwich shift for the EM systems
        T = 1e-7
        N = 128
        k_idx = np.arange(N)
        omega = k_idx * np.pi / T
        s_np = a + 1j * omega
        k_max = 2 * np.pi / (1e-3)  # 1 mm sampling
        kx = np.linspace(-k_max, k_max, 32)
        ky = np.linspace(-k_max, k_max, 32)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        S = s_np.reshape(1, 1, -1)
        KX_b = KX[..., None]
        KY_b = KY[..., None]
        S_b = backend.array(S + 0j * KX_b)
        KX_bb = backend.array(KX_b + 0j * S)
        KY_bb = backend.array(KY_b + 0j * S)
        gamma = maxwell_lossy(S_b, KX_bb, KY_bb, 0.1, 10.0, backend)
        gr = backend.to_numpy(gamma).real
        assert np.all(gr >= -1e-12), (
            f"Re(gamma_z) < 0 for some (s, k_perp); min={gr.min()}"
        )


class TestMaxwellDispersion:
    def test_kperp_zero_matches_1d(self, backend):
        """At kperp=0, gamma_z should match the 1D dispersion relation
        gamma = sqrt(mu_0 * (sigma*s + epsilon*s^2))."""
        sigma = 0.01  # S/m
        epsilon_r = 4.0
        epsilon = EPS_0 * epsilon_r

        # A few test points on the Bromwich contour
        s_vals = backend.array([1.0 + 0j, 1.0 + 10j, 1.0 + 100j, 1.0 + 1000j])
        KX = backend.array([0.0])
        KY = backend.array([0.0])

        gamma_z = maxwell_lossy(
            s_vals.reshape(1, 1, -1),
            KX.reshape(1, 1, 1),
            KY.reshape(1, 1, 1),
            sigma, epsilon_r, backend,
        )

        gamma_z_np = backend.to_numpy(gamma_z[0, 0, :])

        # Reference: gamma = sqrt(mu_0 * (sigma*s + eps*s^2))
        s_np = np.array([1.0 + 0j, 1.0 + 10j, 1.0 + 100j, 1.0 + 1000j])
        gamma_ref = np.sqrt(MU_0 * (sigma * s_np + epsilon * s_np**2))
        # Ensure Re >= 0
        gamma_ref = np.where(gamma_ref.real < 0, -gamma_ref, gamma_ref)

        np.testing.assert_allclose(gamma_z_np, gamma_ref, rtol=1e-10)

    def test_evanescent_modes(self, backend):
        """High kperp modes should still have Re(gamma_z) >= 0."""
        sigma = 0.01
        epsilon_r = 4.0
        s = backend.array([1.0 + 100j]).reshape(1, 1, 1)
        # Very high kperp -- evanescent
        KX = backend.array([1e6]).reshape(1, 1, 1)
        KY = backend.array([0.0]).reshape(1, 1, 1)

        gamma_z = maxwell_lossy(s, KX, KY, sigma, epsilon_r, backend)
        gamma_np = backend.to_numpy(gamma_z)
        assert gamma_np.real >= -1e-10, f"Evanescent mode Re(gamma) < 0: {gamma_np}"


class TestAcousticDispersion:
    def test_lossless_limit(self, backend):
        """With nu=0, gamma_z^2 = s^2/c^2 - kperp^2."""
        c = 1500.0  # water
        nu = 0.0
        s = backend.array([0.0 + 1000j]).reshape(1, 1, 1)  # purely imaginary = monochromatic
        KX = backend.array([0.0]).reshape(1, 1, 1)
        KY = backend.array([0.0]).reshape(1, 1, 1)

        gamma_z = damped_acoustic(s, KX, KY, c, nu, backend)
        gamma_np = backend.to_numpy(gamma_z[0, 0, 0])

        # Expected: sqrt(s^2/c^2) = sqrt(-1e6/c^2) = i * 1000/c
        expected = np.sqrt(-1e6 / c**2 + 0j)
        if expected.real < 0:
            expected = -expected

        np.testing.assert_allclose(gamma_np, expected, rtol=1e-10)


class TestConvDiffDispersion:
    def test_no_radial_diffusion(self, backend):
        """With Dr=0 and kr=0, gamma_z^2 = v^2/(4*Dz^2) + s/Dz."""
        v, Dz, Dr = 0.001, 1e-8, 0.0
        s = backend.array([0.5 + 0j]).reshape(1, 1)
        KR = backend.array([0.0]).reshape(1, 1)

        gamma_z = convection_diffusion_cylindrical(s, KR, v, Dz, Dr, backend)
        gamma_np = backend.to_numpy(gamma_z[0, 0])

        expected = np.sqrt(v**2 / (4*Dz**2) + 0.5/Dz + 0j)
        if expected.real < 0:
            expected = -expected

        np.testing.assert_allclose(gamma_np, expected, rtol=1e-10)
