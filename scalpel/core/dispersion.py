"""
Dispersion relation library for spectral factorization.

Each dispersion relation computes gamma_z(s, kx, ky) — the propagation
constant in the z-direction for a given Laplace variable s and transverse
wavenumbers. All functions operate on broadcastable arrays for batched
GPU evaluation.

Branch cut convention: Re(gamma_z) >= 0 for physical attenuation.
"""

from __future__ import annotations

import math

# Physical constants
MU_0 = 4e-7 * math.pi       # Vacuum permeability [H/m]
EPS_0 = 8.854187817e-12      # Vacuum permittivity [F/m]
C_LIGHT = 299792458.0        # Speed of light [m/s]


_BRANCH_REL_TOL = 1e-14  # fraction of |r| below which Re(r) is treated as +0


def safe_sqrt(z, backend):
    """Complex square root enforcing Re(result) >= 0.

    This is the branch cut handler. For the propagation constant, we need the
    principal branch with non-negative real part to ensure physical attenuation
    (exp(-gamma_z * d) decays for d > 0).

    The sign flip uses a small relative tolerance to suppress spurious flips
    from float round-off near the branch cut, where Re(sqrt(z)) can legitimately
    reach zero. Specifically, we treat Re(r) as non-negative whenever
    Re(r) > -_BRANCH_REL_TOL * |r|. Since the principal complex square root
    returned by NumPy / JAX / PyTorch already has Re >= 0, this guard is
    defensive; the test suite verifies Re(gamma_z) >= 0 across the dispersion
    relations used in the demonstrations.
    """
    r = backend.sqrt(z)
    abs_r = backend.abs(r)
    # Strict inequality keeps purely-imaginary r (Re == 0) in the +1 branch.
    sign = 1.0 - 2.0 * (backend.real(r) < -_BRANCH_REL_TOL * abs_r)
    return r * sign


def maxwell_lossy(s, KX, KY, sigma, epsilon_r, backend):
    """Dispersion relation for lossy Maxwell's equations.

    gamma_z^2 = mu_0 * (sigma * s + epsilon * s^2) - kx^2 - ky^2

    Parameters
    ----------
    s : array, shape (..., N_brom)
        Laplace variable on Bromwich contour.
    KX, KY : array, shape (Nx, Ny, ...)
        Transverse wavenumber grids.
    sigma : float
        Conductivity [S/m].
    epsilon_r : float
        Relative permittivity.
    backend : Backend
        Compute backend.

    Returns
    -------
    gamma_z : array, shape (Nx, Ny, N_brom)
        Propagation constant.
    """
    epsilon = EPS_0 * epsilon_r
    gamma_sq = MU_0 * (sigma * s + epsilon * s**2) - KX**2 - KY**2
    return safe_sqrt(gamma_sq, backend)


def damped_acoustic(s, KX, KY, c, nu, backend):
    """Dispersion relation for damped acoustic waves.

    gamma_z^2 = (nu/c^2)*s + (1/c^2)*s^2 - kx^2 - ky^2

    Parameters
    ----------
    s : array
        Laplace variable.
    KX, KY : array
        Transverse wavenumber grids.
    c : float
        Sound speed [m/s].
    nu : float
        Kinematic viscosity (damping) [m^2/s].
    backend : Backend
        Compute backend.

    Returns
    -------
    gamma_z : array
        Propagation constant.
    """
    gamma_sq = (nu / c**2) * s + (1.0 / c**2) * s**2 - KX**2 - KY**2
    return safe_sqrt(gamma_sq, backend)


def convection_diffusion_cylindrical(s, KR, v, Dz, Dr, backend):
    """Dispersion relation for convection-diffusion in cylindrical geometry.

    gamma_z^2 = v^2/(4*Dz^2) + s/Dz + Dr*kr^2/Dz

    This comes from the axial ODE after Hankel transform in r and
    Laplace transform in t, with the convective term absorbed via
    the substitution C = C_hat * exp(v*z/(2*Dz)).

    Parameters
    ----------
    s : array
        Laplace variable.
    KR : array
        Radial wavenumber (Bessel eigenvalues).
    v : float
        Axial velocity [m/s].
    Dz : float
        Axial dispersion coefficient [m^2/s].
    Dr : float
        Radial diffusion coefficient [m^2/s].
    backend : Backend
        Compute backend.

    Returns
    -------
    gamma_z : array
        Propagation constant.
    """
    gamma_sq = v**2 / (4 * Dz**2) + s / Dz + Dr * KR**2 / Dz
    return safe_sqrt(gamma_sq, backend)


def elastic_pwave(s, KX, KY, c_p, rho, eta_p=0.0, eta_s=0.0, backend=None):
    """P-wave potential dispersion from the Helmholtz decomposition of Navier.

    The scalar potential phi satisfies
        ∂²phi/∂t² = c_p²(s) ∇² phi
    with Kelvin-Voigt effective speed
        c_p²(s) = (lambda + 2 mu + (eta_p + 2 eta_s) s) / rho
                = c_p² + (eta_p + 2 eta_s) s / rho

    After 2D FFT in (x, y) and Laplace in t:
        gamma_p² = s² / c_p²(s) + kx² + ky²

    The elastic limit (eta_p = eta_s = 0) recovers gamma_p² = s²/c_p² + k_perp²,
    which has the same structural form as the damped acoustic relation at zero
    viscosity. The attenuation channel mirrors the sigma/epsilon crossover of
    lossy Maxwell: at omega >> omega_cross = M/(eta_p+2*eta_s) the wave is
    dispersive-elastic, below omega_cross it is viscously damped.
    """
    c_eff_sq = c_p ** 2 + (eta_p + 2.0 * eta_s) * s / rho
    gamma_sq = s ** 2 / c_eff_sq + KX ** 2 + KY ** 2
    return safe_sqrt(gamma_sq, backend)


def elastic_swave(s, KX, KY, c_s, rho, eta_s=0.0, backend=None):
    """S-wave potential dispersion from the Helmholtz decomposition of Navier.

    Each component of the vector potential psi (subject to ∇·psi = 0) satisfies
        ∂²psi/∂t² = c_s²(s) ∇² psi
    with Kelvin-Voigt effective shear speed
        c_s²(s) = (mu + eta_s s) / rho = c_s² + eta_s s / rho.

    After 2D FFT in (x, y) and Laplace in t:
        gamma_s² = s² / c_s²(s) + kx² + ky²

    The gauge constraint ∇·psi = 0 is preserved automatically under the 2D FFT:
    in spectral space it becomes i(kx psi_x + ky psi_y) + ∂_z psi_z = 0, which
    is algebraic and invertible pointwise at each (s, k_perp).
    """
    c_eff_sq = c_s ** 2 + eta_s * s / rho
    gamma_sq = s ** 2 / c_eff_sq + KX ** 2 + KY ** 2
    return safe_sqrt(gamma_sq, backend)


def diffusion(s, KX, KY, D, backend):
    """Dispersion relation for the heat/diffusion equation.

    gamma_z^2 = s/D + kx^2 + ky^2

    This is the isotropic diffusion equation ∂u/∂t = D∇²u after
    2D FFT in (x,y) and Laplace transform in t. Unlike the wave-type
    equations (Maxwell, acoustic), here kperp^2 ADDS to gamma_z
    (faster transverse variation requires more z-propagation),
    rather than subtracting (cutoff/evanescent modes).

    The per-mode analytical inverse is the shifted Lévy distribution:
        h(t) = (d/√D) / (2√(πt³)) · exp(-d²/(4Dt) - D·kperp²·t)

    Parameters
    ----------
    s : array
        Laplace variable.
    KX, KY : array
        Transverse wavenumber grids.
    D : float
        Diffusion coefficient [m²/s].
    backend : Backend
        Compute backend.

    Returns
    -------
    gamma_z : array
        Propagation constant.
    """
    gamma_sq = s / D + KX**2 + KY**2
    return safe_sqrt(gamma_sq, backend)


def subdiffusion_caputo(s, KX, KY, D, alpha, backend):
    """Dispersion relation for fractional Caputo subdiffusion.

    For ∂_t^α u = D ∇²u with α ∈ (0, 1), the Laplace-domain operator
    s^α replaces s in the integer-order diffusion dispersion:

        gamma_z^2 = s^α / D + kx^2 + ky^2

    α = 1 reduces to integer-order diffusion. α < 1 is anomalous
    subdiffusion (sluggish spread, heavy temporal tails). The
    factorization handles fractional time derivatives natively because
    s^α is just a different elementary function on the Bromwich contour;
    no time-history convolution is needed.

    Parameters
    ----------
    s : array
        Laplace variable on the Bromwich contour (Re(s) > 0, so s^α is
        well-defined on the principal branch).
    KX, KY : array
        Transverse wavenumber grids.
    D : float
        Generalized diffusion coefficient [m²/s^α].
    alpha : float
        Fractional order, 0 < alpha <= 1.
    backend : Backend
        Compute backend.

    Returns
    -------
    gamma_z : array
        Propagation constant with Re(gamma_z) >= 0.
    """
    s_alpha = s ** alpha
    gamma_sq = s_alpha / D + KX**2 + KY**2
    return safe_sqrt(gamma_sq, backend)
