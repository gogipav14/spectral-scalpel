"""Universal crossover constants: delta/lambda = 1/Lambda_- across systems.

Verifies the claim (Section 5, main text) that all hyperbolic two-term-pencil
systems (Maxwell lossy, damped acoustics) produce the same skin-depth-to-
wavelength ratio at their own crossover frequency omega_x = alpha/beta:

    delta / lambda |_{omega_x} = tan(3*pi/8) / (2*pi) = (1 + sqrt(2)) / (2*pi)
                                                      ~= 0.384199...

One-line derivation:
    gamma^2(s) = alpha*s + beta*s^2.
    At s = i*omega_x with omega_x = alpha/beta,
        gamma^2 = alpha*(i*omega_x) + beta*(i*omega_x)^2
               = i*alpha^2/beta - alpha^2/beta
               = (alpha^2/beta)*(i - 1).
    So gamma(i*omega_x) = (alpha/sqrt(beta)) * sqrt(i - 1)
                       = (alpha/sqrt(beta)) * 2^{1/4} * exp(i*3*pi/8).
    Hence Re(gamma) = (alpha/sqrt(beta)) * 2^{1/4} * cos(3*pi/8)
          Im(gamma) = (alpha/sqrt(beta)) * 2^{1/4} * sin(3*pi/8),
    and
        delta/lambda = |Im(gamma)| / (2*pi*Re(gamma))
                     = tan(3*pi/8) / (2*pi),
    which cancels (alpha, sqrt(beta)) entirely, proving material-independence.
"""
from __future__ import annotations

import numpy as np
import pytest

from scalpel.backends import get_backend
from scalpel.core.dispersion import maxwell_lossy, damped_acoustic, MU_0, EPS_0


LAMBDA_MINUS = 2 * np.pi * (np.sqrt(2.0) - 1.0)           # ~= 2.6034
DELTA_OVER_LAMBDA_THEORY = 1.0 / LAMBDA_MINUS             # ~= 0.384199


def _delta_over_lambda_at(gamma_cplx: complex) -> float:
    re = float(np.real(gamma_cplx))
    im = float(abs(np.imag(gamma_cplx)))
    delta = 1.0 / re
    wavelength = 2.0 * np.pi / im
    return delta / wavelength


@pytest.fixture
def backend():
    return get_backend()


@pytest.mark.parametrize(
    "name, alpha, beta, eval_gamma",
    [
        # Maxwell dry sand: sigma=1e-4 S/m, epsilon_r=4
        (
            "maxwell_dry_sand",
            MU_0 * 1e-4,
            MU_0 * (4.0 * EPS_0),
            lambda s, b: maxwell_lossy(
                s, np.zeros_like(s), np.zeros_like(s), 1e-4, 4.0, b
            ),
        ),
        # Maxwell wet clay: sigma=0.1 S/m, epsilon_r=10
        (
            "maxwell_wet_clay",
            MU_0 * 0.1,
            MU_0 * (10.0 * EPS_0),
            lambda s, b: maxwell_lossy(
                s, np.zeros_like(s), np.zeros_like(s), 0.1, 10.0, b
            ),
        ),
        # Acoustics (soft tissue-ish): c = 1500 m/s, nu = 45 m^2/s
        (
            "acoustic_tissue",
            45.0 / (1500.0 ** 2),
            1.0 / (1500.0 ** 2),
            lambda s, b: damped_acoustic(
                s, np.zeros_like(s), np.zeros_like(s), 1500.0, 45.0, b
            ),
        ),
    ],
)
def test_delta_over_lambda_matches_theory(name, alpha, beta, eval_gamma, backend):
    """delta/lambda at omega_x should equal (1+sqrt(2))/(2*pi) to >=5 sig figs."""
    omega_x = alpha / beta
    s = backend.array(np.array([1j * omega_x]))

    gamma_arr = eval_gamma(s, backend)
    gamma = complex(backend.to_numpy(gamma_arr).ravel()[0])

    ratio = _delta_over_lambda_at(gamma)

    rel_err = abs(ratio - DELTA_OVER_LAMBDA_THEORY) / DELTA_OVER_LAMBDA_THEORY
    assert rel_err < 1e-5, (
        f"{name}: delta/lambda = {ratio:.8f}, theory = {DELTA_OVER_LAMBDA_THEORY:.8f}, "
        f"relative error = {rel_err:.2e} (target < 1e-5)"
    )


def test_lambda_minus_value():
    """Lambda_- identities: 2*pi*cot(3*pi/8) = 2*pi*(sqrt(2)-1)."""
    form_a = 2 * np.pi * (1.0 / np.tan(3 * np.pi / 8))
    form_b = 2 * np.pi * (np.sqrt(2.0) - 1.0)
    assert abs(form_a - form_b) < 1e-14
    assert abs(form_a - LAMBDA_MINUS) < 1e-14
