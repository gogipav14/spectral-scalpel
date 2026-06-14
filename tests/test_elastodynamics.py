"""
Elastodynamics validation: the P- and S-wave potentials propagate as decoupled
scalar wave equations whose transfer functions are pure delays. Their Laplace
transfer functions are

    H_p(s) = exp(-s d / c_p),   H_s(s) = exp(-s d / c_s)

so convolution with a causal Gaussian pulse g(t) = N(t; t0, pw) produces a
rigid shift: the output peaks at t0 + d/c_p and t0 + d/c_s respectively.

We verify:
  1. Arrival-time accuracy (better than one delta_t).
  2. Speed ratio c_p/c_s reproduced from peak-time ratio.
  3. Viscoelastic (Kelvin-Voigt) attenuation: the arrival amplitude shrinks
     when eta_s > 0 relative to the lossless case.
"""

import numpy as np
import pytest

from scalpel.core.dispersion import elastic_pwave, elastic_swave
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.core.nilt import nilt_scalar
from scalpel.systems.elastodynamics import MATERIALS


def laplace_gaussian(s, pw, t0):
    """Bilateral Laplace of g(t) = (1/(pw*sqrt(2pi))) exp(-(t-t0)^2/(2 pw^2)).

    Uses the normalised form so that peak amplitude in time domain is
    1/(pw*sqrt(2pi)).
    """
    import cmath
    return pw * cmath.exp(-s * t0 + 0.5 * pw ** 2 * s ** 2) * np.sqrt(2 * np.pi) / (pw * np.sqrt(2 * np.pi))


def _peak_parabolic(t, f):
    """Return the sub-sample peak location using three-point parabolic fit."""
    i = int(np.argmax(f))
    if i <= 0 or i >= len(f) - 1:
        return t[i]
    y0, y1, y2 = f[i - 1], f[i], f[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0.0:
        return t[i]
    delta = 0.5 * (y0 - y2) / denom
    return t[i] + delta * (t[1] - t[0])


def _transfer_dc(mat_name, depth, wave, pw, t0, eta_s=0.0, eta_p=0.0):
    """Return (t, f(t)) from NILT of H(s)*G(s) at k_perp = 0."""
    mat = MATERIALS[mat_name]
    if wave == "p":
        speed = mat.c_p
    else:
        speed = mat.c_s

    t_transit = depth / speed
    t_end = 3 * t_transit + t0
    params = tune_params(t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
                         eps_tail=1e-6, N_init=512, rho=2.0 / pw)

    import cmath

    def F(s):
        if wave == "p":
            c_eff_sq = speed ** 2 + (eta_p + 2.0 * eta_s) * s / mat.rho
        else:
            c_eff_sq = speed ** 2 + eta_s * s / mat.rho
        gamma = cmath.sqrt(s ** 2 / c_eff_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-gamma * depth) * laplace_gaussian(s, pw, t0)

    refined = refine_until_accept(F, params, t_end,
                                  eps_im_max=1e-2, eps_conv=1e-2,
                                  N_max=4096, t_eval_min=t_end * 0.01)
    f, t, _ = nilt_scalar(F, refined.a, refined.T, refined.N)
    return t, f, refined


@pytest.mark.parametrize("mat_name", ["sandstone", "steel", "ice"])
def test_p_arrival_time(mat_name):
    mat = MATERIALS[mat_name]
    depth = 0.05
    t_p = depth / mat.c_p
    pw = 0.15 * t_p
    t0 = 3 * pw
    t, f, refined = _transfer_dc(mat_name, depth, "p", pw, t0)
    dt = (2 * refined.T) / refined.N
    peak_t = t[int(np.argmax(f))]
    expected = t0 + t_p
    assert abs(peak_t - expected) < 2 * dt, (
        f"{mat_name}: P-peak at {peak_t*1e6:.3f} us, expected {expected*1e6:.3f} us, dt={dt*1e6:.3f}")


@pytest.mark.parametrize("mat_name", ["sandstone", "steel", "ice"])
def test_s_arrival_time(mat_name):
    mat = MATERIALS[mat_name]
    depth = 0.05
    t_s = depth / mat.c_s
    pw = 0.15 * (depth / mat.c_p)
    t0 = 3 * pw
    t, f, refined = _transfer_dc(mat_name, depth, "s", pw, t0)
    dt = (2 * refined.T) / refined.N
    peak_t = t[int(np.argmax(f))]
    expected = t0 + t_s
    assert abs(peak_t - expected) < 2 * dt, (
        f"{mat_name}: S-peak at {peak_t*1e6:.3f} us, expected {expected*1e6:.3f} us, dt={dt*1e6:.3f}")


@pytest.mark.parametrize("mat_name", ["sandstone", "steel", "ice"])
def test_speed_ratio_from_peaks(mat_name):
    """Ratio of P/S peak delays after subtracting the source offset must match c_p/c_s."""
    mat = MATERIALS[mat_name]
    depth = 0.05
    pw = 0.15 * (depth / mat.c_p)
    t0 = 3 * pw

    t_P, f_P, _ = _transfer_dc(mat_name, depth, "p", pw, t0)
    t_S, f_S, _ = _transfer_dc(mat_name, depth, "s", pw, t0)

    delay_P = _peak_parabolic(t_P, f_P) - t0
    delay_S = _peak_parabolic(t_S, f_S) - t0
    ratio = delay_S / delay_P

    expected = mat.speed_ratio   # c_p / c_s
    assert abs(ratio - expected) / expected < 5e-3, (
        f"{mat_name}: peak-delay ratio {ratio:.4f}, expected c_p/c_s={expected:.4f}")


def test_kelvin_voigt_attenuation_reduces_amplitude():
    """For a viscoelastic material, |f_peak| must drop below the lossless reference."""
    mat = MATERIALS["rubber"]
    depth = 0.05
    pw = 0.15 * (depth / mat.c_p)
    t0 = 3 * pw

    _, f_lossless, _ = _transfer_dc("rubber", depth, "p", pw, t0, eta_s=0.0, eta_p=0.0)
    _, f_damped,   _ = _transfer_dc("rubber", depth, "p", pw, t0,
                                    eta_s=mat.eta_s, eta_p=mat.eta_p)

    amp_l = float(np.max(np.abs(f_lossless)))
    amp_d = float(np.max(np.abs(f_damped)))
    assert amp_d < amp_l, (
        f"Viscoelastic amplitude {amp_d:.3e} should fall below lossless {amp_l:.3e}")


def test_dispersion_signatures_agree_with_engine_path():
    """The stand-alone dispersion functions must produce the same gamma_z that the
    scalar F(s) above implicitly uses. Guards against regressions where the engine
    and the analytical reference drift apart."""
    import numpy as _np
    from scalpel.backends import get_backend
    b = get_backend()

    mat = MATERIALS["steel"]
    s = b.array(1.0e5 + 0j)
    KX = b.array(0.0 + 0j)
    KY = b.array(0.0 + 0j)

    gp = elastic_pwave(s, KX, KY, mat.c_p, mat.rho, 0.0, 0.0, b)
    gs = elastic_swave(s, KX, KY, mat.c_s, mat.rho, 0.0, b)

    expected_p = 1.0e5 / mat.c_p
    expected_s = 1.0e5 / mat.c_s
    assert _np.isclose(float(b.to_numpy(b.real(gp))), expected_p, rtol=1e-10)
    assert _np.isclose(float(b.to_numpy(b.real(gs))), expected_s, rtol=1e-10)
