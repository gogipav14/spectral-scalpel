"""
CFL-informed feasibility checking for NILT parameters.

Ported from nilt-cfl/tuner.py. These are pure numpy functions that run
on CPU — they determine the NILT parameters before GPU computation begins.

The key constraint (Theorem 1, CES paper):
    alpha_c * t_max + ln(C / eps_tail) < L - delta_s

where L = ln(DBL_MAX) ~ 709.8 for float64.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


L_FLOAT64 = 709.8  # ln(DBL_MAX) for float64


@dataclass
class TunedParams:
    """Container for tuned NILT parameters."""
    a: float           # Bromwich shift
    T: float           # Half-period
    N: int             # FFT size
    delta_t: float     # Time step
    t_max: float       # Maximum time = 2T
    a_min: float       # Lower bound after floor
    a_max: float       # Upper bound from dynamic range
    margin: float      # a_max - a
    feasible: bool     # Whether CFL conditions are satisfied


def tune_params(
    t_end: float,
    alpha_c: float,
    C: float = 1.0,
    kappa: float = 1.0,
    eps_tail: float = 1e-6,
    delta_min: float = 1e-3,
    delta_floor: float = 1e-3,
    delta_s: float = 10.0,
    N_init: int = 512,
    rho: Optional[float] = None,
    gamma: float = 1.5,
    L: float = L_FLOAT64,
) -> TunedParams:
    """Tune NILT parameters using CFL-informed framework (Algorithm 1 Phase 1).

    Parameters
    ----------
    t_end : float
        End time for evaluation.
    alpha_c : float
        Abscissa of convergence.
    C : float
        Tail envelope constant.
    kappa : float
        Period factor T = kappa * t_end.
    eps_tail : float
        Aliasing tolerance.
    delta_min : float
        Singularity margin.
    delta_floor : float
        Minimum positive shift floor.
    delta_s : float
        Dynamic range safety margin.
    N_init : int
        Initial FFT size.
    rho : float, optional
        Spectral radius for frequency heuristic.
    gamma : float
        Oversampling factor.
    L : float
        ln(DBL_MAX) for precision.
    """
    T = kappa * t_end
    t_max = 2 * T

    a_max = (L - delta_s) / t_max

    alias_factor = (2 * kappa - 1) * t_end
    if alias_factor > 0:
        a_alias = alpha_c + np.log(C / eps_tail) / alias_factor
    else:
        a_alias = alpha_c + delta_min

    a_sing = alpha_c + delta_min
    a_min_star = max(a_alias, a_sing)
    a_min = max(a_min_star, delta_floor)

    feasible = a_min <= a_max
    a = a_min if feasible else a_max

    if rho is not None:
        omega_max_init = gamma * rho
        N_from_spectrum = 2 ** int(np.ceil(np.log2(2 * T * omega_max_init / np.pi)))
        N = max(N_init, N_from_spectrum)
    else:
        N = N_init

    N = 2 ** int(np.ceil(np.log2(N)))
    delta_t = 2 * T / N

    return TunedParams(
        a=a, T=T, N=N, delta_t=delta_t, t_max=t_max,
        a_min=a_min, a_max=a_max, margin=a_max - a, feasible=feasible,
    )


def check_feasibility(
    alpha_c: float,
    t_max: float,
    C: float = 1.0,
    eps_tail: float = 1e-6,
    delta_s: float = 10.0,
    L: float = L_FLOAT64,
) -> tuple[bool, float, float]:
    """Check CFL feasibility condition (Theorem 1).

    Returns
    -------
    feasible : bool
    lhs : float
        alpha_c * t_max + ln(C / eps_tail)
    rhs : float
        L - delta_s
    """
    lhs = alpha_c * t_max + np.log(C / eps_tail)
    rhs = L - delta_s
    return lhs <= rhs, float(lhs), float(rhs)


def refine_until_accept(
    F_scalar,
    params: TunedParams,
    t_end: float,
    eps_im_max: float = 1e-2,
    eps_conv: float = 1e-2,
    N_max: int = 16384,
    t_eval_min: float = 0.1,
) -> TunedParams:
    """Algorithm 1 Phase 2: adaptively double N until quality criteria met.

    Uses eps_im (imaginary leakage) and N-doubling convergence on a
    representative scalar transfer function (e.g., the DC mode).

    Parameters
    ----------
    F_scalar : callable
        F(s) -> complex. A single representative transfer function.
    params : TunedParams
        Initial parameters from tune_params().
    t_end : float
        End time.
    eps_im_max : float
        Imaginary leakage threshold.
    eps_conv : float
        N-doubling convergence threshold.
    N_max : int
        Maximum N.
    t_eval_min : float
        Minimum time for evaluation.

    Returns
    -------
    refined : TunedParams
        Parameters with possibly increased N.
    """
    from .nilt import nilt_scalar, eps_im

    a, T = params.a, params.T
    N = params.N

    max_iter = int(np.log2(N_max / N)) + 2
    for _ in range(max_iter):
        _, t_full, z_ifft = nilt_scalar(F_scalar, a, T, N)

        mask = (t_full >= t_eval_min) & (t_full <= t_end)
        current_eps = eps_im(z_ifft[mask]) if np.any(mask) else np.inf
        E_N = n_doubling_error(F_scalar, a, T, N, t_eval_min, t_end)

        if current_eps <= eps_im_max and E_N <= eps_conv:
            break
        N = 2 * N
        if N > N_max:
            N = N_max
            break

    delta_t = 2 * T / N
    return TunedParams(
        a=a, T=T, N=N, delta_t=delta_t, t_max=2*T,
        a_min=params.a_min, a_max=params.a_max,
        margin=params.margin, feasible=params.feasible,
    )


def n_doubling_error(F_scalar, a, T, N, t_eval_min, t_end):
    """Compute N-doubling convergence metric E_N = RMS(f_N - f_2N)/RMS(f_2N)."""
    from .nilt import nilt_scalar

    f_N, t_N, _ = nilt_scalar(F_scalar, a, T, N)
    f_2N, t_2N, _ = nilt_scalar(F_scalar, a, T, 2 * N)

    mask_2N = (t_2N >= t_eval_min) & (t_2N <= t_end)
    t_eval = t_2N[mask_2N]
    f_2N_eval = f_2N[mask_2N]

    f_N_eval = np.interp(t_eval, t_N, f_N)

    rms_diff = np.sqrt(np.mean((f_N_eval - f_2N_eval)**2))
    rms_2N = np.sqrt(np.mean(f_2N_eval**2))

    if rms_2N < 1e-300:
        return np.inf
    return float(rms_diff / rms_2N)


def feasibility_map(
    alpha_c_fn,
    kperp_values: np.ndarray,
    t_end: float,
    kappa: float = 1.0,
    C: float = 1.0,
    eps_tail: float = 1e-6,
    delta_s: float = 10.0,
    L: float = L_FLOAT64,
) -> np.ndarray:
    """Compute feasibility for an array of transverse wavenumbers.

    Parameters
    ----------
    alpha_c_fn : callable
        alpha_c(kperp) -> float. Abscissa of convergence per mode.
    kperp_values : ndarray
        Array of transverse wavenumber magnitudes.
    t_end : float
        End time.

    Returns
    -------
    mask : ndarray of bool
        True where the mode is feasible.
    """
    t_max = 2 * kappa * t_end
    rhs = L - delta_s
    mask = np.array([
        alpha_c_fn(kp) * t_max + np.log(C / eps_tail) <= rhs
        for kp in kperp_values
    ])
    return mask
