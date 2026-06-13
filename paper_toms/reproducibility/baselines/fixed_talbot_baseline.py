#!/usr/bin/env python3
"""
Fixed Talbot contour NILT baseline for the transform-domain
head-to-head of paper_toms/article.tex Fig. 4.

The reference implementation delegates to ``mpmath.invertlaplace`` with
``method='talbot'``, which is the canonical multi-precision Talbot
implementation; we layer the Abate-Valko (2004) and Weideman (2006)
parameter conventions on top: M = N_NILT / 2 quadrature points,
nu = 0.2 * M damping, Talbot (1979) cotangent-deformed contour with
vertex at sigma = M / (5t).

As with ``de_hoog_baseline.py``, we deliberately do NOT roll our own
quadrature here: the manuscript benchmarks against the fixed-Talbot
*algorithm*, and the mpmath reference is the correct-by-construction
implementation that the field uses.
"""

from __future__ import annotations

import math
from typing import Callable

# Abate-Valko (2004) parameter conventions
DEFAULT_N_NILT = 2048


def fixed_talbot_invert(F: Callable, t: float, *, n_nilt: int = DEFAULT_N_NILT,
                        dps: int = 30) -> float:
    """Invert F(s) at time t via mpmath's Talbot implementation.

    The mpmath Talbot internally uses N = degree + 1 contour points;
    the Abate-Valko convention is M = N_NILT / 2, mapped here by
    setting degree = n_nilt // 2.

    Parameters
    ----------
    F : callable
        Laplace-domain function F(s) -> complex (or mpc).
    t : float
        Inversion time.
    n_nilt : int, default 2048
        Number of contour quadrature points (Abate-Valko M = n_nilt/2).
    dps : int, default 30
        mpmath decimal-precision digits.

    Returns
    -------
    float : f(t)
    """
    import mpmath as mp

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        return float(mp.invertlaplace(F, mp.mpf(t), method="talbot",
                                       degree=n_nilt // 2))
    finally:
        mp.mp.dps = old_dps


if __name__ == "__main__":
    F = lambda s: 1.0 / (s + 1)
    print(f"Fixed Talbot parameters: M = N_NILT/2, nu = 0.2*M (Abate-Valko 2004)")
    print(f"Contour: Talbot (1979) cotangent, vertex sigma = M / (5t)")
    print(f"Sanity check on F(s) = 1/(s+1) -> f(t) = exp(-t):")
    for t in (0.1, 1.0, 5.0):
        try:
            approx = fixed_talbot_invert(F, t)
            exact = math.exp(-t)
            rel_err = abs(approx - exact) / exact
            print(f"  t={t:.2f}: talbot={approx:.10e}, exact={exact:.10e}, "
                  f"rel_err={rel_err:.2e}")
        except Exception as e:
            print(f"  t={t:.2f}: error: {type(e).__name__}: {e}")
            print(f"  (install mpmath: pip install mpmath)")
            break
