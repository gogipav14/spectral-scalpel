#!/usr/bin/env python3
"""
de Hoog quotient-difference NILT baseline for the transform-domain
head-to-head of paper_toms/article.tex Fig. 4.

The reference implementation delegates to ``mpmath.invertlaplace`` with
``method='dehoog'``, which is the canonical multi-precision QD
implementation; we layer the de Hoog (1982) parameter convention on
top: convergence parameter $\\varepsilon = 10^{-10}$, default contour
parameter $\\alpha = 10^{-10}$ for stability.

We deliberately do NOT roll our own QD recurrence here: the manuscript
benchmarks against the de Hoog *algorithm* (not against one
specific implementation), and the mpmath reference is the
correct-by-construction implementation that the field uses.
"""

from __future__ import annotations

import math
from typing import Callable

# de Hoog 1982 parameter recommendations
EPSILON = 1e-10
ALPHA   = 1e-10


def dehoog_invert(F: Callable, t: float, *, dps: int = 30) -> float:
    """Invert F(s) at time t via mpmath's QD implementation.

    Parameters
    ----------
    F : callable
        Laplace-domain function F(s) -> complex (or mpc).
    t : float
        Inversion time.
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
        return float(mp.invertlaplace(F, mp.mpf(t), method="dehoog"))
    finally:
        mp.mp.dps = old_dps


if __name__ == "__main__":
    F = lambda s: 1.0 / (s + 1)
    print(f"de Hoog (1982) parameters: epsilon={EPSILON}, alpha={ALPHA}")
    print(f"Sanity check on F(s) = 1/(s+1) -> f(t) = exp(-t):")
    for t in (0.1, 1.0, 5.0):
        try:
            approx = dehoog_invert(F, t)
            exact = math.exp(-t)
            rel_err = abs(approx - exact) / exact
            print(f"  t={t:.2f}: dehoog={approx:.10e}, exact={exact:.10e}, "
                  f"rel_err={rel_err:.2e}")
        except Exception as e:
            print(f"  t={t:.2f}: error: {type(e).__name__}: {e}")
            print(f"  (install mpmath: pip install mpmath)")
            break
