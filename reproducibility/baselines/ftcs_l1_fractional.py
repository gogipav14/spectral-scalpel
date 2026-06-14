#!/usr/bin/env python3
"""
FTCS + L1 baseline for the fractional Caputo head-to-head of
article.tex Fig. 3(b).

Parameter choices:
    - L1 quadrature (Lin & Xu 2007 / Sun & Wu 2006) for the Caputo derivative
    - CFL = 0.5 (largest stable for explicit FTCS at alpha = 0.7)
    - Second-order central spatial differences
    - alpha = 0.7 single-term Caputo

This file is a thin wrapper around the canonical L1 implementation in
``reports/claims_audit/benchmark_fractional_heat_3d.py``; the wrapper
exists so the baseline parameter choices are visible in one place.

Usage:
    python ftcs_l1_fractional.py
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Baseline parameters
ALPHA = 0.7
CFL = 0.5
SPATIAL_ORDER = 2
QUADRATURE = "L1 (Lin-Xu 2007 / Sun-Wu 2006)"


def build_l1_solver(n_t: int):
    """Construct a JIT-compiled L1+FTCS scan over n_t fractional time steps.

    The actual numerical kernel lives in
    ``reproducibility/scripts/benchmark_fractional_heat_3d.py:make_l1_scan_fn``.
    """
    HERE = os.path.dirname(os.path.abspath(__file__))
    SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    from benchmark_fractional_heat_3d import make_l1_scan_fn
    return make_l1_scan_fn(n_t)


def show_parameters() -> None:
    print(f"FTCS+L1 baseline parameters:")
    print(f"  alpha (Caputo order)  = {ALPHA}")
    print(f"  CFL                   = {CFL}")
    print(f"  spatial order         = {SPATIAL_ORDER}")
    print(f"  quadrature            = {QUADRATURE}")
    print(f"  canonical kernel      = "
          f"reproducibility/scripts/benchmark_fractional_heat_3d.py:make_l1_scan_fn")


if __name__ == "__main__":
    show_parameters()
    try:
        solver = build_l1_solver(n_t=128)
        print(f"\nsolver built: {solver}")
    except Exception as e:
        print(f"\nbuild failed: {type(e).__name__}: {e}")
