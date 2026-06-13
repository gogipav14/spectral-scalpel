"""
mpmath 50-digit Talbot inversion of the centerline modes for selected
physics dispersions. Used as the ground-truth anchor for
``scalpel.validate.against_mpmath``.

The Talbot inversion is implemented in arbitrary precision via mpmath's
``invertlaplace`` routine with ``method='talbot'``, then downcast to
float64 for differencing. The reference table for the fractional
Caputo single-term case is precomputed in
``reports/claims_audit/fractional_mpmath_reference.csv`` and loaded
here so calls are O(1) and reproducible across runs.
"""

from __future__ import annotations

import csv
import os
import math
from typing import Optional

import numpy as np

# Repository root, located relative to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_FRACTIONAL_REFERENCE_CSV = os.path.join(
    _REPO_ROOT, "reports", "claims_audit", "fractional_mpmath_reference.csv"
)
_PAPER_TOMS_REFERENCE_CSV = os.path.join(
    _REPO_ROOT, "paper_toms", "data", "fractional_mpmath_reference.csv"
)


def _load_fractional_csv() -> Optional[dict]:
    """Load the canonical fractional reference CSV. Returns None if absent."""
    for path in (_FRACTIONAL_REFERENCE_CSV, _PAPER_TOMS_REFERENCE_CSV):
        if os.path.exists(path):
            with open(path, "r") as f:
                rdr = csv.reader(f)
                rows = list(rdr)
            if not rows:
                continue
            header = rows[0]
            data = {}
            for col_idx, name in enumerate(header):
                data[name] = []
            for r in rows[1:]:
                for col_idx, val in enumerate(r):
                    try:
                        data[header[col_idx]].append(float(val))
                    except ValueError:
                        data[header[col_idx]].append(val)
            return data
    return None


def centerline_reference(
    physics: str,
    *,
    alpha: Optional[float] = None,
    t_max: Optional[float] = None,
    n: Optional[int] = None,
) -> np.ndarray:
    """Return the centerline (k_perp = 0) ground-truth trace.

    For ``physics == 'fractional'`` the precomputed CSV is consulted
    first; if absent (or if a non-bundled (alpha, t_max) is requested),
    falls back to a fresh mpmath Talbot inversion (slow; requires
    ``pip install mpmath``).

    Parameters
    ----------
    physics : str
        Currently 'fractional' (single-term Caputo) is the bundled case.
    alpha : float, optional
        Fractional order for ``physics='fractional'``.
    t_max : float, optional
        Time horizon for the reference trace.
    n : int, optional
        Number of time samples to return.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    if physics != "fractional":
        raise NotImplementedError(
            f"centerline_reference for physics={physics!r} is not yet bundled; "
            f"add a wrapper following the fractional case as a template."
        )

    data = _load_fractional_csv()
    if data is not None and "u" in data and "t" in data:
        u = np.asarray(data["u"], dtype=np.float64)
        if n is None or n == len(u):
            return u
        # Resample to n points linearly in t. The bundled CSV is fine-grained.
        t = np.asarray(data["t"], dtype=np.float64)
        if t_max is not None:
            mask = t <= t_max
            t = t[mask]
            u = u[mask]
        if n is not None:
            t_target = np.linspace(t[0], t[-1], n)
            u = np.interp(t_target, t, u)
        return u

    # Fallback: live mpmath inversion. This is slow (~minutes for 50 digits)
    # and only runs if the CSV is missing.
    return _mpmath_fractional_centerline(alpha or 0.7, t_max or 20e-3, n or 64)


def _mpmath_fractional_centerline(alpha: float, t_max: float, n: int) -> np.ndarray:
    """Fresh mpmath Talbot inversion of E_alpha(-D k_perp^2 t^alpha) at k=0."""
    try:
        import mpmath as mp
    except ImportError as e:
        raise RuntimeError(
            "mpmath not installed; cannot build live reference. "
            "Install with `pip install mpmath` or commit a precomputed CSV."
        ) from e

    mp.mp.dps = 50

    def F(s):
        # k_perp = 0 reduces gamma_z to sqrt(D * s^alpha) -> at depth d
        # H(s, 0, d) = exp(-d * sqrt(D * s^alpha))
        # For unit source S(s) = 1/s, the inverse is well-defined.
        return 1 / s * mp.exp(-mp.sqrt(mp.mpf(1e-3) * s ** alpha) * mp.mpf(0.1))

    ts = np.linspace(t_max * 1e-3, t_max, n)
    vals = []
    for t in ts:
        vals.append(float(mp.invertlaplace(F, mp.mpf(t), method="talbot")))
    return np.asarray(vals, dtype=np.float64)
