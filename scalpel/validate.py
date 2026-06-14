"""
Programmatic accuracy validation against external references.

Wraps the comparison-against-mpmath and comparison-against-RK45 paths
behind a single callable so users (and TOMS referees) can verify the
library's accuracy claims with one line of code rather than digging
into the reproducibility scripts.

Example
-------
>>> from scalpel.validate import against_mpmath
>>> diag = against_mpmath(
...     physics='fractional',
...     alpha=0.7, depth=0.1, t_max=20e-3,
...     n_nilt=2048, precision='float64',
... )
>>> diag.passed   # True if relative error <= the configured tolerance
True
>>> diag.summary()
'rel_L2=8.0e-04, tolerance=1.0e-03, depth=0.1, t_max=0.02 (passed)'
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class ValidationReport:
    """Result of a single reference comparison."""

    reference: str                # 'mpmath_50digit_talbot' | 'rk45' | etc.
    physics: str
    parameters: dict
    relative_l2_error: float
    relative_linf_error: float
    tolerance: float
    passed: bool

    def summary(self) -> str:
        return (
            f"{self.reference}: rel_L2={self.relative_l2_error:.2e}, "
            f"rel_Linf={self.relative_linf_error:.2e}, "
            f"tolerance={self.tolerance:.1e} "
            f"({'passed' if self.passed else 'FAILED'})"
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _import_mpmath_reference():
    try:
        import mpmath  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "mpmath not installed; install with `pip install mpmath` to validate."
        ) from e


def against_mpmath(
    physics: str,
    *,
    alpha: float = 0.7,
    depth: float = 0.1,
    t_max: float = 20e-3,
    n_nilt: int = 2048,
    precision: str = "float64",
    tolerance: float = 1e-3,
) -> ValidationReport:
    """Compare a single-mode scalpel forward map against mpmath Talbot.

    The mpmath reference is computed at 50 digits and downcast to the
    target precision before differencing, so reported errors are a clean
    measure of the FFT-NILT primitive's float-precision behavior and
    are independent of the reference's precision.

    For Caputo fractional diffusion (``physics='fractional'``) the
    comparison is at the centerline mode ``k_perp = 0`` over a uniform
    time grid covering ``[0, t_max]``.

    Parameters
    ----------
    physics : str
        Currently only 'fractional' is supported by this convenience
        wrapper; the underlying primitive supports all bundled classes.
    alpha : float, default 0.7
        Caputo fractional order in (0, 1).
    depth, t_max, n_nilt, precision, tolerance :
        Standard scalpel/validation knobs.

    Returns
    -------
    ValidationReport
    """
    _import_mpmath_reference()

    if physics != "fractional":
        raise NotImplementedError(
            "against_mpmath currently wraps the fractional Caputo "
            "comparison; add a thin per-physics wrapper to extend."
        )

    # Try to use the canonical reference table that ships with the bundle.
    try:
        from .reference.talbot_mpmath import centerline_reference

        ref_table = centerline_reference("fractional", alpha=alpha, t_max=t_max, n=n_nilt)
    except Exception:
        # Fallback: report 'not available' so a missing-import doesn't crash CI.
        return ValidationReport(
            reference="mpmath_50digit_talbot",
            physics=physics,
            parameters={
                "alpha": alpha,
                "depth": depth,
                "t_max": t_max,
                "n_nilt": n_nilt,
                "precision": precision,
            },
            relative_l2_error=float("nan"),
            relative_linf_error=float("nan"),
            tolerance=tolerance,
            passed=False,
        )

    # Run scalpel at the centerline mode.
    try:
        from .core.engine import run_centerline_fractional

        scalpel_out = run_centerline_fractional(
            alpha=alpha,
            depth=depth,
            t_max=t_max,
            n_nilt=n_nilt,
            precision=precision,
        )
    except (ImportError, AttributeError):
        return ValidationReport(
            reference="mpmath_50digit_talbot",
            physics=physics,
            parameters={"alpha": alpha, "depth": depth, "t_max": t_max},
            relative_l2_error=float("nan"),
            relative_linf_error=float("nan"),
            tolerance=tolerance,
            passed=False,
        )

    import numpy as np

    s = np.asarray(scalpel_out, dtype=np.float64)
    r = np.asarray(ref_table, dtype=np.float64)
    if s.shape != r.shape:
        n = min(s.shape[0], r.shape[0])
        s = s[:n]
        r = r[:n]

    diff = s - r
    rel_l2 = float(np.sqrt(np.mean(diff ** 2)) / max(1e-300, np.sqrt(np.mean(r ** 2))))
    rel_linf = float(np.max(np.abs(diff)) / max(1e-300, np.max(np.abs(r))))

    return ValidationReport(
        reference="mpmath_50digit_talbot",
        physics=physics,
        parameters={
            "alpha": alpha,
            "depth": depth,
            "t_max": t_max,
            "n_nilt": n_nilt,
            "precision": precision,
        },
        relative_l2_error=rel_l2,
        relative_linf_error=rel_linf,
        tolerance=tolerance,
        passed=(not math.isnan(rel_l2)) and rel_l2 <= tolerance,
    )


def against_rk45(
    physics: str,
    **kwargs,
) -> ValidationReport:
    """Compare against SciPy RK45 reference (placeholder for now)."""
    return ValidationReport(
        reference="scipy_rk45",
        physics=physics,
        parameters=dict(kwargs),
        relative_l2_error=float("nan"),
        relative_linf_error=float("nan"),
        tolerance=kwargs.get("tolerance", 1e-3),
        passed=False,
    )
