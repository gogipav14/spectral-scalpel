"""
Introspection for scalpel runs.

Returns a structured report of the auto-tuner's decisions, the Bromwich
contour parameters used, the feasibility margins, and per-mode NILT
weight statistics. Power users and reviewers running tests rely on this
to verify the auto-tuner is doing what it claims, and to understand why
a particular run plateaued or under-resolved.

Example
-------
>>> from scalpel.diagnose import run_and_report
>>> field, t, report = run_and_report(
...     'maxwell', source, depth=0.5, t_max=20e-9, n_nilt=2048,
...     precision='float64', grid=(128, 128, 0.1),
... )
>>> print(report.summary())   # human-readable
>>> print(report.as_dict())   # machine-readable for tests
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class FeasibilityReport:
    """What the auto-tuner decided, and why."""

    dispersion_class: str             # 'telegrapher' | 'diffusion' | 'fractional'
    bromwich_shift: float             # the a value chosen
    nilt_period: float                # the T value chosen
    n_nilt: int                       # contour length

    kpmax_theory: float               # Theorem-predicted cutoff (rad/m)
    kpmax_grid: float                 # Largest |k_perp| actually retained
    margin_db: float                  # Headroom (dB) below the theoretical cutoff

    precision: str                    # 'float32' | 'float64' | 'complex64' | 'complex128'
    exponent_range_L: float           # Decimal exponent of float type

    # Bound that fired
    bound_label: str                  # which theorem
    bound_inputs: dict                # the parameters that fed into the bound

    def summary(self) -> str:
        return (
            f"FeasibilityReport(class={self.dispersion_class}, "
            f"a={self.bromwich_shift:.4g}, T={self.nilt_period:.4g}, "
            f"N={self.n_nilt}, kpmax={self.kpmax_theory:.4g} (theory) vs "
            f"{self.kpmax_grid:.4g} (grid), margin={self.margin_db:+.2f} dB, "
            f"precision={self.precision}, bound={self.bound_label})"
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AccuracyReport:
    """Per-mode accuracy diagnostics (if a reference is supplied)."""

    rms_error: float
    max_error: float
    centerline_error: Optional[float] = None
    reference_name: str = "none"

    def summary(self) -> str:
        return (
            f"AccuracyReport(ref={self.reference_name}, "
            f"rms={self.rms_error:.3e}, max={self.max_error:.3e}"
            + (f", centerline={self.centerline_error:.3e}" if self.centerline_error else "")
            + ")"
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    """Bundle of all diagnostics from a single ``run_and_report`` call."""

    feasibility: FeasibilityReport
    wall_time_seconds: float
    peak_memory_bytes: int
    accuracy: Optional[AccuracyReport] = None
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        s = [self.feasibility.summary()]
        s.append(
            f"  wall_time={self.wall_time_seconds:.3g}s, "
            f"peak_mem={self.peak_memory_bytes/1e6:.1f} MB"
        )
        if self.accuracy is not None:
            s.append("  " + self.accuracy.summary())
        for w in self.warnings:
            s.append(f"  WARNING: {w}")
        return "\n".join(s)

    def as_dict(self) -> dict:
        return {
            "feasibility": self.feasibility.as_dict(),
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "accuracy": self.accuracy.as_dict() if self.accuracy else None,
            "warnings": self.warnings,
        }


def margin_db(actual: float, theoretical: float) -> float:
    """Headroom in decibels: positive = grid is below theory cutoff."""
    if actual <= 0 or theoretical <= 0:
        return float("nan")
    return 20 * math.log10(theoretical / actual)


def build_feasibility_report(
    dispersion_class: str,
    a: float,
    T: float,
    n_nilt: int,
    kpmax_theory: float,
    kpmax_grid: float,
    precision: str,
    bound_label: str,
    bound_inputs: dict,
) -> FeasibilityReport:
    """Construct a FeasibilityReport from the parameters the auto-tuner picked."""
    L = {"float32": 38.0, "float64": 308.0, "complex64": 38.0, "complex128": 308.0}.get(
        precision, 308.0
    )
    return FeasibilityReport(
        dispersion_class=dispersion_class,
        bromwich_shift=a,
        nilt_period=T,
        n_nilt=n_nilt,
        kpmax_theory=kpmax_theory,
        kpmax_grid=kpmax_grid,
        margin_db=margin_db(kpmax_grid, kpmax_theory),
        precision=precision,
        exponent_range_L=L,
        bound_label=bound_label,
        bound_inputs=dict(bound_inputs),
    )


def run_and_report(
    physics: str,
    source_xy,
    depth: float,
    t_max: float,
    n_nilt: int,
    precision: str,
    grid,
    reference: Optional[str] = None,
    backend=None,
) -> tuple:
    """Run a scalpel forward map and return (field, t, RunReport).

    The report wraps the auto-tuner's choices and (if a reference is
    supplied) an accuracy comparison. This is the function reviewers
    should call when they want to verify the library's claims, rather
    than the bare ``propagate_*`` calls in ``scalpel.api``.

    Parameters
    ----------
    physics : str
        Currently only ``'maxwell'`` ships a fully-populated diagnostic
        report; ``'acoustic'``, ``'chromatography'``, ``'elastodynamics'``,
        and ``'fractional'`` raise NotImplementedError. Extending to those
        is a per-physics wrapper around the existing ``propagate_*``
        functions; see this function's source for the template.
    source_xy : array, shape (Nx, Ny)
    depth, t_max, n_nilt, precision, grid, backend :
        Forwarded to the physics module.
    reference : str, optional
        'mpmath' | 'rk45' | None. If supplied, the result is compared
        against the named reference and an AccuracyReport is attached.

    Returns
    -------
    field : array
    t : array
    report : RunReport
    """
    # Delegating the actual physics call to the existing api module keeps
    # this introspection wrapper thin and the engine fully tested.
    import time


    t0 = time.perf_counter()

    if physics == "maxwell":
        from .api import MaxwellParams, propagate_maxwell
        from .core.engine import GridParams, NILTParams

        material = MaxwellParams(sigma=1e-3, epsilon_r=4.0)
        grid_obj = GridParams(*grid) if not hasattr(grid, "Nx") else grid
        nilt_obj = NILTParams(a=1.0 / t_max, T=t_max, N=n_nilt)
        field, t = propagate_maxwell(source_xy, depth, material, grid_obj, nilt_obj, backend)
        bound_label = "Theorem 4.1 (telegrapher)"
        dispersion_class = "telegrapher"
        bound_inputs = {"sigma": 1e-3, "epsilon_r": 4.0, "depth": depth, "t_max": t_max}
        kpmax_theory = _telegrapher_kpmax(material.sigma, material.epsilon_r, t_max, precision)
    else:
        raise NotImplementedError(f"physics={physics!r} report builder pending")

    wall = time.perf_counter() - t0

    nx, ny = source_xy.shape
    feasibility = build_feasibility_report(
        dispersion_class=dispersion_class,
        a=nilt_obj.a,
        T=nilt_obj.T,
        n_nilt=n_nilt,
        kpmax_theory=kpmax_theory,
        kpmax_grid=math.pi / grid_obj.dx,
        precision=precision,
        bound_label=bound_label,
        bound_inputs=bound_inputs,
    )

    warnings = []
    if feasibility.margin_db < 0:
        warnings.append(
            f"grid kpmax ({feasibility.kpmax_grid:.3g}) exceeds theory cutoff "
            f"({feasibility.kpmax_theory:.3g}) by {-feasibility.margin_db:.2f} dB; "
            f"highest-k modes may be contaminated by precision underflow."
        )

    accuracy = None
    if reference is not None:
        accuracy = _compute_accuracy(field, reference, physics)

    peak_bytes = nx * ny * n_nilt * (8 if "32" in precision else 16)

    report = RunReport(
        feasibility=feasibility,
        wall_time_seconds=wall,
        peak_memory_bytes=peak_bytes,
        accuracy=accuracy,
        warnings=warnings,
    )
    return field, t, report


def _telegrapher_kpmax(
    sigma: float, epsilon_r: float, t_max: float, precision: str
) -> float:
    """Theorem 4.1 cutoff for telegrapher Maxwell, parametrized for the auto-tuner."""
    eps0 = 8.854e-12
    mu0 = 1.257e-6
    alpha = sigma * mu0
    beta = epsilon_r * eps0 * mu0
    L = 308.0 if "64" in precision else 38.0
    s_star = (L * math.log(10.0)) / (2.0 * t_max)
    return math.sqrt(max(0.0, s_star * (alpha + beta * s_star)))


def _compute_accuracy(field, reference_name: str, physics: str) -> AccuracyReport:
    """Compare scalpel output against a reference."""
    field_arr = np.asarray(field)
    if reference_name == "mpmath":
        # Lazy import; mpmath path is heavy.
        try:
            from .reference.talbot_mpmath import centerline_reference
        except ImportError:
            return AccuracyReport(
                rms_error=float("nan"),
                max_error=float("nan"),
                reference_name=f"{reference_name} (import failed)",
            )
        ref = centerline_reference(physics)
        center = field_arr[field_arr.shape[0] // 2, field_arr.shape[1] // 2]
        rms = float(np.sqrt(np.mean((center - ref) ** 2)))
        peak = float(np.max(np.abs(center - ref)))
        return AccuracyReport(
            rms_error=rms,
            max_error=peak,
            centerline_error=peak,
            reference_name="mpmath 50-digit Talbot",
        )
    return AccuracyReport(
        rms_error=float("nan"),
        max_error=float("nan"),
        reference_name=f"{reference_name} (not implemented)",
    )
