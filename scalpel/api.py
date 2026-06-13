"""
High-level public API for spectral scalpel.

Convenience functions that wire up the engine with specific dispersion
relations for each physics system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .backends import get_backend
from .core.engine import SpectralEngine, GridParams, NILTParams
from .core import dispersion as disp


@dataclass
class MaxwellParams:
    """Material parameters for lossy Maxwell."""
    sigma: float       # Conductivity [S/m]
    epsilon_r: float   # Relative permittivity


@dataclass
class AcousticParams:
    """Material parameters for damped acoustics."""
    c: float           # Sound speed [m/s]
    nu: float          # Kinematic viscosity [m^2/s]


@dataclass
class ChromatographyParams:
    """Material parameters for preparative-scale chromatography."""
    v: float           # Mean axial velocity [m/s]
    Dz: float          # Axial dispersion coefficient [m^2/s]
    Dr: float          # Radial diffusion coefficient [m^2/s]


@dataclass
class ElastodynamicsParams:
    """Material parameters for P/S waves in layered viscoelastic media."""
    c_p: float         # P-wave speed [m/s]
    c_s: float         # S-wave speed [m/s]
    rho: float         # Density [kg/m^3]
    eta_p: float = 0.0 # Kelvin-Voigt P-wave viscosity [Pa.s]
    eta_s: float = 0.0 # Kelvin-Voigt S-wave viscosity [Pa.s]


def propagate_maxwell(
    source_xy,
    depth: float,
    material: MaxwellParams,
    grid: GridParams,
    nilt: NILTParams,
    backend=None,
):
    """Propagate EM field through lossy medium via spectral factorization.

    Parameters
    ----------
    source_xy : array, shape (Nx, Ny)
        Source field distribution.
    depth : float
        Propagation distance [m].
    material : MaxwellParams
        Conductivity and permittivity.
    grid : GridParams
        Spatial grid.
    nilt : NILTParams
        NILT parameters.
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    field : array, shape (Nx, Ny, Nt)
        Time-domain field.
    t : array, shape (Nt,)
        Time points.
    """
    b = backend or get_backend()

    def dispersion_fn(s, KX, KY, _b):
        return disp.maxwell_lossy(s, KX, KY, material.sigma, material.epsilon_r, _b)

    engine = SpectralEngine(dispersion_fn, b)
    return engine.forward(source_xy, depth, grid, nilt)


def propagate_chromatography(
    source_xy,
    depth: float,
    column: ChromatographyParams,
    grid: GridParams,
    nilt: NILTParams,
    backend=None,
):
    """Propagate a chromatographic pulse through a column slab (Cartesian path).

    Wraps ``scalpel.core.dispersion.convection_diffusion_cylindrical`` on
    a Cartesian (KX, KY) grid, using ``KR = sqrt(KX^2 + KY^2)`` as a
    drop-in radial wavenumber. This path is fast on GPU (standard 2D FFT
    machinery) and correct for axisymmetric sources on a square grid.

    For the natively cylindrical / Hankel-transform path, see
    ``propagate_chromatography_hankel``. The two paths agree on
    axisymmetric problems; the Hankel path is more efficient when
    ``N_modes << Nx*Ny`` and is the mathematically natural construction
    for a column.
    """
    b = backend or get_backend()

    def dispersion_fn(s, KX, KY, _b):
        # Use |k| as radial wavenumber for the unified-engine signature.
        KR = (KX ** 2 + KY ** 2) ** 0.5
        return disp.convection_diffusion_cylindrical(s, KR, column.v, column.Dz, column.Dr, _b)

    engine = SpectralEngine(dispersion_fn, b)
    return engine.forward(source_xy, depth, grid, nilt)


def propagate_chromatography_hankel(
    source_r,
    depth: float,
    column: ChromatographyParams,
    R: float,
    nilt: NILTParams,
    backend=None,
):
    """Propagate an axisymmetric chromatographic pulse via Hankel + NILT.

    Uses the natively cylindrical pipeline
    (``scalpel.core.engine.CylindricalEngine``) with the
    quasi-discrete Hankel transform of
    Guizar-Sicairos & Gutierrez-Vega (2004) for the radial direction
    and the batched FFT-NILT primitive for the time / Bromwich direction.
    The convective-substitution phase ``v/(2 Dz)`` is applied
    automatically.

    Parameters
    ----------
    source_r : ndarray, shape (N_modes,)
        Source profile evaluated at the Hankel radial grid
        ``HankelTransform(R, N_modes).r``.
    depth : float
        Column length (axial propagation distance) in metres.
    column : ChromatographyParams
        Velocity, axial dispersion, radial diffusion.
    R : float
        Radial domain extent (column radius) in metres.
    nilt : NILTParams
        Bromwich parameters.
    backend : Backend, optional

    Returns
    -------
    field : ndarray, shape (N_modes, Nt)
        Real-valued field at the radial sample points for each time step.
    t : ndarray, shape (Nt,)
        Time points.

    Notes
    -----
    The Hankel grid is non-uniform in r (Bessel zeros scaled by R);
    callers wanting a uniform-r output should interpolate via
    ``numpy.interp(uniform_r, ht.r, field[:, t_idx])``.
    """
    from .core.engine import CylindricalEngine
    from .core.hankel import HankelTransform

    b = backend or get_backend()
    n_modes = source_r.shape[0]
    ht = HankelTransform(R=R, N=n_modes)

    def dispersion_fn(s, KR, _b):
        # The cylindrical engine passes (s, KR, backend). The KR axis is
        # the radial wavenumber from the Hankel grid; convert to KX-like
        # for the existing dispersion signature.
        return disp.convection_diffusion_cylindrical(
            s, KR, column.v, column.Dz, column.Dr, _b
        )

    engine = CylindricalEngine(dispersion_fn, ht, b)
    # Convective substitution v/(2 Dz) factors the column ODE into a
    # symmetric Helmholtz problem; the CylindricalEngine takes the
    # exponent directly.
    conv_phase = column.v / (2.0 * column.Dz) if column.Dz > 0 else 0.0
    return engine.forward(source_r, depth, nilt, conv_phase=conv_phase)


def propagate_elastodynamics(
    source_xy,
    depth: float,
    material: ElastodynamicsParams,
    grid: GridParams,
    nilt: NILTParams,
    wave: str = "p",
    backend=None,
):
    """Propagate elastic P or S potential through a slab.

    ``wave='p'`` selects the curl-free P-wave dispersion (Helmholtz
    decomposition's longitudinal potential); ``wave='s'`` selects the
    divergence-free S-wave dispersion. The two are decoupled in the
    spectral domain; a full vector field would composite both with
    the gauge constraint enforced on reconstruction (see Methods).
    """
    b = backend or get_backend()
    wave = wave.lower()
    if wave == "p":
        def dispersion_fn(s, KX, KY, _b):
            return disp.elastic_pwave(s, KX, KY, material.c_p, material.rho,
                                       material.eta_p, material.eta_s, _b)
    elif wave == "s":
        def dispersion_fn(s, KX, KY, _b):
            return disp.elastic_swave(s, KX, KY, material.c_s, material.rho,
                                       material.eta_s, _b)
    else:
        raise ValueError(f"wave must be 'p' or 's' (got {wave!r})")

    engine = SpectralEngine(dispersion_fn, b)
    return engine.forward(source_xy, depth, grid, nilt)


def propagate_acoustic(
    source_xy,
    depth: float,
    medium: AcousticParams,
    grid: GridParams,
    nilt: NILTParams,
    backend=None,
):
    """Propagate acoustic field through viscous medium.

    Parameters
    ----------
    source_xy : array, shape (Nx, Ny)
        Source pressure field.
    depth : float
        Propagation distance [m].
    medium : AcousticParams
        Sound speed and viscosity.
    grid : GridParams
        Spatial grid.
    nilt : NILTParams
        NILT parameters.
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    field : array, shape (Nx, Ny, Nt)
        Time-domain pressure field.
    t : array, shape (Nt,)
        Time points.
    """
    b = backend or get_backend()

    def dispersion_fn(s, KX, KY, _b):
        return disp.damped_acoustic(s, KX, KY, medium.c, medium.nu, _b)

    engine = SpectralEngine(dispersion_fn, b)
    return engine.forward(source_xy, depth, grid, nilt)
