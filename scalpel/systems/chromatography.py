"""
Chromatography system: column parameters, velocity profiles, sources.

The PDE in cylindrical coordinates:
    dC/dt + v(r)*dC/dz = Dr*(1/r)*d/dr(r*dC/dr) + Dz*d²C/dz²

After Hankel transform in r and Laplace in t:
    Dz * d²Ĉ/dz² - v*dĈ/dz - (s + Dr*kr²)*Ĉ = 0

With convective substitution Ĉ = Ĝ*exp(v*z/(2*Dz)):
    Dz * d²Ĝ/dz² - gamma_z² * Ĝ = 0
    gamma_z² = v²/(4*Dz²) + s/Dz + Dr*kr²/Dz
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class ColumnParams:
    """Packed chromatography column parameters."""
    name: str
    v: float           # Mean axial velocity [m/s]
    Dz: float          # Axial dispersion coefficient [m²/s]
    Dr: float          # Radial diffusion coefficient [m²/s]
    R: float           # Column radius [m]
    L: float           # Column length [m]
    porosity: float = 0.4   # Bed porosity

    @property
    def Pe(self) -> float:
        """Axial Péclet number Pe = v*L/Dz."""
        return self.v * self.L / self.Dz

    @property
    def Pe_radial(self) -> float:
        """Radial Péclet number Pe_r = v*R²/(Dr*L)."""
        return self.v * self.R**2 / (self.Dr * self.L)

    @property
    def residence_time(self) -> float:
        """Mean residence time L/v [s]."""
        return self.L / self.v


# ── Column database ─────────────────────────────────────────────────

COLUMNS = {
    "hplc": ColumnParams(
        name="HPLC analytical",
        v=1e-3,          # 1 mm/s
        Dz=1e-8,         # typical HPLC axial dispersion
        Dr=1e-9,         # radial molecular diffusion
        R=2.3e-3,        # 4.6 mm ID column
        L=0.15,          # 15 cm column
    ),
    "preparative": ColumnParams(
        name="Preparative",
        v=1e-4,          # 0.1 mm/s
        Dz=5e-8,         # larger dispersion
        Dr=1e-8,         # enhanced radial mixing
        R=12.5e-3,       # 25 mm ID column
        L=0.30,          # 30 cm column
    ),
    "process": ColumnParams(
        name="Process scale",
        v=5e-4,
        Dz=1e-7,
        Dr=5e-9,
        R=0.15,          # 30 cm ID
        L=0.50,
    ),
}


def get_column(name: str) -> ColumnParams:
    if name not in COLUMNS:
        raise ValueError(f"Unknown column: {name!r}. "
                         f"Available: {list(COLUMNS.keys())}")
    return COLUMNS[name]


def pulse_injection(t: np.ndarray, t_inject: float, width: float) -> np.ndarray:
    """Rectangular pulse injection of duration `width` starting at t=0.

    Parameters
    ----------
    t : ndarray
        Time array [s].
    t_inject : float
        Injection start time [s].
    width : float
        Pulse duration [s].

    Returns
    -------
    C_in : ndarray
        Inlet concentration (0 or 1).
    """
    return ((t >= t_inject) & (t < t_inject + width)).astype(float)


def gaussian_injection(t: np.ndarray, t0: float, sigma_t: float) -> np.ndarray:
    """Gaussian pulse injection."""
    return np.exp(-0.5 * ((t - t0) / sigma_t)**2) / (sigma_t * math.sqrt(2 * math.pi))
