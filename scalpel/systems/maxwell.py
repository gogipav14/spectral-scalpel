"""
Maxwell/EM system: material database, source wavelets, GPR scenarios.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

from ..core.dispersion import MU_0, EPS_0


@dataclass
class MaxwellMaterial:
    """Lossy dielectric material."""
    name: str
    sigma: float       # Conductivity [S/m]
    epsilon_r: float   # Relative permittivity
    mu_r: float = 1.0  # Relative permeability

    @property
    def epsilon(self) -> float:
        return EPS_0 * self.epsilon_r

    @property
    def mu(self) -> float:
        return MU_0 * self.mu_r

    @property
    def omega_cross(self) -> float:
        """Crossover frequency sigma/epsilon [rad/s]."""
        return self.sigma / self.epsilon

    @property
    def skin_depth_dc(self) -> float:
        """DC skin depth sqrt(2/(mu*sigma*omega_cross)) [m]."""
        return math.sqrt(2 / (self.mu * self.sigma * self.omega_cross))

    @property
    def wave_speed(self) -> float:
        """Phase velocity in lossless limit [m/s]."""
        return 1.0 / math.sqrt(self.mu * self.epsilon)


# ── Material database ────────────────────────────────────────────────

MATERIALS = {
    "vacuum":    MaxwellMaterial("Vacuum",    sigma=0.0,    epsilon_r=1.0),
    "dry_sand":  MaxwellMaterial("Dry sand",  sigma=1e-4,   epsilon_r=4.0),
    "wet_sand":  MaxwellMaterial("Wet sand",  sigma=1e-2,   epsilon_r=25.0),
    "wet_clay":  MaxwellMaterial("Wet clay",  sigma=0.1,    epsilon_r=10.0),
    "concrete":  MaxwellMaterial("Concrete",  sigma=1e-3,   epsilon_r=6.0),
    "seawater":  MaxwellMaterial("Seawater",  sigma=4.0,    epsilon_r=80.0),
    "freshwater": MaxwellMaterial("Fresh water", sigma=0.01, epsilon_r=80.0),
    "muscle":    MaxwellMaterial("Muscle",    sigma=0.5,    epsilon_r=50.0),
}


def get_material(name: str) -> MaxwellMaterial:
    if name not in MATERIALS:
        raise ValueError(f"Unknown material: {name!r}. "
                         f"Available: {list(MATERIALS.keys())}")
    return MATERIALS[name]


# ── Source wavelets ──────────────────────────────────────────────────

def ricker_wavelet(t: np.ndarray, f_center: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet.

    Parameters
    ----------
    t : ndarray
        Time array [s].
    f_center : float
        Center frequency [Hz].

    Returns
    -------
    w : ndarray
        Wavelet amplitude.
    """
    tau = t - 1.0 / f_center
    a = (math.pi * f_center * tau) ** 2
    return (1 - 2 * a) * np.exp(-a)


def gaussian_pulse(t: np.ndarray, t0: float, sigma_t: float) -> np.ndarray:
    """Gaussian pulse centered at t0 with width sigma_t."""
    return np.exp(-0.5 * ((t - t0) / sigma_t) ** 2) / (sigma_t * math.sqrt(2 * math.pi))


def ricker_spectrum(s, f_center: float):
    """Laplace-domain Ricker wavelet (for direct spectral injection).

    F(s) = (2/sqrt(pi)) * (s/fp)^2 * exp(-s^2/fp^2)
    where fp = pi * f_center.
    """
    import cmath
    fp = math.pi * f_center
    return (2 / math.sqrt(math.pi)) * (s / fp) ** 2 * cmath.exp(-(s / fp) ** 2)


# ── GPR scenario helper ─────────────────────────────────────────────

@dataclass
class GPRScenario:
    """Ground-penetrating radar simulation setup."""
    material: MaxwellMaterial
    target_depth: float     # [m]
    f_center: float         # Source center frequency [Hz]
    grid_nx: int
    grid_dx: float          # [m]

    @property
    def t_end(self) -> float:
        """Observation window: 5x two-way travel time."""
        v = self.material.wave_speed
        return 5 * (2 * self.target_depth / v)
