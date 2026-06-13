"""
Acoustic system: tissue database, transducer sources.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class AcousticMedium:
    """Viscous acoustic medium."""
    name: str
    c: float       # Sound speed [m/s]
    nu: float      # Kinematic viscosity [m^2/s]
    rho: float     # Density [kg/m^3]

    @property
    def omega_cross(self) -> float:
        """Crossover frequency c^2/nu [rad/s]."""
        return self.c**2 / self.nu

    @property
    def attenuation_per_wavelength(self) -> float:
        """Attenuation in Np per wavelength at crossover."""
        return math.pi * self.nu / self.c


MEDIA = {
    "water":        AcousticMedium("Water",       c=1500.0, nu=1e-6,  rho=1000.0),
    "soft_tissue":  AcousticMedium("Soft tissue", c=1540.0, nu=1e-3,  rho=1050.0),
    "liver":        AcousticMedium("Liver",       c=1590.0, nu=2e-3,  rho=1060.0),
    "bone":         AcousticMedium("Bone",        c=3000.0, nu=0.01,  rho=1900.0),
    "air":          AcousticMedium("Air",         c=343.0,  nu=1.5e-5, rho=1.2),
}


def get_medium(name: str) -> AcousticMedium:
    if name not in MEDIA:
        raise ValueError(f"Unknown medium: {name!r}. "
                         f"Available: {list(MEDIA.keys())}")
    return MEDIA[name]


def transducer_source(x, y, aperture: float, focus: float = None):
    """Circular piston transducer source profile.

    Parameters
    ----------
    x, y : ndarray
        Spatial coordinates [m].
    aperture : float
        Transducer diameter [m].
    focus : float, optional
        Focal distance [m]. If None, unfocused (plane piston).

    Returns
    -------
    source : ndarray
        Spatial source amplitude (0 or 1 for unfocused, phase-curved for focused).
    """
    r = np.sqrt(x**2 + y**2)
    radius = aperture / 2
    mask = (r <= radius).astype(float)

    if focus is not None:
        # Geometric focusing: phase curvature
        phase = np.exp(-1j * np.pi * r**2 / (focus * aperture))
        return mask * phase

    return mask
