"""
Elastodynamic system: Helmholtz decomposition of the Navier equation into
P-wave (curl-free) and S-wave (divergence-free) scalar potentials.

Governing PDE (homogeneous isotropic elastic medium):
    rho * ∂²u/∂t² = (lambda + 2*mu) ∇(∇·u) - mu ∇ × ∇ × u

With the Helmholtz decomposition
    u = ∇phi + ∇ × psi,   ∇·psi = 0   (Coulomb gauge)
the vector equation decouples into two scalar wave equations:
    ∂²phi/∂t² = c_p² ∇² phi,      c_p² = (lambda + 2 mu) / rho
    ∂²psi/∂t² = c_s² ∇² psi,      c_s² = mu / rho

The gauge constraint ∇·psi = 0 is the curl-constraint that closes the
spectral factorization: after 2D FFT in (x, y), the two potentials propagate
independently in z with their own dispersion relations, and the physical
displacement is reconstructed by differentiating in the spectral domain
before the inverse FFT.

Viscoelastic attenuation is modelled with the Kelvin-Voigt form
    mu(s) = mu + eta_s s,   lambda(s) = lambda + eta_p s
so in Laplace space the effective wave speeds become
    c_p²(s) = (lambda + 2 mu + (eta_p + 2 eta_s) s) / rho
    c_s²(s) = (mu + eta_s s) / rho
and the crossover frequencies omega_cross = mu/eta_s, (lambda+2*mu)/(eta_p+2*eta_s)
play the role that sigma/epsilon plays for lossy Maxwell.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class ElasticMaterial:
    """Isotropic linear elastic (optionally Kelvin-Voigt viscoelastic) solid."""
    name: str
    rho: float         # Density [kg/m^3]
    c_p: float         # P-wave speed [m/s]
    c_s: float         # S-wave speed [m/s]
    eta_p: float = 0.0  # P-wave viscous modulus [Pa s]
    eta_s: float = 0.0  # S-wave viscous modulus [Pa s]

    @property
    def mu(self) -> float:
        """Shear modulus mu = rho * c_s^2 [Pa]."""
        return self.rho * self.c_s ** 2

    @property
    def lam(self) -> float:
        """Lame's first parameter lambda = rho*(c_p^2 - 2 c_s^2) [Pa]."""
        return self.rho * (self.c_p ** 2 - 2 * self.c_s ** 2)

    @property
    def M(self) -> float:
        """P-wave modulus M = lambda + 2 mu = rho * c_p^2 [Pa]."""
        return self.rho * self.c_p ** 2

    @property
    def poisson(self) -> float:
        """Poisson's ratio nu = (c_p^2 - 2 c_s^2) / (2 (c_p^2 - c_s^2))."""
        r = (self.c_p / self.c_s) ** 2
        return (r - 2) / (2 * (r - 1))

    @property
    def speed_ratio(self) -> float:
        """c_p / c_s (= sqrt((1-nu) / (0.5 - nu)); bounded below by sqrt(2))."""
        return self.c_p / self.c_s

    @property
    def omega_cross_s(self) -> float:
        """S-wave crossover frequency mu / eta_s [rad/s]. Infinite if undamped."""
        if self.eta_s <= 0.0:
            return math.inf
        return self.mu / self.eta_s

    @property
    def omega_cross_p(self) -> float:
        """P-wave crossover frequency M / (eta_p + 2 eta_s) [rad/s]. Infinite if undamped."""
        denom = self.eta_p + 2 * self.eta_s
        if denom <= 0.0:
            return math.inf
        return self.M / denom

    @property
    def skin_depth_dc_s(self) -> float:
        """Low-frequency S-wave diffusive skin depth sqrt(2 eta_s / (rho omega_cross_s)) [m]."""
        if self.eta_s <= 0.0:
            return math.inf
        return math.sqrt(2 * self.eta_s / (self.rho * self.omega_cross_s))


# Material database: seismic / industrial / biomedical targets.
# c_p, c_s from standard references (Mavko, Rock Physics Handbook; Sarvazyan,
# ultrasound elastography review; Ensminger, Ultrasonics).
MATERIALS = {
    "steel":      ElasticMaterial("Steel",      rho=7850.0, c_p=5900.0, c_s=3200.0),
    "aluminum":   ElasticMaterial("Aluminum",   rho=2700.0, c_p=6320.0, c_s=3130.0),
    "concrete":   ElasticMaterial("Concrete",   rho=2400.0, c_p=3700.0, c_s=2300.0),
    "granite":    ElasticMaterial("Granite",    rho=2700.0, c_p=5500.0, c_s=3200.0),
    "limestone":  ElasticMaterial("Limestone",  rho=2600.0, c_p=4600.0, c_s=2500.0),
    "sandstone":  ElasticMaterial("Sandstone",  rho=2200.0, c_p=3000.0, c_s=1700.0),
    "ice":        ElasticMaterial("Ice",        rho=920.0,  c_p=3800.0, c_s=1900.0),
    # Kelvin-Voigt viscoelastic cases (eta fitted to ~1% attenuation per wavelength
    # at a representative frequency in the quoted band; order-of-magnitude only):
    "rubber":     ElasticMaterial("Rubber",     rho=1100.0, c_p=1800.0, c_s=50.0,
                                  eta_p=1.0e4, eta_s=5.0),
    "soft_tissue": ElasticMaterial("Soft tissue (shear-elastography)",
                                   rho=1050.0, c_p=1540.0, c_s=3.0,
                                   eta_p=0.0, eta_s=0.02),
}


def get_material(name: str) -> ElasticMaterial:
    if name not in MATERIALS:
        raise ValueError(f"Unknown elastic material: {name!r}. "
                         f"Available: {list(MATERIALS.keys())}")
    return MATERIALS[name]


# --- Source wavelets ------------------------------------------------

def ricker_wavelet(t: np.ndarray, f_center: float) -> np.ndarray:
    """Ricker wavelet (canonical seismic source)."""
    tau = t - 1.0 / f_center
    a = (math.pi * f_center * tau) ** 2
    return (1 - 2 * a) * np.exp(-a)


def ricker_spectrum(s, f_center: float):
    """Laplace-domain Ricker wavelet."""
    import cmath
    fp = math.pi * f_center
    return (2 / math.sqrt(math.pi)) * (s / fp) ** 2 * cmath.exp(-(s / fp) ** 2)


def vertical_point_force(x, y, width: float) -> np.ndarray:
    """Gaussian-blurred vertical point force on the (x, y) source plane.

    Returns the z-component of the body-force density (couples into the
    P potential via rho * ddot u_z -> ∂phi/∂z after Helmholtz decomposition).
    """
    r2 = x ** 2 + y ** 2
    return np.exp(-0.5 * r2 / width ** 2) / (2 * math.pi * width ** 2)


# --- Scenario helper ------------------------------------------------

@dataclass
class ElastodynamicScenario:
    """Canonical layered-halfspace seismic scan (one-sided propagation in z)."""
    material: ElasticMaterial
    target_depth: float   # [m] observation distance
    f_center: float       # [Hz] source centre frequency
    grid_nx: int
    grid_dx: float        # [m]

    @property
    def t_end(self) -> float:
        """Observation window: 4x the two-way S-wave traveltime."""
        return 4 * (2 * self.target_depth / self.material.c_s)

    @property
    def t_p(self) -> float:
        """One-way P-wave traveltime to the target."""
        return self.target_depth / self.material.c_p

    @property
    def t_s(self) -> float:
        """One-way S-wave traveltime to the target."""
        return self.target_depth / self.material.c_s
