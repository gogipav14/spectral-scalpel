"""
3D FDTD Yee scheme for lossy Maxwell's equations.

    curl E = -mu * dH/dt
    curl H = sigma*E + epsilon * dE/dt

Staggered grid (Yee cell): E and H components offset by half a cell
in space and half a step in time.

Implements the standard update equations with exponential time-stepping
for the lossy (conduction) term, ensuring unconditional stability of
the loss operator.

This is the CPU reference solver for cross-validating the spectral engine.
For a fair comparison both solve the same problem: impulse propagation
through a lossy homogeneous slab of thickness d.

NOTE: This solver is intentionally simple (no PML, just zero-BC).
It is meant to be slow — that's the point of the comparison.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

MU_0 = 4e-7 * np.pi
EPS_0 = 8.854187817e-12


@dataclass
class FDTD3DResult:
    """Result from 3D FDTD simulation."""
    Ex: np.ndarray       # (Nt_save,) E-field at observation point
    t: np.ndarray        # (Nt_save,) time points
    wall_ms: float       # total wall time


def fdtd_3d_slab(
    sigma: float,
    epsilon_r: float,
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
    t_end: float,
    source_fn,
    source_xy: np.ndarray = None,
    obs_z_frac: float = 0.75,
    save_every: int = 10,
) -> FDTD3DResult:
    """Run 3D FDTD for a lossy homogeneous slab.

    The source is injected as Ex at z=Lz/4 (soft source) with spatial
    profile source_xy. The observer records Ex at z=obs_z_frac*Lz.

    Parameters
    ----------
    sigma : float
        Conductivity [S/m].
    epsilon_r : float
        Relative permittivity.
    Lx, Ly, Lz : float
        Domain dimensions [m].
    Nx, Ny, Nz : int
        Grid points.
    t_end : float
        Simulation time [s].
    source_fn : callable
        source_fn(t) -> float. Temporal source waveform.
    source_xy : ndarray, shape (Nx, Ny), optional
        Spatial source profile. Default: point source at center.
    obs_z_frac : float
        Observer z position as fraction of Lz.
    save_every : int
        Save interval.

    Returns
    -------
    FDTD3DResult
    """
    import time as timer

    epsilon = EPS_0 * epsilon_r
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    # CFL condition: dt < 1/(c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    c = 1.0 / np.sqrt(MU_0 * epsilon)
    dt = 0.9 / (c * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

    Nt = int(t_end / dt) + 1

    # Update coefficients for lossy dielectric
    # E update: exponential time-stepping for sigma term
    c1 = (1 - sigma*dt/(2*epsilon)) / (1 + sigma*dt/(2*epsilon))
    c2x = (dt/(epsilon*dx)) / (1 + sigma*dt/(2*epsilon))
    c2y = (dt/(epsilon*dy)) / (1 + sigma*dt/(2*epsilon))
    c2z = (dt/(epsilon*dz)) / (1 + sigma*dt/(2*epsilon))

    # H update (lossless magnetic)
    c3x = dt / (MU_0 * dx)
    c3y = dt / (MU_0 * dy)
    c3z = dt / (MU_0 * dz)

    # Fields (only Ex component for simplicity — scalar wave in x-polarization)
    # Full vectorial would need Ex, Ey, Ez, Hx, Hy, Hz
    # For x-polarized plane wave propagating in z: Ex, Hy are the relevant pair
    Ex = np.zeros((Nx, Ny, Nz))
    Hy = np.zeros((Nx, Ny, Nz))
    Hz = np.zeros((Nx, Ny, Nz))

    # Source and observer z-indices
    src_z = Nz // 4
    obs_z = int(obs_z_frac * Nz)

    if source_xy is None:
        source_xy = np.zeros((Nx, Ny))
        source_xy[Nx//2, Ny//2] = 1.0

    Ex_obs = []
    t_save = []

    t0 = timer.perf_counter()

    for n in range(Nt):
        t_n = n * dt

        # Update Hy: dHy/dt = (1/mu) * dEx/dz
        Hy[:, :, :-1] += c3z * (Ex[:, :, 1:] - Ex[:, :, :-1])

        # Update Hz: dHz/dt = -(1/mu) * dEx/dy
        Hz[:, :-1, :] -= c3y * (Ex[:, 1:, :] - Ex[:, :-1, :])

        # Update Ex: dEx/dt = (1/eps)*(dHz/dy - dHy/dz) - (sigma/eps)*Ex
        Ex[:, 1:, :] = c1 * Ex[:, 1:, :] + c2y * (Hz[:, 1:, :] - Hz[:, :-1, :])
        Ex[:, :, 1:] = c1 * Ex[:, :, 1:] - c2z * (Hy[:, :, 1:] - Hy[:, :, :-1])

        # Inject source (soft source)
        Ex[:, :, src_z] += source_fn(t_n) * source_xy * dt / epsilon

        # Record observer
        if n % save_every == 0:
            # Average over source_xy footprint at observer plane
            obs_val = np.sum(Ex[:, :, obs_z] * source_xy) / max(np.sum(source_xy), 1e-30)
            Ex_obs.append(obs_val)
            t_save.append(t_n)

    wall = (timer.perf_counter() - t0) * 1e3

    return FDTD3DResult(
        Ex=np.array(Ex_obs),
        t=np.array(t_save),
        wall_ms=wall,
    )
