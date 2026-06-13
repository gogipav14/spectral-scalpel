"""
2D staggered-grid FDTD for damped acoustic waves.

    dp/dt + rho*c^2 * (dvx/dx + dvy/dy) = 0
    rho * dvx/dt + dp/dx = -eta * dvx    (viscous damping)
    rho * dvy/dt + dp/dy = -eta * dvy

Staggered grid: pressure at cell centers, velocities at cell edges.
This is the numpy reference solver for cross-validation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class AcousticFDTDResult:
    p: np.ndarray       # (Nx, Ny, Nt_save) pressure field
    t: np.ndarray       # (Nt_save,) saved time points
    x: np.ndarray       # (Nx,) spatial grid
    y: np.ndarray       # (Ny,)


def fdtd_acoustic_2d(
    c: float,
    nu: float,
    rho: float,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    t_end: float,
    source_fn,          # source_fn(t) -> float
    source_ix: int = None,
    source_iy: int = None,
    save_every: int = 20,
) -> AcousticFDTDResult:
    """Run 2D acoustic FDTD with viscous damping.

    Parameters
    ----------
    c : float
        Sound speed [m/s].
    nu : float
        Kinematic viscosity [m^2/s].
    rho : float
        Density [kg/m^3].
    Lx, Ly : float
        Domain size [m].
    Nx, Ny : int
        Grid points.
    t_end : float
        Simulation end time [s].
    source_fn : callable
        Pressure source function of time.
    source_ix, source_iy : int, optional
        Source cell index. Default: center.
    save_every : int
        Save every N steps.

    Returns
    -------
    AcousticFDTDResult with pressure snapshots.
    """
    dx = Lx / Nx
    dy = Ly / Ny
    dt = 0.5 * min(dx, dy) / c   # CFL

    if source_ix is None:
        source_ix = Nx // 2
    if source_iy is None:
        source_iy = Ny // 2

    Nt = int(t_end / dt) + 1
    eta = rho * nu  # dynamic viscosity

    # Damping coefficient for velocity update
    # dvx/dt = -(1/rho)*dp/dx - (eta/rho)*vx
    # Using exponential integrator for stability:
    damp = np.exp(-eta * dt / rho)
    c1_v = damp
    c2_v = (1 - damp) / (eta / rho * dt) * (dt / rho) if eta > 1e-30 \
           else dt / rho

    # Pressure update coefficient
    c_p = rho * c**2 * dt

    p = np.zeros((Nx, Ny))
    vx = np.zeros((Nx + 1, Ny))   # staggered in x
    vy = np.zeros((Nx, Ny + 1))   # staggered in y

    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy

    p_save = []
    t_save = []

    for n in range(Nt):
        t_n = n * dt

        # Update velocities
        # vx: dp/dx at cell edges
        vx[1:-1, :] = c1_v * vx[1:-1, :] - c2_v * (p[1:, :] - p[:-1, :]) / dx
        vy[:, 1:-1] = c1_v * vy[:, 1:-1] - c2_v * (p[:, 1:] - p[:, :-1]) / dy

        # Absorbing BC (zero velocity at boundaries)
        vx[0, :] = 0
        vx[-1, :] = 0
        vy[:, 0] = 0
        vy[:, -1] = 0

        # Update pressure
        div_v = (vx[1:, :] - vx[:-1, :]) / dx + (vy[:, 1:] - vy[:, :-1]) / dy
        p -= c_p * div_v

        # Add source
        p[source_ix, source_iy] += source_fn(t_n)

        if n % save_every == 0:
            p_save.append(p.copy())
            t_save.append(t_n)

    return AcousticFDTDResult(
        p=np.array(p_save).transpose(1, 2, 0),  # (Nx, Ny, Nt_save)
        t=np.array(t_save),
        x=x, y=y,
    )
