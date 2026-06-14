"""
Method of Lines (MOL) reference solver for chromatography columns.

Solves the axial dispersion model:
    dC/dt + v*dC/dz = Dz * d²C/dz²

using central finite differences in z and scipy.integrate.solve_ivp for time.
Optionally includes radial diffusion on a (r, z) grid.

This is the slow reference solver for validating the spectral engine.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class MOLResult:
    """Result from MOL simulation."""
    C: np.ndarray       # (Nz, Nt) or (Nr, Nz, Nt) concentration
    t: np.ndarray       # (Nt,) time points
    z: np.ndarray       # (Nz,) axial grid
    r: np.ndarray = None  # (Nr,) radial grid (if 2D)


def mol_column_1d(
    v: float,
    Dz: float,
    L: float,
    Nz: int,
    t_end: float,
    inlet_fn=None,
    Nt_save: int = 200,
) -> MOLResult:
    """1D axial dispersion model via Method of Lines.

    Parameters
    ----------
    v : float
        Axial velocity [m/s].
    Dz : float
        Axial dispersion [m²/s].
    L : float
        Column length [m].
    Nz : int
        Number of axial cells.
    t_end : float
        Simulation end time [s].
    inlet_fn : callable, optional
        C_in(t) -> float. Default: delta-like pulse at t=0.
    Nt_save : int
        Number of time points to save.

    Returns
    -------
    MOLResult with concentration field.
    """
    dz = L / Nz
    z = np.arange(Nz) * dz + dz / 2  # cell centers

    if inlet_fn is None:
        # Approximate delta: inject for one time step
        pulse_dt = dz / v
        def inlet_fn(t):
            return 1.0 / pulse_dt if t < pulse_dt else 0.0

    def rhs(t, C):
        dCdt = np.zeros(Nz)

        # Interior: central differences for diffusion, upwind for advection
        # Diffusion: Dz * (C_{i+1} - 2*C_i + C_{i-1}) / dz^2
        dCdt[1:-1] += Dz * (C[2:] - 2*C[1:-1] + C[:-2]) / dz**2

        # Advection: upwind  -v * (C_i - C_{i-1}) / dz
        dCdt[1:] -= v * (C[1:] - C[:-1]) / dz

        # Inlet BC: Danckwerts
        C_in = inlet_fn(t)
        dCdt[0] += Dz * (C[1] - C[0]) / dz**2
        dCdt[0] -= v * (C[0] - C_in) / dz

        # Outlet BC: zero gradient
        dCdt[-1] += Dz * (C[-2] - C[-1]) / dz**2

        return dCdt

    C0 = np.zeros(Nz)
    t_eval = np.linspace(0, t_end, Nt_save)

    sol = solve_ivp(rhs, [0, t_end], C0, t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10)

    return MOLResult(C=sol.y, t=sol.t, z=z)


def mol_column_2d(
    v: float,
    Dz: float,
    Dr: float,
    R: float,
    L: float,
    Nz: int,
    Nr: int,
    t_end: float,
    inlet_fn=None,
    Nt_save: int = 200,
) -> MOLResult:
    """2D (r, z) axial dispersion + radial diffusion via MOL.

    Parameters
    ----------
    v : float
        Axial velocity [m/s] (uniform, no parabolic profile for simplicity).
    Dz : float
        Axial dispersion [m²/s].
    Dr : float
        Radial diffusion [m²/s].
    R : float
        Column radius [m].
    L : float
        Column length [m].
    Nz, Nr : int
        Grid points in z and r.
    t_end : float
        Simulation end time [s].
    inlet_fn : callable, optional
        C_in(t) -> float (uniform inlet).
    Nt_save : int
        Time points to save.

    Returns
    -------
    MOLResult with (Nr, Nz, Nt) concentration.
    """
    dz = L / Nz
    dr = R / Nr
    z = np.arange(Nz) * dz + dz / 2
    r = np.arange(Nr) * dr + dr / 2

    if inlet_fn is None:
        pulse_dt = dz / v
        def inlet_fn(t):
            return 1.0 / pulse_dt if t < pulse_dt else 0.0

    def rhs(t, C_flat):
        C = C_flat.reshape(Nr, Nz)
        dCdt = np.zeros_like(C)

        # Axial dispersion (z direction)
        dCdt[:, 1:-1] += Dz * (C[:, 2:] - 2*C[:, 1:-1] + C[:, :-2]) / dz**2

        # Axial advection (upwind)
        dCdt[:, 1:] -= v * (C[:, 1:] - C[:, :-1]) / dz

        # Inlet BC
        C_in = inlet_fn(t)
        dCdt[:, 0] += Dz * (C[:, 1] - C[:, 0]) / dz**2
        dCdt[:, 0] -= v * (C[:, 0] - C_in) / dz

        # Outlet BC (zero gradient)
        dCdt[:, -1] += Dz * (C[:, -2] - C[:, -1]) / dz**2

        # Radial diffusion: Dr * (1/r) * d/dr(r * dC/dr)
        # Interior
        for i in range(1, Nr - 1):
            ri = r[i]
            dCdt[i, :] += Dr * (
                (r[i] + dr/2) * (C[i+1, :] - C[i, :]) / dr
                - (r[i] - dr/2) * (C[i, :] - C[i-1, :]) / dr
            ) / (ri * dr)

        # r=0 (symmetry BC: dC/dr = 0)
        dCdt[0, :] += Dr * 2 * (C[1, :] - C[0, :]) / dr**2

        # r=R (wall BC: dC/dr = 0)
        dCdt[-1, :] += Dr * (
            -(r[-1] - dr/2) * (C[-1, :] - C[-2, :]) / dr
        ) / (r[-1] * dr)

        return dCdt.ravel()

    C0 = np.zeros(Nr * Nz)
    t_eval = np.linspace(0, t_end, Nt_save)

    sol = solve_ivp(rhs, [0, t_end], C0, t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-8)

    C_out = sol.y.reshape(Nr, Nz, -1)
    return MOLResult(C=C_out, t=sol.t, z=z, r=r)
