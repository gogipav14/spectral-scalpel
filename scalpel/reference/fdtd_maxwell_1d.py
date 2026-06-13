"""
1D FDTD (Yee scheme) for the telegrapher's equation.

    mu*sigma * dE/dt + mu*epsilon * d²E/dt² = d²E/dz²

Uses first-order Mur ABC at the right boundary. The left boundary
is used as a prescribed-field source: E[0] = source_fn(t).
This produces a clean unidirectional pulse with no backward radiation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

MU_0 = 4e-7 * np.pi
EPS_0 = 8.854187817e-12


@dataclass
class FDTDResult:
    E: np.ndarray        # (Nz, Nt_save) electric field
    t: np.ndarray        # (Nt_save,) saved time points
    z: np.ndarray        # (Nz,) spatial grid


def fdtd_1d(
    sigma: float,
    epsilon_r: float,
    Lz: float,
    Nz: int,
    t_end: float,
    source_fn,             # source_fn(t) -> float
    obs_z: float = None,   # observation depth [m], default Lz/2
    save_every: int = 10,
) -> tuple[FDTDResult, np.ndarray, np.ndarray]:
    """Run 1D FDTD with boundary source at z=0, Mur ABC at z=Lz.

    The source is applied as E[0] = source_fn(t), producing a
    unidirectional rightward-propagating pulse.

    Parameters
    ----------
    sigma : float
        Conductivity [S/m].
    epsilon_r : float
        Relative permittivity.
    Lz : float
        Domain length [m].
    Nz : int
        Number of spatial cells.
    t_end : float
        Simulation end time [s].
    source_fn : callable
        Source function of time: source_fn(t) -> float.
    obs_z : float, optional
        Observation depth [m]. Default: Lz/2.
    save_every : int
        Save full field every N steps.

    Returns
    -------
    result : FDTDResult
        Full field snapshots.
    obs_signal : ndarray
        E-field at observation depth, every time step.
    obs_time : ndarray
        Time array for obs_signal (every step, not decimated).
    """
    epsilon = EPS_0 * epsilon_r
    dz = Lz / Nz

    # CFL: dt < dz / c_material, use Courant number 0.5
    c_mat = 1.0 / np.sqrt(MU_0 * epsilon)
    dt = 0.5 * dz / c_mat

    Nt = int(t_end / dt) + 1

    # Update coefficients (lossy medium)
    c1 = (1 - sigma*dt/(2*epsilon)) / (1 + sigma*dt/(2*epsilon))
    c2 = (dt/(epsilon*dz)) / (1 + sigma*dt/(2*epsilon))
    c3 = dt / (MU_0 * dz)

    E = np.zeros(Nz)
    H = np.zeros(Nz - 1)

    z = np.arange(Nz) * dz

    # Mur ABC at right boundary
    mur_coeff = (c_mat * dt - dz) / (c_mat * dt + dz)
    E_prev_right = 0.0

    # Observer
    if obs_z is None:
        obs_z = Lz / 2
    obs_idx = min(int(obs_z / dz), Nz - 1)

    E_save = []
    t_save = []
    obs_signal = np.zeros(Nt)
    obs_time = np.arange(Nt) * dt

    for n in range(Nt):
        t_n = n * dt

        # Store right boundary neighbor
        Em2_old = E[-2]

        # Update H
        H = H + c3 * (E[1:] - E[:-1])

        # Update E (interior only, not boundaries)
        E[1:-1] = c1 * E[1:-1] + c2 * (H[1:] - H[:-1])

        # Left boundary: prescribed source
        E[0] = source_fn(t_n)

        # Right boundary: first-order Mur ABC
        E[-1] = E_prev_right + mur_coeff * (E[-2] - E[-1])
        E_prev_right = Em2_old

        # Record observer
        obs_signal[n] = E[obs_idx]

        if n % save_every == 0:
            E_save.append(E.copy())
            t_save.append(t_n)

    result = FDTDResult(
        E=np.array(E_save).T,
        t=np.array(t_save),
        z=z,
    )
    return result, obs_signal, obs_time
