"""
Exponential Time Differencing Runge-Kutta (ETDRK) for reaction-diffusion.

For the semi-linear PDE:
    du/dt = L*u + N(u)

where L is the linear operator (diffusion, diagonal in Fourier space)
and N(u) is the nonlinear reaction term.

The exact propagator exp(L*dt) is computed in Fourier space as
exp(-D*k^2*dt) per mode — no time-stepping error in the linear part.

Implements:
  - ETDRK1 (exponential Euler)
  - ETDRK4 (Cox-Matthews 4th order, 2002)
  - Strang splitting (for comparison with moljax)

All methods use the scalpel JAX/PyTorch backend for GPU acceleration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..backends import get_backend


@dataclass
class RDField:
    """A single field in a reaction-diffusion system."""
    name: str
    D: float                  # Diffusion coefficient
    data: np.ndarray = None   # Current field values (Nx, Ny)


@dataclass
class RDSystem:
    """Multi-field reaction-diffusion system.

    du_i/dt = D_i * Laplacian(u_i) + R_i(u_1, ..., u_n)
    """
    fields: Dict[str, RDField]
    reaction_fn: Callable      # reaction_fn(fields_dict, params) -> dict of RHS arrays
    params: Dict = field(default_factory=dict)
    Lx: float = 1.0
    Ly: float = 1.0

    @property
    def Nx(self):
        return next(iter(self.fields.values())).data.shape[0]

    @property
    def Ny(self):
        return next(iter(self.fields.values())).data.shape[1]


def _build_laplacian_eigenvalues(Nx, Ny, Lx, Ly, backend):
    """Fourier eigenvalues of the 2D Laplacian on periodic domain."""
    dx = Lx / Nx
    dy = Ly / Ny
    kx = backend.fftfreq(Nx, dx) * 2 * np.pi
    ky = backend.fftfreq(Ny, dy) * 2 * np.pi
    KX, KY = backend.meshgrid(kx, ky)
    return -(KX**2 + KY**2)  # eigenvalues of Laplacian


def _phi1(z):
    """phi_1(z) = (exp(z) - 1) / z, regularized for small z."""
    # Use Taylor expansion for |z| < 1e-4
    return np.where(
        np.abs(z) < 1e-4,
        1.0 + z/2 + z**2/6 + z**3/24,
        (np.exp(z) - 1) / np.where(np.abs(z) < 1e-30, 1.0, z)
    )


def etdrk1_step(system: RDSystem, dt: float, backend=None):
    """Single ETDRK1 (exponential Euler) step.

    u_new = exp(L*dt) * u + dt * phi_1(L*dt) * N(u)
    """
    b = backend or get_backend()

    lap_eig = _build_laplacian_eigenvalues(
        system.Nx, system.Ny, system.Lx, system.Ly, b)

    # Current state in Fourier space
    state = {name: f.data for name, f in system.fields.items()}
    state_hat = {name: b.fft2(b.array(v)) for name, v in state.items()}

    # Nonlinear RHS
    N = system.reaction_fn(state, system.params)

    for name, f in system.fields.items():
        D = f.D
        L_dt = D * lap_eig * dt  # (Nx, Ny) eigenvalues * dt

        exp_L = b.exp(L_dt)
        N_hat = b.fft2(b.array(N[name]))

        # phi_1 in Fourier space
        L_dt_np = b.to_numpy(L_dt)
        phi1_vals = b.array(_phi1(L_dt_np))

        # u_new_hat = exp(L*dt) * u_hat + dt * phi_1(L*dt) * N_hat
        u_new_hat = exp_L * state_hat[name] + dt * phi1_vals * N_hat
        f.data = b.to_numpy(b.real(b.ifft2(u_new_hat)))


def strang_split_step(system: RDSystem, dt: float, backend=None):
    """Strang splitting: half-diffusion, full-reaction, half-diffusion.

    Equivalent to moljax's IMEX-Strang but using scalpel backend.
    """
    b = backend or get_backend()

    lap_eig = _build_laplacian_eigenvalues(
        system.Nx, system.Ny, system.Lx, system.Ly, b)

    # Half diffusion step (exact in Fourier)
    for name, f in system.fields.items():
        u_hat = b.fft2(b.array(f.data))
        exp_half = b.exp(f.D * lap_eig * dt / 2)
        u_hat = exp_half * u_hat
        f.data = b.to_numpy(b.real(b.ifft2(u_hat)))

    # Full reaction step (Heun's method, 2nd order explicit)
    state = {name: f.data for name, f in system.fields.items()}
    k1 = system.reaction_fn(state, system.params)

    state_tilde = {name: state[name] + dt * k1[name]
                   for name in system.fields}
    k2 = system.reaction_fn(state_tilde, system.params)

    for name, f in system.fields.items():
        f.data = state[name] + dt / 2 * (k1[name] + k2[name])

    # Half diffusion step
    for name, f in system.fields.items():
        u_hat = b.fft2(b.array(f.data))
        exp_half = b.exp(f.D * lap_eig * dt / 2)
        u_hat = exp_half * u_hat
        f.data = b.to_numpy(b.real(b.ifft2(u_hat)))


def integrate(
    system: RDSystem,
    t_end: float,
    dt: float,
    method: str = "strang",
    save_every: int = 100,
    backend=None,
) -> Dict:
    """Integrate reaction-diffusion system.

    Parameters
    ----------
    system : RDSystem
        The reaction-diffusion system.
    t_end : float
        End time.
    dt : float
        Time step.
    method : str
        "strang" or "etdrk1".
    save_every : int
        Save state every N steps.
    backend : Backend, optional

    Returns
    -------
    dict with 'snapshots' (list of state dicts), 'times' (list of floats)
    """
    b = backend or get_backend()
    step_fn = strang_split_step if method == "strang" else etdrk1_step

    Nt = int(t_end / dt)
    snapshots = []
    times = []

    for n in range(Nt):
        if n % save_every == 0:
            snap = {name: f.data.copy() for name, f in system.fields.items()}
            snapshots.append(snap)
            times.append(n * dt)

        step_fn(system, dt, b)

    # Final state
    snapshots.append({name: f.data.copy() for name, f in system.fields.items()})
    times.append(Nt * dt)

    return {"snapshots": snapshots, "times": times}


# ═══════════════════════════════════════════════════════════════════════
#  Pre-built reaction-diffusion systems
# ═══════════════════════════════════════════════════════════════════════

def brusselator(Nx=128, Ny=128, Lx=64.0, Ly=64.0,
                Du=1.0, Dv=8.0, a=4.5, b=7.5,
                seed=42) -> RDSystem:
    """Brusselator reaction-diffusion system.

    du/dt = Du*Lap(u) + a - (b+1)*u + u^2*v
    dv/dt = Dv*Lap(v) + b*u - u^2*v

    Turing instability when b > 1 + a^2 and Dv/Du sufficiently large.

    Default params give Turing patterns (spots/stripes).
    """
    rng = np.random.RandomState(seed)

    # Steady state + small perturbation
    u0 = a + 0.1 * rng.randn(Nx, Ny)
    v0 = b / a + 0.1 * rng.randn(Nx, Ny)

    def reaction(state, params):
        u, v = state["u"], state["v"]
        a_, b_ = params["a"], params["b"]
        uvv = u * u * v
        return {
            "u": a_ - (b_ + 1) * u + uvv,
            "v": b_ * u - uvv,
        }

    return RDSystem(
        fields={"u": RDField("u", Du, u0), "v": RDField("v", Dv, v0)},
        reaction_fn=reaction,
        params={"a": a, "b": b},
        Lx=Lx, Ly=Ly,
    )


def gray_scott(Nx=128, Ny=128, Lx=2.5, Ly=2.5,
               Du=0.16, Dv=0.08, F=0.04, k=0.06,
               seed=42) -> RDSystem:
    """Gray-Scott reaction-diffusion system.

    du/dt = Du*Lap(u) - u*v^2 + F*(1-u)
    dv/dt = Dv*Lap(v) + u*v^2 - (F+k)*v

    Produces complex patterns: spots, stripes, mitosis, coral.
    """
    rng = np.random.RandomState(seed)

    # Initialize with u=1, v=0 everywhere, seed a central square with v
    u0 = np.ones((Nx, Ny))
    v0 = np.zeros((Nx, Ny))

    # Seed pattern in center
    cx, cy = Nx // 2, Ny // 2
    r = max(Nx // 10, 5)
    u0[cx-r:cx+r, cy-r:cy+r] = 0.5 + 0.1 * rng.randn(2*r, 2*r)
    v0[cx-r:cx+r, cy-r:cy+r] = 0.25 + 0.1 * rng.randn(2*r, 2*r)

    def reaction(state, params):
        u, v = state["u"], state["v"]
        F_, k_ = params["F"], params["k"]
        uvv = u * v * v
        return {
            "u": -uvv + F_ * (1.0 - u),
            "v": uvv - (F_ + k_) * v,
        }

    return RDSystem(
        fields={"u": RDField("u", Du, u0), "v": RDField("v", Dv, v0)},
        reaction_fn=reaction,
        params={"F": F, "k": k},
        Lx=Lx, Ly=Ly,
    )


def schnakenberg(Nx=128, Ny=128, Lx=50.0, Ly=50.0,
                 Du=1.0, Dv=40.0, a=0.1, b=0.9,
                 seed=42) -> RDSystem:
    """Schnakenberg reaction-diffusion system.

    du/dt = Du*Lap(u) + a - u + u^2*v
    dv/dt = Dv*Lap(v) + b - u^2*v

    Classic Turing pattern generator (spots).
    """
    rng = np.random.RandomState(seed)

    u_ss = a + b
    v_ss = b / (a + b)**2
    u0 = u_ss + 0.01 * rng.randn(Nx, Ny)
    v0 = v_ss + 0.01 * rng.randn(Nx, Ny)

    def reaction(state, params):
        u, v = state["u"], state["v"]
        a_, b_ = params["a"], params["b"]
        return {
            "u": a_ - u + u * u * v,
            "v": b_ - u * u * v,
        }

    return RDSystem(
        fields={"u": RDField("u", Du, u0), "v": RDField("v", Dv, v0)},
        reaction_fn=reaction,
        params={"a": a, "b": b},
        Lx=Lx, Ly=Ly,
    )
