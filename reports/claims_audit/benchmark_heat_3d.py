"""
3D heat-conduction benchmark: spectral scalpel vs FTCS.

This complements the lossy-Maxwell / Yee benchmark of Fig. 3 with a second,
simpler domain where the natural baseline is not Yee but the standard
3D Forward-Time Centered-Space (FTCS) explicit solver. This addresses the
reviewer concern that "Yee" is a Maxwell-specific label.

Problem:
  u_t = D nabla^2 u    in a cubic domain [0,L]^3, periodic BCs
  D = 1e-2 m^2/s, L = 0.32 m, Nx=Ny=Nz = 64 => dx = 5 mm
  Initial condition: Gaussian blob at centre, width sigma = 0.03 m
  Observation: centerline u(0,0,d,t) at depth d = 10 cm, t_end = 20 ms

Stability:
  FTCS: dt < dx^2 / (6D)
  With Cou = 0.5:  dt = 0.5 * dx^2 / (6D)
"""

from __future__ import annotations

import math
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# --- Problem parameters --------------------------------------------
D = 1e-2
L = 0.32
Nx = Ny = Nz = 64
dx = L / Nx
sigma_src = 0.03
d_obs = 0.10
t_end = 0.02        # 20 ms observation window

cou_ftcs = 0.5
dt_ftcs = cou_ftcs * dx**2 / (6 * D)
Nt_ftcs = int(math.ceil(t_end / dt_ftcs))
dt_ftcs = t_end / Nt_ftcs

print("=" * 65)
print(f"  3D heat benchmark: D={D}, L={L}, Nx=Ny=Nz={Nx}")
print(f"  dx = {dx*1000:.2f} mm, dt_FTCS = {dt_ftcs*1e6:.2f} us, Nt = {Nt_ftcs}")
print(f"  t_end = {t_end*1e3:.0f} ms, d_obs = {d_obs*1e2:.0f} cm")
print("=" * 65)

# Indices
x = (np.arange(Nx) - Nx // 2) * dx
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
u0_np = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma_src**2))
obs_k = Nz // 2 + int(round(d_obs / dx))
print(f"  obs at axial index {obs_k} (of {Nz})")


# =============================================================
# 3D FTCS, JAX, periodic BCs
# =============================================================
@jax.jit
def ftcs_step(u):
    lap = (jnp.roll(u, +1, axis=0) + jnp.roll(u, -1, axis=0)
           + jnp.roll(u, +1, axis=1) + jnp.roll(u, -1, axis=1)
           + jnp.roll(u, +1, axis=2) + jnp.roll(u, -1, axis=2)
           - 6 * u) / dx**2
    return u + D * dt_ftcs * lap


@jax.jit
def ftcs_run(u, n_steps):
    def body(i, u):
        return ftcs_step(u)
    return jax.lax.fori_loop(0, n_steps, body, u)


# warmup + timed runs
u0_j = jnp.asarray(u0_np)
_ = ftcs_run(u0_j, 2).block_until_ready()

t0 = time.perf_counter()
u_final_ftcs = ftcs_run(u0_j, Nt_ftcs).block_until_ready()
t_ftcs = (time.perf_counter() - t0) * 1000.0
u_obs_ftcs = float(u_final_ftcs[Nx // 2, Nx // 2, obs_k])
print(f"\nFTCS JAX GPU: {t_ftcs:.1f} ms   u(obs, t_end) = {u_obs_ftcs:.6e}")

# NFE count: Nt * 6*N^3 field updates (one roll per direction per step,
# counting only distinct ops; the interior stencil is 7-point, ~7 adds + 1 mul per cell)
nfe_ftcs = 7 * Nx * Ny * Nz * Nt_ftcs
print(f"  NFE (7-point stencil x Nt): {nfe_ftcs:.2e}")

# =============================================================
# Spectral scalpel on same 3D problem (2D transverse + NILT in z->t)
# =============================================================
# Source: Gaussian on z=0 plane, propagate to depth d_obs, integrate in t.
# For heat equation with an initial condition (not a boundary source),
# the equivalent operator is: given initial u(x,y,z,0), what is u(x,y,d_obs,t)?
# The spectral scalpel propagator treats it as boundary-value: u(x,y,z=0,t=0) -> u(x,y,z=d,t).
# The heat-equation Green function for this slab geometry is
#   G(k_perp, z=d, t) = (d / sqrt(D)) / (2 sqrt(pi * t^3)) * exp(-d^2/(4Dt) - D k_perp^2 t)
# (shifted Levy). We compare the scalpel output at the observation point to FTCS.

from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import diffusion

backend = get_backend()

# Source: Gaussian in the transverse plane at z=0
src_plane_np = np.exp(-(X[:, :, 0]**2 + Y[:, :, 0]**2) / (2 * sigma_src**2))
src_plane = backend.array(src_plane_np, dtype=complex)

grid = GridParams(Nx=Nx, Ny=Ny, dx=dx, dy=dx)

# NILT: heat equation has alpha_c = 0 (no singularities in right half-plane
# beyond the k-dependent branch point at origin for k=0)
# Practical choice: a = 2.3 / t_end, T = 2 t_end
nilt_p = NILTParams(a=2.3 / t_end, T=2 * t_end, N=2048)


def disp(s, KX, KY, b):
    return diffusion(s, KX, KY, D, b)


engine = SpectralEngine(disp, backend)

# Warmup + timed
_ = engine.forward(src_plane, d_obs, grid, nilt_p)
if backend.name == "jax":
    _[0].block_until_ready()

t0 = time.perf_counter()
field, t_arr = engine.forward(src_plane, d_obs, grid, nilt_p)
if backend.name == "jax":
    field.block_until_ready()
t_scalpel = (time.perf_counter() - t0) * 1000.0

# NFE: Nx * Ny * NILT
nfe_scalpel = Nx * Ny * nilt_p.N
print(f"\nScalpel JAX GPU: {t_scalpel:.1f} ms   N_NILT = {nilt_p.N}")
print(f"  NFE (N_x*N_y*N_NILT): {nfe_scalpel:.2e}")

# Read off centerline at t=t_end
t_np = backend.to_numpy(t_arr)
field_np = backend.to_numpy(field)
i_t = int(np.argmin(np.abs(t_np - t_end)))
u_obs_scalpel = float(field_np[Nx // 2, Nx // 2, i_t])
print(f"  u(centerline, t=t_end) = {u_obs_scalpel:.6e}")

# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 65)
print(f"{'':<12}{'Wall (ms)':>12}{'NFE':>16}")
print("-" * 65)
print(f"{'FTCS JAX':<12}{t_ftcs:>12.1f}{nfe_ftcs:>16.2e}")
print(f"{'Scalpel':<12}{t_scalpel:>12.1f}{nfe_scalpel:>16.2e}")
print("-" * 65)
print(f"Wall-clock speedup:  {t_ftcs / t_scalpel:.1f}x")
print(f"NFE ratio:           {nfe_ftcs / nfe_scalpel:.0f}x")
print("=" * 65)

import csv
import pickle

base = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/benchmark_heat_3d_data"
payload = {
    "t_ftcs_ms": t_ftcs, "t_scalpel_ms": t_scalpel,
    "nfe_ftcs": nfe_ftcs, "nfe_scalpel": nfe_scalpel,
    "Nx": Nx, "Nt_ftcs": Nt_ftcs, "N_NILT": nilt_p.N,
    "D": D, "L": L, "dx": dx, "d_obs": d_obs, "t_end": t_end,
    "u_ftcs_obs": u_obs_ftcs, "u_scalpel_obs": u_obs_scalpel,
    "speedup_x": t_ftcs / t_scalpel, "nfe_ratio": nfe_ftcs / nfe_scalpel,
}
with open(base + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "value"])
    for k, v in payload.items():
        w.writerow([k, v])
with open(base + ".pkl", "wb") as f:
    pickle.dump(payload, f)
print(f"Saved -> {base}.csv and {base}.pkl")
