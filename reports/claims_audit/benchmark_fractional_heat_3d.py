"""
Fractional Caputo subdiffusion benchmark, panel (b) v2.

Replaces the integer-order FTCS comparison: that test put the spectral
factorization on a problem where it had no structural advantage (vanilla
linear diffusion in a periodic box, where a closed-form Fourier propagator
e^{-D|k|^2 t} dominates everything). This test puts both methods on a
problem where Laplace structure is the natural language.

PDE
---
    ∂_t^α u = D ∇²u,    α ∈ (0, 1)    (Caputo derivative)

Geometry: 3D slab [0, Lx]² × [0, Lz], periodic in xy, Dirichlet in z.
Source: u(x, y, z=0, t) = f(x, y) · H(t), Heaviside step in time.
Far face: u(x, y, Lz, t) = 0.
Observation: u(0, 0, d_obs, t) for t in [0, t_end].

Methods
-------
1. FTCS + L1 fractional time scheme. Honest baseline. Cost O(N_t² · N³)
   from the L1 history convolution.
2. Spectral scalpel with the new `subdiffusion_caputo` dispersion.
   Cost O(N_x · N_y · N_NILT), independent of t_end.

Reference
---------
We compute scalpel at N_NILT = 4096 as a high-precision self-consistency
reference and report relative error of (a) FTCS+L1 and (b) scalpel at
N_NILT = 2048 against it. An analytical Mittag-Leffler / Wright-function
reference for this slab geometry is non-elementary; self-consistency at
two NILT resolutions is the same approach used in the recoverability-bound
audit.
"""

from __future__ import annotations

import csv
import math
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Problem parameters (kept at 32^3 so the L1 history fits comfortably in 8 GB
# GPU memory; with α = 0.7 and Cou = 0.5 the L1-stable dt gives ~420 steps for
# 100 ms, history footprint ~110 MB, total work ~3e9 ops, a couple seconds on
# RTX 5060)
# -----------------------------------------------------------------------------
ALPHA = 0.7
D = 1e-2
L = 0.32
Nx = Ny = Nz = 32
dx = L / Nx
sigma_src = 0.03
d_obs = 0.10
t_end = 0.10

cou = 0.5
gamma_2_minus_alpha = math.gamma(2.0 - ALPHA)
dt_stab_alpha = dx**2 / (6.0 * D * gamma_2_minus_alpha)
dt_max = dt_stab_alpha ** (1.0 / ALPHA)
dt = cou * dt_max
N_t = int(math.ceil(t_end / dt))
dt = t_end / N_t

print("=" * 70)
print(f"  Fractional subdiffusion benchmark, α = {ALPHA}")
print(f"  D = {D}, L = {L}, N = {Nx}, dx = {dx*1000:.2f} mm")
print(f"  t_end = {t_end*1e3:.0f} ms, dt = {dt*1e6:.1f} us, N_t = {N_t}")
print(f"  d_obs = {d_obs*1e2:.0f} cm")
print(f"  L1 history footprint: {N_t * Nx * Ny * Nz * 8 / 2**20:.1f} MB")
print("=" * 70)

# -----------------------------------------------------------------------------
# Source: Gaussian on the z=0 plane, Heaviside in time
# -----------------------------------------------------------------------------
x_grid = (np.arange(Nx) - Nx / 2 + 0.5) * dx
X, Y = np.meshgrid(x_grid, x_grid, indexing="ij")
src_plane_np = np.exp(-(X**2 + Y**2) / (2.0 * sigma_src**2)).astype(np.float64)
src_plane = jnp.asarray(src_plane_np)

obs_k = int(round(d_obs / dx))
print(f"  obs at axial index {obs_k} of {Nz}")

# -----------------------------------------------------------------------------
# 3D Laplacian: periodic in xy, Dirichlet (zero) in z (boundary applied
# separately at z=0 by overwriting the source plane each step; z=Lz held at 0)
# -----------------------------------------------------------------------------


@jax.jit
def laplacian(u):
    lap_x = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) - 2.0 * u) / dx**2
    lap_y = (jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 2.0 * u) / dx**2
    zero_pad = jnp.zeros_like(u[:, :, :1])
    u_zp1 = jnp.concatenate([zero_pad, u[:, :, :-1]], axis=2)  # u(z-dz)
    u_zm1 = jnp.concatenate([u[:, :, 1:], zero_pad], axis=2)   # u(z+dz)
    lap_z = (u_zp1 + u_zm1 - 2.0 * u) / dx**2
    return lap_x + lap_y + lap_z


# -----------------------------------------------------------------------------
# L1 scheme weights b_k = (k+1)^(1-α) - k^(1-α), k = 0, 1, ..., N_t
# (b_0 = 1, b_k decays like α k^(-α) for large k)
# -----------------------------------------------------------------------------
k_arr = np.arange(N_t + 1)
b = ((k_arr + 1.0) ** (1.0 - ALPHA) - k_arr ** (1.0 - ALPHA)).astype(np.float64)
b_jax = jnp.asarray(b)

coef = gamma_2_minus_alpha * (dt ** ALPHA) * D


def make_l1_scan_fn(N_t_static):
    """Builds a JIT-compiled scan over N_t_static fractional time steps.

    Returns the centerline trace u(Nx/2, Ny/2, :, t_n) for n = 1, ..., N_t.
    """
    j_idx = jnp.arange(N_t_static)

    @jax.jit
    def run(u0, src_plane):
        diffs = jnp.zeros((N_t_static,) + u0.shape, dtype=u0.dtype)

        def step_fn(carry, n):
            u_prev, diffs = carry
            lap = laplacian(u_prev)
            valid = j_idx < (n - 1)
            safe = jnp.where(valid, n - 1 - j_idx, 0)
            w = jnp.where(valid, b_jax[safe], 0.0)
            # Weighted sum of all diffs slots (masked).
            weighted_sum = jnp.tensordot(w, diffs, axes=1)
            u_new = u_prev - weighted_sum + coef * lap
            u_new = u_new.at[:, :, 0].set(src_plane)  # Heaviside-step BC
            diffs = diffs.at[n - 1].set(u_new - u_prev)
            return (u_new, diffs), u_new[Nx // 2, Ny // 2, :]

        ns = jnp.arange(1, N_t_static + 1)
        (u_final, _), trace = jax.lax.scan(step_fn, (u0, diffs), ns)
        return u_final, trace

    return run


# Initial condition: u(x, y, z, 0) = 0 in interior, src on z=0 plane (the step
# turns on at t=0+; the L1 scheme treats u^0 as the pre-step state, so we set
# u^0 to zero and let the BC overwrite z=0 from step n=1 onwards). Equivalently
# we could put src on u^0[..., 0]; the difference shows up only in the very
# first time bin and washes out within ~2 steps.
u0 = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float64)

print("\n--- FTCS + L1 ---")
l1_run = make_l1_scan_fn(N_t)
# warmup (compile)
_, _ = l1_run(u0, src_plane)
jax.block_until_ready(_)

t0 = time.perf_counter()
u_final_l1, trace_l1 = l1_run(u0, src_plane)
jax.block_until_ready(u_final_l1)
t_l1_ms = (time.perf_counter() - t0) * 1000.0
trace_l1_np = np.asarray(trace_l1)  # shape (N_t, Nz)
u_obs_l1_t = trace_l1_np[:, obs_k]
print(f"  wallclock: {t_l1_ms:.1f} ms")
print(f"  u(obs, t_end) = {u_obs_l1_t[-1]:.6e}")

# NFE accounting: per step, 1 Laplacian (~7 N^3) + 1 weighted sum (~ N_t * N^3)
# integrated: 7 N^3 N_t + N^3 sum_{n=1}^{N_t} n  =  7 N^3 N_t + N^3 N_t(N_t+1)/2
nfe_l1 = 7 * Nx * Ny * Nz * N_t + Nx * Ny * Nz * N_t * (N_t + 1) // 2
print(f"  NFE: {nfe_l1:.2e}")

# -----------------------------------------------------------------------------
# Scalpel with new fractional Caputo dispersion
# -----------------------------------------------------------------------------
from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import subdiffusion_caputo

backend = get_backend()
src_plane_c = backend.array(src_plane_np, dtype=complex)
grid = GridParams(Nx=Nx, Ny=Ny, dx=dx, dy=dx)


def disp(s, KX, KY, b_):
    return subdiffusion_caputo(s, KX, KY, D, ALPHA, b_)


engine = SpectralEngine(disp, backend)


def run_scalpel(N_NILT):
    nilt_p = NILTParams(a=2.3 / t_end, T=2.0 * t_end, N=N_NILT)
    # warmup
    field, t_arr = engine.forward(src_plane_c, d_obs, grid, nilt_p)
    if backend.name == "jax":
        field.block_until_ready()
    t0 = time.perf_counter()
    field, t_arr = engine.forward(src_plane_c, d_obs, grid, nilt_p)
    if backend.name == "jax":
        field.block_until_ready()
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, field, t_arr, nilt_p


def to_step_response(field_impulse, t_arr):
    """Scalpel returns u for delta-in-time source. Convert to step response by
    cumulative trapezoidal integration along the time axis."""
    f = backend.to_numpy(field_impulse)
    t = backend.to_numpy(t_arr)
    dt_n = np.diff(t, prepend=t[0])
    return np.cumsum(f * dt_n[None, None, :], axis=2)


print("\n--- Scalpel @ N_NILT = 2048 (impulse, then cumulative integration to step) ---")
t_sc_ms, field_sc, t_arr_sc, _ = run_scalpel(2048)
step_sc = to_step_response(field_sc, t_arr_sc)
t_arr_np = backend.to_numpy(t_arr_sc)
i_t_end = int(np.argmin(np.abs(t_arr_np - t_end)))
u_obs_sc = float(step_sc[Nx // 2, Ny // 2, i_t_end])
print(f"  wallclock: {t_sc_ms:.1f} ms")
print(f"  u(obs, t_end) = {u_obs_sc:.6e}")
nfe_sc = Nx * Ny * 2048
print(f"  NFE (Nx*Ny*N_NILT): {nfe_sc:.2e}")

print("\n--- Scalpel @ N_NILT = 4096 (high-precision self-consistency reference) ---")
t_sc_ref_ms, field_sc_ref, t_arr_sc_ref, _ = run_scalpel(4096)
step_sc_ref = to_step_response(field_sc_ref, t_arr_sc_ref)
t_arr_ref_np = backend.to_numpy(t_arr_sc_ref)
i_t_end_ref = int(np.argmin(np.abs(t_arr_ref_np - t_end)))
u_obs_ref = float(step_sc_ref[Nx // 2, Ny // 2, i_t_end_ref])
print(f"  wallclock: {t_sc_ref_ms:.1f} ms")
print(f"  u(obs, t_end) = {u_obs_ref:.6e}")

# -----------------------------------------------------------------------------
# Accuracy
# -----------------------------------------------------------------------------
err_l1 = abs(u_obs_l1_t[-1] - u_obs_ref) / max(abs(u_obs_ref), 1e-30)
err_sc = abs(u_obs_sc - u_obs_ref) / max(abs(u_obs_ref), 1e-30)

print("\n" + "=" * 70)
print(f"{'method':<24}{'wall (ms)':>12}{'NFE':>14}{'rel err':>14}")
print("-" * 70)
print(f"{'FTCS+L1 (JAX GPU)':<24}{t_l1_ms:>12.1f}{nfe_l1:>14.2e}{err_l1:>14.2e}")
print(f"{'Scalpel N_NILT=2048':<24}{t_sc_ms:>12.1f}{nfe_sc:>14.2e}{err_sc:>14.2e}")
print(f"{'Scalpel N_NILT=4096 (ref)':<24}{t_sc_ref_ms:>12.1f}{Nx*Ny*4096:>14.2e}{0.0:>14.2e}")
print("-" * 70)
if t_sc_ms > 0:
    print(f"Wallclock speedup, scalpel(2048) over FTCS+L1: {t_l1_ms / t_sc_ms:.1f}x")
print("=" * 70)

# -----------------------------------------------------------------------------
# CSV outputs (preferred over pickle for downstream tokenization)
# -----------------------------------------------------------------------------
import os
out_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(out_dir, "benchmark_fractional_heat_3d_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "wall_ms", "nfe", "rel_err_vs_ref",
                "alpha", "Nx", "Nz", "t_end", "d_obs", "D", "L", "dx",
                "N_t", "N_NILT"])
    w.writerow(["FTCS+L1 JAX GPU", t_l1_ms, nfe_l1, err_l1,
                ALPHA, Nx, Nz, t_end, d_obs, D, L, dx, N_t, ""])
    w.writerow(["Scalpel JAX GPU", t_sc_ms, nfe_sc, err_sc,
                ALPHA, Nx, Nz, t_end, d_obs, D, L, dx, "", 2048])
    w.writerow(["Scalpel JAX GPU (ref)", t_sc_ref_ms, Nx * Ny * 4096, 0.0,
                ALPHA, Nx, Nz, t_end, d_obs, D, L, dx, "", 4096])
print(f"Saved -> {csv_path}")

trace_csv = os.path.join(out_dir, "benchmark_fractional_heat_3d_trace.csv")
n_trace = min(len(u_obs_l1_t), 256)
stride = max(1, len(u_obs_l1_t) // n_trace)
t_l1_arr = (np.arange(N_t) + 1) * dt
with open(trace_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "u_l1", "u_scalpel_step"])
    sc_interp = np.interp(t_l1_arr, t_arr_np,
                          step_sc[Nx // 2, Ny // 2, :])
    for i in range(0, len(t_l1_arr), stride):
        w.writerow([t_l1_arr[i], float(u_obs_l1_t[i]), float(sc_interp[i])])
print(f"Saved -> {trace_csv}")
