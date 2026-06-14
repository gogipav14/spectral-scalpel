"""
JAX-CPU-only sidecar for the fractional fractional heat benchmark.

Sets JAX_PLATFORMS=cpu BEFORE importing jax so CUDA is never initialized.
Prints a single JSON line on the last line of stdout for the parent script
to parse.
"""

from __future__ import annotations

import os
# MUST come before any `import jax` to actually disable the CUDA platform.
os.environ["JAX_PLATFORMS"] = "cpu"

import json
import math
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
assert all(d.platform == "cpu" for d in jax.devices()), \
    f"expected CPU-only, got {jax.devices()}"

# ---- Problem parameters (must mirror the parent multi-backend script) ----
ALPHA = 0.7
D = 1e-2
L = 0.32
Nx = Ny = Nz = 32
dx = L / Nx
sigma_src = 0.03
d_obs = 0.10
t_end = 0.020
N_NILT = 2048

cou = 0.5
gamma_2_minus_alpha = math.gamma(2.0 - ALPHA)
dt_max = (dx**2 / (6.0 * D * gamma_2_minus_alpha)) ** (1.0 / ALPHA)
dt = cou * dt_max
N_t = int(math.ceil(t_end / dt))
dt = t_end / N_t
coef = gamma_2_minus_alpha * (dt ** ALPHA) * D

k_arr = np.arange(N_t + 1)
b_np = ((k_arr + 1.0) ** (1.0 - ALPHA) - k_arr ** (1.0 - ALPHA)).astype(np.float64)

x_grid = (np.arange(Nx) - Nx / 2 + 0.5) * dx
X, Y = np.meshgrid(x_grid, x_grid, indexing="ij")
src_plane_np = np.exp(-(X**2 + Y**2) / (2.0 * sigma_src**2)).astype(np.float64)


# ---- Scalpel JAX (CPU-only here because of JAX_PLATFORMS=cpu) ----
a_nilt = 2.3 / t_end
T_nilt = 2.0 * t_end
kx = jnp.fft.fftfreq(Nx, dx) * 2 * math.pi
ky = jnp.fft.fftfreq(Ny, dx) * 2 * math.pi
KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
omega = jnp.arange(N_NILT, dtype=jnp.float64) * (math.pi / T_nilt)
s = a_nilt + 1j * omega
half = jnp.ones(N_NILT, dtype=jnp.float64).at[0].set(0.5)
dt_n = 2.0 * T_nilt / N_NILT
t_arr = jnp.arange(N_NILT, dtype=jnp.float64) * dt_n
correction = jnp.exp(a_nilt * t_arr) / T_nilt
src = jnp.asarray(src_plane_np.astype(np.complex128))

s_alpha = s[None, None, :] ** ALPHA
g2 = s_alpha / D + KX[:, :, None]**2 + KY[:, :, None]**2
gz = jnp.sqrt(g2)
gz = gz * (1.0 - 2.0 * (gz.real < 0))
H = jnp.exp(-gz * d_obs)
H_half = H * half[None, None, :]


@jax.jit
def go(src, H_half, correction):
    S = jnp.fft.fft2(src)
    G = S[:, :, None] * H_half
    z = N_NILT * jnp.fft.ifft(G, axis=-1)
    fkt = z.real * correction[None, None, :]
    return jnp.fft.ifft2(fkt, axes=(0, 1)).real


_ = go(src, H_half, correction).block_until_ready()
t0 = time.perf_counter()
for _ in range(5):
    _ = go(src, H_half, correction).block_until_ready()
scalpel_ms = (time.perf_counter() - t0) / 5 * 1e3


# ---- FTCS+L1 JAX (CPU) ----
src_j = jnp.asarray(src_plane_np)
b_j = jnp.asarray(b_np)
u0 = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float64)
j_idx = jnp.arange(N_t)


def laplacian(u):
    lap_x = (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) - 2.0 * u) / dx**2
    lap_y = (jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1) - 2.0 * u) / dx**2
    zp = jnp.zeros_like(u[:, :, :1])
    u_zp1 = jnp.concatenate([zp, u[:, :, :-1]], axis=2)
    u_zm1 = jnp.concatenate([u[:, :, 1:], zp], axis=2)
    lap_z = (u_zp1 + u_zm1 - 2.0 * u) / dx**2
    return lap_x + lap_y + lap_z


@jax.jit
def run_l1(u0, src):
    diffs = jnp.zeros((N_t,) + u0.shape, dtype=u0.dtype)

    def step_fn(carry, n):
        u_prev, diffs = carry
        lap = laplacian(u_prev)
        valid = j_idx < (n - 1)
        safe = jnp.where(valid, n - 1 - j_idx, 0)
        w = jnp.where(valid, b_j[safe], 0.0)
        weighted_sum = jnp.tensordot(w, diffs, axes=1)
        u_new = u_prev - weighted_sum + coef * lap
        u_new = u_new.at[:, :, 0].set(src)
        diffs = diffs.at[n - 1].set(u_new - u_prev)
        return (u_new, diffs), None

    ns = jnp.arange(1, N_t + 1)
    (u_final, _), _ = jax.lax.scan(step_fn, (u0, diffs), ns)
    return u_final


_ = run_l1(u0, src_j).block_until_ready()
t0 = time.perf_counter()
res = run_l1(u0, src_j)
res.block_until_ready()
ftcs_l1_ms = (time.perf_counter() - t0) * 1e3

print("Devices:", jax.devices())
print("Scalpel JAX CPU:", scalpel_ms, "ms")
print("FTCS+L1 JAX CPU:", ftcs_l1_ms, "ms")
print(json.dumps({"scalpel_ms": scalpel_ms, "ftcs_l1_ms": ftcs_l1_ms}))
