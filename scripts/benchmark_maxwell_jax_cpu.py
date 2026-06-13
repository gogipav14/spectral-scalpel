"""
Panel (a) sidecar: JAX CPU run for the Maxwell wet-clay benchmark.

Mirrors the JAX-GPU section of scripts/benchmark_all.py with the JAX
backend pinned to CPU at process startup (JAX_PLATFORMS=cpu) so CUDA is
not initialized at all. Result is in the same units as the other panel-(a)
backends and can be appended to figure_benchmark_data.csv.
"""

from __future__ import annotations

import os
# Must be set BEFORE `import jax` to actually disable the CUDA platform
# (jax.jit(..., backend='cpu') still spins up CUDA at import otherwise).
os.environ["JAX_PLATFORMS"] = "cpu"

import math
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
assert all(d.platform == "cpu" for d in jax.devices()), \
    f"expected CPU-only, got {jax.devices()}"

# Same physics as benchmark_all.py
MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12
sigma = 0.1
eps_r = 10.0
epsilon = EPS_0 * eps_r
depth = 0.5
c_mat = 1.0 / math.sqrt(MU_0 * epsilon)
t_transit = depth / c_mat
t_end = 13 * t_transit

a_nilt = 6.7167e7
T_nilt = 1.3713e-7
N_NILT = 2048

Lz = depth * 3
Nz = 500
dz = Lz / Nz
dt_fdtd = 0.5 * dz / c_mat
Nt_fdtd = int(t_end / dt_fdtd) + 1

Nx = 64
n_modes = Nx * Nx

c1_v = (1 - sigma * dt_fdtd / (2 * epsilon)) / (1 + sigma * dt_fdtd / (2 * epsilon))
c2_v = (dt_fdtd / (epsilon * dz)) / (1 + sigma * dt_fdtd / (2 * epsilon))
c3_v = dt_fdtd / (MU_0 * dz)
mur_v = (c_mat * dt_fdtd - dz) / (c_mat * dt_fdtd + dz)
obs_idx = int(depth / dz)
tc = 3 * t_transit
tw = t_transit * 0.15

dev = jax.devices("cpu")[0]
print(f"JAX CPU device: {dev}")

# --- Scalpel pipeline (matches scalpel.core.engine) ---------------------------
kx = jnp.fft.fftfreq(Nx, 0.01) * 2 * math.pi
ky = jnp.fft.fftfreq(Nx, 0.01) * 2 * math.pi
KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
omega = jnp.arange(N_NILT, dtype=jnp.float64) * (math.pi / T_nilt)
s = a_nilt + 1j * omega
half = jnp.ones(N_NILT, dtype=jnp.float64).at[0].set(0.5)
dt_n = 2 * T_nilt / N_NILT
t_arr = jnp.arange(N_NILT, dtype=jnp.float64) * dt_n
correction = jnp.exp(a_nilt * t_arr) / T_nilt
src_np = np.zeros((Nx, Nx))
src_np[Nx // 2, Nx // 2] = 1.0
src = jnp.asarray(src_np, dtype=jnp.complex128)

# Precompute dispersion outside the timed pipeline (caching like the engine)
g2 = (MU_0 * (sigma * s[None, None, :] + epsilon * s[None, None, :] ** 2)
      - KX[:, :, None] ** 2 - KY[:, :, None] ** 2)
gz = jnp.sqrt(g2)
gz = gz * (1.0 - 2.0 * (gz.real < 0))
H = jnp.exp(-gz * depth)
H_half = H * half[None, None, :]

src = jax.device_put(src, dev)
H_half = jax.device_put(H_half, dev)
correction = jax.device_put(correction, dev)


def _scalpel(src, H_half, correction):
    S = jnp.fft.fft2(src)
    G = S[:, :, None] * H_half
    z = N_NILT * jnp.fft.ifft(G, axis=-1)
    fkt = z.real * correction[None, None, :]
    return jnp.fft.ifft2(fkt, axes=(0, 1)).real


scalpel = jax.jit(_scalpel)

# warmup
_ = scalpel(src, H_half, correction).block_until_ready()
t0 = time.perf_counter()
for _ in range(5):
    _ = scalpel(src, H_half, correction).block_until_ready()
scalpel_ms = (time.perf_counter() - t0) / 5 * 1e3
print(f"Scalpel JAX CPU: {scalpel_ms:.1f} ms")

# --- 3D Yee FDTD (batched 1D columns over n_modes transverse modes) ----------
def _fdtd():
    def step(carry, t_n):
        E, H_, Ep = carry
        em2 = E[:, -2]
        H_ = H_ + c3_v * (E[:, 1:] - E[:, :-1])
        E_int = c1_v * E[:, 1:-1] + c2_v * (H_[:, 1:] - H_[:, :-1])
        E_left = jnp.full((n_modes, 1),
                          jnp.exp(-0.5 * ((t_n - tc) / tw) ** 2))
        E_right = (Ep + mur_v * (E[:, -2] - E[:, -1]))[:, None]
        E_new = jnp.concatenate([E_left, E_int, E_right], axis=1)
        return (E_new, H_, em2), E_new[:, obs_idx]

    E0 = jnp.zeros((n_modes, Nz))
    H0 = jnp.zeros((n_modes, Nz - 1))
    Ep0 = jnp.zeros(n_modes)
    t_steps = jnp.arange(Nt_fdtd) * dt_fdtd
    _, obs = jax.lax.scan(step, (E0, H0, Ep0), t_steps)
    return obs


fdtd = jax.jit(_fdtd)

_ = fdtd().block_until_ready()
t0 = time.perf_counter()
for _ in range(3):
    _ = fdtd().block_until_ready()
fdtd_ms = (time.perf_counter() - t0) / 3 * 1e3
print(f"3D Yee FDTD JAX CPU: {fdtd_ms:.1f} ms")
print(f"Wall ratio: {fdtd_ms / scalpel_ms:.0f}x")
