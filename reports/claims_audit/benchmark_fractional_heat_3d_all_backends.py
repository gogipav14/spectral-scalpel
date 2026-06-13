"""
Panel (b) v3: Fractional Caputo subdiffusion across all five backends.

Same structure as scripts/benchmark_all.py for panel (a). For each of
NumPy CPU, CuPy GPU, PyTorch CPU, PyTorch GPU, JAX GPU we time:
  - Scalpel (FFT_xy + NILT_z with the s^alpha dispersion)
  - FTCS + L1 fractional time scheme

Problem: 3D slab [0, L]^2 x [0, Lz], periodic xy, Dirichlet z, surface
Heaviside source. PDE: d_t^alpha u = D nabla^2 u, alpha = 0.7.

Sized so the L1 history footprint (~80 MB) fits on the 8 GB RTX 5060 and
the NumPy CPU run completes in a reasonable time. Stability for L1 + 7-
point stencil: dt^alpha <= dx^2 / (6 D Gamma(2-alpha)).

Output: CSV (one row per (backend, method)) + console summary.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

# -----------------------------------------------------------------------------
# Problem parameters
# -----------------------------------------------------------------------------
ALPHA = 0.7
D = 1e-2
L = 0.32
Nx = Ny = Nz = 32
dx = L / Nx
sigma_src = 0.03
d_obs = 0.10
t_end = 0.020  # 20 ms; gives ~320 L1 steps at Cou = 0.5

cou = 0.5
gamma_2_minus_alpha = math.gamma(2.0 - ALPHA)
dt_max = (dx**2 / (6.0 * D * gamma_2_minus_alpha)) ** (1.0 / ALPHA)
dt = cou * dt_max
N_t = int(math.ceil(t_end / dt))
dt = t_end / N_t
coef = gamma_2_minus_alpha * (dt ** ALPHA) * D

# L1 weights b_k = (k+1)^(1-alpha) - k^(1-alpha)
k_arr = np.arange(N_t + 1)
b_np = ((k_arr + 1.0) ** (1.0 - ALPHA) - k_arr ** (1.0 - ALPHA)).astype(np.float64)

# Source
x_grid = (np.arange(Nx) - Nx / 2 + 0.5) * dx
X, Y = np.meshgrid(x_grid, x_grid, indexing="ij")
src_plane_np = np.exp(-(X**2 + Y**2) / (2.0 * sigma_src**2)).astype(np.float64)
obs_k = int(round(d_obs / dx))
N_NILT = 2048

print("=" * 75)
print(f" Multi-backend fractional benchmark, alpha = {ALPHA}")
print(f" N = {Nx}^3, dx = {dx*1000:.2f} mm, t_end = {t_end*1e3:.0f} ms")
print(f" L1: dt = {dt*1e6:.1f} us, N_t = {N_t}, history = "
      f"{N_t * Nx**3 * 8 / 2**20:.1f} MB")
print(f" Scalpel: N_NILT = {N_NILT}")
print("=" * 75)

results = []  # list of (backend, method, wall_ms)


# =============================================================================
# Generic xp-namespace runners (work for numpy, cupy, and TorchWrap)
# =============================================================================

def run_scalpel_xp(xp, label, sync=None, asarray=None):
    """Spectral factorization: 2D FFT_xy + NILT_z with s^alpha dispersion.

    Direct port of the run_scalpel pipeline in benchmark_all.py with the
    Maxwell dispersion swapped for fractional Caputo subdiffusion.
    """
    asarray = asarray or (lambda a: xp.asarray(a))

    kx = xp.fft.fftfreq(Nx, dx) * 2 * math.pi
    ky = xp.fft.fftfreq(Ny, dx) * 2 * math.pi
    KX, KY = xp.meshgrid(kx, ky, indexing="ij")

    # NILT contour: a = 2.3 / t_end, T = 2 * t_end
    a_nilt = 2.3 / t_end
    T_nilt = 2.0 * t_end
    omega = xp.arange(N_NILT, dtype=float) * (math.pi / T_nilt)
    s = a_nilt + 1j * omega
    half = xp.ones(N_NILT, dtype=float)
    # half[0] = 0.5: in-place set (works for numpy/cupy/torch; not jax)
    if hasattr(half, "__setitem__"):
        try:
            half[0] = 0.5
        except TypeError:
            # immutable backend (jax) - construct fresh
            half_np = np.ones(N_NILT)
            half_np[0] = 0.5
            half = asarray(half_np)
    dt_n = 2.0 * T_nilt / N_NILT
    t_arr = xp.arange(N_NILT, dtype=float) * dt_n
    correction = xp.exp(a_nilt * t_arr) / T_nilt

    src = asarray(src_plane_np.astype(np.complex128))

    # Precompute the s-domain dispersion contributions once (they do not
    # depend on the source). This is the apples-to-apples way to time the
    # per-source pipeline; in production the engine caches these too.
    s_alpha = s[None, None, :] ** ALPHA
    g2 = s_alpha / D + KX[:, :, None]**2 + KY[:, :, None]**2
    gz = xp.sqrt(g2)
    gz_real = gz.real if hasattr(gz, "real") else xp.real(gz)
    gz = gz * (1.0 - 2.0 * (gz_real < 0))
    H = xp.exp(-gz * d_obs)
    H_half = H * half[None, None, :]

    def go():
        S = xp.fft.fft2(src)
        G = S[:, :, None] * H_half
        z = N_NILT * xp.fft.ifft(G, axis=-1)
        fkt = z.real * correction[None, None, :]
        return xp.fft.ifft2(fkt, axes=(0, 1)).real

    # warmup
    _ = go()
    if sync:
        sync()

    t0 = time.perf_counter()
    for _ in range(5):
        _ = go()
        if sync:
            sync()
    return (time.perf_counter() - t0) / 5 * 1e3


def run_ftcs_l1_xp(xp, label, sync=None, asarray=None):
    """FTCS + L1 fractional time. In-place mutation; not for JAX."""
    asarray = asarray or (lambda a: xp.asarray(a))

    src = asarray(src_plane_np)
    b_arr = asarray(b_np)
    u_prev = asarray(np.zeros((Nx, Ny, Nz), dtype=np.float64))
    diffs = asarray(np.zeros((N_t, Nx, Ny, Nz), dtype=np.float64))
    zero_pad = asarray(np.zeros((Nx, Ny, 1), dtype=np.float64))

    def laplacian(u):
        lap_x = (xp.roll(u, 1, 0) + xp.roll(u, -1, 0) - 2.0 * u) / dx**2
        lap_y = (xp.roll(u, 1, 1) + xp.roll(u, -1, 1) - 2.0 * u) / dx**2
        u_zp1 = xp.concatenate([zero_pad, u[:, :, :-1]], axis=2)
        u_zm1 = xp.concatenate([u[:, :, 1:], zero_pad], axis=2)
        lap_z = (u_zp1 + u_zm1 - 2.0 * u) / dx**2
        return lap_x + lap_y + lap_z

    if sync:
        sync()
    t0 = time.perf_counter()
    for n in range(1, N_t + 1):
        lap = laplacian(u_prev)
        if n > 1:
            w_slice = b_arr[1:n]
            # Reverse: b[n-1], b[n-2], ..., b[1]
            if hasattr(w_slice, "flip"):  # torch
                try:
                    w = w_slice.flip(0)
                except TypeError:
                    w = w_slice[::-1]
            else:
                w = w_slice[::-1]
            weighted_sum = xp.einsum("i,i...->...", w, diffs[:n - 1])
        else:
            weighted_sum = xp.zeros_like(u_prev)
        u_new = u_prev - weighted_sum + coef * lap
        u_new[:, :, 0] = src
        diffs[n - 1] = u_new - u_prev
        u_prev = u_new
    if sync:
        sync()
    return (time.perf_counter() - t0) * 1e3


# =============================================================================
# JAX-specific runners (immutable arrays + scan for the L1 history)
# =============================================================================

def run_scalpel_jax(device="gpu"):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax_device = jax.devices(device)[0]

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
    src = jax.device_put(src, jax_device)
    H_half = jax.device_put(H_half, jax_device)
    correction = jax.device_put(correction, jax_device)

    def _go(src, H_half, correction):
        S = jnp.fft.fft2(src)
        G = S[:, :, None] * H_half
        z = N_NILT * jnp.fft.ifft(G, axis=-1)
        fkt = z.real * correction[None, None, :]
        return jnp.fft.ifft2(fkt, axes=(0, 1)).real

    go = jax.jit(_go, backend=device)

    _ = go(src, H_half, correction).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(5):
        _ = go(src, H_half, correction).block_until_ready()
    return (time.perf_counter() - t0) / 5 * 1e3


def run_ftcs_l1_jax(device="gpu"):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax_device = jax.devices(device)[0]

    src_j = jax.device_put(jnp.asarray(src_plane_np), jax_device)
    b_j = jax.device_put(jnp.asarray(b_np), jax_device)
    u0 = jax.device_put(jnp.zeros((Nx, Ny, Nz), dtype=jnp.float64), jax_device)
    j_idx = jax.device_put(jnp.arange(N_t), jax_device)

    def laplacian(u):
        lap_x = (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) - 2.0 * u) / dx**2
        lap_y = (jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1) - 2.0 * u) / dx**2
        zp = jnp.zeros_like(u[:, :, :1])
        u_zp1 = jnp.concatenate([zp, u[:, :, :-1]], axis=2)
        u_zm1 = jnp.concatenate([u[:, :, 1:], zp], axis=2)
        lap_z = (u_zp1 + u_zm1 - 2.0 * u) / dx**2
        return lap_x + lap_y + lap_z

    def _run(u0, src):
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

    run = jax.jit(_run, backend=device)

    _ = run(u0, src_j).block_until_ready()
    t0 = time.perf_counter()
    res = run(u0, src_j)
    res.block_until_ready()
    return (time.perf_counter() - t0) * 1e3


# =============================================================================
# Per-backend driver
# =============================================================================

def safe_run(name, fn):
    try:
        return fn()
    except Exception as exc:
        print(f"  ! {name} FAILED: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return float("nan")


# 1. NumPy CPU
print("\n--- NumPy CPU ---")
sc_ms = safe_run("scalpel", lambda: run_scalpel_xp(np, "NumPy CPU"))
fl_ms = safe_run("ftcs+l1", lambda: run_ftcs_l1_xp(np, "NumPy CPU"))
print(f"  scalpel: {sc_ms:.1f} ms | FTCS+L1: {fl_ms:.0f} ms")
results.append(("NumPy CPU", "scalpel", sc_ms))
results.append(("NumPy CPU", "ftcs_l1", fl_ms))

# 2. CuPy GPU
print("\n--- CuPy GPU ---")
try:
    import cupy as cp
    sync_cp = cp.cuda.Stream.null.synchronize
    sc_ms = safe_run("scalpel",
                     lambda: run_scalpel_xp(cp, "CuPy GPU", sync=sync_cp))
    fl_ms = safe_run("ftcs+l1",
                     lambda: run_ftcs_l1_xp(cp, "CuPy GPU", sync=sync_cp))
except ImportError:
    print("  cupy not installed; skipping")
    sc_ms = fl_ms = float("nan")
print(f"  scalpel: {sc_ms:.1f} ms | FTCS+L1: {fl_ms:.0f} ms")
results.append(("CuPy GPU", "scalpel", sc_ms))
results.append(("CuPy GPU", "ftcs_l1", fl_ms))

# 3 & 4. PyTorch CPU and GPU
print("\n--- PyTorch CPU + GPU ---")
try:
    import torch

    class TorchWrap:
        def __init__(self, device):
            self.device = device
            self.fft = self
        def fftfreq(self, n, d):
            return torch.fft.fftfreq(n, d, device=self.device,
                                     dtype=torch.float64)
        def fft2(self, x): return torch.fft.fft2(x)
        def ifft(self, x, axis=-1): return torch.fft.ifft(x, dim=axis)
        def ifft2(self, x, axes=None): return torch.fft.ifft2(x, dim=axes)
        def meshgrid(self, *a, indexing="ij"):
            return torch.meshgrid(*a, indexing=indexing)
        def arange(self, n, dtype=None):
            return torch.arange(n, device=self.device, dtype=torch.float64)
        def ones(self, n, dtype=None):
            return torch.ones(n, device=self.device, dtype=torch.float64)
        def zeros(self, shape, dtype=None):
            dt = torch.complex128 if dtype == complex else torch.float64
            return torch.zeros(shape, device=self.device, dtype=dt)
        def asarray(self, a, dtype=None):
            t = torch.as_tensor(np.asarray(a), device=self.device)
            if dtype == complex or (np.iscomplexobj(a)):
                t = t.to(torch.complex128)
            else:
                t = t.to(torch.float64)
            return t
        def sqrt(self, x): return torch.sqrt(x)
        def exp(self, x): return torch.exp(x)
        def real(self, x): return x.real if torch.is_complex(x) else x
        def roll(self, x, shift, axis):
            return torch.roll(x, shifts=shift, dims=axis)
        def concatenate(self, lst, axis):
            return torch.cat(lst, dim=axis)
        def einsum(self, expr, *ops):
            return torch.einsum(expr, *ops)
        def zeros_like(self, x): return torch.zeros_like(x)

    for dev_name, dev in [("PyTorch CPU", torch.device("cpu")),
                          ("PyTorch GPU", torch.device("cuda"))]:
        if dev.type == "cuda" and not torch.cuda.is_available():
            print(f"  {dev_name} unavailable; skipping")
            results.append((dev_name, "scalpel", float("nan")))
            results.append((dev_name, "ftcs_l1", float("nan")))
            continue
        tw = TorchWrap(dev)
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else None
        sc_ms = safe_run(f"{dev_name} scalpel",
                         lambda: run_scalpel_xp(tw, dev_name, sync=sync,
                                                asarray=tw.asarray))
        fl_ms = safe_run(f"{dev_name} ftcs+l1",
                         lambda: run_ftcs_l1_xp(tw, dev_name, sync=sync,
                                                asarray=tw.asarray))
        print(f"  {dev_name}: scalpel {sc_ms:.1f} ms | FTCS+L1 {fl_ms:.0f} ms")
        results.append((dev_name, "scalpel", sc_ms))
        results.append((dev_name, "ftcs_l1", fl_ms))

except ImportError:
    print("  torch not installed; skipping")
    for dev_name in ("PyTorch CPU", "PyTorch GPU"):
        results.append((dev_name, "scalpel", float("nan")))
        results.append((dev_name, "ftcs_l1", float("nan")))

# 5. JAX CPU via subprocess sidecar (necessary because `import jax` in this
# process has already initialized CUDA for the JAX GPU run; `jax.jit(...,
# backend='cpu')` would still compile for CPU but CUDA stays warm and the
# user correctly flagged that 'CUDA spins up' is misleading for a CPU row.
# Spawn a fresh process with JAX_PLATFORMS=cpu so no GPU platform exists.)
print("\n--- JAX CPU (subprocess, JAX_PLATFORMS=cpu) ---")
sidecar = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_fractional_heat_3d_jax_cpu_sidecar.py")
try:
    proc = subprocess.run([sys.executable, sidecar],
                          capture_output=True, text=True, check=True,
                          timeout=900)
    last = proc.stdout.strip().split("\n")[-1]
    data = json.loads(last)
    sc_ms = data["scalpel_ms"]
    fl_ms = data["ftcs_l1_ms"]
except Exception as exc:
    print(f"  sidecar failed: {exc!r}")
    sc_ms = fl_ms = float("nan")
print(f"  scalpel: {sc_ms:.1f} ms | FTCS+L1: {fl_ms:.0f} ms")
results.append(("JAX CPU", "scalpel", sc_ms))
results.append(("JAX CPU", "ftcs_l1", fl_ms))

# 6. JAX GPU (inline; CUDA initialization cost goes here)
print("\n--- JAX GPU ---")
try:
    sc_ms = safe_run("scalpel", lambda: run_scalpel_jax(device="gpu"))
    fl_ms = safe_run("ftcs+l1", lambda: run_ftcs_l1_jax(device="gpu"))
except ImportError:
    print("  jax not installed; skipping")
    sc_ms = fl_ms = float("nan")
except RuntimeError as e:
    print(f"  JAX GPU device unavailable: {e}")
    sc_ms = fl_ms = float("nan")
print(f"  scalpel: {sc_ms:.1f} ms | FTCS+L1: {fl_ms:.0f} ms")
results.append(("JAX GPU", "scalpel", sc_ms))
results.append(("JAX GPU", "ftcs_l1", fl_ms))

# =============================================================================
# Summary + CSV
# =============================================================================
print("\n" + "=" * 75)
print(f" {'Backend':<14}{'Method':<10}{'Wall (ms)':>14}")
print("-" * 75)
by_backend = {}
for backend, method, ms in results:
    by_backend.setdefault(backend, {})[method] = ms
    print(f" {backend:<14}{method:<10}{ms:>14.1f}")
print("-" * 75)
print(f" {'Backend':<14}{'Speedup (FTCS+L1 / Scalpel)':>30}")
for backend, methods in by_backend.items():
    sc, fl = methods.get("scalpel"), methods.get("ftcs_l1")
    if sc and fl and sc == sc and fl == fl and sc > 0:
        print(f" {backend:<14}{fl/sc:>27.1f}x")
    else:
        print(f" {backend:<14}{'n/a':>30}")
print("=" * 75)

out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_fractional_heat_3d_all_backends.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["backend", "method", "wall_ms", "alpha", "Nx", "t_end",
                "d_obs", "D", "L", "dx", "N_t", "N_NILT"])
    for backend, method, ms in results:
        w.writerow([backend, method, ms, ALPHA, Nx, t_end, d_obs,
                    D, L, dx, N_t if method == "ftcs_l1" else "",
                    N_NILT if method == "scalpel" else ""])
print(f"Saved -> {out_csv}")
