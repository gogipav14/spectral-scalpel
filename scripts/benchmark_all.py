"""
Complete benchmark: Scalpel vs FDTD across all backends.
Same problem, same observable, same hardware.

Backends: NumPy (CPU), CuPy (GPU), PyTorch CPU, PyTorch GPU, JAX GPU.
"""

import time
import math
import numpy as np

MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12

sigma = 0.1
eps_r = 10.0
epsilon = EPS_0 * eps_r
depth = 0.5
c_mat = 1.0 / math.sqrt(MU_0 * epsilon)
t_transit = depth / c_mat
t_end = 13 * t_transit

a_nilt = 6.7167e+07
T_nilt = 1.3713e-07
N_NILT = 2048

Lz = depth * 3
Nz = 500
dz = Lz / Nz
dt_fdtd = 0.5 * dz / c_mat
Nt_fdtd = int(t_end / dt_fdtd) + 1

Nx = 64
n_modes = Nx * Nx

c1_v = (1 - sigma*dt_fdtd/(2*epsilon)) / (1 + sigma*dt_fdtd/(2*epsilon))
c2_v = (dt_fdtd/(epsilon*dz)) / (1 + sigma*dt_fdtd/(2*epsilon))
c3_v = dt_fdtd / (MU_0 * dz)
mur_v = (c_mat*dt_fdtd - dz) / (c_mat*dt_fdtd + dz)
obs_idx = int(depth / dz)
tc = 3 * t_transit
tw = t_transit * 0.15

nfe_scalpel = n_modes * N_NILT
nfe_fdtd = n_modes * 2 * Nz * Nt_fdtd

print("=" * 75)
print(f" ALL-BACKEND BENCHMARK: {Nx}x{Nx}, wet clay, d={depth}m")
print(f" Scalpel: N_NILT={N_NILT} | FDTD: Nz={Nz}, Nt={Nt_fdtd}")
print(f" NFE ratio: {nfe_fdtd/nfe_scalpel:.0f}x")
print("=" * 75)

results = []


# ── Generic scalpel/FDTD using array library xp ────────────────────

def run_scalpel(xp, label, sync=None):
    """Run scalpel using array library xp (numpy, cupy, torch-like)."""
    kx = xp.fft.fftfreq(Nx, 0.01) * 2 * math.pi
    ky = xp.fft.fftfreq(Nx, 0.01) * 2 * math.pi
    KX, KY = xp.meshgrid(kx, ky, indexing='ij')
    omega = xp.arange(N_NILT, dtype=float) * (math.pi / T_nilt)
    s = a_nilt + 1j * omega
    half = xp.ones(N_NILT, dtype=float)
    half[0] = 0.5
    dt_n = 2 * T_nilt / N_NILT
    t_arr = xp.arange(N_NILT, dtype=float) * dt_n
    correction = xp.exp(a_nilt * t_arr) / T_nilt

    src = xp.zeros((Nx, Nx), dtype=complex)
    src[Nx//2, Nx//2] = 1.0

    def go():
        S = xp.fft.fft2(src)
        g2 = (MU_0 * (sigma * s[None, None, :] + epsilon * s[None, None, :]**2)
              - KX[:, :, None]**2 - KY[:, :, None]**2)
        gz = xp.sqrt(g2)
        gz = gz * (1.0 - 2.0 * (gz.real < 0))
        H = xp.exp(-gz * depth)
        G = S[:, :, None] * H * half[None, None, :]
        z = N_NILT * xp.fft.ifft(G, axis=-1)
        fkt = z.real * correction[None, None, :]
        return xp.fft.ifft2(fkt, axes=(0, 1)).real

    # warmup
    _ = go()
    if sync: sync()

    t0 = time.perf_counter()
    for _ in range(20):
        _ = go()
        if sync: sync()
    ms = (time.perf_counter() - t0) / 20 * 1e3
    return ms


def run_fdtd_batched(xp, label, sync=None):
    """Batched FDTD: n_modes 1D columns in parallel using xp."""
    E = xp.zeros((n_modes, Nz), dtype=float)
    H = xp.zeros((n_modes, Nz - 1), dtype=float)
    E_prev = xp.zeros(n_modes, dtype=float)

    def go():
        nonlocal E, H, E_prev
        E[:] = 0; H[:] = 0; E_prev[:] = 0
        for n in range(Nt_fdtd):
            t_n = n * dt_fdtd
            em2 = E[:, -2].copy() if hasattr(E, 'copy') else E[:, -2].clone()
            H = H + c3_v * (E[:, 1:] - E[:, :-1])
            E[:, 1:-1] = c1_v * E[:, 1:-1] + c2_v * (H[:, 1:] - H[:, :-1])
            E[:, 0] = math.exp(-0.5 * ((t_n - tc) / tw)**2)
            E[:, -1] = E_prev + mur_v * (E[:, -2] - E[:, -1])
            E_prev = em2
        return E[:, obs_idx]

    # warmup
    _ = go()
    if sync: sync()

    t0 = time.perf_counter()
    _ = go()
    if sync: sync()
    ms = (time.perf_counter() - t0) * 1e3
    return ms


# ═══════════════════════════════════════════════════════════════════
#  1. NumPy CPU
# ═══════════════════════════════════════════════════════════════════
print("\n─── NumPy CPU ───")
s_ms = run_scalpel(np, "NumPy CPU")
f_ms = run_fdtd_batched(np, "NumPy CPU")
print(f"  Scalpel: {s_ms:.1f} ms | FDTD: {f_ms:.0f} ms | Wall ratio: {f_ms/s_ms:.0f}x")
results.append(("NumPy CPU", s_ms, f_ms))

# ═══════════════════════════════════════════════════════════════════
#  2. CuPy GPU
# ═══════════════════════════════════════════════════════════════════
print("\n─── CuPy GPU ───")
import cupy as cp

s_ms = run_scalpel(cp, "CuPy GPU", sync=cp.cuda.Stream.null.synchronize)
f_ms = run_fdtd_batched(cp, "CuPy GPU", sync=cp.cuda.Stream.null.synchronize)
print(f"  Scalpel: {s_ms:.1f} ms | FDTD: {f_ms:.0f} ms | Wall ratio: {f_ms/s_ms:.0f}x")
results.append(("CuPy GPU", s_ms, f_ms))

# ═══════════════════════════════════════════════════════════════════
#  3. PyTorch CPU
# ═══════════════════════════════════════════════════════════════════
print("\n─── PyTorch CPU ───")
import torch

class TorchWrap:
    """Minimal wrapper to make torch look like numpy for the benchmark."""
    def __init__(self, device):
        self.device = device
        self.fft = self
    def fftfreq(self, n, d):
        return torch.fft.fftfreq(n, d, device=self.device, dtype=torch.float64)
    def fft2(self, x):
        return torch.fft.fft2(x)
    def ifft(self, x, axis=-1):
        return torch.fft.ifft(x, dim=axis)
    def ifft2(self, x, axes=None):
        return torch.fft.ifft2(x, dim=axes)
    def meshgrid(self, *a, indexing='ij'):
        return torch.meshgrid(*a, indexing=indexing)
    def arange(self, n, dtype=None):
        return torch.arange(n, device=self.device, dtype=torch.float64)
    def ones(self, n, dtype=None):
        return torch.ones(n, device=self.device, dtype=torch.float64)
    def zeros(self, shape, dtype=None):
        dt = torch.complex128 if dtype == complex else torch.float64
        return torch.zeros(shape, device=self.device, dtype=dt)
    def sqrt(self, x): return torch.sqrt(x)
    def exp(self, x): return torch.exp(x)

tw_cpu = TorchWrap(torch.device("cpu"))
s_ms = run_scalpel(tw_cpu, "PyTorch CPU")

# FDTD batched with torch on CPU
def run_fdtd_torch(device):
    E = torch.zeros(n_modes, Nz, device=device, dtype=torch.float64)
    H = torch.zeros(n_modes, Nz-1, device=device, dtype=torch.float64)
    Ep = torch.zeros(n_modes, device=device, dtype=torch.float64)

    E.zero_(); H.zero_(); Ep.zero_()
    for n in range(Nt_fdtd):
        t_n = n * dt_fdtd
        em2 = E[:, -2].clone()
        H = H + c3_v * (E[:, 1:] - E[:, :-1])
        E[:, 1:-1] = c1_v * E[:, 1:-1] + c2_v * (H[:, 1:] - H[:, :-1])
        E[:, 0] = math.exp(-0.5 * ((t_n - tc) / tw)**2)
        E[:, -1] = Ep + mur_v * (E[:, -2] - E[:, -1])
        Ep = em2
    return E[:, obs_idx]

_ = run_fdtd_torch(torch.device("cpu"))
t0 = time.perf_counter()
_ = run_fdtd_torch(torch.device("cpu"))
f_ms = (time.perf_counter() - t0) * 1e3

print(f"  Scalpel: {s_ms:.1f} ms | FDTD: {f_ms:.0f} ms | Wall ratio: {f_ms/s_ms:.0f}x")
results.append(("PyTorch CPU", s_ms, f_ms))

# ═══════════════════════════════════════════════════════════════════
#  4. PyTorch GPU
# ═══════════════════════════════════════════════════════════════════
print("\n─── PyTorch GPU ───")
tw_gpu = TorchWrap(torch.device("cuda"))
s_ms = run_scalpel(tw_gpu, "PyTorch GPU", sync=torch.cuda.synchronize)

_ = run_fdtd_torch(torch.device("cuda"))
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = run_fdtd_torch(torch.device("cuda"))
torch.cuda.synchronize()
f_ms = (time.perf_counter() - t0) * 1e3

print(f"  Scalpel: {s_ms:.1f} ms | FDTD: {f_ms:.0f} ms | Wall ratio: {f_ms/s_ms:.0f}x")
results.append(("PyTorch GPU", s_ms, f_ms))

# ═══════════════════════════════════════════════════════════════════
#  5. JAX GPU
# ═══════════════════════════════════════════════════════════════════
print("\n─── JAX GPU ───")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import maxwell_lossy

backend = get_backend("jax")
grid = GridParams(Nx=Nx, Ny=Nx, dx=0.01, dy=0.01)
nilt_p = NILTParams(a=a_nilt, T=T_nilt, N=N_NILT)
def disp_fn(s, KX, KY, b): return maxwell_lossy(s, KX, KY, sigma, eps_r, b)
engine = SpectralEngine(disp_fn, backend)

src_jax = backend.array(np.eye(Nx, Nx) * 0 + 0j)
src_np = np.zeros((Nx, Nx)); src_np[Nx//2, Nx//2] = 1.0
src_jax = backend.array(src_np, dtype=complex)

f, t = engine.forward(src_jax, depth, grid, nilt_p)
f.block_until_ready()

t0 = time.perf_counter()
for _ in range(20):
    f, t = engine.forward(src_jax, depth, grid, nilt_p)
    f.block_until_ready()
s_ms = (time.perf_counter() - t0) / 20 * 1e3

# JAX FDTD via lax.scan
@jax.jit
def fdtd_jax_batched():
    def step(carry, t_n):
        E, H, Ep = carry
        em2 = E[:, -2]
        H = H + c3_v * (E[:, 1:] - E[:, :-1])
        E_int = c1_v * E[:, 1:-1] + c2_v * (H[:, 1:] - H[:, :-1])
        E_left = jnp.full((n_modes, 1),
                          jnp.exp(-0.5 * ((t_n - tc) / tw)**2))
        E_right = (Ep + mur_v * (E[:, -2] - E[:, -1]))[:, None]
        E = jnp.concatenate([E_left, E_int, E_right], axis=1)
        Ep = em2
        return (E, H, Ep), E[:, obs_idx]

    E0 = jnp.zeros((n_modes, Nz))
    H0 = jnp.zeros((n_modes, Nz-1))
    Ep0 = jnp.zeros(n_modes)
    t_steps = jnp.arange(Nt_fdtd) * dt_fdtd
    _, obs = jax.lax.scan(step, (E0, H0, Ep0), t_steps)
    return obs

_ = fdtd_jax_batched().block_until_ready()
t0 = time.perf_counter()
for _ in range(3):
    _ = fdtd_jax_batched().block_until_ready()
f_ms = (time.perf_counter() - t0) / 3 * 1e3

print(f"  Scalpel: {s_ms:.1f} ms | FDTD: {f_ms:.0f} ms | Wall ratio: {f_ms/s_ms:.0f}x")
results.append(("JAX GPU", s_ms, f_ms))

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print(f" SUMMARY: {Nx}x{Nx} grid, Nz={Nz}, Nt={Nt_fdtd}")
print(f" NFE: scalpel={nfe_scalpel:.2e}, fdtd={nfe_fdtd:.2e}, ratio={nfe_fdtd/nfe_scalpel:.0f}x")
print(f"{'='*75}")
print(f" {'Backend':>14s}  {'Scalpel':>10s}  {'FDTD':>10s}  {'Wall ratio':>10s}")
for name, s, f in results:
    print(f" {name:>14s}  {s:>9.1f}ms  {f:>9.0f}ms  {f/s:>9.0f}x")
print(f"{'='*75}")
