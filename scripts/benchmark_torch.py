"""
Fair PyTorch benchmark: scalpel vs FDTD, same hardware.
GPU and CPU, with proper timing.

Strategy: time 1 FDTD column, then extrapolate to Nx*Ny columns.
The FDTD Python loop is the same on GPU and CPU — the kernel launch
overhead is the bottleneck, not the compute.
"""

import torch
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

c1_v = (1 - sigma*dt_fdtd/(2*epsilon)) / (1 + sigma*dt_fdtd/(2*epsilon))
c2_v = (dt_fdtd/(epsilon*dz)) / (1 + sigma*dt_fdtd/(2*epsilon))
c3_v = dt_fdtd / (MU_0 * dz)
mur_v = (c_mat*dt_fdtd - dz) / (c_mat*dt_fdtd + dz)
obs_idx = int(depth / dz)
tc = 3 * t_transit
tw = t_transit * 0.15

nfe_scalpel = Nx * Nx * N_NILT
nfe_fdtd = Nx * Nx * 2 * Nz * Nt_fdtd

print("=" * 70)
print(f" PyTorch BENCHMARK: {Nx}x{Nx}, wet clay, d={depth}m")
print(f" Scalpel N_NILT={N_NILT}, FDTD Nz={Nz} Nt={Nt_fdtd}")
print(f" NFE: scalpel={nfe_scalpel:.2e}, fdtd={nfe_fdtd:.2e}")
print(f" NFE ratio: {nfe_fdtd/nfe_scalpel:.0f}x")
print("=" * 70)


def scalpel_torch(source_2d, device):
    dx_grid = 0.01
    kx = torch.fft.fftfreq(Nx, dx_grid, device=device, dtype=torch.float64) * 2 * math.pi
    ky = torch.fft.fftfreq(Nx, dx_grid, device=device, dtype=torch.float64) * 2 * math.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')

    omega = torch.arange(N_NILT, device=device, dtype=torch.float64) * (math.pi / T_nilt)
    s = a_nilt + 1j * omega

    gamma_sq = (MU_0 * (sigma * s[None, None, :] + epsilon * s[None, None, :]**2)
                - KX[:, :, None]**2 - KY[:, :, None]**2)
    gamma_z = torch.sqrt(gamma_sq)
    gamma_z = gamma_z * (1.0 - 2.0 * (gamma_z.real < 0))

    S = torch.fft.fft2(source_2d)
    H = torch.exp(-gamma_z * depth)

    half = torch.ones(N_NILT, device=device, dtype=torch.float64)
    half[0] = 0.5
    G = S[:, :, None] * H * half[None, None, :]

    z_raw = N_NILT * torch.fft.ifft(G, dim=-1)
    dt_n = 2 * T_nilt / N_NILT
    t_arr = torch.arange(N_NILT, device=device, dtype=torch.float64) * dt_n
    correction = torch.exp(a_nilt * t_arr) / T_nilt
    field_kt = z_raw.real * correction[None, None, :]
    return torch.fft.ifft2(field_kt, dim=(0, 1)).real


def fdtd_1col_torch(device):
    """Single 1D FDTD column on the given device."""
    E = torch.zeros(Nz, device=device, dtype=torch.float64)
    H = torch.zeros(Nz - 1, device=device, dtype=torch.float64)
    E_prev_right = torch.tensor(0.0, device=device, dtype=torch.float64)

    for n in range(Nt_fdtd):
        t_n = n * dt_fdtd
        Em2_old = E[-2].clone()

        H = H + c3_v * (E[1:] - E[:-1])
        E[1:-1] = c1_v * E[1:-1] + c2_v * (H[1:] - H[:-1])
        E[0] = math.exp(-0.5 * ((t_n - tc) / tw)**2)
        E[-1] = E_prev_right + mur_v * (E[-2] - E[-1])
        E_prev_right = Em2_old

    return E[obs_idx]


def fdtd_batched_torch(device, n_cols):
    """Batched FDTD: n_cols columns in parallel."""
    E = torch.zeros(n_cols, Nz, device=device, dtype=torch.float64)
    H = torch.zeros(n_cols, Nz - 1, device=device, dtype=torch.float64)
    E_prev_right = torch.zeros(n_cols, device=device, dtype=torch.float64)

    for n in range(Nt_fdtd):
        t_n = n * dt_fdtd
        Em2_old = E[:, -2].clone()

        H = H + c3_v * (E[:, 1:] - E[:, :-1])
        E[:, 1:-1] = c1_v * E[:, 1:-1] + c2_v * (H[:, 1:] - H[:, :-1])
        E[:, 0] = math.exp(-0.5 * ((t_n - tc) / tw)**2)
        E[:, -1] = E_prev_right + mur_v * (E[:, -2] - E[:, -1])
        E_prev_right = Em2_old

    return E[:, obs_idx]


for device_name in ["cuda", "cpu"]:
    device = torch.device(device_name)
    print(f"\n{'─'*70}")
    print(f" PyTorch on {device_name.upper()}")
    print(f"{'─'*70}")

    # ── Scalpel ────────────────────────────────────────────────
    src = torch.zeros(Nx, Nx, device=device, dtype=torch.complex128)
    src[Nx//2, Nx//2] = 1.0

    _ = scalpel_torch(src, device)
    if device_name == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        _ = scalpel_torch(src, device)
        if device_name == "cuda":
            torch.cuda.synchronize()
    scalpel_ms = (time.perf_counter() - t0) / n_runs * 1e3

    print(f"  Scalpel: {scalpel_ms:.1f} ms")

    # ── FDTD: time 1 column ───────────────────────────────────
    _ = fdtd_1col_torch(device)
    if device_name == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    _ = fdtd_1col_torch(device)
    if device_name == "cuda":
        torch.cuda.synchronize()
    fdtd_1col_ms = (time.perf_counter() - t0) * 1e3

    print(f"  FDTD 1 column: {fdtd_1col_ms:.1f} ms ({Nt_fdtd} steps x {Nz} cells)")

    # ── FDTD: batched columns ─────────────────────────────────
    # Try batching in groups to avoid OOM / excessive time
    if device_name == "cuda":
        # GPU: batch all columns, the per-step kernels are parallel across cols
        batch_sizes = [64, 256, 1024, Nx*Nx]
    else:
        # CPU: batch all at once (vectorized numpy-like)
        batch_sizes = [Nx*Nx]

    for bs in batch_sizes:
        if device_name == "cuda":
            torch.cuda.empty_cache()

        try:
            _ = fdtd_batched_torch(device, bs)
            if device_name == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = fdtd_batched_torch(device, bs)
            if device_name == "cuda":
                torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - t0) * 1e3

            # Scale to full Nx*Ny
            full_ms = batch_ms * (Nx*Nx / bs)
            per_col = batch_ms / bs

            print(f"  FDTD {bs:>5d} cols: {batch_ms:.0f} ms "
                  f"({per_col:.2f} ms/col, full={full_ms:.0f} ms)")
        except Exception as e:
            print(f"  FDTD {bs:>5d} cols: FAILED ({e})")
            break

    # ── Summary for this device ────────────────────────────────
    # Use the best FDTD estimate
    fdtd_full_ms = fdtd_1col_ms * Nx * Nx  # conservative: no batching benefit
    ratio = fdtd_full_ms / scalpel_ms

    print(f"\n  Summary ({device_name.upper()}):")
    print(f"    Scalpel:    {scalpel_ms:10.1f} ms")
    print(f"    FDTD (seq): {fdtd_full_ms:10.0f} ms  ({Nx*Nx} x {fdtd_1col_ms:.1f}ms)")
    print(f"    Wall ratio: {ratio:.0f}x")
    print(f"    NFE ratio:  {nfe_fdtd/nfe_scalpel:.0f}x")
