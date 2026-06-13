"""
Maxwell dispersion sign-convention verification.

A reviewer-style sanity check raised by GPT-5.5: deriving the telegrapher
Fourier-Laplace transform from first principles gives

    gamma_z^2 = mu_0 (sigma s + epsilon s^2) + k_perp^2          (PLUS sign)

while the code in scalpel/core/dispersion.py:maxwell_lossy uses

    gamma_z^2 = mu_0 (sigma s + epsilon s^2) - k_perp^2          (MINUS sign)

The 1D validation (k_perp = 0) is insensitive to this term, so the standard
Bessel-I_1 Green's function check does not exercise the disputed sign.

This script tests a high-k_perp mode against an independent reference: the
3D Yee FDTD impulse response at a transverse-shifted observation point. If
the spectral engine's sign convention is correct, the spectral and FDTD
responses match at the high-k_perp observation. If they disagree, the sign
in the code (and the paper) is wrong.

Outputs:
- Time-series comparison plot (PNG)
- CSV with both responses and the relative error
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np

# Setup -----------------------------------------------------------------
MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12

# Wet clay parameters from the manuscript
sigma = 0.1
eps_r = 10.0
epsilon = EPS_0 * eps_r
mu = MU_0
c_mat = 1.0 / math.sqrt(mu * epsilon)
t_transit = 0.5 / c_mat
t_end = 13.0 * t_transit

# Test point: high transverse k_perp, fixed depth
# We will compare on a 64x64 transverse grid, observing at a transverse
# shifted location to exercise non-DC modes.
Nx = Ny = 64
dx = 0.01           # 1 cm transverse pixel
depth = 0.5         # 50 cm slab

print("=" * 70)
print(" Maxwell sign-convention verification, wet clay")
print(f" sigma = {sigma} S/m, eps_r = {eps_r}, c_mat = {c_mat:.3e} m/s")
print(f" depth = {depth} m, t_transit = {t_transit*1e9:.2f} ns")
print("=" * 70)

# Source: Gaussian-pulse-in-time, delta-in-space at center
tc = 3 * t_transit
tw = t_transit * 0.15

# Observation points to test the k_perp sign:
# - (Nx/2, Ny/2): centerline (DC mode dominates, k_perp = 0)
# - (Nx/2 + 8, Ny/2): off-axis by 8 cells = 8 cm (excites non-DC modes)
obs_points = [
    ("centerline", Nx // 2, Ny // 2),
    ("off_axis_8cm", Nx // 2 + 8, Ny // 2),
]


# -----------------------------------------------------------------------
# Method A: spectral engine with current code convention (gamma_z^2 - k_perp^2)
# -----------------------------------------------------------------------
def spectral_engine_minus_sign(N_NILT=2048, a_nilt=6.72e7, T_nilt=1.37e-7):
    """As implemented in scalpel/core/dispersion.py:maxwell_lossy."""
    omega = np.arange(N_NILT) * (math.pi / T_nilt)
    s = a_nilt + 1j * omega

    kx = np.fft.fftfreq(Nx, dx) * 2 * math.pi
    ky = np.fft.fftfreq(Ny, dx) * 2 * math.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    # Source: Gaussian pulse in time delta in space
    src_xy = np.zeros((Nx, Ny), dtype=complex)
    src_xy[Nx // 2, Ny // 2] = 1.0
    Shat = np.fft.fft2(src_xy)
    g_t = np.exp(-0.5 * ((np.arange(N_NILT) * 2 * T_nilt / N_NILT - tc) / tw) ** 2)
    # In s-domain: Laplace transform of Gaussian centered at tc, width tw
    g_s = tw * math.sqrt(2 * math.pi) * np.exp(-0.5 * (s * tw) ** 2 - s * tc)

    # ----- THE DISPUTED SIGN: code uses -k_perp^2 -----
    g2_minus = (
        mu * (sigma * s[None, None, :] + epsilon * s[None, None, :] ** 2)
        - KX[:, :, None] ** 2 - KY[:, :, None] ** 2
    )
    gz_minus = np.sqrt(g2_minus)
    gz_minus = gz_minus * (1.0 - 2.0 * (gz_minus.real < 0))
    H_minus = np.exp(-gz_minus * depth)

    G_minus = Shat[:, :, None] * H_minus * g_s[None, None, :]
    # Half-weight at k=0
    half = np.ones(N_NILT)
    half[0] = 0.5
    G_minus = G_minus * half[None, None, :]
    z_minus = N_NILT * np.fft.ifft(G_minus, axis=-1)
    dt_n = 2 * T_nilt / N_NILT
    t_arr = np.arange(N_NILT) * dt_n
    correction = np.exp(a_nilt * t_arr) / T_nilt
    fkt_minus = np.real(z_minus) * correction[None, None, :]
    field_minus = np.real(np.fft.ifft2(fkt_minus, axes=(0, 1)))

    # Also compute the +sign version for comparison
    g2_plus = (
        mu * (sigma * s[None, None, :] + epsilon * s[None, None, :] ** 2)
        + KX[:, :, None] ** 2 + KY[:, :, None] ** 2
    )
    gz_plus = np.sqrt(g2_plus)
    gz_plus = gz_plus * (1.0 - 2.0 * (gz_plus.real < 0))
    H_plus = np.exp(-gz_plus * depth)
    G_plus = Shat[:, :, None] * H_plus * g_s[None, None, :] * half[None, None, :]
    z_plus = N_NILT * np.fft.ifft(G_plus, axis=-1)
    fkt_plus = np.real(z_plus) * correction[None, None, :]
    field_plus = np.real(np.fft.ifft2(fkt_plus, axes=(0, 1)))

    return t_arr, field_minus, field_plus


# -----------------------------------------------------------------------
# Method B: 3D Yee FDTD reference
# -----------------------------------------------------------------------
def yee_fdtd_3d(N_steps=4000):
    """Minimal 3D Yee FDTD for E_y mode, lossy Maxwell. Single source pulse
    at (Nx/2, Ny/2, 0), observe at every transverse point at z = depth."""
    Lz = 1.5 * depth  # Some headroom for absorbing BC
    Nz = 75
    dz = Lz / Nz
    dt = 0.4 * dz / c_mat  # Conservative Courant number
    Nt = int(math.ceil(t_end / dt))

    print(f" FDTD Nz={Nz}, dz={dz*1e3:.2f} mm, dt={dt*1e12:.2f} ps, Nt={Nt}")

    c1 = (1 - sigma * dt / (2 * epsilon)) / (1 + sigma * dt / (2 * epsilon))
    c2 = (dt / (epsilon * dz)) / (1 + sigma * dt / (2 * epsilon))
    c3 = dt / (mu * dz)

    # Fields. Use E_y, H_x, H_z for the TE mode propagating in z.
    # For simplicity, use a scalar telegrapher model on a 3D grid:
    E = np.zeros((Nx, Ny, Nz))
    E_prev = np.zeros((Nx, Ny, Nz))
    E_pprev = np.zeros((Nx, Ny, Nz))

    obs_idx = int(round(depth / dz))
    print(f" obs_idx = {obs_idx}")

    # We solve the telegrapher equation directly via centered differences,
    # avoiding the H_x/H_z bookkeeping for clarity.
    # nabla^2 E = mu sigma dE/dt + mu eps d^2E/dt^2
    # Discretize: lap E = mu sigma (E_n - E_{n-1})/dt + mu eps (E_n - 2E_{n-1} + E_{n-2})/dt^2
    # Solve for E_n:
    #   E_n [mu sigma/dt + mu eps/dt^2] = lap E_{n-1} + mu sigma E_{n-1}/dt + mu eps (2 E_{n-1} - E_{n-2})/dt^2
    A = mu * sigma / dt + mu * epsilon / (dt * dt)
    B = mu * sigma / dt + 2 * mu * epsilon / (dt * dt)
    C = mu * epsilon / (dt * dt)

    traces = np.zeros((Nt, len(obs_points)))

    t0 = time.perf_counter()
    for n in range(Nt):
        t_n = n * dt
        # Source: Gaussian pulse in time at z=0 plane center
        src = math.exp(-0.5 * ((t_n - tc) / tw) ** 2)

        # Laplacian via 7-point stencil
        lap = np.zeros_like(E_prev)
        lap[1:-1, 1:-1, 1:-1] = (
            (E_prev[2:, 1:-1, 1:-1] + E_prev[:-2, 1:-1, 1:-1] - 2 * E_prev[1:-1, 1:-1, 1:-1]) / dx ** 2
            + (E_prev[1:-1, 2:, 1:-1] + E_prev[1:-1, :-2, 1:-1] - 2 * E_prev[1:-1, 1:-1, 1:-1]) / dx ** 2
            + (E_prev[1:-1, 1:-1, 2:] + E_prev[1:-1, 1:-1, :-2] - 2 * E_prev[1:-1, 1:-1, 1:-1]) / dz ** 2
        )

        E_new = (lap + B * E_prev - C * E_pprev) / A
        # Boundary: source at z=0 center
        E_new[Nx // 2, Ny // 2, 0] = src
        # Absorbing top (z = Nz-1): simple radiation BC
        E_new[:, :, -1] = E_prev[:, :, -2]

        # Record observations
        for i, (name, ox, oy) in enumerate(obs_points):
            traces[n, i] = E_new[ox, oy, obs_idx]

        E_pprev = E_prev
        E_prev = E_new

        if n % 500 == 0:
            print(f"   step {n}/{Nt}, t = {t_n*1e9:.2f} ns, src={src:.3e}")

    elapsed = time.perf_counter() - t0
    print(f" FDTD done in {elapsed:.1f}s")

    t_arr_fdtd = np.arange(Nt) * dt
    return t_arr_fdtd, traces


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
print("\n--- Spectral engine (both signs) ---")
t_sp, field_minus, field_plus = spectral_engine_minus_sign()
print(f" spectral t_arr length: {len(t_sp)}, dt = {t_sp[1]-t_sp[0]:.3e} s")

print("\n--- 3D Yee FDTD reference ---")
t_fdtd, traces_fdtd = yee_fdtd_3d()

# Compare at each observation point
results = []
for i, (name, ox, oy) in enumerate(obs_points):
    sp_trace_minus = field_minus[ox, oy, :]
    sp_trace_plus = field_plus[ox, oy, :]
    fdtd_trace = traces_fdtd[:, i]

    # Interpolate spectral to FDTD time grid for direct comparison
    sp_on_fdtd_minus = np.interp(t_fdtd, t_sp, sp_trace_minus)
    sp_on_fdtd_plus = np.interp(t_fdtd, t_sp, sp_trace_plus)

    # Compute RMS error against FDTD on the FDTD-resolved interval
    valid = (t_fdtd > 0.5 * t_transit) & (t_fdtd < t_end)
    fdtd_norm = np.linalg.norm(fdtd_trace[valid])
    err_minus = np.linalg.norm(sp_on_fdtd_minus[valid] - fdtd_trace[valid]) / fdtd_norm
    err_plus = np.linalg.norm(sp_on_fdtd_plus[valid] - fdtd_trace[valid]) / fdtd_norm

    print(f"\n  {name} (obs at ({ox},{oy})):")
    print(f"    spectral (-k_perp^2 convention) rel error vs FDTD: {err_minus:.3e}")
    print(f"    spectral (+k_perp^2 convention) rel error vs FDTD: {err_plus:.3e}")
    if err_plus < err_minus:
        print(f"    >>> +k_perp^2 convention is closer to FDTD (ratio {err_minus/err_plus:.2f}x)")
    else:
        print(f"    >>> -k_perp^2 convention is closer to FDTD (ratio {err_plus/err_minus:.2f}x)")

    results.append((name, err_minus, err_plus))

# Save CSV
out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "maxwell_sign_convention_check.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["observation_point", "rel_err_minus_sign", "rel_err_plus_sign"])
    for name, em, ep in results:
        w.writerow([name, em, ep])
print(f"\nSaved -> {out_csv}")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
print(" Compare 'rel_err_minus_sign' (paper/code convention) and")
print(" 'rel_err_plus_sign' (first-principles derivation).")
print(" - At centerline (k_perp = 0): both signs should agree.")
print(" - Off-axis: the convention that matches FDTD is correct.")
print()
print(" Action required:")
print(" - If off-axis shows -sign matches: paper convention is right.")
print("   Document the convention explicitly in the manuscript.")
print(" - If off-axis shows +sign matches: paper convention is wrong.")
print("   Update dispersion.py and re-validate.")
