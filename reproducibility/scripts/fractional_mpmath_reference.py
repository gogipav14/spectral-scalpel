"""
High-precision reference for the fractional-Caputo half-space Heaviside
benchmark, via arbitrary-precision numerical Laplace inversion (mpmath
`invertlaplace`, Talbot algorithm, 50-digit precision).

Purpose: break the ``scalpel vs scalpel'' self-consistency circularity
identified in the editorial pass. The transfer function of the factorization
is analytically known; mpmath inverts it independently of the float64 NILT
pipeline used by scalpel. Any residual mismatch bounds the NILT error of
scalpel at N_NILT=2048, float64.

Setup: k_perp = 0 centerline mode of the 3D fractional slab benchmark
(Sec. 3, Fig. 3b, SI Table 3). The 1D PDE is d_t^alpha u = D u_zz with
u(0, t) = H(t), u(z -> infty, t) = 0. Laplace solution at depth z:
   u_hat(z, s) = exp(-z sqrt(s^alpha / D)) / s.
Invert at the benchmark's observation point z = d_obs, several t in
[1 ms, 20 ms].
"""

from __future__ import annotations

import csv
import os
import time

import mpmath as mp
import numpy as np

# Problem parameters (must match reproducibility/scripts/benchmark_fractional_*)
ALPHA = 0.7
D = 1e-2
d_obs = 0.10
t_end = 0.020
t_samples_ms = [1.0, 2.5, 5.0, 10.0, 15.0, 20.0]  # ms

mp.mp.dps = 50  # 50 decimal digits


def u_hat(s):
    """Laplace-space centerline response at z = d_obs for Heaviside surface
    source in 1D fractional-Caputo half-space diffusion."""
    gamma = mp.sqrt(s ** ALPHA / D)
    return mp.exp(-d_obs * gamma) / s


print("=" * 70)
print(f" mpmath high-precision reference, dps = {mp.mp.dps}")
print(f" 1D fractional Caputo, alpha = {ALPHA}, D = {D}, d_obs = {d_obs}")
print(f" times: {t_samples_ms} ms")
print("=" * 70)

mpmath_values = {}
for t_ms in t_samples_ms:
    t_s = t_ms * 1e-3
    t0 = time.perf_counter()
    val = mp.invertlaplace(u_hat, t_s, method="talbot")
    elapsed = time.perf_counter() - t0
    mpmath_values[t_ms] = float(val)
    print(f"  t = {t_ms:5.1f} ms:  u(d_obs, t) = {float(val): .8e}   "
          f"({elapsed*1000:.0f} ms)")

# ---------------------------------------------------------------------------
# Compare to scalpel (float64, N_NILT = 2048 and 4096) at k_perp = 0 mode.
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print(" Scalpel (float64) at same problem, k_perp = 0 centerline mode")
print("-" * 70)

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # avoid CUDA warm-up for this tiny test
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def scalpel_1d(n_nilt: int):
    """1D fractional-Caputo NILT at k_perp = 0 (single Bromwich inversion).

    Uses the same contour family (Dubner-Abate / half-period Fourier cosine)
    as scalpel.core.engine, so any mismatch with mpmath measures the NILT
    discretization error at float64 / N_NILT = n_nilt.
    """
    a = 2.3 / t_end
    T = 2.0 * t_end
    n = jnp.arange(n_nilt, dtype=jnp.float64)
    omega = n * (jnp.pi / T)
    s = a + 1j * omega
    gamma = jnp.sqrt(s ** ALPHA / D)
    U = jnp.exp(-d_obs * gamma) / s     # impulse divided by s for Heaviside
    half = jnp.ones(n_nilt, dtype=jnp.float64).at[0].set(0.5)
    G = U * half
    z = n_nilt * jnp.fft.ifft(G)
    dt_n = 2 * T / n_nilt
    t_arr = n * dt_n
    correction = jnp.exp(a * t_arr) / T
    return jnp.real(z) * correction, t_arr


for n_nilt in (2048, 4096):
    field, t_arr = scalpel_1d(n_nilt)
    t_np = np.asarray(t_arr)
    field_np = np.asarray(field)
    print(f"\n  N_NILT = {n_nilt}")
    for t_ms in t_samples_ms:
        t_s = t_ms * 1e-3
        idx = int(np.argmin(np.abs(t_np - t_s)))
        scv = float(field_np[idx])
        ref = mpmath_values[t_ms]
        rel = abs(scv - ref) / max(abs(ref), 1e-30)
        print(f"    t = {t_ms:5.1f} ms:  scalpel = {scv: .8e}   "
              f"rel_err vs mpmath = {rel:.2e}")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPRO_DATA = os.path.abspath(os.path.join(_HERE, "..", "data"))
os.makedirs(_REPRO_DATA, exist_ok=True)
out_csv = os.path.join(_REPRO_DATA, "fractional_mpmath_reference.csv")
rows = []
for n_nilt in (2048, 4096):
    field, t_arr = scalpel_1d(n_nilt)
    t_np = np.asarray(t_arr)
    field_np = np.asarray(field)
    for t_ms in t_samples_ms:
        t_s = t_ms * 1e-3
        idx = int(np.argmin(np.abs(t_np - t_s)))
        scv = float(field_np[idx])
        ref = mpmath_values[t_ms]
        rel = abs(scv - ref) / max(abs(ref), 1e-30)
        rows.append((t_ms, n_nilt, scv, ref, rel))

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_ms", "N_NILT", "scalpel_float64", "mpmath_50digit",
                "rel_err"])
    for r in rows:
        w.writerow(r)
print(f"\nSaved -> {out_csv}")
