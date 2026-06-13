"""
Head-to-head benchmark: spectral scalpel ETDRK vs moljax IMEX-Strang.

Same Gray-Scott problem, same grid, same dt, same number of steps.
Reports wall time per step and total wall time.

Key difference:
  - moljax: (I - dt/2*D*Δ)^{-1} implicit Helmholtz solve via FFT
  - scalpel: exp(D*Δ*dt/2) exact exponential propagator via FFT
Both are O(N log N) per step. The question is the constant factor.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Common parameters
# ═══════════════════════════════════════════════════════════════════════

Du, Dv = 2e-5, 1e-5
F, k = 0.04, 0.06
Lx, Ly = 2.5, 2.5

grid_sizes = [64, 128, 256]
dt = 1.0
n_steps = 1000

print("="*70)
print("  Gray-Scott benchmark: spectral scalpel vs moljax")
print(f"  Du={Du}, Dv={Dv}, F={F}, k={k}")
print(f"  L={Lx}, dt={dt}, n_steps={n_steps}")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════
#  Scalpel: JIT-compiled Strang (exact exponential diffusion)
# ═══════════════════════════════════════════════════════════════════════

from scalpel.core.etdrk_jax import integrate_jit, integrate_jit_rfft
from scalpel.core.etdrk import gray_scott as make_gs

def bench_scalpel(N, use_rfft=False):
    sys = make_gs(Nx=N, Ny=N, Lx=Lx, Ly=Ly, Du=Du, Dv=Dv, F=F, k=k)
    u0 = jnp.asarray(sys.fields["u"].data)
    v0 = jnp.asarray(sys.fields["v"].data)

    integrate_fn = integrate_jit_rfft if use_rfft else integrate_jit

    # Warmup / compile
    us, vs, ts = integrate_fn("gray_scott", u0, v0, Du, Dv, Lx, Ly,
                               {"F": F, "k": k}, dt,
                               n_steps=10, save_every=5)
    us.block_until_ready()

    # Timed
    t0 = time.perf_counter()
    us, vs, ts = integrate_fn("gray_scott", u0, v0, Du, Dv, Lx, Ly,
                               {"F": F, "k": k}, dt,
                               n_steps=n_steps, save_every=n_steps)
    us.block_until_ready()
    wall = (time.perf_counter() - t0) * 1e3

    return wall, np.asarray(vs[-1])

# ═══════════════════════════════════════════════════════════════════════
#  Moljax: IMEX-Strang (implicit Helmholtz diffusion)
# ═══════════════════════════════════════════════════════════════════════

from moljax.core.grid import Grid2D
from moljax.core.model import create_gray_scott_periodic_fft
from moljax.core.stepping import integrate_imex_fixed_dt
from moljax.core.utils import get_interior

def bench_moljax(N):
    grid = Grid2D.uniform(N, N, 0.0, Lx, 0.0, Ly, n_ghost=1)

    model, fft_cache, diffusivities = create_gray_scott_periodic_fft(
        grid=grid, Du=Du, Dv=Dv, F=F, k=k, dtype=jnp.float64
    )

    # Initial condition (same as scalpel)
    X, Y = grid.meshgrid(include_ghost=True)
    u0 = jnp.ones((grid.ny_total, grid.nx_total), dtype=jnp.float64)
    v0 = jnp.zeros((grid.ny_total, grid.nx_total), dtype=jnp.float64)
    cx = Lx / 2
    cy = Ly / 2
    mask = (jnp.abs(X - cx) < 0.1*Lx) & (jnp.abs(Y - cy) < 0.1*Ly)
    key = jax.random.PRNGKey(42)
    u0 = jnp.where(mask, 0.5, u0) + 0.01 * jax.random.uniform(key, u0.shape, minval=-1, maxval=1)
    v0 = jnp.where(mask, 0.25, v0) + 0.01 * jax.random.uniform(jax.random.split(key)[0], v0.shape, minval=-1, maxval=1)
    y0 = {'u': u0, 'v': v0}

    # Warmup
    _t, _y, _f = integrate_imex_fixed_dt(
        model, y0, 0.0, 10*dt, dt, fft_cache, diffusivities,
        use_strang=True, save_every=5
    )
    jax.block_until_ready(_f)

    # Timed
    t0 = time.perf_counter()
    t_hist, y_hist, y_final = integrate_imex_fixed_dt(
        model, y0, 0.0, n_steps*dt, dt, fft_cache, diffusivities,
        use_strang=True, save_every=n_steps
    )
    wall = (time.perf_counter() - t0) * 1e3

    v_final = np.asarray(get_interior(y_final['v'], grid))
    return wall, v_final


# ═══════════════════════════════════════════════════════════════════════
#  Run benchmarks
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'Grid':>8s}  {'Scalpel fft':>13s}  {'Scalpel rfft':>14s}  "
      f"{'Moljax':>10s}  {'fft/moljax':>11s}  {'rfft/moljax':>12s}")
print("-"*80)

results = []
for N in grid_sizes:
    wall_fft, v_fft = bench_scalpel(N, use_rfft=False)
    wall_rfft, v_rfft = bench_scalpel(N, use_rfft=True)
    wall_m, v_m = bench_moljax(N)

    ratio_fft = wall_m / wall_fft
    ratio_rfft = wall_m / wall_rfft

    results.append(dict(N=N, wall_fft=wall_fft, wall_rfft=wall_rfft,
                        wall_m=wall_m, ratio_fft=ratio_fft,
                        ratio_rfft=ratio_rfft))

    print(f"{N:>4d}x{N:<3d}  {wall_fft:>11.1f} ms  {wall_rfft:>12.1f} ms  "
          f"{wall_m:>8.1f} ms  {ratio_fft:>10.2f}x  {ratio_rfft:>11.2f}x")

# ═══════════════════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

Ns = [r["N"] for r in results]
w = 0.25

# (a) Wall time — three bars
ax = axes[0]
x = np.arange(len(Ns))
ax.bar(x - w, [r["wall_fft"] for r in results], w, label="Scalpel fft2", color="C0")
ax.bar(x,     [r["wall_rfft"] for r in results], w, label="Scalpel rfft2", color="C2")
ax.bar(x + w, [r["wall_m"] for r in results], w, label="Moljax IMEX", color="C1")
ax.set_xticks(x)
ax.set_xticklabels([f"{n}x{n}" for n in Ns])
ax.set_ylabel("Wall time [ms]")
ax.set_title(f"(a) Total time ({n_steps} steps)", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.25)

# (b) Per-step cost
ax = axes[1]
ax.bar(x - w, [r["wall_fft"]/n_steps*1e3 for r in results], w,
       label="fft2", color="C0")
ax.bar(x,     [r["wall_rfft"]/n_steps*1e3 for r in results], w,
       label="rfft2", color="C2")
ax.bar(x + w, [r["wall_m"]/n_steps*1e3 for r in results], w,
       label="Moljax", color="C1")
ax.set_xticks(x)
ax.set_xticklabels([f"{n}x{n}" for n in Ns])
ax.set_ylabel("Time per step [us]")
ax.set_title("(b) Per-step cost", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.25)

# (c) Speedup ratio (moljax / scalpel)
ax = axes[2]
ax.bar(x - w/2, [r["ratio_fft"] for r in results], w,
       label="fft2 vs moljax", color="C0", alpha=0.8)
ax.bar(x + w/2, [r["ratio_rfft"] for r in results], w,
       label="rfft2 vs moljax", color="C2", alpha=0.8)
ax.axhline(1.0, color="black", ls="--", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}x{n}" for n in Ns])
ax.set_ylabel("Speedup (moljax / scalpel)")
ax.set_title("(c) Scalpel speedup over moljax", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.25)

fig.suptitle(
    "Gray-Scott: Scalpel fft2 vs rfft2 vs Moljax IMEX\n"
    f"Du={Du}, Dv={Dv}, F={F}, k={k}, dt={dt}, {n_steps} steps",
    fontsize=11, y=1.01)

plt.tight_layout()
out = "scripts/benchmark_vs_moljax.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
