"""
#2  N_NILT convergence study
─────────────────────────────
Fix spatial grid (64x64), sweep N_NILT from 64 to 8192.
Compare engine output vs analytical (diffusion Lévy kernel).
Expect spectral (exponential) convergence in N.

Outputs parsable data tables before plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import diffusion
from scalpel.core.feasibility import tune_params

# ── physics ─────────────────────────────────────────────────────────────
D     = 1e-4
d     = 0.005
w     = 0.008
t_end = 2.0

# ── CFL Phase 1 (shared a, T across all N values) ──────────────────────
params = tune_params(t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
                     eps_tail=1e-6, N_init=64, rho=D/d**2)
a, T = params.a, params.T
print(f"CFL-tuned: a={a:.4f}, T={T:.1f}, feasible={params.feasible}")

# ── fixed grid ──────────────────────────────────────────────────────────
Nx, dx = 64, 0.001
grid = GridParams(Nx=Nx, Ny=Nx, dx=dx, dy=dx)
backend = get_backend()
print(f"Backend: {backend.name}")

# Source
x = (np.arange(Nx) - Nx//2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")
source_np = np.exp(-(X**2 + Y**2) / (2*w**2))
source = backend.array(source_np, dtype=complex)

# Analytical reference
kx_arr = np.fft.fftfreq(Nx, dx) * 2 * np.pi
KX_k, KY_k = np.meshgrid(kx_arr, kx_arr, indexing="ij")
kperp_sq = KX_k**2 + KY_k**2
S_hat = np.fft.fft2(source_np)
a_levy = d / np.sqrt(D)

def analytical_field(t_arr):
    Nt = len(t_arr)
    field = np.zeros((Nx, Nx, Nt))
    for i, tv in enumerate(t_arr):
        if tv <= 0:
            continue
        h = (a_levy / (2*np.sqrt(np.pi*tv**3))
             * np.exp(-a_levy**2/(4*tv) - D*kperp_sq*tv))
        field[:,:,i] = np.real(np.fft.ifft2(S_hat * h))
    return field

# ── sweep N_NILT ────────────────────────────────────────────────────────
N_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

def disp_fn(s, KX, KY, b):
    return diffusion(s, KX, KY, D, b)

engine = SpectralEngine(disp_fn, backend)

print(f"\n{'N':>6s}  {'RMS_relL2':>11s}  {'Peak_relL2':>11s}  {'Mean_relL2':>11s}  "
      f"{'eps_im':>11s}  {'GPU_ms':>8s}  {'MaxAbsErr':>11s}")
print("-" * 82)

results = []
for N_nilt in N_values:
    nilt_p = NILTParams(a=a, T=T, N=N_nilt)

    # Warmup (includes JIT compile)
    _f, _t = engine.forward(source, d, grid, nilt_p)
    if backend.name == "jax":
        _f.block_until_ready()

    # Timed (5 runs, take median)
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        field, t_arr = engine.forward(source, d, grid, nilt_p)
        if backend.name == "jax":
            field.block_until_ready()
        elif backend.name == "torch":
            import torch; torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1e3)
    wall = np.median(timings)

    field_np = backend.to_numpy(field)
    t_np = backend.to_numpy(t_arr)

    # Analytical at same time points
    field_exact = analytical_field(t_np)

    # Metrics (within observation window)
    valid = (t_np > 0.005) & (t_np <= t_end)
    l2_err = np.array([np.sqrt(np.mean((field_np[:,:,i]-field_exact[:,:,i])**2))
                       for i in range(len(t_np))])
    l2_ref = np.array([np.sqrt(np.mean(field_exact[:,:,i]**2))
                       for i in range(len(t_np))])

    # RMS relative L2 over the window (the honest metric)
    sig = valid & (l2_ref > 0.01 * l2_ref[valid].max())
    rms_rel = np.sqrt(np.mean((l2_err[sig] / l2_ref[sig])**2)) if np.any(sig) else np.nan

    # Also report peak and mean for comparison
    rel_l2 = np.full(len(t_np), np.nan)
    rel_l2[sig] = l2_err[sig] / l2_ref[sig]
    peak_rel = np.nanmin(rel_l2[sig]) if np.any(sig) else np.nan
    mean_rel = np.nanmean(rel_l2[sig & (t_np < 0.5*t_end)]) \
               if np.any(sig & (t_np < 0.5*t_end)) else np.nan
    max_abs = np.max(l2_err[valid])

    results.append(dict(N=N_nilt, dt=nilt_p.delta_t, wall=wall,
                        rms_rel=rms_rel, peak_rel=peak_rel, mean_rel=mean_rel,
                        max_abs=max_abs, t=t_np, rel_l2=rel_l2,
                        l2_err=l2_err, l2_ref=l2_ref, sig=sig,
                        field=field_np, field_exact=field_exact))

    print(f"{N_nilt:>6d}  {rms_rel:>11.3e}  {peak_rel:>11.3e}  {mean_rel:>11.3e}  "
          f"{'N/A':>11s}  {wall:>8.1f}  {max_abs:>11.3e}")

# ═══════════════════════════════════════════════════════════════════════
#  Figure (2x2)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── (a) Convergence: error vs N ──────────────────────────────────────
ax = axes[0, 0]
Ns = [r["N"] for r in results]
rms_vals = [r["rms_rel"] for r in results]
mean_vals = [r["mean_rel"] for r in results]
ax.semilogy(Ns, rms_vals, "ko-", lw=2, ms=7, label="RMS rel L$_2$")
ax.semilogy(Ns, mean_vals, "s--", color="C1", lw=1.5, ms=6,
            label="Mean rel L$_2$ (t<1s)")
ax.axhline(1e-2, color="red", ls="--", lw=0.7, alpha=0.5, label="1%")
ax.axhline(1e-6, color="green", ls="--", lw=0.7, alpha=0.5, label="1 ppm")
ax.set_xlabel("N$_{\\rm NILT}$ (Bromwich points)")
ax.set_ylabel("Relative L$_2$ error")
ax.set_title("(a) Convergence rate", fontsize=11)
ax.set_xscale("log", base=2)
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns], fontsize=8)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.25)
ax.set_ylim(1e-13, 1e1)  # show all the way down to N=8192

# ── (b) Rel L2 vs time for selected N values ─────────────────────────
ax = axes[0, 1]
show_Ns = [64, 256, 1024, 4096, 8192]
colors_b = ["C3", "C0", "C2", "C4", "C5"]
for N_show, col in zip(show_Ns, colors_b):
    r = next(r for r in results if r["N"] == N_show)
    t_ms = r["t"] * 1e3
    sm = r["sig"] & (t_ms > 5) & (t_ms < t_end*1e3)
    if np.any(sm):
        ax.semilogy(t_ms[sm], r["rel_l2"][sm], color=col, lw=1.2,
                    label=f"N={N_show}")
ax.axhline(1e-2, color="red", ls="--", lw=0.7, alpha=0.5)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Relative L$_2$")
ax.set_title("(b) Error vs time for selected N", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)
ax.set_ylim(1e-13, 1e1)  # match panel (a)

# ── (c) Center waveform for selected N ───────────────────────────────
ax = axes[1, 0]
cx = Nx // 2
r_best = results[-1]
t_ms_b = r_best["t"] * 1e3
pm = (t_ms_b > 5) & (t_ms_b < t_end*1e3)
ax.plot(t_ms_b[pm], r_best["field_exact"][cx, cx, pm], "k-", lw=2.5,
        label="Analytical", zorder=5)
# Small N: use markers; large N: use lines
for N_show, col, ms_size in [(64, "C3", 4), (256, "C0", 3), (1024, "C2", 0)]:
    r = next(r for r in results if r["N"] == N_show)
    t_ms_r = r["t"] * 1e3
    pm_r = (t_ms_r > 5) & (t_ms_r < t_end*1e3)
    if ms_size > 0:
        # Few points — show markers
        ax.plot(t_ms_r[pm_r], r["field"][cx, cx, pm_r], "o", color=col,
                ms=ms_size, label=f"N={N_show}", alpha=0.7)
    else:
        ax.plot(t_ms_r[pm_r], r["field"][cx, cx, pm_r], "--", color=col, lw=1,
                label=f"N={N_show}")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("u(0,0,d,t)")
ax.set_title("(c) Center waveform convergence", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# ── (d) Wall time vs N ───────────────────────────────────────────────
ax = axes[1, 1]
walls = [r["wall"] for r in results]
ax.loglog(Ns, walls, "ko-", lw=2, ms=7)
for r in results:
    ax.annotate(f" {r['wall']:.1f}ms", (r["N"], r["wall"]),
                fontsize=7, va="bottom")
# Anchor at last point (GPU saturated, asymptotic regime)
N_ref = np.array([Ns[0], Ns[-1]])
w_last = walls[-1]
N_last = Ns[-1]
ax.loglog(N_ref, w_last * (N_ref / N_last), "r--", lw=0.8, alpha=0.5,
          label="$O(N)$ from 8192")
ax.loglog(N_ref,
          w_last * (N_ref / N_last) * np.log2(N_ref + 1) / np.log2(N_last + 1),
          "b--", lw=0.8, alpha=0.5, label="$O(N \\log N)$ from 8192")
ax.set_xlabel("N$_{\\rm NILT}$")
ax.set_ylabel("Wall time [ms]")
ax.set_title("(d) GPU timing vs N$_{\\rm NILT}$ (median of 5)", fontsize=11)
ax.set_xscale("log", base=2)
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns], fontsize=8)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.25)

fig.suptitle(
    f"N$_{{\\rm NILT}}$ convergence study — diffusion benchmark (64x64 grid)\n"
    f"D={D:.0e} m$^2$/s,  d={d*1e3:.0f} mm,  "
    f"a={a:.4f},  T={T:.1f} s  (CFL-tuned)",
    fontsize=11, y=1.01)

plt.tight_layout()
out = "scripts/nilt_convergence.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)

# Also save to paper/figures/
import shutil
shutil.copy(out, "paper/figures/nilt_convergence.png")
print(f"Copied -> paper/figures/nilt_convergence.png")
