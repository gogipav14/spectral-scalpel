"""
Combined convergence + grid-validation figure (2 rows x 3 cols).

Row 1: N_NILT convergence (from plot_nilt_convergence.py)
  (a) Error vs N
  (b) Error vs time for selected N
  (c) Center waveform convergence

Row 2: Grid independence (from plot_analytical_validation.py)
  (d) Relative L2 vs time for all grids
  (e) Accuracy vs grid size (flat line = grid-independent)
  (f) GPU timing vs N_NILT
"""

import numpy as np
import matplotlib.pyplot as plt
import time, cmath

from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import diffusion
from scalpel.core.feasibility import tune_params

D = 1e-4; d = 0.005; w = 0.008; t_end = 2.0
params = tune_params(t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
                     eps_tail=1e-6, N_init=64, rho=D/d**2)
a, T = params.a, params.T
Nx, dx = 64, 0.001
a_levy = d / np.sqrt(D)

backend = get_backend()
print(f"CFL: a={a:.4f}, T={T:.1f}, Backend: {backend.name}")

# Source + analytical
x = (np.arange(Nx) - Nx//2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")
source_np = np.exp(-(X**2 + Y**2) / (2*w**2))
source = backend.array(source_np, dtype=complex)
kx_arr = np.fft.fftfreq(Nx, dx) * 2 * np.pi
KX_k, KY_k = np.meshgrid(kx_arr, kx_arr, indexing="ij")
kperp_sq = KX_k**2 + KY_k**2
S_hat = np.fft.fft2(source_np)

def analytical_field(t_arr):
    Nt = len(t_arr)
    field = np.zeros((Nx, Nx, Nt))
    for i, tv in enumerate(t_arr):
        if tv <= 0: continue
        h = (a_levy / (2*np.sqrt(np.pi*tv**3))) * np.exp(-a_levy**2/(4*tv) - D*kperp_sq*tv)
        field[:,:,i] = np.real(np.fft.ifft2(S_hat * h))
    return field

def disp_fn(s, KX, KY, b):
    return diffusion(s, KX, KY, D, b)

engine = SpectralEngine(disp_fn, backend)
grid = GridParams(Nx=Nx, Ny=Nx, dx=dx, dy=dx)

# ═══════════════════════════════════════════════════════════════════
# Part 1: N_NILT sweep at fixed 64x64
# ═══════════════════════════════════════════════════════════════════
N_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
conv_results = []
print(f"\n{'N':>6s}  {'RMS_relL2':>11s}  {'GPU_ms':>8s}")
print("-"*30)

for N_nilt in N_values:
    nilt_p = NILTParams(a=a, T=T, N=N_nilt)
    _f, _t = engine.forward(source, d, grid, nilt_p)
    if backend.name == "jax": _f.block_until_ready()

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        field, t_arr = engine.forward(source, d, grid, nilt_p)
        if backend.name == "jax": field.block_until_ready()
        timings.append((time.perf_counter() - t0) * 1e3)
    wall = np.median(timings)

    field_np = backend.to_numpy(field)
    t_np = backend.to_numpy(t_arr)
    field_exact = analytical_field(t_np)

    valid = (t_np > 0.005) & (t_np <= t_end)
    l2_err = np.array([np.sqrt(np.mean((field_np[:,:,i]-field_exact[:,:,i])**2)) for i in range(len(t_np))])
    l2_ref = np.array([np.sqrt(np.mean(field_exact[:,:,i]**2)) for i in range(len(t_np))])
    sig = valid & (l2_ref > 0.01 * l2_ref[valid].max())
    rms_rel = np.sqrt(np.mean((l2_err[sig] / l2_ref[sig])**2)) if np.any(sig) else np.nan
    rel_l2 = np.full(len(t_np), np.nan)
    rel_l2[sig] = l2_err[sig] / l2_ref[sig]

    conv_results.append(dict(N=N_nilt, wall=wall, rms_rel=rms_rel,
                             t=t_np, rel_l2=rel_l2, sig=sig,
                             field=field_np, field_exact=field_exact))
    print(f"{N_nilt:>6d}  {rms_rel:>11.3e}  {wall:>8.1f}")

# ═══════════════════════════════════════════════════════════════════
# Part 2: Grid sweep at fixed N_NILT=2048
# ═══════════════════════════════════════════════════════════════════
grid_configs = [(32, 0.002), (64, 0.001), (96, 0.001), (128, 0.001)]
grid_results = []
print(f"\n{'Grid':>10s}  {'Peak_relL2':>11s}  {'Mean_relL2':>11s}  {'GPU_ms':>8s}")
print("-"*50)

for gNx, gdx in grid_configs:
    gg = GridParams(Nx=gNx, Ny=gNx, dx=gdx, dy=gdx)
    nilt_p = NILTParams(a=a, T=T, N=2048)
    gx = (np.arange(gNx) - gNx//2) * gdx
    gX, gY = np.meshgrid(gx, gx, indexing="ij")
    gsrc_np = np.exp(-(gX**2 + gY**2) / (2*w**2))
    gsrc = backend.array(gsrc_np, dtype=complex)

    _f, _t = engine.forward(gsrc, d, gg, nilt_p)
    if backend.name == "jax": _f.block_until_ready()
    t0 = time.perf_counter()
    gfield, gt_arr = engine.forward(gsrc, d, gg, nilt_p)
    if backend.name == "jax": gfield.block_until_ready()
    gwall = (time.perf_counter() - t0) * 1e3

    gf_np = backend.to_numpy(gfield)
    gt_np = backend.to_numpy(gt_arr)

    gkx = np.fft.fftfreq(gNx, gdx) * 2 * np.pi
    gKX, gKY = np.meshgrid(gkx, gkx, indexing="ij")
    gkp2 = gKX**2 + gKY**2
    gSh = np.fft.fft2(gsrc_np)
    gfe = np.zeros_like(gf_np)
    for i, tv in enumerate(gt_np):
        if tv <= 0: continue
        h = (a_levy/(2*np.sqrt(np.pi*tv**3))) * np.exp(-a_levy**2/(4*tv) - D*gkp2*tv)
        gfe[:,:,i] = np.real(np.fft.ifft2(gSh * h))

    gl2e = np.array([np.sqrt(np.mean((gf_np[:,:,i]-gfe[:,:,i])**2)) for i in range(len(gt_np))])
    gl2r = np.array([np.sqrt(np.mean(gfe[:,:,i]**2)) for i in range(len(gt_np))])
    gsig = (gt_np > 0.005) & (gt_np <= t_end) & (gl2r > 0.01*gl2r.max())
    grel = np.full(len(gt_np), np.nan)
    grel[gsig] = gl2e[gsig] / gl2r[gsig]
    pk = np.nanmin(grel[gsig]) if np.any(gsig) else np.nan
    mn = np.nanmean(grel[gsig & (gt_np < t_end*0.5)]) if np.any(gsig & (gt_np < t_end*0.5)) else np.nan

    grid_results.append(dict(Nx=gNx, pk=pk, mn=mn, wall=gwall,
                             t=gt_np, rel_l2=grel, sig=gsig))
    print(f"{gNx:>4d}x{gNx:<4d}  {pk:>11.3e}  {mn:>11.3e}  {gwall:>8.1f}")

# ═══════════════════════════════════════════════════════════════════
# Figure: 2 rows x 3 cols
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# (a) Convergence rate
ax = axes[0, 0]
Ns = [r["N"] for r in conv_results]
rms = [r["rms_rel"] for r in conv_results]
ax.semilogy(Ns, rms, "ko-", lw=2, ms=7)
ax.axhline(1e-2, color="red", ls="--", lw=0.7, alpha=0.5, label="1%")
ax.axhline(1e-6, color="green", ls="--", lw=0.7, alpha=0.5, label="1 ppm")
ax.set_xlabel("$N_{\\rm NILT}$"); ax.set_ylabel("RMS relative $L_2$")
ax.set_title("(a) Spectral convergence in $N$", fontsize=10)
ax.set_xscale("log", base=2); ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns], fontsize=7)
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.2)
ax.set_ylim(1e-11, 1e1)

# (b) Error vs time
ax = axes[0, 1]
for N_show, col in [(64, "C3"), (256, "C0"), (1024, "C2"), (8192, "C5")]:
    r = next(r for r in conv_results if r["N"] == N_show)
    sm = r["sig"] & (r["t"]*1e3 > 5) & (r["t"]*1e3 < t_end*1e3)
    if np.any(sm):
        ax.semilogy(r["t"][sm]*1e3, r["rel_l2"][sm], color=col, lw=1.2, label=f"N={N_show}")
ax.axhline(1e-2, color="red", ls="--", lw=0.7, alpha=0.5)
ax.set_xlabel("Time [ms]"); ax.set_ylabel("Relative $L_2$")
ax.set_title("(b) Error vs time", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.2)
ax.set_ylim(1e-11, 1e1)

# (c) Center waveform
ax = axes[0, 2]
cx = Nx // 2
r_best = conv_results[-1]
tm = r_best["t"] * 1e3
pm = (tm > 5) & (tm < t_end*1e3)
ax.plot(tm[pm], r_best["field_exact"][cx,cx,pm], "k-", lw=2.5, label="Analytical", zorder=5)
for N_show, col, ms in [(64, "C3", 3), (256, "C0", 2), (1024, "C2", 0)]:
    r = next(r for r in conv_results if r["N"] == N_show)
    tr = r["t"]*1e3; pr = (tr > 5) & (tr < t_end*1e3)
    if ms > 0:
        ax.plot(tr[pr], r["field"][cx,cx,pr], "o", color=col, ms=ms, label=f"N={N_show}", alpha=0.7)
    else:
        ax.plot(tr[pr], r["field"][cx,cx,pr], "--", color=col, lw=1, label=f"N={N_show}")
ax.set_xlabel("Time [ms]"); ax.set_ylabel("$u(0,0,d,t)$")
ax.set_title("(c) Waveform convergence", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# (d) Grid-independent rel L2
ax = axes[1, 0]
colors = ["C0", "C1", "C2", "C3"]
for i, r in enumerate(grid_results):
    sm = r["sig"] & (r["t"]*1e3 > 1) & (r["t"]*1e3 < t_end*1e3)
    ax.semilogy(r["t"][sm]*1e3, r["rel_l2"][sm], color=colors[i], lw=1.2,
                label=f"{r['Nx']}$^2$")
ax.axhline(1e-2, color="red", ls="--", lw=0.7, alpha=0.5)
ax.set_xlabel("Time [ms]"); ax.set_ylabel("Relative $L_2$")
ax.set_title("(d) Grid-independent accuracy", fontsize=10)
ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", alpha=0.2)
ax.set_ylim(1e-8, 1e0)

# (e) Peak error vs grid size
ax = axes[1, 1]
Nxs = [r["Nx"] for r in grid_results]
pks = [r["pk"] for r in grid_results]
mns = [r["mn"] for r in grid_results]
ax.semilogy(Nxs, pks, "ko-", lw=2, ms=7, label="Best rel $L_2$")
ax.semilogy(Nxs, mns, "s--", color="C1", lw=1.5, ms=6, label="Mean rel $L_2$")
ax.axhline(1e-2, color="red", ls="--", lw=0.7, alpha=0.5)
ax.set_xlabel("Grid size $N_x$"); ax.set_ylabel("Relative $L_2$")
ax.set_title("(e) Accuracy vs grid size", fontsize=10)
ax.set_xticks(Nxs); ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.2)
ax.set_ylim(1e-8, 1e0)

# (f) GPU timing vs N_NILT
ax = axes[1, 2]
walls = [r["wall"] for r in conv_results]
ax.loglog(Ns, walls, "ko-", lw=2, ms=7)
for r in conv_results:
    ax.annotate(f" {r['wall']:.0f}", (r["N"], r["wall"]), fontsize=6, va="bottom")
w_last = walls[-1]; N_last = Ns[-1]
N_ref = np.array([Ns[0], Ns[-1]])
ax.loglog(N_ref, w_last*(N_ref/N_last), "r--", lw=0.8, alpha=0.5, label="$O(N)$")
ax.set_xlabel("$N_{\\rm NILT}$"); ax.set_ylabel("Wall time [ms]")
ax.set_title("(f) GPU timing", fontsize=10)
ax.set_xscale("log", base=2); ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns], fontsize=7)
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.2)

fig.suptitle("Spectral convergence and grid-independent accuracy\n"
             f"Diffusion benchmark, $D$={D:.0e}, $d$={d*1e3:.0f} mm, "
             f"CFL-tuned $a$={a:.2f}, $T$={T:.0f} s",
             fontsize=11)
plt.tight_layout()
out = "paper/figures/convergence_combined.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"\nSaved -> {out}")
plt.close(fig)
