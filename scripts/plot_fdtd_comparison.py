"""
Matched-accuracy FDTD vs Spectral Scalpel benchmark.

Fair comparison: same observable (impulse response at depth d),
same error tolerance, same hardware. Reports cost-per-solution-point.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import cmath

from scalpel.core.nilt import nilt_scalar, eps_im
from scalpel.core.dispersion import MU_0, EPS_0
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import maxwell_lossy
from scalpel.backends import get_backend
from scalpel.reference.fdtd_maxwell_1d import fdtd_1d

# ── Materials ───────────────────────────────────────────────────────
materials = {
    "Dry sand": {"sigma": 1e-4, "eps_r": 4.0},
    "Wet clay": {"sigma": 0.1,  "eps_r": 10.0},
    "Seawater": {"sigma": 4.0,  "eps_r": 80.0},
}

depth = 0.5  # m

# ── Helper: analytical 1D impulse response (Levy distribution) ─────
def analytical_1d(t, sigma, eps_r, d):
    """Analytical impulse response for 1D lossy Maxwell at kperp=0.

    For the diffusion limit (sigma-dominated):
    h(t) = (d*sqrt(mu*sigma)) / (2*sqrt(pi*t^3)) * exp(-d^2*mu*sigma/(4t))

    For the full telegrapher's equation, use scalar NILT as reference.
    """
    epsilon = EPS_0 * eps_r
    alpha = MU_0 * sigma
    beta = MU_0 * epsilon

    def F(s):
        gamma_sq = alpha * s + beta * s**2
        gamma = cmath.sqrt(gamma_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-gamma * d)

    return F


# ── Run benchmarks ──────────────────────────────────────────────────
backend = get_backend()
print(f"Backend: {backend.name}")
print()

results = []

for mat_name, mat_params in materials.items():
    sigma = mat_params["sigma"]
    eps_r = mat_params["eps_r"]
    epsilon = EPS_0 * eps_r

    # Determine timescale: need long enough to capture the full response
    omega_cross = sigma / (EPS_0 * eps_r)
    t_transit = depth * np.sqrt(MU_0 * epsilon)  # wave transit time
    t_diffuse = depth**2 * MU_0 * sigma          # diffusion time
    # Use diffusion timescale (longer) — the tail matters for accuracy
    t_end = max(t_diffuse * 5, t_transit * 20, 1e-6)
    t_end = min(t_end, 0.1)  # cap at 100ms

    print(f"{'='*60}")
    print(f" {mat_name}: sigma={sigma}, eps_r={eps_r}")
    print(f" omega_cross = {omega_cross:.2e} rad/s")
    print(f" t_end = {t_end:.2e} s")
    print(f"{'='*60}")

    # ── Spectral scalpel ────────────────────────────────────────
    F_dc = analytical_1d(None, sigma, eps_r, depth)

    params = tune_params(
        t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
        eps_tail=1e-6, N_init=512,
        rho=max(omega_cross, 1.0/t_end),
    )
    refined = refine_until_accept(
        F_dc, params, t_end,
        eps_im_max=1e-2, eps_conv=1e-2, N_max=16384,
        t_eval_min=t_end * 0.01,
    )

    # Run spectral engine (1x1 grid = 1D)
    grid = GridParams(Nx=1, Ny=1, dx=1.0, dy=1.0)
    nilt_p = NILTParams(a=refined.a, T=refined.T, N=refined.N)

    def disp_fn(s, KX, KY, b):
        return maxwell_lossy(s, KX, KY, sigma, eps_r, b)

    engine = SpectralEngine(disp_fn, backend)
    source = backend.array([[1.0]], dtype=complex)

    # Warmup
    _ = engine.forward(source, depth, grid, nilt_p)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(10):
        field, t_arr = engine.forward(source, depth, grid, nilt_p)
        if backend.name == "jax":
            field.block_until_ready()
    scalpel_wall = (time.perf_counter() - t0) / 10

    scalpel_f = backend.to_numpy(field[0, 0, :])
    scalpel_t = backend.to_numpy(t_arr)

    # Reference solution (high-N scalar NILT)
    f_ref, t_ref, _ = nilt_scalar(F_dc, refined.a, refined.T, max(refined.N * 4, 16384))

    # Interpolate reference to scalpel time grid
    mask = (scalpel_t > t_end * 0.01) & (scalpel_t <= t_end)
    f_ref_interp = np.interp(scalpel_t[mask], t_ref, f_ref)
    scalpel_err = np.sqrt(np.mean((scalpel_f[mask] - f_ref_interp)**2))
    ref_rms = np.sqrt(np.mean(f_ref_interp**2))
    scalpel_rel = scalpel_err / ref_rms if ref_rms > 1e-300 else np.inf

    print(f"  Scalpel: N={refined.N}, a={refined.a:.4f}, T={refined.T:.2e}")
    print(f"  Scalpel wall: {scalpel_wall*1e3:.2f} ms")
    print(f"  Scalpel rel L2: {scalpel_rel:.2e}")

    # ── FDTD ────────────────────────────────────────────────────
    # Sweep grid refinement to match scalpel accuracy
    Lz = depth * 2.5
    source_idx = 10

    # Gaussian pulse source
    t_pulse = t_end * 0.001
    def source_fn(t_val):
        return np.exp(-0.5 * (t_val / t_pulse)**2) / (t_pulse * np.sqrt(2*np.pi))

    fdtd_results = {}
    for Nz in [200, 500, 1000, 2000, 5000]:
        t0 = time.perf_counter()
        res = fdtd_1d(sigma, eps_r, Lz, Nz, t_end,
                      source_z_idx=source_idx, source_fn=source_fn,
                      save_every=max(1, int(Nz / 50)))
        fdtd_wall = time.perf_counter() - t0

        # Extract signal at depth
        obs_idx = int(depth / (Lz / Nz)) + source_idx
        if obs_idx >= Nz:
            obs_idx = Nz - 1
        fdtd_signal = res.E[obs_idx, :]
        fdtd_time = res.t

        # Compute FDTD error against reference
        valid_fdtd = (fdtd_time > t_end * 0.01) & (fdtd_time <= t_end)
        if np.sum(valid_fdtd) > 10 and np.max(np.abs(fdtd_signal[valid_fdtd])) > 1e-30:
            f_ref_fdtd = np.interp(fdtd_time[valid_fdtd], t_ref, f_ref)
            # Normalize FDTD to match reference amplitude
            fdtd_peak = np.max(np.abs(fdtd_signal[valid_fdtd]))
            ref_peak = np.max(np.abs(f_ref_fdtd))
            if fdtd_peak > 1e-30 and ref_peak > 1e-30:
                scale = ref_peak / fdtd_peak
                fdtd_err = np.sqrt(np.mean((fdtd_signal[valid_fdtd] * scale - f_ref_fdtd)**2))
                fdtd_ref_rms = np.sqrt(np.mean(f_ref_fdtd**2))
                fdtd_rel = fdtd_err / fdtd_ref_rms if fdtd_ref_rms > 1e-300 else np.inf
            else:
                fdtd_rel = np.inf
        else:
            fdtd_rel = np.inf

        fdtd_results[Nz] = {
            "wall": fdtd_wall, "rel_l2": fdtd_rel,
            "signal": fdtd_signal, "time": fdtd_time,
            "Nt": len(fdtd_time) * max(1, int(Nz / 50)),
        }
        print(f"  FDTD Nz={Nz:5d}: wall={fdtd_wall*1e3:8.1f} ms, rel L2={fdtd_rel:.2e}")

    # Find FDTD config with closest accuracy to scalpel
    best_nz = min(fdtd_results, key=lambda nz: abs(np.log10(max(fdtd_results[nz]["rel_l2"], 1e-20)) - np.log10(max(scalpel_rel, 1e-20))))
    best_fdtd = fdtd_results[best_nz]

    speedup = best_fdtd["wall"] / scalpel_wall if scalpel_wall > 0 else np.inf

    results.append({
        "material": mat_name,
        "sigma": sigma,
        "eps_r": eps_r,
        "scalpel_wall_ms": scalpel_wall * 1e3,
        "scalpel_rel_l2": scalpel_rel,
        "scalpel_N": refined.N,
        "fdtd_Nz": best_nz,
        "fdtd_wall_ms": best_fdtd["wall"] * 1e3,
        "fdtd_rel_l2": best_fdtd["rel_l2"],
        "speedup": speedup,
        "all_fdtd": fdtd_results,
    })

    print(f"\n  >>> Matched comparison: Nz={best_nz}")
    print(f"      Scalpel: {scalpel_wall*1e3:.2f} ms (rel L2 = {scalpel_rel:.2e})")
    print(f"      FDTD:    {best_fdtd['wall']*1e3:.1f} ms (rel L2 = {best_fdtd['rel_l2']:.2e})")
    print(f"      Speedup: {speedup:.0f}x")
    print()

# ── Summary table ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print(" MATCHED-ACCURACY BENCHMARK SUMMARY")
print("=" * 80)
print(f" {'Material':12s}  {'Scalpel [ms]':>12s}  {'FDTD [ms]':>10s}  "
      f"{'Scalpel L2':>10s}  {'FDTD L2':>10s}  {'Speedup':>8s}")
print(f" {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
for r in results:
    print(f" {r['material']:12s}  {r['scalpel_wall_ms']:12.2f}  "
          f"{r['fdtd_wall_ms']:10.1f}  {r['scalpel_rel_l2']:10.2e}  "
          f"{r['fdtd_rel_l2']:10.2e}  {r['speedup']:7.0f}x")

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, r in enumerate(results):
    ax = axes[idx]
    # FDTD wall time vs accuracy
    nzs = sorted(r["all_fdtd"].keys())
    walls = [r["all_fdtd"][nz]["wall"] * 1e3 for nz in nzs]
    errs = [r["all_fdtd"][nz]["rel_l2"] for nz in nzs]

    valid = [i for i, e in enumerate(errs) if e < 1e2 and e > 1e-15]
    if valid:
        ax.loglog([walls[i] for i in valid], [errs[i] for i in valid],
                  "s-", color="C1", ms=6, lw=1.5, label="FDTD")
        for i in valid:
            ax.annotate(f"Nz={nzs[i]}", (walls[i], errs[i]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")

    # Scalpel point
    ax.plot(r["scalpel_wall_ms"], r["scalpel_rel_l2"],
            "D", color="C0", ms=10, zorder=5, label=f"Scalpel (N={r['scalpel_N']})")

    ax.set_xlabel("Wall time [ms]")
    ax.set_ylabel("Relative $L_2$")
    ax.set_title(f"{r['material']}\n$\\sigma$={r['sigma']}, $\\varepsilon_r$={r['eps_r']}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Matched-Accuracy: Spectral Scalpel vs 1D FDTD", fontsize=12, y=1.02)
fig.tight_layout()
out = "scripts/fdtd_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
