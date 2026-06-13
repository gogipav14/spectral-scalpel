"""
Acoustic demo — ultrasound pulse in viscous tissue.

Properly matched comparison:
  Row 0: 1D waveform — scalar NILT of H(s)*G(s) at kperp=0
         (same as Maxwell: convolve impulse response with source)
  Row 1: 2D pressure field snapshot from full engine
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath, time

from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import damped_acoustic
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.core.nilt import nilt_scalar

backend = get_backend()

media = {
    "Water":       dict(c=1500.0, nu=1e-6,  color="C0"),
    "Soft tissue": dict(c=1540.0, nu=1e-3,  color="C1"),
    "Bone":        dict(c=3000.0, nu=0.01,  color="C2"),
}

depth = 0.02    # 2 cm
Nx = 64
dx = 0.0005     # 0.5 mm
grid = GridParams(Nx=Nx, Ny=Nx, dx=dx, dy=dx)
x = (np.arange(Nx) - Nx//2) * dx

# Source for 2D engine
w_src = 0.0025
X, Y = np.meshgrid(x, x, indexing="ij")
source_np = np.exp(-(X**2 + Y**2) / (2*w_src**2))
source = backend.array(source_np, dtype=complex)


def laplace_gaussian(s, pw):
    """Laplace transform of Gaussian pulse."""
    t0 = 3 * pw
    return pw * cmath.exp(-s*t0 + 0.5*pw**2*s**2) * np.sqrt(2*np.pi)


results = {}

for name, med in media.items():
    c = med["c"]
    nu = med["nu"]

    print(f"\n{'='*55}")
    print(f"  {name}: c={c}, nu={nu}")
    print(f"{'='*55}")

    t_char = depth / c
    omega_cross = c**2 / nu
    pulse_width = t_char * 0.3
    t_end = 5 * t_char

    source_bw = 1.0 / pulse_width
    rho_eff = max(2*np.pi*2e6, source_bw)

    params = tune_params(
        t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
        eps_tail=1e-6, N_init=512, rho=rho_eff,
    )

    # Composite transfer function: propagator × source spectrum
    def F_composite(s, _c=c, _nu=nu, _d=depth, _pw=pulse_width):
        gamma_sq = (_nu/_c**2)*s + (1/_c**2)*s**2
        gamma = cmath.sqrt(gamma_sq)
        if gamma.real < 0:
            gamma = -gamma
        H = cmath.exp(-gamma * _d)
        G = laplace_gaussian(s, _pw)
        return H * G

    refined = refine_until_accept(F_composite, params, t_end,
                                  eps_im_max=1e-2, eps_conv=1e-2,
                                  N_max=8192, t_eval_min=t_end*0.01)
    print(f"  CFL: a={refined.a:.2f}, T={refined.T:.4e}, N={refined.N}")

    # ── 1D: scalar NILT of composite ──────────────────────────────
    f_spec, t_spec, _ = nilt_scalar(F_composite, refined.a, refined.T, refined.N)

    # Reference (high-N)
    N_ref = min(refined.N * 4, 16384)
    f_ref, t_ref, _ = nilt_scalar(F_composite, refined.a, refined.T, N_ref)

    # L2 error
    mask = (t_spec > 0.5*t_char) & (t_spec < t_end*0.9)
    mask_r = (t_ref > 0.5*t_char) & (t_ref < t_end*0.9)
    if np.any(mask):
        f_interp = np.interp(t_spec[mask], t_ref[mask_r], f_ref[mask_r])
        rms = np.sqrt(np.mean(f_interp**2))
        rel_l2 = np.sqrt(np.mean((f_spec[mask]-f_interp)**2)) / rms if rms > 1e-30 else np.nan
    else:
        rel_l2 = np.nan
    print(f"  Rel L2: {rel_l2:.2e}")

    # ── 2D engine (for spatial snapshot) ──────────────────────────
    def disp_fn(s, KX, KY, b, _c=c, _nu=nu):
        return damped_acoustic(s, KX, KY, _c, _nu, b)

    nilt_p = NILTParams(a=refined.a, T=refined.T, N=refined.N)
    engine = SpectralEngine(disp_fn, backend)

    _ = engine.forward(source, depth, grid, nilt_p)
    if backend.name == "jax":
        _[0].block_until_ready()

    t0 = time.perf_counter()
    field, t_arr = engine.forward(source, depth, grid, nilt_p)
    if backend.name == "jax":
        field.block_until_ready()
    wall = (time.perf_counter() - t0) * 1e3

    field_np = backend.to_numpy(field)
    t_np = backend.to_numpy(t_arr)
    print(f"  2D engine: {wall:.1f} ms")

    results[name] = dict(
        med=med, t_end=t_end, t_char=t_char,
        f_spec=f_spec, t_spec=t_spec,
        f_ref=f_ref, t_ref=t_ref,
        field=field_np, t_2d=t_np,
        wall=wall, refined=refined, rel_l2=rel_l2,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for col, (name, r) in enumerate(results.items()):
    med = r["med"]
    tc = r["t_char"]

    # ── Row 0: 1D waveform ────────────────────────────────────────
    ax = axes[0, col]
    t_rn = r["t_ref"] / tc
    t_sn = r["t_spec"] / tc
    rv = (r["t_ref"] > 0.3*tc) & (r["t_ref"] <= r["t_end"])
    sv = (r["t_spec"] > 0.3*tc) & (r["t_spec"] <= r["t_end"])

    ax.plot(t_rn[rv], r["f_ref"][rv], "k-", lw=2, label="Reference (high-N)")
    ax.plot(t_sn[sv], r["f_spec"][sv], "--", color=med["color"], lw=1.5,
            label=f"Spectral (N={r['refined'].N})")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("$t / t_{transit}$")
    ax.set_ylabel("Pressure at depth $d$")
    ax.set_title(f"{name}\nc={med['c']}, $\\nu$={med['nu']}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── Row 1: 2D pressure snapshot ───────────────────────────────
    ax = axes[1, col]
    # Snapshot near t_transit
    valid_idx = np.where((r["t_2d"] > 0.5*tc) & (r["t_2d"] < r["t_end"]))[0]
    if len(valid_idx) > 0:
        cx = Nx // 2
        center = r["field"][cx, cx, valid_idx]
        peak_idx = valid_idx[np.argmax(np.abs(center))]
    else:
        peak_idx = np.argmin(np.abs(r["t_2d"] - tc))

    snap = r["field"][:, :, peak_idx]
    vmax = np.max(np.abs(snap))
    if vmax < 1e-30:
        vmax = 1.0

    im = ax.imshow(snap.T,
                   extent=[x.min()*1e3, x.max()*1e3, x.min()*1e3, x.max()*1e3],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower", aspect="equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    t_snap = r["t_2d"][peak_idx]
    ax.set_title(f"t = {t_snap/tc:.2f} $t_{{transit}}$  ({r['wall']:.0f} ms)",
                 fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label="pressure")

fig.suptitle(
    "Acoustic spectral scalpel — three damping regimes\n"
    f"depth={depth*100:.0f} cm, grid={Nx}x{Nx}, dx={dx*1e3:.1f} mm",
    fontsize=12, y=1.01)

plt.tight_layout()
out = "paper/figures/acoustic_demo.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
