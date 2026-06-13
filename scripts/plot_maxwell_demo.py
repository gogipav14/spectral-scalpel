"""
Maxwell lossy medium — three conductivity regimes.

Properly matched comparison:
  - BOTH methods solve the 1D problem: source at z=0, observe at z=d
  - Spectral: scalar NILT of H(s)*G(s) where H=exp(-gamma*d), G=Laplace[source]
  - FDTD: 1D Yee scheme with same source waveform at z=0
  - Analytical: high-N scalar NILT as reference

The spectral engine's 2D field is shown separately as a spatial output panel.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath, time

from scalpel.core.nilt import nilt_scalar
from scalpel.core.dispersion import MU_0, EPS_0
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.reference.fdtd_maxwell_1d import fdtd_1d

# ── Materials ────────────────────────────────────────────────────────
materials = {
    "Dry sand":  dict(sigma=1e-4,  epsilon_r=4.0,  color="C0"),
    "Wet clay":  dict(sigma=0.1,   epsilon_r=10.0, color="C1"),
    "Seawater":  dict(sigma=4.0,   epsilon_r=80.0, color="C2"),
}

depth = 0.5  # meters


def make_source_fn(pulse_width):
    """Normalized Gaussian source pulse."""
    def src(t, pw=pulse_width):
        t0 = 3 * pw
        return np.exp(-0.5 * (t - t0)**2 / pw**2)
    return src


def laplace_gaussian(s, pw):
    """Laplace transform of Gaussian pulse exp(-0.5*(t-t0)^2/pw^2)."""
    t0 = 3 * pw
    return pw * cmath.exp(-s*t0 + 0.5*pw**2*s**2) * np.sqrt(2*np.pi)


results = {}

for name, mat in materials.items():
    sigma = mat["sigma"]
    eps_r = mat["epsilon_r"]
    epsilon = EPS_0 * eps_r

    print(f"\n{'='*60}")
    print(f"  {name}: sigma={sigma}, eps_r={eps_r}")
    print(f"{'='*60}")

    # Characteristic scales
    omega_cross = sigma / epsilon
    c_mat = 1.0 / np.sqrt(MU_0 * epsilon)
    transit_time = depth / c_mat

    # t_end: capture the pulse arrival + enough tail to see the shape.
    # Use the larger of transit-based and diffusion-based timescales,
    # but cap the ratio to avoid impossibly wide observation windows.
    t_end = max(15 * transit_time, 10 / omega_cross)
    t_end = min(t_end, 200 * transit_time, 1e-3)  # never more than 200 transits

    # Source pulse: width scales with transit time
    pulse_width = transit_time * 0.3

    source_fn = make_source_fn(pulse_width)

    # ── CFL tuning on the composite transfer function H(s)*G(s) ────
    # The NILT inverts F(s) = H(s)*G(s) = exp(-gamma*d)*G(s)
    # alpha_c is still 0 (branch point of H at s=0)
    # rho: spectral radius must cover BOTH the material crossover
    # AND the source pulse bandwidth 1/pulse_width
    source_bw = 1.0 / pulse_width
    rho_eff = max(omega_cross, source_bw)

    params = tune_params(
        t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
        eps_tail=1e-6, N_init=512, rho=rho_eff,
    )

    # Composite transfer function: propagator × source spectrum
    def F_composite(s, _sigma=sigma, _eps=epsilon, _d=depth, _pw=pulse_width):
        gamma_sq = MU_0 * (_sigma * s + _eps * s**2)
        gamma = cmath.sqrt(gamma_sq)
        if gamma.real < 0:
            gamma = -gamma
        H = cmath.exp(-gamma * _d)
        G = laplace_gaussian(s, _pw)
        return H * G

    # Bare propagator (impulse response, no source convolution)
    def F_bare(s, _sigma=sigma, _eps=epsilon, _d=depth):
        gamma_sq = MU_0 * (_sigma * s + _eps * s**2)
        gamma = cmath.sqrt(gamma_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-gamma * _d)

    # Phase 2 refinement on composite
    refined = refine_until_accept(F_composite, params, t_end,
                                  eps_im_max=1e-2, eps_conv=1e-2,
                                  N_max=8192, t_eval_min=transit_time*0.1)

    print(f"  CFL: a={refined.a:.4f}, T={refined.T:.4e}, N={refined.N}")

    # ── Spectral (scalar NILT of composite) ────────────────────────
    t0 = time.perf_counter()
    f_spec, t_spec, _ = nilt_scalar(F_composite, refined.a, refined.T, refined.N)
    wall_spec = (time.perf_counter() - t0) * 1e3

    # ── Analytical reference (high-N NILT of composite) ────────────
    N_ref = min(refined.N * 4, 16384)
    f_ref, t_ref, _ = nilt_scalar(F_composite, refined.a, refined.T, N_ref)

    # ── FDTD reference ─────────────────────────────────────────────
    Nz_fdtd = 2000
    Lz_fdtd = depth * 3  # domain extends beyond observation
    obs_z = depth

    t0_fdtd = time.perf_counter()
    fdtd_res, obs_signal, obs_time = fdtd_1d(
        sigma=sigma, epsilon_r=eps_r,
        Lz=Lz_fdtd, Nz=Nz_fdtd, t_end=t_end,
        source_fn=source_fn, obs_z=obs_z,
        save_every=max(1, int(t_end / (refined.delta_t * 10))),
    )
    wall_fdtd = (time.perf_counter() - t0_fdtd) * 1e3

    print(f"  Spectral: {wall_spec:.1f} ms  (N={refined.N})")
    print(f"  FDTD:     {wall_fdtd:.1f} ms  (Nz={Nz_fdtd})")

    # Compute L2 error (spectral vs reference)
    mask_ref = (t_ref > transit_time * 0.5) & (t_ref <= t_end * 0.9)
    mask_spec = (t_spec > transit_time * 0.5) & (t_spec <= t_end * 0.9)
    if np.any(mask_ref):
        f_ref_interp = np.interp(t_spec[mask_spec], t_ref[mask_ref], f_ref[mask_ref])
        rms_ref = np.sqrt(np.mean(f_ref_interp**2))
        if rms_ref > 1e-30:
            rel_l2 = np.sqrt(np.mean((f_spec[mask_spec] - f_ref_interp)**2)) / rms_ref
        else:
            rel_l2 = np.nan
    else:
        rel_l2 = np.nan
    print(f"  Rel L2 (spec vs ref): {rel_l2:.2e}")

    results[name] = dict(
        mat=mat, t_end=t_end, transit=transit_time,
        f_spec=f_spec, t_spec=t_spec,
        f_ref=f_ref, t_ref=t_ref,
        obs_signal=obs_signal, obs_time=obs_time,
        wall_spec=wall_spec, wall_fdtd=wall_fdtd,
        refined=refined, rel_l2=rel_l2,
        omega_cross=omega_cross,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Figure: 2 rows x 3 cols
#  Row 0: Spectral vs FDTD waveform (the money plot)
#  Row 1: Source pulse + CFL info
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for col, (name, r) in enumerate(results.items()):
    mat = r["mat"]
    transit = r["transit"]
    t_end = r["t_end"]

    # Time in transit-time units
    t_spec_n = r["t_spec"] / transit
    t_ref_n = r["t_ref"] / transit
    t_fdtd_n = r["obs_time"] / transit
    t_cut = t_end / transit

    # Valid range
    vm_spec = (t_spec_n > 0.5) & (t_spec_n <= t_cut * 0.9)
    vm_ref = (t_ref_n > 0.5) & (t_ref_n <= t_cut * 0.9)
    vm_fdtd = (t_fdtd_n > 0.5) & (t_fdtd_n <= t_cut * 0.9)

    # ── Row 0: Waveform comparison ────────────────────────────────
    ax = axes[0, col]
    ax.plot(t_ref_n[vm_ref], r["f_ref"][vm_ref],
            "k-", lw=2, label="Reference (high-N NILT)", alpha=0.8)
    ax.plot(t_spec_n[vm_spec], r["f_spec"][vm_spec],
            "--", color=mat["color"], lw=1.5, label=f"Spectral (N={r['refined'].N})")
    ax.plot(t_fdtd_n[vm_fdtd], r["obs_signal"][vm_fdtd],
            ":", color="gray", lw=1, label="FDTD (Nz=2000)", alpha=0.7)
    ax.axvline(1.0, color="red", ls="--", lw=0.5, alpha=0.4, label="$t_{transit}$")
    ax.set_xlabel("$t / t_{transit}$")
    ax.set_ylabel("E at depth $d$")
    ax.set_title(f"{name}\n$\\sigma$={mat['sigma']}, $\\varepsilon_r$={mat['epsilon_r']}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.25)

    # ── Row 1: Info panel ─────────────────────────────────────────
    ax = axes[1, col]
    ax.axis("off")
    info = (
        f"CFL-tuned parameters:\n"
        f"  a = {r['refined'].a:.4f}\n"
        f"  T = {r['refined'].T:.4e} s\n"
        f"  N = {r['refined'].N}\n"
        f"  ω_cross = {r['omega_cross']:.2e} rad/s\n"
        f"  t_transit = {transit:.4e} s\n"
        f"\nAccuracy:\n"
        f"  Rel L₂ (spec vs ref): {r['rel_l2']:.2e}\n"
        f"\nTiming:\n"
        f"  Spectral: {r['wall_spec']:.1f} ms\n"
        f"  FDTD:     {r['wall_fdtd']:.1f} ms"
    )
    ax.text(0.1, 0.95, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

fig.suptitle(
    "Maxwell lossy medium — spectral scalpel vs FDTD\n"
    "Same problem: Gaussian source at z=0, observe E-field at z=d=0.5 m",
    fontsize=12, y=1.02)
plt.tight_layout()
out = "paper/figures/maxwell_demo.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
