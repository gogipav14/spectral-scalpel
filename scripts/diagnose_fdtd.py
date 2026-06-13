"""
FDTD vs Scalpel: clean comparison via NILT[H(s)*G(s)].

Instead of convolving the impulse response with the source in the time domain
(which has normalization issues), compute the expected FDTD output directly
in Laplace domain: E(d,s) = H(s) * G(s), then NILT invert.

H(s) = exp(-gamma_z * d)     (propagation transfer function)
G(s) = L{g(t)}               (Laplace transform of Gaussian source)
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

from scalpel.core.nilt import nilt_scalar
from scalpel.core.dispersion import MU_0, EPS_0
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.reference.fdtd_maxwell_1d import fdtd_1d

# ── Materials ──────────────────────────────────────────────────────
materials = {
    "Dry sand":  {"sigma": 1e-4, "eps_r": 4.0},
    "Wet clay":  {"sigma": 0.1,  "eps_r": 10.0},
    "Seawater":  {"sigma": 4.0,  "eps_r": 80.0},
}

depth = 0.5  # m

fig, axes = plt.subplots(len(materials), 3, figsize=(16, 4*len(materials)))

for row, (mat_name, mp) in enumerate(materials.items()):
    sigma = mp["sigma"]
    eps_r = mp["eps_r"]
    epsilon = EPS_0 * eps_r

    c_mat = 1.0 / np.sqrt(MU_0 * epsilon)
    t_transit = depth / c_mat
    omega_cross = sigma / epsilon

    # Source: Gaussian pulse
    t_center = 3 * t_transit
    t_width = t_transit * 0.15
    t_end = t_center + 10 * t_transit

    def source_fn(t, tc=t_center, tw=t_width):
        return np.exp(-0.5 * ((t - tc) / tw)**2)

    # G(s) = Laplace transform of Gaussian: sqrt(2*pi)*tw * exp(-s*tc + s^2*tw^2/2)
    def G_laplace(s, tc=t_center, tw=t_width):
        return tw * np.sqrt(2*np.pi) * cmath.exp(-s * tc + s**2 * tw**2 / 2)

    # H(s) = exp(-gamma_z * d)
    def H_laplace(s):
        gamma_sq = MU_0 * (sigma * s + epsilon * s**2)
        gamma = cmath.sqrt(gamma_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-gamma * depth)

    # Product: F(s) = H(s) * G(s)
    def F_product(s):
        return H_laplace(s) * G_laplace(s)

    # CFL tune for the product
    params = tune_params(t_end=t_end, alpha_c=0.0, kappa=2.0, N_init=1024,
                         rho=max(omega_cross, 1.0/t_width))
    refined = refine_until_accept(F_product, params, t_end,
                                  eps_im_max=1e-2, eps_conv=1e-2,
                                  N_max=16384, t_eval_min=t_transit*0.5)

    # NILT of H*G (the expected FDTD output)
    f_expected, t_nilt, _ = nilt_scalar(F_product, refined.a, refined.T, refined.N)

    print(f"\n{'='*60}")
    print(f" {mat_name}: sigma={sigma}, eps_r={eps_r}")
    print(f" c_mat={c_mat:.2e}, t_transit={t_transit:.2e}s, omega_x={omega_cross:.2e}")
    print(f" NILT: a={refined.a:.4f}, T={refined.T:.4e}, N={refined.N}")

    # ── FDTD at multiple resolutions ──────────────────────────────
    Lz = depth * 3
    Nz_values = [500, 1000, 2000, 4000]

    # (a) Waveform overlay at finest grid
    ax = axes[row, 0]
    best_corr = -1
    for Nz in Nz_values:
        res, obs_sig, obs_t = fdtd_1d(
            sigma, eps_r, Lz, Nz, t_end,
            source_fn=source_fn, obs_z=depth, save_every=1)

        mask_nilt = (t_nilt > t_transit) & (t_nilt < t_end * 0.9)
        fdtd_interp = np.interp(t_nilt[mask_nilt], obs_t, obs_sig)
        ref_vals = f_expected[mask_nilt]
        rms_ref = np.sqrt(np.mean(ref_vals**2))
        err = np.sqrt(np.mean((fdtd_interp - ref_vals)**2))
        rel = err / rms_ref if rms_ref > 1e-300 else np.inf
        corr = np.corrcoef(fdtd_interp, ref_vals)[0, 1] if rms_ref > 1e-300 else 0

        print(f"  Nz={Nz:5d}: rel L2={rel:.4e}, corr={corr:.6f}")

        if Nz == Nz_values[-1]:
            mask_plot = (obs_t > 0) & (obs_t < t_end * 0.9)
            ax.plot(obs_t[mask_plot]*1e9, obs_sig[mask_plot], 'b-', lw=1,
                    label=f"FDTD Nz={Nz}")

    mask_plot = (t_nilt > 0) & (t_nilt < t_end * 0.9)
    ax.plot(t_nilt[mask_plot]*1e9, f_expected[mask_plot], 'r--', lw=1.5,
            label="NILT[H*G]")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("E")
    ax.set_title(f"{mat_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Convergence: rel L2 vs Nz
    ax = axes[row, 1]
    rels = []
    for Nz in [200, 500, 1000, 2000, 4000, 8000]:
        try:
            res, obs_sig, obs_t = fdtd_1d(
                sigma, eps_r, Lz, Nz, t_end,
                source_fn=source_fn, obs_z=depth, save_every=1)
            fdtd_interp = np.interp(t_nilt[mask_nilt], obs_t, obs_sig)
            err = np.sqrt(np.mean((fdtd_interp - ref_vals)**2))
            rel = err / rms_ref if rms_ref > 1e-300 else np.inf
            rels.append((Nz, rel))
        except Exception:
            pass

    if rels:
        nzs, errs = zip(*rels)
        ax.loglog(nzs, errs, 'ko-', lw=1.5, ms=5)
        # Reference: O(dz^2) convergence
        nz_ref = np.array([nzs[0], nzs[-1]])
        ax.loglog(nz_ref, errs[0] * (nz_ref[0]/nz_ref)**2, 'r--', lw=0.8,
                  alpha=0.5, label="$O(\\Delta z^2)$")
    ax.set_xlabel("Nz")
    ax.set_ylabel("Rel $L_2$")
    ax.set_title(f"FDTD convergence ({mat_name})")
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # (c) Timing comparison
    ax = axes[row, 2]
    import time
    # Scalpel timing (NILT of H*G)
    t0 = time.perf_counter()
    for _ in range(100):
        nilt_scalar(F_product, refined.a, refined.T, refined.N)
    scalpel_ms = (time.perf_counter() - t0) / 100 * 1e3

    fdtd_times = []
    for Nz in [500, 1000, 2000, 4000]:
        t0 = time.perf_counter()
        fdtd_1d(sigma, eps_r, Lz, Nz, t_end,
                source_fn=source_fn, obs_z=depth, save_every=Nz)
        fdtd_ms = (time.perf_counter() - t0) * 1e3
        fdtd_times.append((Nz, fdtd_ms))

    if fdtd_times:
        nzs, times = zip(*fdtd_times)
        ax.loglog(nzs, times, 's-', color='C1', lw=1.5, ms=5, label="FDTD")
        ax.axhline(scalpel_ms, color='C0', lw=2, label=f"Scalpel ({scalpel_ms:.1f}ms)")
        for nz, tm in fdtd_times:
            ax.annotate(f"{tm:.0f}ms", (nz, tm), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("Nz")
    ax.set_ylabel("Wall time [ms]")
    ax.set_title(f"Timing ({mat_name})")
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle("FDTD vs Scalpel: matched observable via NILT[H(s)G(s)]",
             fontsize=13, y=1.01)
fig.tight_layout()
out = "scripts/fdtd_diagnostic.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
