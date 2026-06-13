"""
Chromatography demo: properly matched comparisons.

Row 0: 1D outlet comparison — scalar NILT (analytical kr=0) vs MOL
Row 1: Radial concentration profiles from the cylindrical engine
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath, time

from scalpel.core.nilt import nilt_scalar
from scalpel.core.engine import CylindricalEngine, NILTParams
from scalpel.core.hankel import HankelTransform
from scalpel.core.dispersion import convection_diffusion_cylindrical
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.systems.chromatography import get_column
from scalpel.reference.mol_column import mol_column_1d
from scalpel.backends import get_backend

backend = get_backend()

columns = {
    "HPLC": get_column("hplc"),
    "Preparative": get_column("preparative"),
    "Process": get_column("process"),
}

results = {}

for name, col in columns.items():
    print(f"\n{'='*55}")
    print(f"  {name}: Pe={col.Pe:.0f}, tau={col.residence_time:.1f} s")
    print(f"{'='*55}")

    v, Dz, Dr = col.v, col.Dz, col.Dr
    L = col.L
    t_end = 3 * col.residence_time
    tau = col.residence_time

    # ── CFL tuning ────────────────────────────────────────────────
    rho_eff = 2 * np.pi / tau
    params = tune_params(
        t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
        eps_tail=1e-6, N_init=512, rho=rho_eff,
    )

    # DC mode transfer function (kr=0)
    conv_phase = v / (2 * Dz)

    def F_dc(s, _v=v, _Dz=Dz, _L=L):
        gamma_sq = _v**2 / (4*_Dz**2) + s / _Dz
        gamma = cmath.sqrt(gamma_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-(gamma - _v/(2*_Dz)) * _L)

    refined = refine_until_accept(F_dc, params, t_end,
                                  eps_im_max=1e-2, eps_conv=1e-2,
                                  N_max=8192, t_eval_min=t_end*0.01)
    print(f"  CFL: a={refined.a:.4f}, T={refined.T:.2f}, N={refined.N}")

    # ── 1. Analytical reference (high-N scalar NILT, kr=0) ────────
    N_ref = min(refined.N * 4, 16384)
    f_ref, t_ref, _ = nilt_scalar(F_dc, refined.a, refined.T, N_ref)

    # ── 2. Spectral scalar NILT (same N as engine would use) ──────
    f_spec, t_spec, _ = nilt_scalar(F_dc, refined.a, refined.T, refined.N)

    # ── 3. MOL reference (1D, same PDE) ───────────────────────────
    t0 = time.perf_counter()
    mol_res = mol_column_1d(v=v, Dz=Dz, L=L, Nz=500, t_end=t_end,
                             Nt_save=300)
    wall_mol = (time.perf_counter() - t0) * 1e3
    C_out_mol = mol_res.C[-1, :]
    print(f"  MOL: {wall_mol:.1f} ms")

    # Compute L2 error (spectral vs ref)
    mask = (t_spec > 0.1*tau) & (t_spec < 2.5*tau)
    mask_ref = (t_ref > 0.1*tau) & (t_ref < 2.5*tau)
    if np.any(mask):
        f_ref_interp = np.interp(t_spec[mask], t_ref[mask_ref], f_ref[mask_ref])
        rms = np.sqrt(np.mean(f_ref_interp**2))
        if rms > 1e-30:
            rel_l2 = np.sqrt(np.mean((f_spec[mask] - f_ref_interp)**2)) / rms
        else:
            rel_l2 = np.nan
    else:
        rel_l2 = np.nan
    print(f"  Rel L2 (spec vs ref): {rel_l2:.2e}")

    # ── 4. Cylindrical engine (radial structure) ──────────────────
    N_radial = 16
    ht = HankelTransform(col.R, N_radial)
    source_r = np.ones(N_radial)  # uniform inlet

    def disp_fn(s, KR, b, _v=v, _Dz=Dz, _Dr=Dr):
        return convection_diffusion_cylindrical(s, KR, _v, _Dz, _Dr, b)

    nilt_p = NILTParams(a=refined.a, T=refined.T, N=refined.N)
    cyl_engine = CylindricalEngine(disp_fn, ht, backend)

    field_rt, t_cyl = cyl_engine.forward(source_r, L, nilt_p,
                                          conv_phase=conv_phase)

    results[name] = dict(
        col=col, t_end=t_end, tau=tau,
        f_spec=f_spec, t_spec=t_spec,
        f_ref=f_ref, t_ref=t_ref,
        mol_res=mol_res, C_out_mol=C_out_mol,
        field_rt=field_rt, t_cyl=t_cyl,
        ht=ht, refined=refined, rel_l2=rel_l2,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Figure (2 rows x 3 cols)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ci, (name, r) in enumerate(results.items()):
    tau = r["tau"]

    # ── Row 0: Outlet elution profile ─────────────────────────────
    ax = axes[0, ci]

    # Reference (high-N NILT)
    t_rn = r["t_ref"] / tau
    rv = (r["t_ref"] > 0.05*tau) & (r["t_ref"] < r["t_end"]*0.9)
    ax.plot(t_rn[rv], r["f_ref"][rv], "k-", lw=2, label="Reference (high-N NILT)")

    # Spectral scalar (same N as engine)
    t_sn = r["t_spec"] / tau
    sv = (r["t_spec"] > 0.05*tau) & (r["t_spec"] < r["t_end"]*0.9)
    ax.plot(t_sn[sv], r["f_spec"][sv], "--", color="C0", lw=1.5,
            label=f"Spectral (N={r['refined'].N})")

    # MOL
    t_mn = r["mol_res"].t / tau
    mv = (r["mol_res"].t > 0.05*tau) & (r["mol_res"].t < r["t_end"]*0.9)
    ax.plot(t_mn[mv], r["C_out_mol"][mv], ":", color="C1", lw=1.2,
            label="MOL (Nz=500)")

    ax.set_xlabel("$t / \\tau$")
    ax.set_ylabel("$C_{out}$")
    ax.set_title(f"{name}\nPe={r['col'].Pe:.0f}, $\\tau$={tau:.1f} s",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── Row 1: Radial profiles ────────────────────────────────────
    ax = axes[1, ci]
    r_mm = r["ht"].r * 1e3

    for frac, ls, cc, lbl in [(0.5, "--", "C1", "0.5"),
                               (1.0, "-",  "C0", "1.0"),
                               (1.5, ":",  "C2", "1.5")]:
        idx = np.argmin(np.abs(r["t_cyl"] - frac*tau))
        if 0 < r["t_cyl"][idx] < r["t_end"]:
            ax.plot(r_mm, r["field_rt"][:, idx], f"o{ls}", color=cc,
                    lw=1.5, ms=4, label=f"$t = {lbl}\\tau$")

    ax.set_xlabel("$r$ [mm]")
    ax.set_ylabel("$C(r, z\\!=\\!L, t)$")
    ax.set_title("Radial concentration profile", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

fig.suptitle(
    "Chromatography spectral scalpel — Hankel + NILT vs MOL\n"
    "Axial dispersion + radial diffusion in packed bed columns",
    fontsize=12, y=1.01)

plt.tight_layout()
out = "paper/figures/chromatography_demo.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
