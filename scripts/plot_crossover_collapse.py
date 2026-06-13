"""
Cross-system feasibility collapse figure.

Normalize depth by the skin depth at the crossover frequency for each system.
If the crossover constant governs the feasibility scaling, the EM and
acoustic curves should collapse onto each other (both Λ₋ ≈ 2.603),
while chromatography sits on a different curve (Λ₊ ≈ 15.17).

This is the figure that sells the paper.
"""

import numpy as np
import matplotlib.pyplot as plt

from scalpel.core.dispersion import MU_0, EPS_0

# ── Systems ─────────────────────────────────────────────────────────
# For each system, define: omega_cross, gamma_at_cross(kperp=0), skin_depth

Nx = 128
dx = 0.01
kx_arr = np.fft.fftfreq(Nx, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx_arr, kx_arr, indexing="ij")
kperp = np.sqrt(KX**2 + KY**2)
total_modes = Nx * Nx

systems = {}

# --- Maxwell: dry sand ---
sigma_ds = 1e-4
eps_r_ds = 4.0
eps_ds = EPS_0 * eps_r_ds
omega_x_ds = sigma_ds / eps_ds
# gamma at crossover (kperp=0): gamma^2(s=j*omega_x) = mu0*(sigma*j*omega + eps*(j*omega)^2)
s_cross_ds = 1j * omega_x_ds
gamma_sq_ds = MU_0 * (sigma_ds * s_cross_ds + eps_ds * s_cross_ds**2)
gamma_dc_ds = np.sqrt(np.abs(gamma_sq_ds))  # |gamma| at crossover
skin_ds = 1.0 / gamma_dc_ds  # skin depth at crossover

def attn_maxwell(kperp_arr, depth, sigma, eps_r):
    epsilon = EPS_0 * eps_r
    s_x = 1j * sigma / epsilon
    gamma_sq = MU_0 * (sigma * s_x + epsilon * s_x**2) - kperp_arr**2
    gamma_z = np.sqrt(np.abs(gamma_sq))
    return np.exp(-gamma_z * depth)

systems["Maxwell (dry sand)"] = dict(
    omega_x=omega_x_ds, skin=skin_ds,
    attn_fn=lambda kp, d: attn_maxwell(kp, d, sigma_ds, eps_r_ds),
    color="C0", ls="-", marker="o",
    dispersion_class="hyperbolic",
)

# --- Maxwell: wet clay ---
sigma_wc = 0.1
eps_r_wc = 10.0
eps_wc = EPS_0 * eps_r_wc
omega_x_wc = sigma_wc / eps_wc
s_cross_wc = 1j * omega_x_wc
gamma_sq_wc = MU_0 * (sigma_wc * s_cross_wc + eps_wc * s_cross_wc**2)
skin_wc = 1.0 / np.sqrt(np.abs(gamma_sq_wc))

systems["Maxwell (wet clay)"] = dict(
    omega_x=omega_x_wc, skin=skin_wc,
    attn_fn=lambda kp, d: attn_maxwell(kp, d, sigma_wc, eps_r_wc),
    color="C0", ls="--", marker="s",
    dispersion_class="hyperbolic",
)

# --- Acoustics: soft tissue ---
c_st = 1540.0
nu_st = 1e-3
omega_x_st = c_st**2 / nu_st
s_cross_st = 1j * omega_x_st
gamma_sq_st = (nu_st / c_st**2) * s_cross_st + (1/c_st**2) * s_cross_st**2
skin_st = 1.0 / np.sqrt(np.abs(gamma_sq_st))

def attn_acoustic(kperp_arr, depth, c, nu):
    s_x = 1j * c**2 / nu
    gamma_sq = (nu/c**2)*s_x + (1/c**2)*s_x**2 - kperp_arr**2
    gamma_z = np.sqrt(np.abs(gamma_sq))
    return np.exp(-gamma_z * depth)

systems["Acoustics (tissue)"] = dict(
    omega_x=omega_x_st, skin=skin_st,
    attn_fn=lambda kp, d: attn_acoustic(kp, d, c_st, nu_st),
    color="C1", ls="-", marker="^",
    dispersion_class="hyperbolic",
)

# --- Acoustics: bone ---
c_bn = 3000.0
nu_bn = 0.01
omega_x_bn = c_bn**2 / nu_bn
s_cross_bn = 1j * omega_x_bn
gamma_sq_bn = (nu_bn/c_bn**2)*s_cross_bn + (1/c_bn**2)*s_cross_bn**2
skin_bn = 1.0 / np.sqrt(np.abs(gamma_sq_bn))

systems["Acoustics (bone)"] = dict(
    omega_x=omega_x_bn, skin=skin_bn,
    attn_fn=lambda kp, d: attn_acoustic(kp, d, c_bn, nu_bn),
    color="C1", ls="--", marker="v",
    dispersion_class="hyperbolic",
)

# --- Chromatography: HPLC ---
v_hplc = 1e-3
Dz_hplc = 1e-8
Dr_hplc = 1e-9
R_hplc = 2.3e-3
# Crossover: omega_x = v^2/(4*Dz)
omega_x_ch = v_hplc**2 / (4 * Dz_hplc)
# gamma at crossover: gamma^2 = v^2/(4Dz^2) + j*omega_x/Dz
s_cross_ch = 1j * omega_x_ch
gamma_sq_ch = v_hplc**2/(4*Dz_hplc**2) + s_cross_ch/Dz_hplc
skin_ch = 1.0 / np.sqrt(np.abs(gamma_sq_ch))

# For chromatography, use Hankel wavenumbers instead of Cartesian
from scipy.special import jn_zeros
N_hankel = 32
kr_arr = jn_zeros(0, N_hankel) / R_hplc
total_modes_ch = N_hankel

def attn_chrom(kr, depth, v, Dz, Dr):
    s_x = 1j * v**2 / (4*Dz)
    gamma_sq = v**2/(4*Dz**2) + s_x/Dz + Dr*kr**2/Dz
    gamma_z = np.sqrt(np.abs(gamma_sq))
    # Subtract convective phase
    conv = v / (2*Dz)
    net = np.abs(gamma_z - conv)
    return np.exp(-net * depth)

systems["Chromatography (HPLC)"] = dict(
    omega_x=omega_x_ch, skin=skin_ch,
    attn_fn=lambda kp, d: attn_chrom(kp, d, v_hplc, Dz_hplc, Dr_hplc),
    color="C2", ls="-", marker="D",
    dispersion_class="parabolic",
    kperp_arr=kr_arr, total_modes=total_modes_ch,
)


# ═══════════════════════════════════════════════════════════════════════
#  Compute recoverable fraction vs normalized depth
# ═══════════════════════════════════════════════════════════════════════

d_over_skin = np.linspace(0.01, 20.0, 500)
threshold = 1e-6  # micro-precision

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- (a) Recoverable fraction vs d/δ_cross ---
ax = axes[0]

for name, sys in systems.items():
    depths = d_over_skin * sys["skin"]
    fracs = []

    if "kperp_arr" in sys:
        kp = sys["kperp_arr"]
        n_total = sys["total_modes"]
    else:
        kp = kperp
        n_total = total_modes

    for dep in depths:
        attn = sys["attn_fn"](kp, dep)
        fracs.append(np.sum(attn > threshold) / n_total * 100)

    ax.plot(d_over_skin, fracs, color=sys["color"], ls=sys["ls"],
            lw=2, label=name)

ax.set_xlabel("Depth / skin depth at $\\omega_\\times$  ($d / \\delta_{\\mathrm{cross}}$)",
              fontsize=10)
ax.set_ylabel("Recoverable modes [%]", fontsize=10)
ax.set_title("(a) Feasibility collapse: normalized by crossover skin depth",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)
ax.set_ylim(0, 105)
ax.set_xlim(0, 20)

# Annotate Λ regions
ax.annotate("$\\Lambda_- \\approx 2.603$\n(hyperbolic)",
            xy=(5, 60), fontsize=9, color="C0",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
ax.annotate("$\\Lambda_+ \\approx 15.17$\n(parabolic)",
            xy=(12, 60), fontsize=9, color="C2",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

# --- (b) Same but log scale to show collapse more clearly ---
ax = axes[1]

for name, sys in systems.items():
    depths = d_over_skin * sys["skin"]

    if "kperp_arr" in sys:
        kp = sys["kperp_arr"]
        n_total = sys["total_modes"]
    else:
        kp = kperp
        n_total = total_modes

    fracs = []
    for dep in depths:
        attn = sys["attn_fn"](kp, dep)
        fracs.append(np.sum(attn > threshold) / n_total * 100)

    ax.semilogy(d_over_skin, np.maximum(fracs, 0.1),
                color=sys["color"], ls=sys["ls"], lw=2, label=name)

ax.set_xlabel("Depth / skin depth at $\\omega_\\times$", fontsize=10)
ax.set_ylabel("Recoverable modes [%]", fontsize=10)
ax.set_title("(b) Log scale — collapse of hyperbolic systems", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.25)
ax.set_xlim(0, 20)
ax.set_ylim(0.1, 200)

fig.suptitle(
    "Cross-system feasibility collapse\n"
    "Hyperbolic systems (EM, acoustics) collapse under $\\Lambda_-$;  "
    "parabolic (chromatography) governed by $\\Lambda_+$",
    fontsize=11, y=1.02)

plt.tight_layout()
out = "scripts/crossover_collapse.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)
