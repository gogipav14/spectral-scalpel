"""
#3  Feasibility map
────────────────────
For Maxwell lossy medium, show which (kx,ky) modes are recoverable
at different depths and conductivities.

The key insight: the NILT feasibility constraint a_min <= a_max
becomes mode-dependent when the abscissa of convergence depends on
kperp. For Maxwell: the branch point shifts with kperp^2, changing
which modes can be accurately inverted at a given depth/time.

We also show the SNR-based mode limit: even if feasible, modes whose
signal is below machine precision contribute only noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

from scalpel.core.dispersion import MU_0, EPS_0

# ═══════════════════════════════════════════════════════════════════════
#  Physics: mode-dependent attenuation for Maxwell
# ═══════════════════════════════════════════════════════════════════════

def mode_attenuation(kperp, sigma, epsilon_r, depth, t_end, N_brom=2048):
    """For each kperp, compute the signal attenuation at the peak time.

    The transfer function per mode is H(s) = exp(-gamma_z * d) where
    gamma_z = sqrt(mu_0*(sigma*s + eps*s^2) - kperp^2).

    At the "characteristic" Laplace point s = sigma/eps (the crossover),
    gamma_z is real-valued and gives the DC-like attenuation.
    """
    epsilon = EPS_0 * epsilon_r
    # Evaluate at the crossover frequency s_cross = sigma/epsilon
    s_cross = sigma / epsilon
    gamma_sq = MU_0 * (sigma * s_cross + epsilon * s_cross**2) - kperp**2
    gamma_sq = np.where(gamma_sq.real > 0, gamma_sq, gamma_sq)
    gamma_z = np.sqrt(np.abs(gamma_sq))
    attenuation = np.exp(-gamma_z * depth)
    return attenuation


def cfl_feasibility(kperp, sigma, epsilon_r, depth, t_end,
                    kappa=2.0, eps_tail=1e-6, delta_s=10.0, L=709.8):
    """Check CFL feasibility per mode.

    The mode-dependent alpha_c shifts with kperp because higher
    transverse wavenumbers change the singularity structure.
    For Maxwell: alpha_c ~ 0 for all modes (branch point at s=0),
    but the EFFECTIVE observation difficulty increases because
    the signal decays faster for high-kperp modes.

    Returns: feasible (bool array), margin (float array)
    """
    T = kappa * t_end
    t_max = 2 * T
    a_max = (L - delta_s) / t_max

    # alpha_c = 0 for all modes (branch point is at s=0)
    alpha_c = 0.0
    alias_factor = (2*kappa - 1) * t_end
    a_alias = alpha_c + np.log(1.0 / eps_tail) / alias_factor
    a_min = max(a_alias, 1e-3)

    feasible = a_min <= a_max
    margin = a_max - a_min

    # But: signal must be above machine epsilon
    attn = mode_attenuation(kperp, sigma, epsilon_r, depth, t_end)
    snr_ok = attn > 1e-15  # above machine epsilon

    return feasible & snr_ok, margin, attn


# ═══════════════════════════════════════════════════════════════════════
#  Compute feasibility maps
# ═══════════════════════════════════════════════════════════════════════

materials = [
    ("Dry sand",  1e-4,  4.0),
    ("Wet clay",  0.1,   10.0),
    ("Seawater",  4.0,   80.0),
]

Nx = 128
dx = 0.01  # 1 cm
kx_arr = np.fft.fftfreq(Nx, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx_arr, kx_arr, indexing="ij")
kperp = np.sqrt(KX**2 + KY**2)

depths = [0.1, 0.5, 1.0, 2.0, 5.0]

# ═══════════════════════════════════════════════════════════════════════
#  Figure 1: Feasibility vs depth for three materials (3x5 grid)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, len(depths), figsize=(16, 10))

for row, (mat_name, sigma, eps_r) in enumerate(materials):
    epsilon = EPS_0 * eps_r
    omega_cross = sigma / epsilon
    t_end = min(max(50 / omega_cross, 1e-8), 1e-3)

    for col, dep in enumerate(depths):
        ax = axes[row, col]

        _, _, attn = cfl_feasibility(kperp, sigma, eps_r, dep, t_end)

        # Log attenuation map (shifted to positive for display)
        log_attn = np.log10(np.clip(attn, 1e-16, 1.0))

        im = ax.imshow(
            np.fft.fftshift(log_attn),
            extent=[kx_arr.min(), kx_arr.max(),
                    kx_arr.min(), kx_arr.max()],
            cmap="RdYlGn", vmin=-16, vmax=0,
            origin="lower", aspect="equal",
        )

        # Contour at -15 (machine epsilon)
        ax.contour(
            np.fft.fftshift(KX), np.fft.fftshift(KY),
            np.fft.fftshift(log_attn),
            levels=[-15], colors="red", linewidths=1.5,
        )
        # Contour at -6 (micro-precision)
        ax.contour(
            np.fft.fftshift(KX), np.fft.fftshift(KY),
            np.fft.fftshift(log_attn),
            levels=[-6], colors="orange", linewidths=1,
            linestyles="--",
        )

        if row == 0:
            ax.set_title(f"d = {dep} m", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{mat_name}\n$k_y$ [rad/m]", fontsize=9)
        else:
            ax.set_ylabel("")
        if row == 2:
            ax.set_xlabel("$k_x$ [rad/m]", fontsize=9)

        ax.tick_params(labelsize=7)

# Colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
                    label="log$_{10}$(attenuation)")

fig.suptitle(
    "Mode attenuation map — which (k$_x$, k$_y$) modes survive propagation?\n"
    "Red contour: machine epsilon (10$^{-15}$).  "
    "Orange dashed: 10$^{-6}$ (micro-precision)",
    fontsize=11, y=1.02)

plt.tight_layout()
out1 = "scripts/feasibility_map.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
print(f"Saved -> {out1}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
#  Figure 2: Recoverable mode count vs depth (1D summary)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

depth_sweep = np.linspace(0.01, 5.0, 200)

for i, (mat_name, sigma, eps_r) in enumerate(materials):
    ax = axes[i]
    epsilon = EPS_0 * eps_r
    omega_cross = sigma / epsilon
    t_end = min(max(50 / omega_cross, 1e-8), 1e-3)

    total_modes = Nx * Nx
    n_recoverable_15 = []
    n_recoverable_6 = []
    n_recoverable_3 = []

    for dep in depth_sweep:
        _, _, attn = cfl_feasibility(kperp, sigma, eps_r, dep, t_end)
        n_recoverable_15.append(np.sum(attn > 1e-15))
        n_recoverable_6.append(np.sum(attn > 1e-6))
        n_recoverable_3.append(np.sum(attn > 1e-3))

    ax.plot(depth_sweep, np.array(n_recoverable_15)/total_modes*100,
            "r-", lw=2, label="> 10$^{-15}$")
    ax.plot(depth_sweep, np.array(n_recoverable_6)/total_modes*100,
            "--", color="orange", lw=1.5, label="> 10$^{-6}$")
    ax.plot(depth_sweep, np.array(n_recoverable_3)/total_modes*100,
            "g:", lw=1.5, label="> 10$^{-3}$")

    ax.set_xlabel("Depth [m]")
    ax.set_ylabel("Recoverable modes [%]")
    ax.set_title(f"{mat_name} ($\\sigma$={sigma})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", ls="--", lw=0.5, alpha=0.5)

fig.suptitle(
    "Recoverable mode fraction vs propagation depth\n"
    f"Grid: {Nx}x{Nx}, dx={dx*100:.0f} cm",
    fontsize=11, y=1.02)
plt.tight_layout()
out2 = "scripts/feasibility_vs_depth.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved -> {out2}")
plt.close(fig)
