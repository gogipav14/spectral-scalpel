"""
Ablation study: what breaks when NILT constraints are violated.

Shows four failure modes:
(a) Too-small a → aliasing (oscillations in time response)
(b) Too-small N → truncation (noisy late-time tails)
(c) Too-large depth → modes collapse to zero (feasibility failure)
(d) Near-crossover modes → highest stiffness ratio
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

from scalpel.core.nilt import nilt_scalar, eps_im
from scalpel.core.dispersion import MU_0, EPS_0
from scalpel.core.feasibility import tune_params

# ── Diffusion benchmark (analytical reference) ─────────────────────
D = 1e-4
d = 0.005
t_end = 2.0

def F_diffusion(s):
    gamma = cmath.sqrt(s / D)
    if gamma.real < 0:
        gamma = -gamma
    return cmath.exp(-gamma * d)

def analytical(t):
    a_levy = d / np.sqrt(D)
    mask = t > 0
    result = np.zeros_like(t)
    result[mask] = (a_levy / (2 * np.sqrt(np.pi * t[mask]**3))
                    * np.exp(-a_levy**2 / (4 * t[mask])))
    return result

# CFL-tuned reference (good parameters)
params_good = tune_params(t_end=t_end, alpha_c=0.0, kappa=2.0, N_init=2048, rho=D/d**2)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── (a) Too-small a: aliasing ──────────────────────────────────────
ax = axes[0, 0]
a_good = params_good.a
T_good = params_good.T
N_good = params_good.N

# Hand-pick a value that's too small (below the aliasing floor)
a_values = [0.001, 0.01, a_good]
labels = ["$a = 0.001$ (too small)", "$a = 0.01$", f"$a = {a_good:.2f}$ (CFL-tuned)"]
colors = ["red", "orange", "C0"]

for a_val, label, color in zip(a_values, labels, colors):
    f, t, _ = nilt_scalar(F_diffusion, a_val, T_good, N_good)
    mask = (t > 0.01) & (t <= t_end)
    ax.plot(t[mask], f[mask], color=color, lw=1.5, label=label)

# Analytical
t_an = np.linspace(0.01, t_end, 500)
ax.plot(t_an, analytical(t_an), "k--", lw=1, label="Analytical", alpha=0.5)

ax.set_xlabel("Time [s]")
ax.set_ylabel("$h(t)$")
ax.set_title("(a) Ablation: Bromwich shift $a$\n(too small → aliasing)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── (b) Too-small N: truncation ───────────────────────────────────
ax = axes[0, 1]
N_values = [64, 256, 1024, N_good]
for N_val in N_values:
    f, t, z = nilt_scalar(F_diffusion, a_good, T_good, N_val)
    mask = (t > 0.01) & (t <= t_end)
    eps = eps_im(z[mask]) if np.any(mask) else np.inf
    ax.plot(t[mask], f[mask], lw=1.5,
            label=f"$N = {N_val}$, $\\varepsilon_{{Im}}$ = {eps:.1e}")

ax.plot(t_an, analytical(t_an), "k--", lw=1, label="Analytical", alpha=0.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("$h(t)$")
ax.set_title("(b) Ablation: FFT size $N$\n(too small → truncation)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ── (c) Too-large depth: modes collapse ───────────────────────────
ax = axes[1, 0]
# Use Maxwell (dry sand) for this — show attenuation at different depths
sigma, eps_r = 0.01, 4.0
epsilon = EPS_0 * eps_r
alpha_em = MU_0 * sigma
beta_em = MU_0 * epsilon

depths_em = [0.01, 0.1, 0.5, 2.0]
t_end_em = 1e-5

# Compute per-mode attenuation at different depths
k_perp = np.linspace(0, 100, 200)
for d_em in depths_em:
    # Attenuation of each mode: |H(s=a, k_perp, d)| where s is on the real axis
    s_test = 1.0  # representative real s
    gamma_sq = alpha_em * s_test + beta_em * s_test**2 - k_perp**2
    gamma_z = np.sqrt(np.abs(gamma_sq))
    attenuation = np.exp(-gamma_z * d_em)
    ax.semilogy(k_perp, np.clip(attenuation, 1e-20, None), lw=1.5,
                label=f"$d = {d_em}$ m")

ax.axhline(1e-15, color="red", ls="--", lw=0.8, alpha=0.6, label="Machine $\\varepsilon$")
ax.set_xlabel("$k_\\perp$ [rad/m]")
ax.set_ylabel("Mode amplitude $|H|$")
ax.set_title("(c) Ablation: propagation depth\n(deeper → fewer recoverable modes)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(1e-20, 10)

# ── (d) Stiffness ratio near crossover ───────────────────────────
ax = axes[1, 1]

# Modewise stiffness for two materials
materials = [
    ("Dry sand", 1e-4, 4.0, "C0"),
    ("Wet clay", 0.1,  10.0, "C1"),
]
t_end_stiff = 1e-6
t_max_stiff = 2 * 2.0 * t_end_stiff  # kappa=2

k_range = np.logspace(-3, 5, 500)
for name, sig, er, color in materials:
    al = MU_0 * sig
    be = MU_0 * EPS_0 * er
    omega_x = al / be
    k_x = al / np.sqrt(be)

    # Branch point: s* = [-alpha + sqrt(alpha^2 + 4*beta*k^2)] / (2*beta)
    discriminant = al**2 + 4 * be * k_range**2
    s_star = (-al + np.sqrt(discriminant)) / (2 * be)

    sigma_nilt = (s_star * t_max_stiff + np.log(1.0 / 1e-6)) / (709.8 - 10.0)

    ax.semilogx(k_range, sigma_nilt, color=color, lw=2, label=name)
    # Mark crossover wavenumber
    ax.axvline(k_x, color=color, ls=":", lw=1, alpha=0.5)
    ax.annotate(f"$k_{{\\perp,\\times}}$", (k_x, 0.05),
                color=color, fontsize=8, ha="right")

ax.axhline(1.0, color="red", ls="--", lw=1.5, label="$\\sigma_{NILT} = 1$ (infeasible)")
ax.set_xlabel("$k_\\perp$ [rad/m]")
ax.set_ylabel("$\\sigma_{NILT}(k_\\perp)$")
ax.set_title("(d) Modewise stiffness ratio\n(crossover $k_{\\perp,\\times}$ marks scaling transition)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.5)

fig.suptitle("Ablation study: what breaks when constraints are violated", fontsize=12, y=1.01)
fig.tight_layout()
out = "scripts/ablation.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)
