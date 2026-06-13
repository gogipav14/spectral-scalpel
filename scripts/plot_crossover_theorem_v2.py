"""
Figure 5: Crossover-Limited Recoverability (Theorem 2 verification).

Two panels, both log-log, all three EM materials:
  (a) Branch point s*(k_perp) — shows k^2 and k asymptotic regimes
  (b) Stiffness ratio sigma_NILT(k_perp) — shows where sigma=1 (k_perp,max)

Outputs parsable data table.
"""

import numpy as np
import matplotlib.pyplot as plt
from scalpel.core.dispersion import MU_0, EPS_0

# ── materials ────────────────────────────────────────────────────────
materials = [
    ("Dry sand",  1e-4, 4.0,  "#2176AE"),
    ("Wet clay",  0.1,  10.0, "#E8871E"),
    ("Seawater",  4.0,  80.0, "#3A8C5C"),
]

# ── NILT feasibility parameters ──────────────────────────────────────
L = 709.8; delta_s = 10.0; eps_tail = 1e-6
t_end = 1e-7  # 100 ns observation window
kappa = 2.0
T = kappa * t_end
t_max = 2 * T
s_star_max = (L - delta_s - np.log(1.0 / eps_tail)) / t_max

# ── compute ──────────────────────────────────────────────────────────
k_perp = np.logspace(-3, 6, 2000)

print("="*80)
print("CROSSOVER THEOREM DATA")
print("="*80)
print(f"t_end = {t_end:.0e} s,  kappa = {kappa},  T = {T:.0e} s,  t_max = {t_max:.0e} s")
print(f"s*_max = {s_star_max:.4e}")
print()

print(f"{'Material':<12s}  {'alpha':>12s}  {'beta':>12s}  {'omega_x':>12s}  "
      f"{'k_x [rad/m]':>12s}  {'k_max [rad/m]':>14s}")
print("-"*80)

mat_data = []
for name, sigma, eps_r, color in materials:
    alpha = MU_0 * sigma
    beta = MU_0 * EPS_0 * eps_r
    omega_cross = alpha / beta
    k_cross = alpha / np.sqrt(beta)

    # Branch point: s*(k) = (-alpha + sqrt(alpha^2 + 4*beta*k^2)) / (2*beta)
    s_star = (-alpha + np.sqrt(alpha**2 + 4*beta*k_perp**2)) / (2*beta)

    # Asymptotes
    s_diff = k_perp**2 / alpha           # k << k_cross
    s_wave = k_perp / np.sqrt(beta)      # k >> k_cross

    # Stiffness ratio
    sigma_nilt = (s_star * t_max + np.log(1.0 / eps_tail)) / (L - delta_s)

    # k_perp_max from closed-form
    k_max = np.sqrt(s_star_max * (beta * s_star_max + alpha))

    print(f"{name:<12s}  {alpha:>12.4e}  {beta:>12.4e}  {omega_cross:>12.4e}  "
          f"{k_cross:>12.4e}  {k_max:>14.4e}")

    mat_data.append(dict(name=name, sigma=sigma, eps_r=eps_r, color=color,
                         alpha=alpha, beta=beta, omega_cross=omega_cross,
                         k_cross=k_cross, k_max=k_max,
                         s_star=s_star, s_diff=s_diff, s_wave=s_wave,
                         sigma_nilt=sigma_nilt))

# Stiffness at key k values
print(f"\n{'Material':<12s}  {'k_perp':>10s}  {'s*':>12s}  {'sigma_NILT':>12s}  {'Status':>10s}")
print("-"*65)
for md in mat_data:
    for kp in [1, 10, 100, 1000, md["k_max"]]:
        alpha, beta = md["alpha"], md["beta"]
        s = (-alpha + np.sqrt(alpha**2 + 4*beta*kp**2)) / (2*beta)
        sig = (s * t_max + np.log(1.0/eps_tail)) / (L - delta_s)
        lbl = "k_max" if kp == md["k_max"] else ""
        print(f"{md['name']:<12s}  {kp:>10.2f}  {s:>12.3e}  {sig:>12.4f}  "
              f"{'FEAS' if sig < 1 else 'INFEAS':>10s}  {lbl}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure: 1 row x 2 cols
# ═══════════════════════════════════════════════════════════════════════
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── (a) Branch point s*(k_perp) log-log ──────────────────────────────
for md in mat_data:
    ax_a.loglog(k_perp, md["s_star"], "-", color=md["color"], lw=2,
                label=md["name"])
    # Mark k_cross
    ax_a.axvline(md["k_cross"], color=md["color"], ls=":", lw=0.8, alpha=0.5)

# Asymptotic reference lines (from first material)
md0 = mat_data[0]
ax_a.loglog(k_perp, md0["s_diff"], "--", color="gray", lw=0.8, alpha=0.6,
            label="$k^2/\\alpha$ (diffusive)")
ax_a.loglog(k_perp, md0["s_wave"], "-.", color="gray", lw=0.8, alpha=0.6,
            label="$k/\\sqrt{\\beta}$ (wave)")

# s*_max line
ax_a.axhline(s_star_max, color="red", ls="--", lw=1.2, alpha=0.7,
             label=f"$s^*_{{\\max}}$ = {s_star_max:.2e}")

ax_a.set_xlabel("$k_\\perp$ [rad/m]", fontsize=10)
ax_a.set_ylabel("$s^*(k_\\perp)$", fontsize=10)
ax_a.set_title("(a) Branch point: two asymptotic regimes", fontsize=11)
ax_a.legend(fontsize=7.5, loc="upper left")
ax_a.grid(True, which="both", alpha=0.2)
ax_a.set_xlim(k_perp[0], k_perp[-1])

# ── (b) Stiffness ratio sigma_NILT(k_perp) ──────────────────────────
for md in mat_data:
    ax_b.loglog(k_perp, md["sigma_nilt"], "-", color=md["color"], lw=2,
                label=md["name"])
    # Mark k_max with a dot
    ax_b.plot(md["k_max"], 1.0, "o", color=md["color"], ms=8, zorder=5)
    ax_b.annotate(f"  $k_{{\\perp,\\max}}$={md['k_max']:.1f}",
                  (md["k_max"], 1.0), fontsize=7, color=md["color"],
                  va="bottom")

# sigma = 1 threshold
ax_b.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.8,
             label="$\\sigma_{\\mathrm{NILT}} = 1$ (infeasible)")

# Shade infeasible region
ax_b.axhspan(1.0, 1e6, color="red", alpha=0.05)
ax_b.text(k_perp[10], 10, "INFEASIBLE", fontsize=9, color="red",
          alpha=0.5, fontweight="bold")
ax_b.text(k_perp[10], 0.1, "RECOVERABLE", fontsize=9, color="green",
          alpha=0.5, fontweight="bold")

ax_b.set_xlabel("$k_\\perp$ [rad/m]", fontsize=10)
ax_b.set_ylabel("$\\sigma_{\\mathrm{NILT}}(k_\\perp)$", fontsize=10)
ax_b.set_title(f"(b) Modewise stiffness ($t_{{\\mathrm{{end}}}}$ = {t_end*1e9:.0f} ns)",
               fontsize=11)
ax_b.legend(fontsize=7.5, loc="upper left")
ax_b.grid(True, which="both", alpha=0.2)
ax_b.set_xlim(k_perp[0], k_perp[-1])
ax_b.set_ylim(1e-4, 1e4)

fig.suptitle("Crossover-Limited Recoverability (Theorem 2)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out = "paper/figures/crossover_theorem.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"\nSaved -> {out}")
plt.close(fig)
