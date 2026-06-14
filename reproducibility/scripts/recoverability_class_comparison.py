"""
Side-by-side comparison of class-dependent recoverability cutoffs vs precision.

Plots k_{perp,max} versus floating-point exponent range L on log-log axes,
overlaying the two universality classes:

- Hyperbolic (p,q) = (1,2): k_{perp,max}^2 = s*_max(beta s*_max + alpha).
  Branch-point-abscissa bound. Empirical scaling sqrt(L) (diffusive) to L
  (wave) depending on regime within the class.

- Parabolic (p,q) = (0,1): k_{perp,max} ~ (L - margins) / d.
  Direct-underflow bound. Empirical scaling near L.

This combines:
- reproducibility/scripts/precision_scaling_data.pkl       (hyperbolic, Maxwell)
- reproducibility/scripts/precision_scaling_parabolic_truth_data.pkl  (parabolic, diffusion)

and produces a single comparative figure for the SISC paper.
"""

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

L_FLOAT16 = 11.09     # ln(np.finfo(float16).max)
L_FLOAT32 = 88.72     # ln(np.finfo(float32).max)
L_FLOAT64 = 709.78    # ln(np.finfo(float64).max)
L_FLOAT128 = 11355.6  # ln(np.finfo(longdouble).max), platform-dependent

import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(REPRO_ROOT, "data")
# REGEN dir is where freshly produced CSVs land so csv_diff can
# compare them against the archive in DATA. Defaults to DATA for
# local "refresh the archive" workflows; reproduce.sh overrides to
# reproducibility/regen for clean RCR runs.
REGEN = os.environ.get("REGEN_DIR", DATA)
# FIGURES_DIR is set by reproduce.sh to /results/figures on
# CodeOcean (so the snapshot timeline captures the PNG) and to
# reproducibility/figures/ for local dev.
OUT = os.environ.get("FIGURES_DIR", os.path.join(REPRO_ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(REGEN, exist_ok=True)

# Load the previously-computed sweep results
with open(os.path.join(DATA, "precision_scaling_data.pkl"), "rb") as f:
    hyp_data = pickle.load(f)            # [r64, r32] from verify_precision_scaling.py
with open(os.path.join(DATA, "precision_scaling_parabolic_truth_data.pkl"), "rb") as f:
    par_data = pickle.load(f)            # [r64, r32, metadata] from parabolic test


def first_cross(grid, errs, valids, threshold=1e-2):
    """First k_perp at which rel error exceeds threshold."""
    e = errs.copy()
    e[~valids] = 10.0
    crossing = np.where(e > threshold)[0]
    if len(crossing) == 0:
        return float("nan")
    return float(grid[crossing[0]])


def predicted_hyperbolic_kmax(L, alpha, beta, t_max, delta_s=10.0, ln_C_eps=13.82):
    """Branch-point-abscissa bound for hyperbolic class."""
    sm = (L - delta_s - ln_C_eps) / t_max
    if sm <= 0:
        return float("nan")
    return math.sqrt(sm * (beta * sm + alpha))


def predicted_parabolic_kmax(L, depth, delta_s=10.0, ln_C_eps=13.82):
    """Direct-underflow bound for parabolic class."""
    return max(0.0, (L - delta_s - ln_C_eps) / depth)


# ----- Hyperbolic empirical points (wet-clay Maxwell) -----
MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12
alpha_hyp = MU_0 * 0.1
beta_hyp = MU_0 * EPS_0 * 10.0
depth_hyp = 0.1
t_max_hyp = 4.0 * 1e-7   # 2 * kappa * t_end with kappa=2, t_end=1e-7

hyp_emp = {
    "L_64": first_cross(hyp_data[0]["k_grid"], hyp_data[0]["errs"], hyp_data[0]["valids"]),
    "L_32": first_cross(hyp_data[1]["k_grid"], hyp_data[1]["errs"], hyp_data[1]["valids"]),
}
hyp_pred = {
    "L_64": predicted_hyperbolic_kmax(L_FLOAT64, alpha_hyp, beta_hyp, t_max_hyp),
    "L_32": predicted_hyperbolic_kmax(L_FLOAT32, alpha_hyp, beta_hyp, t_max_hyp),
}

# ----- Parabolic empirical points (pure diffusion) -----
depth_par = 0.05
par_emp = {
    "L_64": first_cross(par_data[0]["k_grid"], par_data[0]["errs"], par_data[0]["valids"]),
    "L_32": first_cross(par_data[1]["k_grid"], par_data[1]["errs"], par_data[1]["valids"]),
}
par_pred = {
    "L_64": predicted_parabolic_kmax(L_FLOAT64, depth_par),
    "L_32": predicted_parabolic_kmax(L_FLOAT32, depth_par),
}

# ----- Theoretical scaling curves -----
L_range = np.linspace(20, 12000, 200)
hyp_curve = np.array([predicted_hyperbolic_kmax(L, alpha_hyp, beta_hyp, t_max_hyp) for L in L_range])
par_curve = np.array([predicted_parabolic_kmax(L, depth_par) for L in L_range])

# ----- Print summary table -----
print("=" * 80)
print("  Class-dependent recoverability cutoffs vs precision")
print("=" * 80)
print(f"{'Class':<20s}{'Precision':<12s}{'Predicted [rad/m]':>20s}{'Empirical [rad/m]':>20s}")
print("-" * 80)
print(f"{'Hyperbolic (1,2)':<20s}{'float64':<12s}{hyp_pred['L_64']:>20.2f}{hyp_emp['L_64']:>20.2f}")
print(f"{'Hyperbolic (1,2)':<20s}{'float32':<12s}{hyp_pred['L_32']:>20.2f}{hyp_emp['L_32']:>20.2f}")
print(f"{'Parabolic (0,1)':<20s}{'float64':<12s}{par_pred['L_64']:>20.2f}{par_emp['L_64']:>20.2f}")
print(f"{'Parabolic (0,1)':<20s}{'float32':<12s}{par_pred['L_32']:>20.2f}{par_emp['L_32']:>20.2f}")
print()
print(f"Hyperbolic predicted ratio: {hyp_pred['L_64'] / hyp_pred['L_32']:.3f}x")
print(f"Hyperbolic empirical ratio: {hyp_emp['L_64'] / max(hyp_emp['L_32'], 1e-10):.3f}x")
print(f"Parabolic predicted ratio: {par_pred['L_64'] / max(par_pred['L_32'], 1e-10):.3f}x")
print(f"Parabolic empirical ratio: {par_emp['L_64'] / max(par_emp['L_32'], 1e-10):.3f}x")

# CSV
out_csv = os.path.join(REGEN, "recoverability_class_comparison.csv")
with open(out_csv, "w") as f:
    f.write("class,precision,L,k_perp_predicted,k_perp_empirical\n")
    f.write(f"hyperbolic,float64,{L_FLOAT64},{hyp_pred['L_64']:.4f},{hyp_emp['L_64']:.4f}\n")
    f.write(f"hyperbolic,float32,{L_FLOAT32},{hyp_pred['L_32']:.4f},{hyp_emp['L_32']:.4f}\n")
    f.write(f"parabolic,float64,{L_FLOAT64},{par_pred['L_64']:.4f},{par_emp['L_64']:.4f}\n")
    f.write(f"parabolic,float32,{L_FLOAT32},{par_pred['L_32']:.4f},{par_emp['L_32']:.4f}\n")
print(f"\nSaved -> {out_csv}")

# Plot
fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Theoretical curves
ax.loglog(L_range, hyp_curve, "-", color="C3", lw=1.6, alpha=0.65,
          label=r"Hyperbolic $(p,q)=(1,2)$: $k_{\perp,\max}^{2}=s^{*}_{\max}(\beta s^{*}_{\max}+\alpha)$")
ax.loglog(L_range, par_curve, "-", color="C0", lw=1.6, alpha=0.65,
          label=r"Parabolic $(p,q)=(0,1)$: $k_{\perp,\max} \approx L/d$")

# Empirical points
ax.loglog([L_FLOAT32, L_FLOAT64], [hyp_emp["L_32"], hyp_emp["L_64"]],
          "s", color="C3", ms=10, mfc="white", mew=2,
          label="Hyperbolic empirical (Maxwell, wet clay)")
ax.loglog([L_FLOAT32, L_FLOAT64], [par_emp["L_32"], par_emp["L_64"]],
          "o", color="C0", ms=10, mfc="white", mew=2,
          label="Parabolic empirical (pure diffusion)")

# Precision vertical lines
for L, name in [(L_FLOAT16, "float16"), (L_FLOAT32, "float32"),
                (L_FLOAT64, "float64"), (L_FLOAT128, "float128")]:
    ax.axvline(L, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.text(L * 1.05, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 1 else 1,
            name, fontsize=8, color="gray", rotation=90, va="top")

ax.set_xlabel(r"Floating-point exponent range $L = \ln(\mathrm{MAX\_FLOAT})$")
ax.set_ylabel(r"$k_{\perp,\max}$ [rad/m]")
ax.set_title("Class-dependent recoverability bounds vs precision\n"
             r"Theory (lines) and self-consistency / truth-comparison empirics (markers)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", alpha=0.2)
ax.set_xlim(8, 15000)

plt.tight_layout()
out_png = os.path.join(OUT, "recoverability_class_comparison.png")
fig.savefig(out_png, dpi=180, bbox_inches="tight")
print(f"Saved -> {out_png}")
