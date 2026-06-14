"""
Redesigned Fig. 4: four-panel 'evidence pack' for the precision-limited
recoverability bound, addressing the GPT / Gemini critique:

 (a) Reduced-coordinate collapse across 3 systems (reused data).
 (b) Physical-units L-shift: same NILT at float32 and float64 on wet-clay
     Maxwell, showing k_perp,max moves from 23.3 to 4.8 rad/m as predicted.
 (c) Classifier view: accept/reject markers from the CFL feasibility
     inequality overlaid on the wet-clay curve, with the theoretically
     infeasible region shaded.
 (d) Ground-truth correlation: for the heat equation (where the shifted-Levy
     analytical inverse is available), a scatter of self-consistency error
     vs relative error against the analytical reference at multiple k_perp.
     Shows that the a-posteriori indicator tracks true error up to the
     failure region.
"""
from __future__ import annotations

import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(REPRO_ROOT, "data")
OUT_DIR = os.path.join(REPRO_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12
L_DBL = 709.78
L_F32 = 88.72
DELTA_S = 10.0
EPS_TAIL = 1e-6
C_TAIL = 1.0
KAPPA = 2.0

# ---------------------------------------------------------------
# Panel (a): reduced coordinate, 3 systems (reuse existing data)
# ---------------------------------------------------------------
with open(os.path.join(DATA, "recoverability_bound_data.pkl"), "rb") as f:
    three_systems = pickle.load(f)

# ---------------------------------------------------------------
# Panel (b): float32 vs float64 in physical k_perp units
# ---------------------------------------------------------------
with open(os.path.join(DATA, "precision_scaling_data.pkl"), "rb") as f:
    prec_data = pickle.load(f)   # [r64, r32]

# ---------------------------------------------------------------
# Panel (c): accept/reject classifier from Eq. (2) on wet-clay
#   For each k_perp in the sweep, classify s*(k) t_max + ln(C/eps) < L-delta
# ---------------------------------------------------------------
# Wet clay params from sweep
ALPHA_WC = MU_0 * 0.1
BETA_WC  = MU_0 * EPS_0 * 10.0
t_end_wc = 1e-7
t_max_wc = 2 * KAPPA * t_end_wc

def classify(k_perp, L):
    s_star = (-ALPHA_WC + math.sqrt(ALPHA_WC**2 + 4*BETA_WC*k_perp**2))/(2*BETA_WC)
    lhs = s_star * t_max_wc + math.log(C_TAIL/EPS_TAIL)
    rhs = L - DELTA_S
    return lhs < rhs   # True = accepted

wc = [s for s in three_systems if s["name"].startswith("Maxwell wet clay")][0]
accept = np.array([classify(k, L_DBL) for k in wc["k_grid"]])

# ---------------------------------------------------------------
# Panel (d): ground-truth correlation on heat equation
# ---------------------------------------------------------------
# Compute relative L2 error vs Levy analytical for several k_perp
# alongside the self-consistency metric used in panels (a-c)
from scalpel.core.nilt import nilt_scalar
import cmath

D = 1e-4
d = 5e-3

def levy_shifted_scalar(t, d, D, kperp_sq):
    a = d / np.sqrt(D)
    out = np.zeros_like(t)
    mask = t > 0
    out[mask] = (a/(2*np.sqrt(np.pi*t[mask]**3))
                 * np.exp(-a**2/(4*t[mask]) - D*kperp_sq*t[mask]))
    return out

# For heat eq, the branch point is s*(k) = D*k^2 (diffusive regime)
# CFL t_max = 2*kappa*t_end
t_end_heat = 4.0  # seconds
t_max_heat = 2*KAPPA*t_end_heat

def heat_s_star_max(L):
    return (L - DELTA_S - math.log(C_TAIL/EPS_TAIL))/t_max_heat
def heat_k_max(L):
    return math.sqrt(heat_s_star_max(L) / D)

k_max_heat_64 = heat_k_max(L_DBL)
print(f"Heat equation predicted k_max at float64: {k_max_heat_64:.3f}")

k_sweep_heat = np.geomspace(k_max_heat_64*1e-3, k_max_heat_64*2.0, 18)

def F_heat(s, k):
    gamma = cmath.sqrt(s/D + k**2)
    if gamma.real < 0: gamma = -gamma
    return cmath.exp(-gamma*d)

corr_self: list[float] = []
corr_truth: list[float] = []
for k in k_sweep_heat:
    a = D*k**2 * 1.2 + 0.5
    T = KAPPA*t_end_heat
    try:
        f_ref, t_ref, _ = nilt_scalar(lambda s: F_heat(s, k), a, T, 4096)
        f_tst, t_tst, _ = nilt_scalar(lambda s: F_heat(s, k), a, T, 512)
    except Exception:
        corr_self.append(float('nan')); corr_truth.append(float('nan')); continue

    if not (np.all(np.isfinite(f_ref)) and np.all(np.isfinite(f_tst))):
        corr_self.append(float('nan')); corr_truth.append(float('nan')); continue

    mask = (t_ref > 0.1*t_end_heat) & (t_ref < 0.9*t_end_heat)
    if not np.any(mask):
        corr_self.append(float('nan')); corr_truth.append(float('nan')); continue

    f_tst_on_ref = np.interp(t_ref[mask], t_tst, f_tst)
    eta = 1e-30
    num_sc = np.linalg.norm(f_ref[mask] - f_tst_on_ref)
    den_sc = np.linalg.norm(f_ref[mask]) + eta
    err_sc = num_sc/den_sc

    truth = levy_shifted_scalar(t_ref[mask], d, D, k**2)
    num_t = np.linalg.norm(f_ref[mask] - truth)
    den_t = np.linalg.norm(truth) + eta
    err_t = num_t/den_t

    corr_self.append(float(err_sc))
    corr_truth.append(float(err_t))
    print(f"  k={k:.3e}  self-cons={err_sc:.3e}  vs truth={err_t:.3e}")

corr_self_arr = np.array(corr_self)
corr_truth_arr = np.array(corr_truth)

# =============================================================
# PLOT
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- Panel (a) ---
ax = axes[0, 0]
colors = {"Maxwell dry sand": "C0", "Maxwell wet clay": "C1", "Acoustic tissue": "C2"}
for s in three_systems:
    k_n = s["k_grid"]/s["k_max"]
    errs = s["errs"].copy()
    errs[~s["valids"]] = 10.0   # catastrophic-overflow marker
    ax.loglog(k_n, errs, "o-", lw=1.5, ms=5, color=colors[s["name"]], label=s["name"])
ax.axvline(1.0, color="red", ls="--", lw=1.2, label=r"$k_{\perp,\max}$ (Eq. 4)")
ax.axhline(1e-2, color="gray", ls=":", lw=0.9, label=r"$1\%$ threshold")
ax.set_xlabel(r"$k_\perp / k_{\perp,\max}^{\mathrm{theorem}}$")
ax.set_ylabel(r"Self-consistency indicator $E_{\mathrm{sc}}$")
ax.set_title(r"(a) Reduced-coordinate collapse (three systems, float64)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.25)

# --- Panel (b) physical units, float32 vs float64 ---
ax = axes[0, 1]
for r, col, mk in [(prec_data[0], "C0", "o"), (prec_data[1], "C3", "s")]:
    errs = r["errs"].copy()
    errs[~r["valids"]] = 10.0
    ax.loglog(r["k_grid"], errs, "-"+mk, lw=1.5, ms=5, color=col,
              label=rf"{r['label']}: $k_{{\perp,\max}}$ = {r['kmax']:.2f} rad/m")
    ax.axvline(r["kmax"], color=col, ls="--", lw=1.0, alpha=0.7)
ax.axhline(1e-2, color="gray", ls=":", lw=0.9)
ax.set_xlabel(r"$k_\perp$ [rad/m]")
ax.set_ylabel(r"$E_{\mathrm{sc}}$")
ratio = prec_data[0]["kmax"] / prec_data[1]["kmax"]
ax.set_title(r"(b) Precision-dependent cutoff, wet-clay Maxwell"
             "\n"
             rf"measured $k_{{\perp,\max}}^{{(64)}} / k_{{\perp,\max}}^{{(32)}} = {ratio:.2f}\times$ (theorem predicts 4.82$\times$)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.25)

# --- Panel (c) classifier view ---
ax = axes[1, 0]
k_n_wc = wc["k_grid"]/wc["k_max"]
errs_wc = wc["errs"].copy()
errs_wc[~wc["valids"]] = 10.0
ax.loglog(k_n_wc[accept], errs_wc[accept], "o", color="C2", ms=7, label=r"feasibility accept")
ax.loglog(k_n_wc[~accept], errs_wc[~accept], "x", color="C3", ms=9, mew=2, label=r"feasibility reject")
ax.loglog(k_n_wc, errs_wc, "-", lw=1.0, color="k", alpha=0.4, zorder=1)
ax.axvline(1.0, color="red", ls="--", lw=1.2, label=r"Eq. (4) cutoff")
ax.axvspan(1.0, ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 3, alpha=0.13, color="red",
           label=r"Eq. (4) infeasible")
ax.axhline(1e-2, color="gray", ls=":", lw=0.9)
ax.set_xlabel(r"$k_\perp / k_{\perp,\max}^{\mathrm{theorem}}$")
ax.set_ylabel(r"$E_{\mathrm{sc}}$")
ax.set_title(r"(c) Feasibility classifier (wet-clay Maxwell)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.25)

# --- Panel (d) ground-truth correlation ---
ax = axes[1, 1]
valid = np.isfinite(corr_self_arr) & np.isfinite(corr_truth_arr)
# Use truth error as x, self-consistency as y to show they correlate
x = corr_truth_arr[valid]
y = corr_self_arr[valid]
ax.loglog(x, y, "o", color="C4", ms=7)
# Identity line
lo = 1e-10
hi = 1e2
ax.loglog([lo, hi], [lo, hi], "k--", lw=1.0, label="$y = x$")
ax.axhline(1e-2, color="gray", ls=":", lw=0.9)
ax.axvline(1e-2, color="gray", ls=":", lw=0.9)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Rel. error vs analytical Lévy, " + r"$E_{\mathrm{truth}}$")
ax.set_ylabel(r"Self-consistency indicator $E_{\mathrm{sc}}$")
ax.set_title("(d) " + r"$E_{\mathrm{sc}}$" + " tracks ground truth (heat eq., " + r"$k_\perp$" + " sweep)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()
out = os.path.join(OUT_DIR, "recoverability_bound.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
