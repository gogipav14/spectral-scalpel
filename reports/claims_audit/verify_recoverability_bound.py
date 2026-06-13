"""
Empirical verification of the corrected recoverability theorem.

Strategy: for each system, sweep k_perp. At each k_perp, run scalpel NILT
at two quadrature orders (N=1024 reference vs N=256 test) and compute the
self-consistency error. Separately compute the CFL-feasibility flag
(s*(k_perp)*t_max < L - delta - ln(C/eps)) and the theorem cutoff
k_perp,max from Eq. \ref{eq:kmax}.

Output: a plot showing relative self-consistency error vs k_perp/k_perp,max,
with the theorem cutoff as a red vertical line, for all three demo systems.
Pass criterion: self-consistency error stays below 1% for k_perp/k_perp,max
<= 1 and explodes past it.

No external reference needed. Assumes that if the NILT is self-consistent
between a low- and a high-N run at the CFL-permitted parameters, it is
converged; divergence between them signals the breakdown.
"""

from __future__ import annotations

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scalpel.core.nilt import nilt_scalar

MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12

L_DBL = 709.8
DELTA_S = 10.0
EPS_TAIL = 1e-6
C_TAIL = 1.0
KAPPA = 2.0


def branch_point(k_perp, alpha, beta):
    return (-alpha + np.sqrt(alpha**2 + 4.0 * beta * k_perp**2)) / (2.0 * beta)


def s_star_max(t_end):
    t_max = 2.0 * KAPPA * t_end
    return (L_DBL - DELTA_S - math.log(C_TAIL / EPS_TAIL)) / t_max


def k_perp_max_theorem(t_end, alpha, beta):
    sm = s_star_max(t_end)
    return math.sqrt(sm * (beta * sm + alpha))


SYSTEMS = [
    dict(name="Maxwell dry sand",
         alpha=MU_0 * 1e-4,  beta=MU_0 * EPS_0 * 4,
         t_end=1e-7, pw=5e-9, d=0.5),
    dict(name="Maxwell wet clay",
         alpha=MU_0 * 0.1,   beta=MU_0 * EPS_0 * 10,
         t_end=1e-7, pw=5e-9, d=0.1),
    dict(name="Acoustic tissue",
         alpha=1e-3 / 1540**2, beta=1.0 / 1540**2,
         t_end=2e-5, pw=1e-6, d=0.02),
]


def make_F(k_perp, alpha, beta, d, pw):
    t0 = 3 * pw
    def F(s):
        gsq = alpha * s + beta * s**2 - k_perp**2
        g = np.sqrt(gsq + 0j)
        if np.real(g) < 0:
            g = -g
        G = np.exp(-s * t0 + 0.5 * pw**2 * s**2)
        return complex(np.exp(-g * d) * G)
    return F


def run_pair(k_perp, alpha, beta, d, pw, t_end, N_ref=4096, N_test=512):
    """Return (rel_consistency_error, both amplitudes valid flag, CFL feasible flag)."""
    t_max = 2 * KAPPA * t_end
    T = KAPPA * t_end
    # Choose Bromwich shift just past the branch point
    a = branch_point(k_perp, alpha, beta) * 1.2 + 1.0 / t_max

    cfl_ok = (branch_point(k_perp, alpha, beta) * t_max
              + math.log(C_TAIL / EPS_TAIL) < L_DBL - DELTA_S)

    F = make_F(k_perp, alpha, beta, d, pw)

    try:
        f_ref, t_ref, _ = nilt_scalar(F, a, T, N_ref)
        f_test, t_test, _ = nilt_scalar(F, a, T, N_test)
    except (OverflowError, FloatingPointError, ValueError):
        return float("inf"), False, cfl_ok

    # Check validity
    if not (np.all(np.isfinite(f_ref)) and np.all(np.isfinite(f_test))):
        return float("inf"), False, cfl_ok

    # Interpolate f_test onto t_ref mask
    mask = (t_ref > 0.1 * t_end) & (t_ref < 0.9 * t_end)
    if not np.any(mask):
        return float("inf"), False, cfl_ok
    f_test_on_ref = np.interp(t_ref[mask], t_test, f_test)
    num = np.linalg.norm(f_ref[mask] - f_test_on_ref)
    den = np.linalg.norm(f_ref[mask]) + 1e-300
    return float(num / den), True, cfl_ok


def sweep_system(sys_, n_k=28):
    k_max = k_perp_max_theorem(sys_["t_end"], sys_["alpha"], sys_["beta"])
    k_grid = np.geomspace(k_max * 1e-3, k_max * 3.0, n_k)

    errs, valids, cfls = [], [], []
    print(f"\n=== {sys_['name']} ===")
    print(f"  theorem k_perp_max = {k_max:.4e}")
    for k in k_grid:
        err, valid, cfl = run_pair(k, sys_["alpha"], sys_["beta"],
                                   sys_["d"], sys_["pw"], sys_["t_end"])
        errs.append(err); valids.append(valid); cfls.append(cfl)
        print(f"    k={k:.3e}  err={err:.3e}  valid={valid}  cfl={cfl}")
    return dict(name=sys_["name"], k_grid=k_grid, errs=np.array(errs),
                valids=np.array(valids), cfls=np.array(cfls), k_max=k_max)


if __name__ == "__main__":
    results = [sweep_system(s) for s in SYSTEMS]

    out_pkl = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/recoverability_bound_data.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {out_pkl}")

    # ==== Plot ====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, r in zip(axes, results):
        k_norm = r["k_grid"] / r["k_max"]
        errs = r["errs"].copy()
        errs[~r["valids"]] = 10.0  # replace NaN/inf with visible marker
        ax.loglog(k_norm, errs, "o-", lw=1.6, ms=5, color="C0",
                  label="N=4096 vs 512")
        ax.axvline(1.0, color="red", ls="--", lw=1.3,
                   label=r"$k_{\perp,\max}$ (Eq. 3)")
        ax.axhline(1e-2, color="gray", ls=":", lw=1.0, label=r"1% error")
        ax.set_xlabel(r"$k_\perp / k_{\perp,\max}^{\mathrm{theorem}}$")
        ax.set_title(r["name"], fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Rel. self-consistency error")
    fig.suptitle("Empirical verification of the corrected recoverability theorem "
                 r"$k_{\perp,\max}^2 = s^*_{\max}(\beta s^*_{\max} + \alpha)$",
                 fontsize=12)
    plt.tight_layout()
    out_png = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/recoverability_bound.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out_png}")
