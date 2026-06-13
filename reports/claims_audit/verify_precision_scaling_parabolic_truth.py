"""
Parabolic-class precision-scaling verification (truth-comparison version).

For the parabolic class, the self-consistency test (verify_precision_scaling_parabolic.py)
loses discriminating power once the signal drops below machine epsilon: both
N=1024 and N=4096 produce the same (machine-zero) result, so relative error
goes to zero rather than diverging.

This script uses the analytical shifted-Levy distribution as ground truth
(diffusion has a closed-form NILT) and reports the relative error at the peak
observation time. The precision-limited cutoff is the smallest k_perp at which
the float precision can no longer represent the true signal accurately.

Predicted cutoff:
    k_{perp,max}(L_p) = (L_p - delta_s - ln(C/eps_tail)) / d

Predicted float64/float32 ratio:
    (L_64 - margins) / (L_32 - margins) ~ 8-10x  (depending on margins)
"""

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

DELTA_S = 10.0
EPS_TAIL = 1e-6
C_TAIL = 1.0

L_FLOAT32 = 88.7228
L_FLOAT64 = 709.7827

D     = 1e-2
DEPTH = 0.05
T_END = 5e-2
T_OBS = 1.0e-2     # observation time within the early-time window


def k_perp_max_parabolic(L, depth=DEPTH):
    return (L - DELTA_S - math.log(C_TAIL / EPS_TAIL)) / depth


def levy_diffusion(t, k_perp, depth=DEPTH, D_=D):
    """Analytical shifted-Levy: NILT of exp(-sqrt(s/D + k_perp^2) * d) for impulse source.

    L^{-1}[exp(-sqrt(s/D + kperp^2) * d)]
       = (d/sqrt(D)) / (2 sqrt(pi * t^3)) * exp(-d^2/(4Dt) - D kperp^2 t)
    """
    if t <= 0:
        return 0.0
    a = depth / math.sqrt(D_)
    prefactor = a / (2.0 * math.sqrt(math.pi * t**3))
    arg = -a**2 / (4.0 * t) - D_ * k_perp**2 * t
    return prefactor * math.exp(arg) if arg > -700 else 0.0


def F_diffusion_pulse(s, k_perp, dtype_complex):
    """Same as previous test (pulse source convolved with diffusion transfer)."""
    real_dtype = np.float32 if dtype_complex is np.complex64 else np.float64
    s = np.array(s, dtype=dtype_complex)
    k2 = np.array(k_perp**2, dtype=real_dtype)
    D_ = real_dtype(D)
    gsq = s / D_ + k2
    g = np.sqrt(gsq)
    if np.real(g) < 0:
        g = -g
    H = np.exp(-g * DEPTH)
    return H.astype(dtype_complex)


def F_diffusion_impulse(s, k_perp, dtype_complex):
    """Pure impulse-source: just exp(-gamma_z * d). Compare directly to Levy."""
    return F_diffusion_pulse(s, k_perp, dtype_complex)


def nilt_at_precision(k_perp, a, T, N, complex_dtype):
    real_dtype = np.float32 if complex_dtype is np.complex64 else np.float64
    omega = np.arange(N, dtype=real_dtype) * (np.pi / real_dtype(T))
    s = (real_dtype(a) + 1j * omega).astype(complex_dtype)
    G = np.array([F_diffusion_impulse(sk, k_perp, complex_dtype) for sk in s], dtype=complex_dtype)
    half = np.ones(N, dtype=real_dtype)
    half[0] = real_dtype(0.5)
    z_raw = complex_dtype(N) * np.fft.ifft(G * half)
    dt = 2 * T / N
    t_arr = np.arange(N, dtype=real_dtype) * real_dtype(dt)
    correction = np.exp(real_dtype(a) * t_arr) / real_dtype(T)
    return np.real(z_raw).astype(real_dtype) * correction, t_arr


def truth_comparison(k_perp, complex_dtype, L):
    """Rel error vs analytical Levy at the peak observation window."""
    T = 2.0 * T_END
    # For diffusion (s* < 0), pick a > 0 small (any positive value works).
    # Choose a such that aT is reasonable: aT ~ ln(1/eps_tail) for accuracy.
    a = math.log(1.0 / EPS_TAIL) / T  # aT = ln(1/eps_tail) ~ 14
    N = 4096
    try:
        f_nilt, t_arr = nilt_at_precision(k_perp, a, T, N, complex_dtype)
    except (FloatingPointError, ValueError, OverflowError):
        return float("inf"), False, 0.0, 0.0

    if not np.all(np.isfinite(f_nilt)):
        return float("inf"), False, 0.0, 0.0

    # Evaluate analytical at the same t grid
    f_truth = np.array([levy_diffusion(float(t), k_perp) for t in t_arr])

    # Compare in the early-time window where signal is largest
    t_peak = DEPTH / (2.0 * D * max(k_perp, 1e-10))   # peak of the Levy
    # Compare around the peak
    t_lo = t_peak * 0.3
    t_hi = t_peak * 3.0 if t_peak * 3.0 < T_END else T_END
    mask = (t_arr > t_lo) & (t_arr < t_hi)
    if not np.any(mask):
        # Fallback: use a fixed mid-time window
        mask = (t_arr > 0.05 * T_END) & (t_arr < 0.5 * T_END)
    if not np.any(mask):
        return float("inf"), False, 0.0, 0.0

    sig_max = float(np.max(np.abs(f_truth[mask])))
    if sig_max < 1e-300:
        # Signal below underflow even for float64 truth; cutoff already passed
        return float("inf"), False, sig_max, float(np.max(np.abs(f_nilt[mask])))

    rel_err = float(np.linalg.norm(f_nilt[mask] - f_truth[mask])) / (
        float(np.linalg.norm(f_truth[mask])) + 1e-300
    )
    return rel_err, True, sig_max, float(np.max(np.abs(f_nilt[mask])))


def sweep(complex_dtype, L, label, n_k=30):
    kmax = k_perp_max_parabolic(L)
    grid = np.geomspace(kmax * 1e-3, kmax * 2.0, n_k)
    errs, valids, sig_truth, sig_nilt = [], [], [], []
    print(f"\n=== {label}   parabolic predicted k_max = {kmax:.3e} rad/m ===")
    for k in grid:
        err, ok, st, sn = truth_comparison(k, complex_dtype, L)
        errs.append(err); valids.append(ok); sig_truth.append(st); sig_nilt.append(sn)
        print(f"    k={k:.3e}  rel_err={err:.3e}  ok={ok}  "
              f"|h_truth|={st:.3e}  |h_nilt|={sn:.3e}")
    return dict(label=label, k_grid=grid, errs=np.array(errs),
                valids=np.array(valids),
                sig_truth=np.array(sig_truth), sig_nilt=np.array(sig_nilt),
                kmax=kmax, L=L)


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore", under="ignore")

    r64 = sweep(np.complex128, L_FLOAT64, "float64")
    r32 = sweep(np.complex64,  L_FLOAT32, "float32")

    def empirical_cutoff(r, threshold=1e-2):
        errs = r["errs"].copy()
        errs[~r["valids"]] = 10.0
        crossing = np.where(errs > threshold)[0]
        if len(crossing) == 0:
            return float("nan")
        return r["k_grid"][crossing[0]]

    k_emp_64 = empirical_cutoff(r64)
    k_emp_32 = empirical_cutoff(r32)

    print(f"\n=== PARABOLIC RECOVERABILITY (vs analytical Levy) ===")
    print(f"  Predicted k_perp,max float64 = {r64['kmax']:.2f} rad/m, "
          f"float32 = {r32['kmax']:.2f} rad/m")
    print(f"  Predicted ratio = {r64['kmax']/r32['kmax']:.3f}x")
    print(f"  Empirical (1% rel err) k_perp,max float64 = {k_emp_64:.2f} rad/m, "
          f"float32 = {k_emp_32:.2f} rad/m")
    if not np.isnan(k_emp_64) and not np.isnan(k_emp_32) and k_emp_32 > 0:
        print(f"  Empirical ratio = {k_emp_64/k_emp_32:.3f}x")

    out_pkl = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/precision_scaling_parabolic_truth_data.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump([r64, r32, dict(
            empirical_64=float(k_emp_64), empirical_32=float(k_emp_32),
            predicted_64=r64['kmax'], predicted_32=r32['kmax'],
        )], f)
    print(f"\nSaved -> {out_pkl}")

    out_csv = out_pkl.replace(".pkl", ".csv")
    with open(out_csv, "w") as f:
        f.write("precision,L,k_perp_predicted,k_perp_empirical_1pct\n")
        for r, kemp in [(r64, k_emp_64), (r32, k_emp_32)]:
            f.write(f"{r['label']},{r['L']},{r['kmax']:.4f},{kemp:.4f}\n")
    print(f"Saved -> {out_csv}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for r, col, mk in [(r64, "C0", "o"), (r32, "C3", "s")]:
        errs = r["errs"].copy()
        errs[~r["valids"]] = 5.0
        ax.loglog(r["k_grid"], errs, "-" + mk, color=col, lw=1.6, ms=6,
                  label=rf"{r['label']}   predicted $k_{{\perp,\max}}$ = {r['kmax']:.0f} rad/m")
        ax.axvline(r["kmax"], color=col, ls="--", lw=1.0, alpha=0.7)
    ax.axhline(1e-2, color="gray", ls=":", lw=0.8, label="1% error")
    ax.set_xlabel(r"$k_\perp$ [rad/m]")
    ax.set_ylabel("Rel. error vs analytical Lévy (truth)")
    ax.set_title("Parabolic recoverability cutoff (truth comparison)\n"
                 rf"pure diffusion, $D = {D}$, $d = {DEPTH}$ m")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    out_png = out_pkl.replace(".pkl", ".png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_png}")
