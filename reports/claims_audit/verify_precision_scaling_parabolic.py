"""
Parabolic-class precision-scaling verification.

The hyperbolic-class bound (verify_precision_scaling.py) comes from the
right-moving branch point of gamma_z^2 = alpha s + beta s^2 - k_perp^2.
For the parabolic class (p,q) = (0,1), gamma_z^2 = A + Bs + k_perp^2 has
all branch points in the closed left half-plane, so no right-moving
abscissa exists. The recoverability bound instead arises from direct
underflow of the transfer function |H| = |exp(-gamma_z d)|.

At high k_perp, the diffusion dispersion gives gamma_z ~ k_perp, hence
|H| ~ exp(-k_perp d). For this to be representable at floating-point
precision L_p = ln(maxfloat_p), we need

    k_perp d  <=  L_p - delta_s

so the predicted cutoff is

    k_{perp,max}^{par}  =  (L_p - delta_s - ln(C/eps_tail)) / d.

Predicted float32 / float64 ratio:
    k_{perp,max}(L_64) / k_{perp,max}(L_32)  =  (L_64 - margins) / (L_32 - margins)
                                              ~  709.78 / 88.72  ~  8.0
with subtractive corrections from delta_s and the eps_tail term.

This is qualitatively different from the hyperbolic case (4.82x) because
the bound is linear in L for parabolic vs sqrt(L) / L for hyperbolic
depending on the asymptotic regime. The empirical ratio should match
~8x.

We use pure diffusion as the parabolic test system: gamma_z^2 = s/D + k_perp^2.
"""

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

DELTA_S = 10.0
EPS_TAIL = 1e-6
C_TAIL = 1.0
KAPPA = 2.0

L_FLOAT32 = 88.7228
L_FLOAT64 = 709.7827

# Pure-diffusion parabolic test system
D     = 1e-2          # m^2 / s (typical heat diffusivity)
DEPTH = 0.05          # m
PW    = 1e-3          # source pulse width
T_END = 5e-2          # 50 ms observation window


def k_perp_max_parabolic(L, depth=DEPTH):
    """Closed-form parabolic recoverability bound.

    Direct-underflow argument: at large k_perp, gamma_z(a, k_perp) ~ k_perp
    and |H| ~ exp(-k_perp * d). Representability requires
    k_perp d < L - delta_s - ln(C/eps_tail).
    """
    return (L - DELTA_S - math.log(C_TAIL / EPS_TAIL)) / depth


def F_diffusion(s, k_perp, dtype_complex):
    """Pure-diffusion Laplace transfer function evaluated in given precision.

    gamma_z^2 = s/D + k_perp^2    (parabolic class, PLUS sign)
    H(s, k_perp, d) = exp(-gamma_z d) * source_spectrum(s)
    """
    real_dtype = np.float32 if dtype_complex is np.complex64 else np.float64
    s = np.array(s, dtype=dtype_complex)
    k2 = np.array(k_perp**2, dtype=real_dtype)
    D_ = real_dtype(D)
    gsq = s / D_ + k2
    g = np.sqrt(gsq)
    if np.real(g) < 0:
        g = -g
    t0 = 3 * PW
    G = np.exp(-s * t0 + 0.5 * PW**2 * s**2)
    H = np.exp(-g * DEPTH)
    return (H * G).astype(dtype_complex)


def nilt_at_precision(k_perp, a, T, N, complex_dtype):
    """Dubner-Abate FFT-based NILT, same machinery as the hyperbolic test."""
    real_dtype = np.float32 if complex_dtype is np.complex64 else np.float64
    omega = np.arange(N, dtype=real_dtype) * (np.pi / real_dtype(T))
    s = (real_dtype(a) + 1j * omega).astype(complex_dtype)
    G = np.array([F_diffusion(sk, k_perp, complex_dtype) for sk in s], dtype=complex_dtype)
    half = np.ones(N, dtype=real_dtype)
    half[0] = real_dtype(0.5)
    z_raw = complex_dtype(N) * np.fft.ifft(G * half)
    dt = 2 * T / N
    t_arr = np.arange(N, dtype=real_dtype) * real_dtype(dt)
    correction = np.exp(real_dtype(a) * t_arr) / real_dtype(T)
    return np.real(z_raw).astype(real_dtype) * correction, t_arr


def self_consistency(k_perp, complex_dtype, L):
    """Rel error between N=1024 and N=4096 NILT runs at given k_perp / precision."""
    # For parabolic, branch points are in LHP. Choose a > 0 small enough that
    # a * t_max is well below L, but large enough for NILT convergence.
    # Standard choice: a ~ 2.3 / t_end (corresponds to truncation tolerance 0.1).
    T = KAPPA * T_END
    a = max(2.3 / T_END, 1.0 / T)

    try:
        f_ref, t_ref = nilt_at_precision(k_perp, a, T, 4096, complex_dtype)
        f_tst, t_tst = nilt_at_precision(k_perp, a, T, 1024, complex_dtype)
    except (FloatingPointError, ValueError, OverflowError):
        return float("inf"), False

    if not (np.all(np.isfinite(f_ref)) and np.all(np.isfinite(f_tst))):
        return float("inf"), False

    mask = (t_ref > 0.1 * T_END) & (t_ref < 0.9 * T_END)
    if not np.any(mask):
        return float("inf"), False
    f_tst_on_ref = np.interp(t_ref[mask], t_tst, f_tst)
    num = np.linalg.norm(f_ref[mask] - f_tst_on_ref)
    den = np.linalg.norm(f_ref[mask]) + 1e-38
    return float(num / den), True


def sweep(complex_dtype, L, label, n_k=24):
    kmax = k_perp_max_parabolic(L)
    grid = np.geomspace(kmax * 1e-3, kmax * 3.0, n_k)
    errs, valids = [], []
    print(f"\n=== {label}   parabolic predicted k_max = {kmax:.3e} rad/m ===")
    for k in grid:
        err, ok = self_consistency(k, complex_dtype, L)
        errs.append(err)
        valids.append(ok)
        print(f"    k={k:.3e}  err={err:.3e}  valid={ok}")
    return dict(label=label, k_grid=grid, errs=np.array(errs),
                valids=np.array(valids), kmax=kmax, L=L)


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")

    r64 = sweep(np.complex128, L_FLOAT64, "float64")
    r32 = sweep(np.complex64,  L_FLOAT32, "float32")

    print(f"\n=== PARABOLIC RECOVERABILITY SUMMARY ===")
    print(f"Predicted k_perp,max (float64): {r64['kmax']:.2f} rad/m")
    print(f"Predicted k_perp,max (float32): {r32['kmax']:.2f} rad/m")
    print(f"Predicted ratio: {r64['kmax']/r32['kmax']:.3f}x")

    # Empirical cutoff: first k at which error crosses 1e-2
    def empirical_cutoff(r, threshold=1e-2):
        errs = r["errs"].copy()
        errs[~r["valids"]] = 10.0
        crossing = np.where(errs > threshold)[0]
        if len(crossing) == 0:
            return float("nan")
        return r["k_grid"][crossing[0]]

    k_emp_64 = empirical_cutoff(r64)
    k_emp_32 = empirical_cutoff(r32)
    print(f"Empirical k_perp,max (float64): {k_emp_64:.2f} rad/m")
    print(f"Empirical k_perp,max (float32): {k_emp_32:.2f} rad/m")
    if k_emp_32 > 0:
        print(f"Empirical ratio: {k_emp_64/k_emp_32:.3f}x")

    out_pkl = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/precision_scaling_parabolic_data.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump([r64, r32, dict(
            empirical_64=float(k_emp_64), empirical_32=float(k_emp_32),
            predicted_64=r64['kmax'], predicted_32=r32['kmax'],
            empirical_ratio=k_emp_64 / max(k_emp_32, 1e-30),
            predicted_ratio=r64['kmax'] / r32['kmax'],
        )], f)
    print(f"\nSaved -> {out_pkl}")

    # CSV (user prefers CSV)
    out_csv = out_pkl.replace(".pkl", ".csv")
    with open(out_csv, "w") as f:
        f.write("precision,L,k_perp_predicted,k_perp_empirical_1pct,ratio_vs_f64\n")
        for r, k_emp in [(r64, k_emp_64), (r32, k_emp_32)]:
            f.write(f"{r['label']},{r['L']:.4f},{r['kmax']:.4f},{k_emp:.4f},"
                    f"{k_emp/max(k_emp_64,1e-30):.4f}\n")
    print(f"Saved -> {out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, col, mk in [(r64, "C0", "o"), (r32, "C3", "s")]:
        errs = r["errs"].copy()
        errs[~r["valids"]] = 5.0
        ax.loglog(r["k_grid"], errs, "-" + mk, color=col, lw=1.6, ms=6,
                  label=rf"{r['label']}   predicted $k_{{\perp,\max}}$ = {r['kmax']:.1f} rad/m")
        ax.axvline(r["kmax"], color=col, ls="--", lw=1.0, alpha=0.7)
    ax.axhline(1e-2, color="gray", ls=":", lw=0.8, label="1% error")
    ax.set_xlabel(r"$k_\perp$ [rad/m]")
    ax.set_ylabel("Rel. self-consistency error (N=4096 vs N=1024)")
    ax.set_title("Parabolic recoverability cutoff, pure diffusion\n"
                 rf"$D = {D}\,$m$^2$/s, $d = {DEPTH}\,$m, "
                 rf"$t_{{\mathrm{{end}}}} = {T_END*1e3:.0f}$ ms")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    out_png = out_pkl.replace(".pkl", ".png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_png}")
