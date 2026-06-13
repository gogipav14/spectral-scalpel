"""
Multi-precision audit of the recoverability bound.

For each precision p with exponent range L_p = ln(maxfloat_p), the theorem
predicts a precision-specific cutoff
    k_{perp,max}(p)^2 = s*_max(p) * (beta*s*_max(p) + alpha)
    s*_max(p)         = (L_p - delta - ln(C/eps)) / t_max

If the argument applies to any NILT or frequency-domain method routed through
exp(-gamma_z d), the empirical breakdown should scale with the precision
exactly as L_p does. We test two precisions in-process:

  - float64: the default scalpel pipeline
  - float32 / complex64: a compiled reference NILT running the same
    Dubner-Abate formula but with 32-bit arithmetic everywhere

For each precision, we sweep k_perp and, for each point, run the NILT at
N=1024 versus N=4096 and measure the relative self-consistency error.
A breakdown within one logarithmic sweep step of the predicted k_max(p)
supports the precision-independent applicability of the bound.

Output:
  reports/claims_audit/precision_scaling.png
"""

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12
DELTA_S = 10.0
EPS_TAIL = 1e-6
C_TAIL = 1.0
KAPPA = 2.0

# L_p = ln(largest finite normal number). Written as constants because
# literal parsing of 1.8e308 rounds up to +inf on some platforms.
L_FLOAT32 = 88.7228       # math.log(np.finfo(np.float32).max)
L_FLOAT64 = 709.7827      # math.log(np.finfo(np.float64).max)

# Wet-clay Maxwell system (primary test)
ALPHA = MU_0 * 0.1
BETA  = MU_0 * EPS_0 * 10.0
D     = 0.1
PW    = 5e-9
T_END = 1e-7


def branch_point(k, alpha=ALPHA, beta=BETA):
    return (-alpha + np.sqrt(alpha**2 + 4.0 * beta * k**2)) / (2.0 * beta)


def s_star_max(L, t_end=T_END):
    t_max = 2.0 * KAPPA * t_end
    return (L - DELTA_S - math.log(C_TAIL / EPS_TAIL)) / t_max


def k_perp_max(L, t_end=T_END, alpha=ALPHA, beta=BETA):
    sm = s_star_max(L, t_end)
    return math.sqrt(sm * (beta * sm + alpha))


def F_np(s, k_perp, dtype_complex):
    """Laplace-domain transfer fn evaluated in the given complex dtype."""
    s = np.array(s, dtype=dtype_complex)
    k2 = np.array(k_perp**2, dtype=np.float32 if dtype_complex is np.complex64 else np.float64)
    alpha = np.array(ALPHA, dtype=k2.dtype)
    beta  = np.array(BETA,  dtype=k2.dtype)
    gsq = alpha * s + beta * s**2 - k2
    g = np.sqrt(gsq)
    if np.real(g) < 0:
        g = -g
    t0 = 3 * PW
    G = np.exp(-s * t0 + 0.5 * PW**2 * s**2)
    H = np.exp(-g * D)
    return (H * G).astype(dtype_complex)


def nilt_at_precision(k_perp, a, T, N, complex_dtype):
    """Dubner-Abate FFT-based NILT at the given complex precision."""
    if complex_dtype is np.complex64:
        real_dtype = np.float32
    else:
        real_dtype = np.float64
    omega = np.arange(N, dtype=real_dtype) * (np.pi / real_dtype(T))
    s = (real_dtype(a) + 1j * omega).astype(complex_dtype)
    G = np.array([F_np(sk, k_perp, complex_dtype) for sk in s], dtype=complex_dtype)
    # Dubner-Abate: half-weight on DC
    half = np.ones(N, dtype=real_dtype)
    half[0] = real_dtype(0.5)
    z_raw = complex_dtype(N) * np.fft.ifft(G * half)
    dt = 2 * T / N
    t_arr = np.arange(N, dtype=real_dtype) * real_dtype(dt)
    correction = np.exp(real_dtype(a) * t_arr) / real_dtype(T)
    return np.real(z_raw).astype(real_dtype) * correction, t_arr


def self_consistency(k_perp, complex_dtype, L):
    """Rel error between N=1024 and N=4096 NILT runs."""
    sm = s_star_max(L)
    a = branch_point(k_perp) * 1.2 + 1.0 / (KAPPA * T_END)
    # Clamp to CFL feasibility: if branch_point*t_max > L, no NILT will work
    T = KAPPA * T_END

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
    kmax = k_perp_max(L)
    grid = np.geomspace(kmax * 1e-3, kmax * 3.0, n_k)
    errs, valids = [], []
    print(f"\n=== {label}   predicted k_max = {kmax:.3e} ===")
    for k in grid:
        err, ok = self_consistency(k, complex_dtype, L)
        errs.append(err); valids.append(ok)
        print(f"    k={k:.3e}  err={err:.3e}  valid={ok}")
    return dict(label=label, k_grid=grid, errs=np.array(errs),
                valids=np.array(valids), kmax=kmax, L=L)


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")  # let overflows produce inf, we detect them

    r64 = sweep(np.complex128, L_FLOAT64, "float64")
    r32 = sweep(np.complex64,  L_FLOAT32, "float32")

    out_pkl = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/precision_scaling_data.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump([r64, r32], f)
    print(f"\nSaved -> {out_pkl}")

    # ==== Plot ====
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, col, mk in [(r64, "C0", "o"), (r32, "C3", "s")]:
        errs = r["errs"].copy()
        errs[~r["valids"]] = 5.0
        ax.loglog(r["k_grid"], errs, "-" + mk, color=col, lw=1.6, ms=6,
                  label=rf"{r['label']}   predicted $k_{{\perp,\max}}$ = {r['kmax']:.2f} rad/m")
        ax.axvline(r["kmax"], color=col, ls="--", lw=1.0, alpha=0.7)
    ax.axhline(1e-2, color="gray", ls=":", lw=0.8, label="1% error")
    ax.set_xlabel(r"$k_\perp$ [rad/m]")
    ax.set_ylabel("Rel. self-consistency error (N=4096 vs N=1024)")
    ax.set_title("Precision-dependent recoverability cutoff, wet-clay Maxwell\n"
                 rf"$d = {D}\,$m, $t_{{\mathrm{{end}}}} = {T_END*1e9:.0f}$ ns")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    out_png = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/precision_scaling.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_png}")
