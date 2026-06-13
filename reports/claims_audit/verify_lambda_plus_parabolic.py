"""
Empirical verification of Lambda_+ = 2*pi*(sqrt(2)+1) ~= 15.17 for parabolic systems.

The master formula (Pavlov 2026, PRE submitted) predicts that for two-mechanism
dispersion gamma^2(s) = A + Bs (class (p,q) = (0,1)), the wavelength-to-
decay-length ratio at the magnitude-balance frequency omega_cross = A/B is
universal:

    delta/lambda |_{omega_cross} = 1/Lambda_+, with Lambda_+ = 2*pi*(sqrt(2)+1).

This script verifies the identity numerically for the parabolic systems in the
spectral-scalpel codebase:
- convection_diffusion_cylindrical (chromatography)
- diffusion (heat / reaction-diffusion analogue with k_r playing the role of A)

The companion test_crossover_constants.py covers the hyperbolic case (Lambda_-)
to ~1e-9 across Maxwell + acoustics. This script extends to the parabolic class.

Output:
- prints delta/lambda for each system
- CSV file with results
- machine-precision agreement against the closed form
"""

import math
import numpy as np


LAMBDA_PLUS = 2.0 * math.pi * (math.sqrt(2.0) + 1.0)
DELTA_OVER_LAMBDA_PLUS_THEORY = 1.0 / LAMBDA_PLUS


def safe_sqrt_principal(z):
    """Principal complex square root with Re >= 0."""
    r = np.sqrt(z + 0j)
    return r if r.real >= 0 else -r


def gamma_parabolic(A, B, s):
    """gamma^2 = A + B s for class (p,q) = (0,1)."""
    return safe_sqrt_principal(A + B * s)


def delta_over_lambda_at(gamma_complex):
    re = float(np.real(gamma_complex))
    im = float(abs(np.imag(gamma_complex)))
    if re == 0 or im == 0:
        return float("nan")
    delta = 1.0 / re
    wavelength = 2.0 * math.pi / im
    return delta / wavelength


def test_system(name, A, B):
    omega_cross = A / B
    s = 1j * omega_cross
    g = gamma_parabolic(A, B, s)
    ratio = delta_over_lambda_at(g)
    rel_err = abs(ratio - DELTA_OVER_LAMBDA_PLUS_THEORY) / DELTA_OVER_LAMBDA_PLUS_THEORY
    return {
        "system": name,
        "A": A, "B": B,
        "omega_cross": omega_cross,
        "lambda_cross": 2.0 * math.pi / abs(g.imag),
        "delta_cross": 1.0 / g.real,
        "delta_over_lambda": ratio,
        "rel_err_vs_theory": rel_err,
    }


# Parabolic systems with their (A, B) coefficients
systems = [
    # Pure diffusion with reaction (Damköhler crossover)
    ("diffusion+reaction (k=1, D=1e-4)", 1.0 / 1e-4, 1.0 / 1e-4),
    ("diffusion+reaction (k=10, D=1e-2)", 10.0 / 1e-2, 1.0 / 1e-2),

    # Convection-diffusion with Peclet crossover
    # gamma^2 = v^2/(4D^2) + s/D, so A = v^2/(4D^2), B = 1/D
    ("conv-diff (v=0.001, D=1e-8)", (0.001) ** 2 / (4 * (1e-8) ** 2), 1.0 / 1e-8),
    ("conv-diff (v=0.01, D=1e-6)", (0.01) ** 2 / (4 * (1e-6) ** 2), 1.0 / 1e-6),
    ("conv-diff (v=0.1, D=1e-4)", (0.1) ** 2 / (4 * (1e-4) ** 2), 1.0 / 1e-4),

    # Black-Scholes (parabolic): gamma^2 = 2r/sigma^2 + 2/sigma^2 s
    ("BS (r=0.03, sigma=0.20)", 2 * 0.03 / 0.20 ** 2, 2.0 / 0.20 ** 2),
    ("BS (r=0.05, sigma=0.30)", 2 * 0.05 / 0.30 ** 2, 2.0 / 0.30 ** 2),
    ("BS (r=0.10, sigma=0.80)", 2 * 0.10 / 0.80 ** 2, 2.0 / 0.80 ** 2),

    # Reaction-diffusion with k spanning orders of magnitude
    ("rxn-diff (k=1e-3, D=1e-5)", 1e-3 / 1e-5, 1.0 / 1e-5),
    ("rxn-diff (k=1e3, D=1e-2)", 1e3 / 1e-2, 1.0 / 1e-2),
]

print("=" * 80)
print(f"  Lambda_+ verification: parabolic class (p,q) = (0,1)")
print(f"  Theory: delta/lambda = 1/(2*pi*(sqrt(2)+1)) = {DELTA_OVER_LAMBDA_PLUS_THEORY:.10f}")
print(f"  Lambda_+ = {LAMBDA_PLUS:.10f}")
print("=" * 80)

results = []
for name, A, B in systems:
    r = test_system(name, A, B)
    results.append(r)
    print(f"\n  {name}")
    print(f"    omega_cross = {r['omega_cross']:.4e} rad/s")
    print(f"    delta/lambda = {r['delta_over_lambda']:.10f}")
    print(f"    rel err vs theory = {r['rel_err_vs_theory']:.2e}")

print("\n" + "=" * 80)
max_err = max(r["rel_err_vs_theory"] for r in results)
print(f"  Max relative error across {len(results)} systems: {max_err:.3e}")
print(f"  Target: < 1e-10  -->  {'PASS' if max_err < 1e-10 else 'FAIL'}")
print(f"  omega_cross spans: {min(r['omega_cross'] for r in results):.2e} "
      f"to {max(r['omega_cross'] for r in results):.2e} rad/s "
      f"({math.log10(max(r['omega_cross'] for r in results) / min(r['omega_cross'] for r in results)):.1f} orders of magnitude)")

# CSV output
out_csv = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/lambda_plus_verification.csv"
with open(out_csv, "w") as f:
    f.write("system,A,B,omega_cross,lambda_cross,delta_cross,delta_over_lambda,rel_err_vs_theory\n")
    for r in results:
        f.write(f"\"{r['system']}\",{r['A']:.6e},{r['B']:.6e},"
                f"{r['omega_cross']:.6e},{r['lambda_cross']:.6e},"
                f"{r['delta_cross']:.6e},{r['delta_over_lambda']:.10f},"
                f"{r['rel_err_vs_theory']:.3e}\n")
print(f"\nSaved -> {out_csv}")
