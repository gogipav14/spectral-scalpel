"""
Maxwell dispersion sign-convention verification via angular spectrum.

The previous FDTD-based test (maxwell_sign_convention_check.py) was
inconclusive because the crude absorbing BC reflected and caused overflow.
This script uses an unambiguous analytical reference: for a single
monochromatic transverse mode at frequency omega and transverse wavenumber
k_perp, the field at depth z is given by angular spectrum:

    E(z) = E(0) * exp(i k_z z)        propagating  (k_perp < omega/c)
    E(z) = E(0) * exp(-kappa z)       evanescent   (k_perp > omega/c)

where k_z = sqrt((omega/c)^2 - k_perp^2) and kappa = sqrt(k_perp^2 - (omega/c)^2).

For lossless Maxwell the correct dispersion is gamma_z^2 = k_perp^2 - omega^2/c^2
in Helmholtz form. In Laplace form (s = -i omega), gamma_z^2 = + s^2/c^2 + k_perp^2.

We feed a complex-exponential input into the spectral engine with both sign
conventions and compare the output amplitude vs the analytical angular-spectrum
prediction. The convention that matches is the correct one.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

# Setup -----------------------------------------------------------------
MU_0 = 4e-7 * math.pi
EPS_0 = 8.854187817e-12
sigma = 0.0       # LOSSLESS for clean angular spectrum
eps_r = 1.0       # vacuum to make things simple
epsilon = EPS_0 * eps_r
mu = MU_0
c = 1.0 / math.sqrt(mu * epsilon)  # speed of light

# Test frequency
omega_test = 2 * math.pi * 1e9     # 1 GHz
k0 = omega_test / c                # free-space wavenumber

depth = 0.05                       # 5 cm slab

print("=" * 70)
print(" Maxwell sign-convention check via angular spectrum")
print(f" Lossless vacuum, omega = {omega_test/2/math.pi/1e9:.1f} GHz")
print(f" c = {c:.3e} m/s, k0 = omega/c = {k0:.3f} rad/m, depth = {depth*100:.0f} cm")
print("=" * 70)

# Test several k_perp values: subcritical (propagating) and supercritical (evanescent)
k_perps = [0.0, 0.5 * k0, 0.9 * k0, 1.2 * k0, 2.0 * k0]

# -----------------------------------------------------------------------
# Analytical angular-spectrum reference
# -----------------------------------------------------------------------
def angular_spectrum(k_perp, omega, d):
    """E(d) / E(0) for monochromatic plane wave with transverse wavenumber k_perp."""
    arg = (omega / c) ** 2 - k_perp ** 2
    if arg >= 0:
        # Propagating: amplitude unity, phase k_z d
        k_z = math.sqrt(arg)
        return complex(math.cos(k_z * d), math.sin(k_z * d))
    else:
        # Evanescent: amplitude exp(-kappa d), no phase
        kappa = math.sqrt(-arg)
        return complex(math.exp(-kappa * d), 0)


# -----------------------------------------------------------------------
# Spectral engine evaluation at a single (k_perp, omega) point
# -----------------------------------------------------------------------
# The spectral engine's transfer function at one mode is
#   H(s, k_perp, d) = exp(-gamma_z(s, k_perp) d)
# Evaluate at s = i omega and compare to angular spectrum.

def gamma_z_minus(k_perp, omega):
    """Paper/code convention: gamma_z^2 = mu (sigma s + epsilon s^2) - k_perp^2"""
    s = 1j * omega
    g2 = mu * (sigma * s + epsilon * s ** 2) - k_perp ** 2
    gz = np.sqrt(g2)
    if gz.real < 0:
        gz = -gz
    return gz


def gamma_z_plus(k_perp, omega):
    """First-principles convention: gamma_z^2 = mu (sigma s + epsilon s^2) + k_perp^2"""
    s = 1j * omega
    g2 = mu * (sigma * s + epsilon * s ** 2) + k_perp ** 2
    gz = np.sqrt(g2)
    if gz.real < 0:
        gz = -gz
    return gz


print(f"\n{'k_perp/k0':>10s}  {'|E(d)/E(0)| analytic':>22s}  "
      f"{'minus conv |H|':>16s}  {'plus conv |H|':>16s}")
print("-" * 70)

rows = []
for kp in k_perps:
    H_analytic = angular_spectrum(kp, omega_test, depth)
    gz_minus = gamma_z_minus(kp, omega_test)
    gz_plus = gamma_z_plus(kp, omega_test)
    H_minus = np.exp(-gz_minus * depth)
    H_plus = np.exp(-gz_plus * depth)

    print(f"{kp/k0:>10.2f}  {abs(H_analytic):>22.6f}  "
          f"{abs(H_minus):>16.6f}  {abs(H_plus):>16.6f}")

    rows.append({
        "k_perp_over_k0": kp / k0,
        "regime": "propagating" if kp < k0 else "evanescent",
        "abs_H_analytic": abs(H_analytic),
        "abs_H_minus_sign": abs(H_minus),
        "abs_H_plus_sign": abs(H_plus),
        "rel_err_minus_sign": abs(abs(H_minus) - abs(H_analytic)) / max(abs(H_analytic), 1e-30),
        "rel_err_plus_sign": abs(abs(H_plus) - abs(H_analytic)) / max(abs(H_analytic), 1e-30),
    })

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
total_err_minus = sum(r["rel_err_minus_sign"] for r in rows)
total_err_plus = sum(r["rel_err_plus_sign"] for r in rows)
print(f" Total rel err (sum), -k_perp^2 convention: {total_err_minus:.3e}")
print(f" Total rel err (sum), +k_perp^2 convention: {total_err_plus:.3e}")
if total_err_plus < total_err_minus * 0.1:
    print(" >>> +k_perp^2 convention matches angular spectrum. Paper convention is WRONG.")
elif total_err_minus < total_err_plus * 0.1:
    print(" >>> -k_perp^2 convention matches angular spectrum. Paper convention is RIGHT.")
else:
    print(" >>> Inconclusive (both conventions disagree with analytic).")

out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "maxwell_sign_angular_spectrum.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved -> {out_csv}")
