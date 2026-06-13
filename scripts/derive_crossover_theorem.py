"""
Theorem 2 derivation: Crossover-Limited Recoverability.

For H(s, k_perp, d) = exp(-gamma_z * d) where gamma_z = sqrt(alpha*s + beta*s^2 - k_perp^2),
derive the maximum recoverable transverse wavenumber k_perp,max(d) from the CFL
feasibility condition, and show that the crossover constant Lambda enters through
the branch point structure.

Steps:
1. Branch point of gamma_z: s* where alpha*s + beta*s^2 = k_perp^2
2. alpha_c(k_perp) = s* (rightmost singularity on real axis)
3. CFL: s* * t_max + ln(C/eps) <= L - delta_s  =>  s* <= s*_max
4. Invert to get k_perp,max from s*_max
5. Show transition from diffusive to wave-like scaling at crossover k_perp
6. Connect to Lambda via skin depth at crossover
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
MU_0 = 4e-7 * np.pi
EPS_0 = 8.854187817e-12

# CFL parameters (from CES paper)
L_DBL = 709.8
delta_s = 10.0
eps_tail = 1e-6
C_tail = 1.0
kappa = 2.0  # T = kappa * t_end

print("=" * 70)
print(" THEOREM 2 DERIVATION: Crossover-Limited Recoverability")
print("=" * 70)

# ─── Step 1: Branch point analysis ──────────────────────────────────
print("\n─── Step 1: Branch point of gamma_z ───")
print("gamma_z^2 = alpha*s + beta*s^2 - k_perp^2")
print("Branch point at gamma_z = 0: alpha*s + beta*s^2 = k_perp^2")
print("Quadratic in s: beta*s^2 + alpha*s - k_perp^2 = 0")
print("s* = [-alpha + sqrt(alpha^2 + 4*beta*k_perp^2)] / (2*beta)")
print("(taking the positive root for physical branch point)")

def branch_point(k_perp, alpha, beta):
    """Rightmost branch point of gamma_z on real s-axis."""
    discriminant = alpha**2 + 4 * beta * k_perp**2
    return (-alpha + np.sqrt(discriminant)) / (2 * beta)

# ─── Step 2: alpha_c(k_perp) = s* ──────────────────────────────────
print("\n─── Step 2: Abscissa of convergence ───")
print("For H(s) = exp(-gamma_z * d), the rightmost singularity is at s*.")
print("Therefore alpha_c(k_perp) = s*(k_perp).")
print()
print("Key property: alpha_c(0) = 0 (branch point at origin for k_perp=0)")
print("              alpha_c(k_perp) > 0 for k_perp > 0")
print("              alpha_c grows with k_perp — higher modes are harder")

# ─── Step 3: CFL condition per mode ─────────────────────────────────
print("\n─── Step 3: CFL feasibility per mode ───")
print("From CES Theorem 1:")
print("  alpha_c(k_perp) * t_max + ln(C/eps) <= L - delta_s")
print()
print("This gives the maximum feasible alpha_c:")

def s_star_max(t_end, kappa_val):
    """Maximum feasible branch point from CFL condition."""
    t_max = 2 * kappa_val * t_end
    return (L_DBL - delta_s - np.log(C_tail / eps_tail)) / t_max

# ─── Step 4: Invert to get k_perp,max ──────────────────────────────
print("\n─── Step 4: k_perp,max from s*_max ───")
print("From s* = [-alpha + sqrt(alpha^2 + 4*beta*k_perp^2)] / (2*beta):")
print("  2*beta*s* + alpha = sqrt(alpha^2 + 4*beta*k_perp^2)")
print("  (2*beta*s* + alpha)^2 = alpha^2 + 4*beta*k_perp^2")
print("  k_perp^2 = [(2*beta*s* + alpha)^2 - alpha^2] / (4*beta)")
print("           = s* * (beta*s* + alpha)")
print()
print("Therefore:")
print("  k_perp,max^2 = s*_max * (beta * s*_max + alpha)")

def k_perp_max(s_max, alpha, beta):
    """Maximum recoverable k_perp from feasibility-limited s*."""
    return np.sqrt(s_max * (beta * s_max + alpha))

# ─── Step 5: Asymptotic regimes ─────────────────────────────────────
print("\n─── Step 5: Two asymptotic regimes ───")
print()
print("Small k_perp (diffusive regime, k_perp << alpha/sqrt(beta)):")
print("  s* ≈ k_perp^2 / alpha   (beta*s^2 negligible)")
print("  alpha_c grows as k_perp^2 — quadratic")
print()
print("Large k_perp (wave regime, k_perp >> alpha/sqrt(beta)):")
print("  s* ≈ k_perp / sqrt(beta)   (alpha*s negligible)")
print("  alpha_c grows as k_perp — linear")
print()
print("Crossover wavenumber:")
print("  k_perp,x where both terms balance: k_perp^2/alpha = k_perp/sqrt(beta)")
print("  k_perp,x = alpha / sqrt(beta)")

# ─── Step 6: Connection to crossover frequency ─────────────────────
print("\n─── Step 6: Connection to omega_cross ───")
print("The crossover frequency is omega_x = alpha/beta")
print("The crossover wavenumber is k_perp,x = alpha/sqrt(beta)")
print()
print("Relationship: k_perp,x^2 = alpha^2/beta = alpha * omega_x")
print("Or equivalently: k_perp,x = sqrt(alpha * omega_x)")
print()
print("At the crossover wavenumber, the branch point is:")

def verify_crossover(alpha, beta):
    k_x = alpha / np.sqrt(beta)
    omega_x = alpha / beta
    s_star = branch_point(k_x, alpha, beta)
    print(f"  alpha = {alpha:.4e}, beta = {beta:.4e}")
    print(f"  omega_x = alpha/beta = {omega_x:.4e}")
    print(f"  k_perp,x = alpha/sqrt(beta) = {k_x:.4e}")
    print(f"  s*(k_perp,x) = {s_star:.4e}")
    print(f"  omega_x / s*(k_perp,x) = {omega_x / s_star:.6f}")
    # Analytical: s* at crossover = (-alpha + sqrt(alpha^2 + 4*alpha^2))/2beta
    #           = (-alpha + alpha*sqrt(5))/2beta = alpha*(sqrt(5)-1)/(2*beta)
    #           = omega_x * (sqrt(5)-1)/2 = omega_x * golden_ratio_conjugate
    golden = (np.sqrt(5) - 1) / 2
    print(f"  Predicted: s*(k_x) = omega_x * (sqrt(5)-1)/2 = omega_x * {golden:.6f}")
    print(f"  Actual ratio: {s_star / omega_x:.6f}")
    return s_star, omega_x, k_x

print("\nMaxwell (dry sand): sigma=1e-4, eps_r=4")
alpha_em = MU_0 * 1e-4
beta_em = MU_0 * EPS_0 * 4
s_em, omega_em, k_em = verify_crossover(alpha_em, beta_em)

print("\nMaxwell (wet clay): sigma=0.1, eps_r=10")
alpha_clay = MU_0 * 0.1
beta_clay = MU_0 * EPS_0 * 10
s_clay, omega_clay, k_clay = verify_crossover(alpha_clay, beta_clay)

print("\nAcoustics (tissue): nu=1e-3, c=1540")
alpha_ac = 1e-3 / 1540**2
beta_ac = 1.0 / 1540**2
s_ac, omega_ac, k_ac = verify_crossover(alpha_ac, beta_ac)

# ─── Step 7: Lambda connection ──────────────────────────────────────
print("\n─── Step 7: Connection to Lambda ───")
print()
print("At the crossover frequency omega_x, the 1D propagation constant is:")
print("  gamma(i*omega_x) = sqrt(alpha*i*omega_x + beta*(i*omega_x)^2)")
print("                   = sqrt(i*alpha*omega_x - beta*omega_x^2)")
print("                   = sqrt(i*alpha^2/beta - alpha^2/beta)")
print("                   = sqrt((alpha^2/beta)*(i - 1))")
print("                   = alpha/sqrt(beta) * sqrt(i - 1)")
print()

def gamma_at_crossover(alpha, beta):
    omega_x = alpha / beta
    s = 1j * omega_x
    gamma_sq = alpha * s + beta * s**2
    gamma = np.sqrt(gamma_sq)
    if gamma.real < 0:
        gamma = -gamma
    return gamma, omega_x

def skin_depth_wavelength_ratio(alpha, beta):
    gamma, omega_x = gamma_at_crossover(alpha, beta)
    # Skin depth: delta = 1/Re(gamma)
    # Wavelength: lambda = 2*pi/Im(gamma)
    delta = 1.0 / gamma.real
    wavelength = 2 * np.pi / abs(gamma.imag)
    ratio = delta / wavelength
    return ratio, delta, wavelength, gamma

print("System                    delta/lambda   Lambda_eff = 1/(delta/lambda)")
print("-" * 70)
for name, al, be in [("Maxwell (dry sand)", alpha_em, beta_em),
                       ("Maxwell (wet clay)", alpha_clay, beta_clay),
                       ("Acoustics (tissue)", alpha_ac, beta_ac)]:
    ratio, delta, wl, gamma = skin_depth_wavelength_ratio(al, be)
    Lambda_eff = 1.0 / ratio
    print(f"{name:25s}  {ratio:.6f}       {Lambda_eff:.4f}")

print()
print("Predicted Lambda_- = 2*pi*cot(3*pi/8) = 2*pi / tan(3*pi/8)")
Lambda_minus = 2 * np.pi / np.tan(3 * np.pi / 8)
print(f"Lambda_- = {Lambda_minus:.4f}")

# ─── Numerical verification: k_perp,max(d) ─────────────────────────
print("\n" + "=" * 70)
print(" NUMERICAL VERIFICATION")
print("=" * 70)

t_end = 1e-6  # 1 microsecond observation window (EM timescale)

print(f"\nObservation window: t_end = {t_end*1e6:.1f} us, kappa = {kappa}")
s_max_val = s_star_max(t_end, kappa)
print(f"Maximum feasible s*: {s_max_val:.4e}")

# For each material, compute k_perp,max analytically and compare
# against brute-force evaluation
print("\n--- Maxwell (dry sand) ---")
k_max_analytical = k_perp_max(s_max_val, alpha_em, beta_em)
print(f"k_perp,max (analytical) = {k_max_analytical:.4f} rad/m")

# Brute-force: sweep k_perp, check which modes are feasible
k_test = np.linspace(0, k_max_analytical * 2, 1000)
s_star_test = branch_point(k_test, alpha_em, beta_em)
t_max_val = 2 * kappa * t_end
feasible = s_star_test * t_max_val + np.log(C_tail / eps_tail) <= L_DBL - delta_s
k_max_brute = k_test[feasible][-1] if np.any(feasible) else 0

print(f"k_perp,max (brute force) = {k_max_brute:.4f} rad/m")
print(f"Match: {abs(k_max_analytical - k_max_brute) / k_max_analytical:.2e} relative error")

# ─── Plot: k_perp,max vs depth for multiple materials ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Branch point s*(k_perp) for different systems
ax = axes[0, 0]
k_range = np.linspace(0, 1000, 500)
for name, al, be, color in [("Dry sand", alpha_em, beta_em, "C0"),
                              ("Wet clay", alpha_clay, beta_clay, "C1"),
                              ("Tissue", alpha_ac, beta_ac, "C2")]:
    s_vals = branch_point(k_range, al, be)
    ax.semilogy(k_range, s_vals, color=color, lw=2, label=name)
    # Mark crossover
    k_x = al / np.sqrt(be)
    if k_x < 1000:
        s_x = branch_point(k_x, al, be)
        ax.plot(k_x, s_x, 'o', color=color, ms=8)

ax.set_xlabel("$k_\\perp$ [rad/m]")
ax.set_ylabel("$s^*(k_\\perp) = \\alpha_c(k_\\perp)$")
ax.set_title("(a) Branch point (abscissa of convergence) vs $k_\\perp$")
ax.legend()
ax.grid(True, alpha=0.3)

# (b) sigma_NILT(k_perp) for fixed depth and t_end
ax = axes[0, 1]
t_end_em = 1e-7  # 100 ns
depths = [0.01, 0.1, 0.5, 1.0]
for d_val in depths:
    # sigma_NILT doesn't depend on d directly — it depends on alpha_c which
    # depends on k_perp but not d. The d enters through the *amplitude*
    # attenuation, not through the feasibility condition.
    # sigma_NILT(k_perp) = [alpha_c(k_perp) * t_max + ln(C/eps)] / (L - delta_s)
    t_max_em = 2 * kappa * t_end_em
    sigma = (branch_point(k_range, alpha_em, beta_em) * t_max_em
             + np.log(C_tail / eps_tail)) / (L_DBL - delta_s)
    ax.plot(k_range, sigma, lw=2, label=f"$t_{{end}}$ = {t_end_em*1e9:.0f} ns")

ax.axhline(1.0, color='red', ls='--', lw=1.5, label="$\\sigma_{NILT} = 1$ (infeasible)")
ax.set_xlabel("$k_\\perp$ [rad/m]")
ax.set_ylabel("$\\sigma_{NILT}(k_\\perp)$")
ax.set_title("(b) Modewise stiffness ratio (dry sand)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.5)

# (c) k_perp,max vs t_end for different materials
ax = axes[1, 0]
t_end_range = np.logspace(-9, -4, 200)
for name, al, be, color in [("Dry sand", alpha_em, beta_em, "C0"),
                              ("Wet clay", alpha_clay, beta_clay, "C1")]:
    k_max_vs_t = []
    for te in t_end_range:
        sm = s_star_max(te, kappa)
        if sm > 0:
            km = k_perp_max(sm, al, be)
        else:
            km = 0
        k_max_vs_t.append(km)
    ax.loglog(t_end_range * 1e6, k_max_vs_t, color=color, lw=2, label=name)
    # Mark crossover wavenumber
    k_x = al / np.sqrt(be)
    ax.axhline(k_x, color=color, ls=':', lw=1, alpha=0.5)

ax.set_xlabel("$t_{end}$ [$\\mu$s]")
ax.set_ylabel("$k_{\\perp,max}$ [rad/m]")
ax.set_title("(c) Maximum recoverable $k_\\perp$ vs observation window")
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# (d) Asymptotic verification: s* ~ k^2/alpha (small k) vs s* ~ k/sqrt(beta) (large k)
ax = axes[1, 1]
k_wide = np.logspace(-2, 6, 500)
s_exact = branch_point(k_wide, alpha_em, beta_em)
s_diffusive = k_wide**2 / alpha_em
s_wavelike = k_wide / np.sqrt(beta_em)

ax.loglog(k_wide, s_exact, 'k-', lw=2, label="Exact $s^*(k_\\perp)$")
ax.loglog(k_wide, s_diffusive, 'b--', lw=1, label="Diffusive: $k_\\perp^2/\\alpha$")
ax.loglog(k_wide, s_wavelike, 'r--', lw=1, label="Wave-like: $k_\\perp/\\sqrt{\\beta}$")
# Mark crossover
k_x_em = alpha_em / np.sqrt(beta_em)
ax.axvline(k_x_em, color='green', ls=':', lw=1.5, alpha=0.7,
           label=f"$k_{{\\perp,\\times}} = \\alpha/\\sqrt{{\\beta}}$ = {k_x_em:.1f}")
ax.set_xlabel("$k_\\perp$ [rad/m]")
ax.set_ylabel("$s^*(k_\\perp)$")
ax.set_title("(d) Asymptotic regimes (dry sand)")
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

fig.suptitle("Crossover-Limited Recoverability", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("scripts/crossover_theorem.png", dpi=200, bbox_inches="tight")
print(f"\nSaved -> scripts/crossover_theorem.png")
plt.close(fig)

# ─── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" THEOREM 2 SUMMARY")
print("=" * 70)
print("""
THEOREM (Crossover-Limited Recoverability):
For the two-term dispersion pencil gamma^2(s) = alpha*s + beta*s^2,
the transfer function H(s, k_perp, d) = exp(-gamma_z * d) has branch point

    s*(k_perp) = [-alpha + sqrt(alpha^2 + 4*beta*k_perp^2)] / (2*beta)

which is the abscissa of convergence of H.
The CFL feasibility condition (Theorem 1, CES paper) gives:

    s*(k_perp) <= s*_max = (L - delta_s - ln(C/eps)) / t_max

Inverting:

    k_perp,max = sqrt(s*_max * (beta * s*_max + alpha))

Two asymptotic regimes:
  - Diffusive (k_perp << alpha/sqrt(beta)):  s* ~ k_perp^2 / alpha  (quadratic)
  - Wave-like (k_perp >> alpha/sqrt(beta)):  s* ~ k_perp / sqrt(beta)  (linear)

The crossover wavenumber k_perp,x = alpha/sqrt(beta) = sqrt(alpha * omega_x)
determines where the scaling transition occurs.

The skin-depth-to-wavelength ratio at the 1D crossover frequency is:
  delta/lambda |_{omega_x} = 1/Lambda_-

where Lambda_- = 2*pi*cot(3*pi/8) ≈ 2.603.

This enters the 2D feasibility because the crossover wavenumber k_perp,x
divides the transverse spectrum into:
  - k_perp < k_perp,x: diffusive band, alpha_c ~ k^2/alpha (fast stiffness growth)
  - k_perp > k_perp,x: wave band, alpha_c ~ k/sqrt(beta) (slow stiffness growth)

The recoverable mode count at depth d is dominated by the diffusive band,
where modes are lost quadratically. The transition at k_perp,x — controlled
by Lambda_- — determines the shape of the feasibility boundary.

QED (modulo the Lambda connection, which is structural but enters through
the depth normalization rather than the mode counting directly).
""")
