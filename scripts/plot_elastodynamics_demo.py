"""
Elastodynamics demo (CMAME target) - four-panel figure:
  (A) P- and S-potentials vs normalised time, three materials overlaid on a
      single panel with tight y-axis. The shared t/t_s axis makes the c_p/c_s
      ratio read directly from the P-peak positions.
  (B) Kelvin-Voigt viscoelastic attenuation: rubber lossless vs damped,
      showing the s-dependent effective speed shrinking peak amplitude.
  (C) 2D P-potential snapshot at t = t_P (steel; representative).
  (D) 2D S-potential snapshot at t = t_S (steel; representative).
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
import time

from scalpel.backends import get_backend
from scalpel.core.engine import SpectralEngine, GridParams, NILTParams
from scalpel.core.dispersion import elastic_pwave, elastic_swave
from scalpel.core.feasibility import tune_params, refine_until_accept
from scalpel.core.nilt import nilt_scalar
from scalpel.systems.elastodynamics import MATERIALS

backend = get_backend()

chosen = ["sandstone", "steel", "ice"]
colors = {"sandstone": "C1", "steel": "C0", "ice": "C2"}

depth = 0.05
Nx = 64
dx = 0.001
grid = GridParams(Nx=Nx, Ny=Nx, dx=dx, dy=dx)
x = (np.arange(Nx) - Nx // 2) * dx
w_src = 3 * dx
X, Y = np.meshgrid(x, x, indexing="ij")
source_np = np.exp(-(X ** 2 + Y ** 2) / (2 * w_src ** 2))
source = backend.array(source_np, dtype=complex)


def laplace_gaussian(s, pw):
    t0 = 3 * pw
    return pw * cmath.exp(-s * t0 + 0.5 * pw ** 2 * s ** 2) * np.sqrt(2 * np.pi)


def run_scalar(c_p, c_s, rho, depth, eta_p=0.0, eta_s=0.0, pw_frac=0.25):
    """Return time axes and normalised P- and S-potentials at the observation depth."""
    t_p = depth / c_p
    t_s = depth / c_s
    pw = pw_frac * (t_s - t_p)
    t0 = 3 * pw
    t_end = 4 * t_s

    params = tune_params(t_end=t_end, alpha_c=0.0, C=1.0, kappa=2.0,
                         eps_tail=1e-6, N_init=512, rho=2.0 / pw)

    def F_phi(s):
        c_eff_sq = c_p ** 2 + (eta_p + 2.0 * eta_s) * s / rho
        gamma = cmath.sqrt(s ** 2 / c_eff_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-gamma * depth) * laplace_gaussian(s, pw)

    def F_psi(s):
        c_eff_sq = c_s ** 2 + eta_s * s / rho
        gamma = cmath.sqrt(s ** 2 / c_eff_sq)
        if gamma.real < 0:
            gamma = -gamma
        return cmath.exp(-gamma * depth) * laplace_gaussian(s, pw)

    refined = refine_until_accept(F_phi, params, t_end,
                                  eps_im_max=1e-2, eps_conv=1e-2,
                                  N_max=8192, t_eval_min=t_end * 0.01)
    phi, t_arr, _ = nilt_scalar(F_phi, refined.a, refined.T, refined.N)
    psi, _,     _ = nilt_scalar(F_psi, refined.a, refined.T, refined.N)
    return dict(t=t_arr, phi=phi, psi=psi, t_p=t_p, t_s=t_s, t0=t0, refined=refined)


# --- Scalar NILT for the three materials (Panel A) -----------------
scalar = {}
for name in chosen:
    mat = MATERIALS[name]
    scalar[name] = run_scalar(mat.c_p, mat.c_s, mat.rho, depth)
    print(f"[{name}]  t_p={scalar[name]['t_p']*1e6:.2f} us  "
          f"t_s={scalar[name]['t_s']*1e6:.2f} us  "
          f"c_p/c_s={mat.speed_ratio:.3f}  N={scalar[name]['refined'].N}")

# --- Viscoelastic contrast: rubber lossless vs Kelvin-Voigt (Panel B) ----
# Pick eta_s so the S-wave crossover omega_cross_s = mu / eta_s lies
# at ~3x the source centre frequency: visible attenuation without full cutoff.
rubber = MATERIALS["rubber"]
t_s_rub = depth / rubber.c_s
pw_rub  = 0.25 * (t_s_rub - depth / rubber.c_p)
omega_source = 2 * np.pi / pw_rub            # dominant angular frequency of Gaussian
omega_cross_target = 1.5 * omega_source    # marginal damping; keeps NILT well-conditioned
eta_s_demo = rubber.mu / omega_cross_target
print(f"[rubber demo] omega_source={omega_source:.3e}, "
      f"eta_s_demo={eta_s_demo:.3g} Pa*s "
      f"(omega_cross={omega_cross_target:.3e})")

ve_lossless = run_scalar(rubber.c_p, rubber.c_s, rubber.rho, depth,
                         eta_p=0.0, eta_s=0.0)
ve_damped   = run_scalar(rubber.c_p, rubber.c_s, rubber.rho, depth,
                         eta_p=0.0, eta_s=eta_s_demo)
print(f"[rubber] lossless max(psi) = {np.max(np.abs(ve_lossless['psi'])):.3e}, "
      f"damped max(psi) = {np.max(np.abs(ve_damped['psi'])):.3e}")

# --- 2D engine run for steel (Panels C, D) -------------------------
steel = MATERIALS["steel"]
c_p, c_s, rho = steel.c_p, steel.c_s, steel.rho
t_p_steel = depth / c_p
t_s_steel = depth / c_s
pw_steel = 0.25 * (t_s_steel - t_p_steel)
params = tune_params(t_end=4 * t_s_steel, alpha_c=0.0, C=1.0, kappa=2.0,
                     eps_tail=1e-6, N_init=512, rho=2.0 / pw_steel)

def F_steel(s, _c=c_p, _d=depth, _pw=pw_steel):
    gamma = cmath.sqrt(s ** 2 / _c ** 2)
    if gamma.real < 0:
        gamma = -gamma
    return cmath.exp(-gamma * _d) * laplace_gaussian(s, _pw)

refined = refine_until_accept(F_steel, params, 4 * t_s_steel,
                              eps_im_max=1e-2, eps_conv=1e-2,
                              N_max=8192, t_eval_min=4 * t_s_steel * 0.01)

nilt_p = NILTParams(a=refined.a, T=refined.T, N=refined.N)

def disp_p(s, KX, KY, b, _c=c_p, _rho=rho):
    return elastic_pwave(s, KX, KY, _c, _rho, 0.0, 0.0, b)

def disp_s(s, KX, KY, b, _c=c_s, _rho=rho):
    return elastic_swave(s, KX, KY, _c, _rho, 0.0, b)

engine_p = SpectralEngine(disp_p, backend)
engine_s = SpectralEngine(disp_s, backend)

# Warm-up + timed
for eng in (engine_p, engine_s):
    _ = eng.forward(source, depth, grid, nilt_p)
    if backend.name == "jax":
        _[0].block_until_ready()

t0 = time.perf_counter()
field_p, t_arr2d = engine_p.forward(source, depth, grid, nilt_p)
field_s, _       = engine_s.forward(source, depth, grid, nilt_p)
if backend.name == "jax":
    field_p.block_until_ready()
    field_s.block_until_ready()
wall = (time.perf_counter() - t0) * 1e3

field_p_np = backend.to_numpy(field_p)
field_s_np = backend.to_numpy(field_s)
t2d = backend.to_numpy(t_arr2d)

idx_p = int(np.argmin(np.abs(t2d - t_p_steel)))
idx_s = int(np.argmin(np.abs(t2d - t_s_steel)))
print(f"[steel 2D] phi at t_p, psi at t_s. Combined engine time: {wall:.1f} ms")

# ==== Figure: 2 rows x 2 cols ====
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

# --- (A) overlay three materials, peak-normalised -----------------
ax = axes[0, 0]
for name in chosen:
    r = scalar[name]
    mat = MATERIALS[name]
    t_norm = r["t"] / r["t_s"]
    phi_n = r["phi"] / np.max(np.abs(r["phi"]))
    psi_n = r["psi"] / np.max(np.abs(r["psi"]))
    mask = (r["t"] > 0.1 * r["t_p"]) & (r["t"] < 3.5 * r["t_s"])
    ax.plot(t_norm[mask], phi_n[mask], "-",  color=colors[name], lw=1.8,
            label=rf"{mat.name}: $c_p/c_s={mat.speed_ratio:.2f}$")
    ax.plot(t_norm[mask], psi_n[mask], "--", color=colors[name], lw=1.2, alpha=0.8)
ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
ax.text(1.01, 1.08, r"$t_S$", fontsize=9)
ax.set_xlabel(r"Observation time $t / t_S$")
ax.set_ylabel("Potential (peak-normalised)")
ax.set_title("(A) P- and S-arrivals across three materials\n"
             r"solid = $\phi$ (P), dashed = $\psi$ (S)",
             fontsize=10, fontweight="bold")
ax.set_xlim(0, 3.5)
ax.set_ylim(-0.35, 1.25)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)

# --- (B) viscoelastic contrast ------------------------------------
ax = axes[0, 1]
t_s_rub = ve_lossless["t_s"]
amp_norm = np.max(np.abs(ve_lossless["psi"]))
mask_ve = (ve_lossless["t"] > 0) & (ve_lossless["t"] < 3 * t_s_rub)
ax.plot(ve_lossless["t"][mask_ve] / t_s_rub,
        ve_lossless["psi"][mask_ve] / amp_norm,
        "-", color="C3", lw=1.8,
        label=r"Lossless ($\eta_s = 0$)")
ax.plot(ve_damped["t"][mask_ve] / t_s_rub,
        ve_damped["psi"][mask_ve] / amp_norm,
        "--", color="C3", lw=1.8, alpha=0.85,
        label=rf"Kelvin-Voigt ($\eta_s = {eta_s_demo:.0f}\,$Pa$\cdot$s)")
ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
ax.set_xlabel(r"Observation time $t / t_S$")
ax.set_ylabel(r"S-potential $\psi$ (normalised to lossless peak)")
ax.set_title("(B) Viscoelastic S-wave attenuation in rubber\n"
             r"same source and depth; only $\eta_s$ changes",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# --- (C) P-potential snapshot at t_P (steel) ----------------------
ax = axes[1, 0]
snap_p = field_p_np[:, :, idx_p]
vmax_p = float(np.max(np.abs(snap_p)))
im = ax.imshow(snap_p.T,
               extent=[x.min() * 1e3, x.max() * 1e3, x.min() * 1e3, x.max() * 1e3],
               cmap="RdBu_r", vmin=-vmax_p, vmax=vmax_p,
               origin="lower", aspect="equal")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title(r"(C) Steel: $\phi(x, y, z = d, t = t_P)$",
             fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.82, label=r"$\phi$")

# --- (D) S-potential snapshot at t_S (steel) ----------------------
ax = axes[1, 1]
snap_s = field_s_np[:, :, idx_s]
vmax_s = float(np.max(np.abs(snap_s)))
im = ax.imshow(snap_s.T,
               extent=[x.min() * 1e3, x.max() * 1e3, x.min() * 1e3, x.max() * 1e3],
               cmap="PuOr_r", vmin=-vmax_s, vmax=vmax_s,
               origin="lower", aspect="equal")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title(r"(D) Steel: $\psi(x, y, z = d, t = t_S)$",
             fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.82, label=r"$\psi$")

fig.suptitle(
    "Elastodynamics via spectral scalpel: Helmholtz decomposition of Navier's equation\n"
    rf"depth = {depth*1e2:.0f} cm, grid = {Nx}$\times${Nx}, "
    rf"dx = {dx*1e3:.1f} mm, steel 2D engine time = {wall:.0f} ms",
    fontsize=12, y=1.01)
plt.tight_layout()
out = "paper/figures/elastodynamics_demo.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
