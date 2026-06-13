"""
Chromatography radial-mode convergence audit.

Reviewers flagged the chromatography demo for not showing convergence as
the Hankel-mode truncation M is varied. We run the axisymmetric preparative
column at M = 4, 8, 16, 32, 64 modes and measure the relative L2 error
of the outlet profile against the M = 128 reference.
"""

from __future__ import annotations

import math
import pickle
import numpy as np

from scalpel.backends import get_backend
from scalpel.core.engine import CylindricalEngine, NILTParams
from scalpel.core.dispersion import convection_diffusion_cylindrical
from scalpel.core.hankel import HankelTransform
from scalpel.core.feasibility import tune_params, refine_until_accept

backend = get_backend()

# Preparative column parameters
v = 1e-3            # m/s axial velocity
Dz = 1e-6           # m^2/s axial dispersion
Dr = 1e-9           # m^2/s radial diffusion
R  = 2.3e-3         # radius
L  = 0.15           # column length
Pe = v * L / Dz
print(f"Peclet = {Pe:.0f}")

import cmath
def F_scalar(s):
    return cmath.exp(-cmath.sqrt(v**2/(4*Dz**2) + s/Dz) * L)

t_end = 3 * L / v
# Manual NILT: Bromwich shift comfortably > 0, period = 2*t_end, N=1024
nilt_p = NILTParams(a=0.1, T=2*t_end, N=1024)
print(f"NILT (manual): a={nilt_p.a:.2e}, T={nilt_p.T:.2e}, N={nilt_p.N}")
conv_phase = v / (2 * Dz)


def run_M(M):
    ht = HankelTransform(N=M, R=R)
    # Gaussian radial source centered at axis
    src_r = np.exp(-(ht.r / (0.3 * R))**2)

    def disp(s, KR, b):
        return convection_diffusion_cylindrical(s, KR, v, Dz, Dr, b)

    engine = CylindricalEngine(disp, ht, backend)
    field, t = engine.forward(src_r, L, nilt_p, conv_phase=conv_phase)
    # Outlet profile at r = 0 (centerline) across time
    return t, field[0, :]


# Reference at M = 128
t_ref, f_ref = run_M(128)
mask = (t_ref > 0.3 * L/v) & (t_ref < 2.5 * L/v)
print(f"Reference peak: {np.max(np.abs(f_ref[mask])):.3e}")

results = []
for M in [4, 8, 16, 32, 64]:
    t, f = run_M(M)
    # Interpolate onto t_ref grid in valid window
    f_interp = np.interp(t_ref[mask], t, f)
    num = np.linalg.norm(f_ref[mask] - f_interp)
    den = np.linalg.norm(f_ref[mask])
    rel = num / den
    results.append((M, rel))
    print(f"  M = {M:4d}   rel L2 vs M=128 ref = {rel:.3e}")

pkl_out = "/home/gogip/github_repos/spectral-scalpel-private/reports/claims_audit/hankel_mode_convergence.pkl"
with open(pkl_out, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved -> {pkl_out}")
