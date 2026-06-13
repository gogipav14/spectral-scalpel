# Option B follow-up: Cole-Cole / dispersive-Maxwell benchmark

**Status:** deferred (not in current PNAS panel set). Captured here so it does
not get lost.

## Why this is the strongest paper story

Frequency-dependent permittivity (Cole-Cole, Debye, Lorentz) is the standard
ground-penetrating-radar / biological-tissue / lossy-soil material model. In
FDTD this forces one of:

1. **ADE-FDTD** (auxiliary differential equations): one or more extra fields
   per cell whose update tracks the polarization current. Roughly 2-4x the
   per-step cost of vanilla FDTD plus a state-vector blow-up.
2. **(P)RC-FDTD** (recursive convolution / piecewise-linear RC): convolves the
   polarization response history each step. Lighter memory than ADE for
   single-pole models, heavier algebra.
3. **Z-transform / JE-FDTD** variants.

All three are well documented in the CEM literature as the practical hot
spot for dispersive-media simulations. Scalpel's cost is unchanged: swap one
dispersion function for the Cole-Cole `epsilon*(s)` form, everything else is
identical. This is exactly the "natural Laplace structure" win that the
factorization claim rests on, in the same physics as panel (a).

## Cole-Cole permittivity in Laplace domain

For a single Cole-Cole pole:

    epsilon*(s) = eps_inf + (eps_s - eps_inf) / (1 + (s * tau)**(1 - alpha))

with relaxation time `tau`, static / high-freq permittivities `eps_s`, `eps_inf`,
and broadening parameter `alpha` in [0, 1) (`alpha = 0` reduces to Debye).

Plug into the lossy Maxwell dispersion in place of `epsilon * s**2`:

    gamma_z**2 = mu_0 * (sigma * s + eps_0 * epsilon*(s) * s**2) - kx**2 - ky**2

This is a one-line addition to `scalpel/core/dispersion.py`.

## Test problem (proposed)

- Cole-Cole wet clay at 100 MHz: `eps_s = 60`, `eps_inf = 6`, `tau = 1e-9 s`,
  `alpha = 0.2`, `sigma = 0.05 S/m`. (Standard Topp-style soil model;
  references: Cassidy 2009, Topp 1980.)
- 3D slab geometry matching panel (a): 64x64 transverse, depth 0.5 m,
  `t_end = 200` ns, periodic transverse, Dirichlet absorbing far face.
- Source: Gaussian-modulated 100 MHz pulse (impulsive in space, banded in time).
- Observe E_y(0, 0, d_obs, t) at d_obs = 0.5 m.

## Methods to compare

| Method              | Per-step cost vs vanilla FDTD   | Memory         | Notes                                       |
| ------------------- | ------------------------------- | -------------- | ------------------------------------------- |
| Vanilla FDTD        | (unsuitable, ignores Cole-Cole) | -              | reference for "what you would naively do"   |
| ADE-FDTD            | ~2-4x                           | +2 fields/cell | Joseph-Hagness style, 1-pole Cole-Cole      |
| (P)RC-FDTD          | ~1.5-2.5x                       | +1 field/cell  | Luebbers-Hunsberger 1991; piecewise-linear  |
| Z-transform FDTD    | ~2x                             | comparable     | Sullivan 1992                               |
| Scalpel (Cole-Cole) | unchanged from vanilla scalpel  | unchanged      | swap dispersion fn                          |

## Why we are deferring

- ADE-FDTD with Cole-Cole alpha != 0 (true Cole-Cole, not just Debye) requires
  a fractional differential equation for the polarization current; the
  standard treatment uses Riemann-Liouville derivatives and an L1-style
  approximation per cell. Implementation cost: ~500 LOC of careful CEM in JAX,
  plus reproducing one of the canonical ADE-FDTD validation cases (Joseph and
  Hagness 1999) for confidence.
- For a 6-page PNAS, panel (a) Maxwell + new panel (b) fractional-heat already
  carries the "scalpel handles Laplace-natural physics natively" point. The
  Cole-Cole Maxwell case is a stronger third panel for a longer venue (CMAME)
  or a follow-up paper.

## When to revisit

- After PNAS submission: add as Fig. 4 / supplement for the CMAME version.
- If a reviewer specifically asks "does this scale to dispersive media?" we
  can implement the ADE-FDTD baseline and ship as a revision figure.
- If a separate methods paper is warranted: the dispersive-media benchmark is
  the natural lead.

## Related followup: CN + L1 implicit baseline for panel (b)

Panel (b) currently compares Scalpel vs FTCS+L1 (explicit fractional). A
reviewer may ask why we didn't use the implicit Crank-Nicolson + L1 scheme
instead, since that allows much larger time steps. Result we expect: still
~10-50x slower than Scalpel because the L1 history convolution is the
dominant cost, not the per-step solve. To make this rigorous we need to
implement CN+L1 cleanly. Sketch:

**PDE:** `(I - (β/2) ∇²) u^n = (I + (β/2) ∇²) u^{n-1} - L1_history_sum`
with `β = Γ(2-α) (Δt)^α D`.

**Solver (JAX-native, no CG):** the operator separates over (FFT_xy basis,
finite-difference z basis):

1. FFT in xy (periodic): diagonal in `|k_perp|^2`, eigenvalues
   `-|k_perp|^2`.
2. Finite difference in z (Dirichlet at z=0 with source overwrite, z=Lz=0):
   tridiagonal.

So per step: 2D FFT_xy → `Nx × Ny` independent tridiagonal solves of size
`Nz` via Thomas → 2D iFFT_xy. Each tridiagonal solve is O(Nz) and trivially
batched with `jax.lax.scan` or `jax.scipy.linalg.solve_banded`.

**Why not in this iteration:** ~150 LOC of careful implementation (Thomas
batched, complex coefficients, source-plane BC handling) + validation
against an analytical limit. Decision: ship Scalpel vs FTCS+L1 (102x win)
for PNAS, queue CN+L1 for the CMAME extended version where the page count
permits a fuller methods comparison.

**Additional implicit baselines worth considering for CMAME version:**

- Backward Euler + L1 (simplest implicit, larger error per step than CN
  but easier to validate)
- ETD-RK4 with the Mittag-Leffler exponential operator (would need a
  Mittag-Leffler function evaluator, e.g. Garrappa 2015 algorithm)
- L2-1_σ scheme for higher temporal order vs L1

## References

- Joseph & Hagness, IEEE Trans. AP 1999 - canonical ADE-FDTD with Lorentz/Debye.
- Luebbers & Hunsberger, IEEE Trans. AP 1991 - PRC-FDTD original.
- Cassidy, "Ground Penetrating Radar Theory and Applications" (Jol ed.) 2009 -
  Cole-Cole soil parameters.
- Sullivan, IEEE Trans. AP 1992 - Z-transform FDTD for dispersive media.
