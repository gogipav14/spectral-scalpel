# Baseline implementations referenced by paper_toms/article.tex Sec 5.7

Each file below is the matched-accuracy baseline used in the
benchmark methodology section of the manuscript. Where the
implementation already lives elsewhere in the scalpel codebase
(typically `scalpel/reference/` for FDTD and `reports/claims_audit/`
for FTCS/NILT baselines), this directory contains either a direct
symlink target or a thin wrapper that selects the appropriate
parameters; otherwise the full baseline source is in this directory.

## Telegrapher / lossy Maxwell baseline (vs Yee FDTD)

| File | Role |
|---|---|
| `yee_fdtd_telegrapher.py` | Front-door wrapper. Delegates to `scalpel.reference.fdtd_maxwell_3d` (or `_1d` for the centerline mode), sets CFL = 0.99, 10 cells per minimum wavelength, second-order central spatial differences, Berenger PML at the z-boundaries. |
| `scalpel/reference/fdtd_maxwell_3d.py` | 3D Yee FDTD reference (already in repo). |

## Fractional Caputo baseline (vs explicit FTCS + L1)

| File | Role |
|---|---|
| `ftcs_l1_fractional.py` | Wrapper. Uses the L1 quadrature (Lin-Xu 2007 / Sun-Wu 2006) for the Caputo derivative with the largest stable time-step (CFL = 0.5 for explicit FTCS), second-order central spatial differences. Delegates to the canonical implementation in `reports/claims_audit/benchmark_fractional_heat_3d.py`. |

## Transform-domain NILT baselines (vs de Hoog and fixed Talbot)

| File | Role |
|---|---|
| `de_hoog_baseline.py` | Reference de Hoog quotient-difference inversion at convergence parameter epsilon = 1e-10 (de Hoog 1982 recommendation). |
| `fixed_talbot_baseline.py` | Reference fixed Talbot contour at Abate-Valko parameter M = N_NILT/2, nu = 0.2 * M (Abate-Valko 2004). |

The baselines are wrapped (not duplicated) so the per-baseline parameter
choices are visible in one place, the source of truth lives in the
existing module, and any change to a parameter has a single point of
control.

## Why this directory exists at all

The manuscript's Section 5.7 "Benchmark methodology and baseline
tuning" promises that "all baseline implementations are in
`reproducibility/baselines/`." The promise is the load-bearing
artifact-vs-claim integrity item flagged in
`paper_toms/reviews/simulated_referee_report.md` round-2 Issue #1.

A TOMS referee `cd`-ing into `paper_toms/reproducibility/baselines/`
should find runnable Python files implementing the baselines at the
parameter choices the manuscript advertises. Each file's first 20
lines should make those parameter choices visible without scrolling.
