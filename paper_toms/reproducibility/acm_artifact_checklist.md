# ACM Artifact Available checklist - paper 1 (scalpel)

Reference: ACM Artifact Review and Badging Version 1.1
(<https://www.acm.org/publications/policies/artifact-review-and-badging-current>).

Companion to manuscript "scalpel: A multi-backend FFT-NILT primitive
for spectral slab PDE solvers, with class-dependent finite-precision
auto-tuning." We request the **Artifact Available** badge.
(Functional and Reusable badges are not requested here, although the
artifact would qualify for both with the additional reviewer
walkthrough that those badges entail.)

## Artifact identification

| Field | Value |
|---|---|
| Title | `scalpel` |
| Version | (to be tagged at acceptance) |
| Persistent identifier | <https://doi.org/10.5281/zenodo.20682437> |
| License | Apache-2.0 |
| Source repository | (Zenodo archive; private mirror at github.com/[redacted]/spectral-scalpel until acceptance) |
| Citation metadata | `CITATION.cff` at the root of the archive |

## Artifact Available badge requirements

Per ACM's policy, the badge is granted if the artifact is:

1. **Placed on a publicly accessible archival repository** with a DOI or link.
   - **Status:** YES. Zenodo archive at the DOI above. Zenodo
   provides long-term archival with a permanent DOI per CoreTrustSeal
   certification.

2. **Available without restriction**.
   - **Status:** YES. Apache-2.0 license. No registration, no
   click-through, no payment.

3. **Documentation sufficient to assess the artifact**.
   - **Status:** YES. The archive includes:
     - The manuscript PDF (`paper_toms/article.pdf`)
     - A README at the archive root with installation, examples,
       and the manuscript-figure reproduction script
     - Per-module docstrings (`scalpel/api.py`, `scalpel/core/*`)
     - A regression test suite (`tests/`)
     - Filled-in version of this checklist

## What ships in the archive

| Path | Purpose |
|---|---|
| `scalpel/` | The library (Python package, Apache-2.0) |
| `scalpel/api.py` | Public API (top-level callables) |
| `scalpel/core/` | NILT engine, dispersion module, feasibility auto-tuner, ETDRK4 reference |
| `scalpel/backends/` | JAX, PyTorch backend dispatch |
| `scalpel/systems/` | Four bundled physics dispersions (Maxwell, chromatography, acoustics, elastodynamics) |
| `scalpel/reference/` | mpmath Talbot reference, RK45 reference |
| `scalpel/benchmarks/` | Wall-clock and accuracy harnesses |
| `tests/` | pytest regression suite |
| `paper_toms/data/` | Raw CSV benchmark data underlying every quantitative claim in the manuscript |
| `paper_toms/figures/` | Figures shown in the manuscript |
| `reproducibility/reproduce.sh` | One-command reproduction driver |
| `reproducibility/acm_artifact_checklist.md` | This file |
| `CITATION.cff` | Machine-readable citation metadata |
| `LICENSE` | Apache-2.0 |
| `pyproject.toml` | Dependencies pinned; backend extras as install groups |

## How to reproduce the manuscript's quantitative claims

From a clean clone of the archive:

```bash
git clone <archive-url> scalpel && cd scalpel
pip install ".[dev]"               # CPU-only by default
bash reproducibility/reproduce.sh
# CPU-only path: measured 21 s wall time end-to-end on the bundle's
# reference workstation (Intel Core Ultra 5 225F, 32 GB, WSL2,
# Python 3.12.3). All 29 regenerated CSV rows within 5% tolerance;
# the GPU-only benchmark CSV is correctly skipped.
# GPU-enabled full reproduction (ENABLE_GPU=1) re-runs the
# benchmark_multi_backend.csv rows; wall time depends on the
# installed backends (estimated <15 min on RTX-5060-class hardware).
```

The `reproduce.sh` script:

1. Runs the four bundled physics demos.
2. Runs the multi-backend wall-clock benchmark (CPU by default;
   enable GPU paths with `ENABLE_GPU=1`).
3. Generates Figures 1-4 of the manuscript.
4. Diffs the regenerated CSVs against the archived CSVs and prints a
   per-row tolerance report.

Per-claim mapping:

| Manuscript claim | Reproducing script | Verifies against |
|---|---|---|
| Fig. 1 spectral convergence | `reports/claims_audit/verify_convergence.py` | Closed-form heat-equation solution |
| Fig. 2 feasibility cutoffs | `reports/claims_audit/recoverability_class_comparison.py` | float32 / float64 sweep |
| Fig. 3 multi-backend benchmark | `reports/claims_audit/render_benchmark_fig_v2.py` (re-render); `reports/claims_audit/benchmark_fractional_heat_3d_all_backends.py` (regenerate) | 3D Yee FDTD via `paper_toms/reproducibility/baselines/yee_fdtd_telegrapher.py`; FTCS+L1 via `paper_toms/reproducibility/baselines/ftcs_l1_fractional.py` |
| Fig. 4 NILT head-to-head | `reports/claims_audit/make_fig4_multipanel.py` | de Hoog and fixed Talbot via `paper_toms/reproducibility/baselines/{de_hoog_baseline.py, fixed_talbot_baseline.py}` (both delegate to mpmath's canonical implementations) |
| mpmath ground-truth (Caputo) | `scalpel/reference/talbot_mpmath.py` (`centerline_reference`) | 50-digit precision Talbot inversion; precomputed CSV at `paper_toms/data/fractional_mpmath_reference.csv` |
| 24-test API-expansion suite | `tests/test_api_expansion.py` | `pytest -q`; CI matrix in `.github/workflows/ci.yml` |

## Hardware / OS dependencies

- **CPU-only path:** Linux, macOS, or Windows; Python 3.10+. No GPU required.
- **GPU path:** NVIDIA GPU with CUDA 13 drivers. We tested on a single workstation: Intel Core Ultra 5 225F (10-core CPU), 32 GB DDR5, NVIDIA RTX 5060 (8 GB VRAM, consumer GPU), running under WSL2 (Linux kernel 6.6) on Windows 11, CUDA 13. Wall-clock numbers reproduce within ±5% on this hardware; on other GPUs the relative ordering of backends is preserved but absolute numbers vary.
- **All backends:** install with `pip install ".[torch,jax,mpmath,viz,dev]"` (NumPy is a base dep; no separate extra). The Julia path additionally needs `pip install ".[julia]"` plus a Julia ≥ 1.10 installation; the CuPy reference wrapper additionally needs `pip install cupy-cuda13x` matched to your CUDA install.

## What this checklist does NOT cover

- This is an Artifact-Available submission; we do not request the
  Functional or Reusable badges in this round. Those would require
  reviewer walkthroughs and explicit functional reviewability that
  are out of scope for the initial review. We are open to upgrading
  if the editor requests.
- The wall-clock numbers in Fig. 3 are hardware-dependent. The
  algorithmic speedup over Yee FDTD and FTCS+L1 is preserved
  across hardware; the absolute multipliers shift.
- The Zenodo DOI given above is the existing archive that already
  contains the prior (SISC) submission's source. At ACM TOMS
  acceptance the archive will be tagged with a version corresponding
  to the accepted manuscript.

## Authorship and AI use

The library was written by the author with assistance from
AI-based coding tools. The author verified all mathematical results,
executed all benchmarks reported, and inspected all code. The author
assumes responsibility for all content. See the "Use of AI tools"
statement at the end of the manuscript for the explicit disclosure.
