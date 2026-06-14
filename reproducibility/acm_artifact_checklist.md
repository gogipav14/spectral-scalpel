# ACM Results Replicated checklist - paper 1 (scalpel)

Reference: ACM Artifact Review and Badging Version 1.1
(<https://www.acm.org/publications/policies/artifact-review-and-badging-current>).

Companion to manuscript "A Multi-Backend FFT-NILT Library for
Spectral Slab PDE Solvers." **We request the Results Replicated
badge** (the strongest of the artifact-evaluation tiers); the
artifact and reproducibility infrastructure are designed to pass
independent third-party re-execution by a TOMS-appointed reviewer.
The artifact also qualifies, by construction, for the Artifact
Available, Artifact Evaluated - Functional, and Artifact Evaluated
- Reusable tiers.

## Artifact identification

| Field | Value |
|---|---|
| Title | `scalpel` |
| Version | v0.2.0 |
| Persistent identifier (Zenodo) | <https://doi.org/10.5281/zenodo.20682437> |
| Source repository (GitHub) | <https://github.com/gogipav14/spectral-scalpel> (tag `v0.2.0`) |
| License | Apache-2.0 |
| Citation metadata | `CITATION.cff` at the root of both the GitHub repo and the Zenodo archive |

## Badge claims and evidence

ACM's policy distinguishes four artifact-evaluation tiers; the
artifact qualifies for all four, in escalating strength:

1. **Artifact Available**: Apache-2.0 source on Zenodo with a permanent
   DOI (CoreTrustSeal-certified) and on GitHub with a tagged release.
2. **Artifact Evaluated - Functional**: artifact is exercisable; the
   regression test suite (`pytest -q`) and `reproduce.sh` both run
   end-to-end on a clean clone in under a minute.
3. **Artifact Evaluated - Reusable**: artifact is a working library
   with documented public API, CI matrix across three OSes and four
   Python versions, semantic versioning (`v0.2.0`), and a public
   issue tracker on GitHub.
4. **Results Replicated** (requested): every quantitative claim in
   the manuscript is derivable row-by-row from the deposited CSVs;
   the `reproduce.sh` script regenerates them with a CSV-diff check
   at 5 percent tolerance; the methodology section
   (`paper_toms/article.tex`, Sec.~Benchmark methodology) details
   the timing protocol so an independent third party can re-execute
   on different hardware and confirm the relative scaling.

## What ships in the archive

| Path | Purpose |
|---|---|
| `scalpel/` | The library (Python package, Apache-2.0) |
| `scalpel/api.py` | Public API (top-level callables) |
| `scalpel/core/` | NILT engine, dispersion module, feasibility auto-tuner, ETDRK4 reference |
| `scalpel/backends/` | Four dispatched backends: NumPy (universal fallback), PyTorch (CPU/CUDA), JAX (CPU/CUDA), CuPy (CUDA). |
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

- The wall-clock numbers in Fig. 3 are hardware-dependent. The
  algorithmic speedup over Yee FDTD and FTCS+L1 is preserved
  across hardware in our spot checks; the absolute multipliers
  shift with backend library version and GPU generation. Section
  bench-methodology documents the timing protocol so an
  independent reproducer can re-execute on different hardware
  and compare relative scalings rather than absolute multipliers.
- The Zenodo DOI given above is the v2 deposit dated
  2026-06-13 and is the version that matches this manuscript's
  source tree; the v1 deposit (10.5281/zenodo.19834321,
  2026-04-27) is a prior version of the library and is not the
  evaluation target.
- Backend coverage is bounded by what the user installs locally.
  The CI matrix exercises NumPy + PyTorch + JAX on three OSes;
  CuPy and Julia paths are exercised only on the development
  workstation since GitHub-hosted runners do not provide GPUs.

## Authorship and AI use

The library was written by the author with assistance from
AI-based coding tools. The author verified all mathematical results,
executed all benchmarks reported, and inspected all code. The author
assumes responsibility for all content. See the "Use of AI tools"
statement at the end of the manuscript for the explicit disclosure.
