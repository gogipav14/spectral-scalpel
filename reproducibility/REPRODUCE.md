# Reproducing the spectral-scalpel TOMS manuscript

This directory is the single source of truth for re-executing every
quantitative claim in the TOMS manuscript. The same tree underpins:

- A CodeOcean GPU capsule (this directory, plus the `Dockerfile` here, is the capsule's `code` + `environment`).
- The Zenodo archival snapshot ([doi:10.5281/zenodo.20682437](https://doi.org/10.5281/zenodo.20682437)).
- The TOMS RCR re-execution path described in the manuscript's Section 7 and in [`acm_artifact_checklist.md`](acm_artifact_checklist.md).

## Quickstart (local, CPU-only verify, ~21 s on the reference workstation)

```bash
git clone https://github.com/gogipav14/spectral-scalpel
cd spectral-scalpel
pip install -e ".[mpmath,viz,dev]"
bash reproducibility/reproduce.sh
```

A clean exit 0 means every regenerated CSV matched its archived
counterpart in `reproducibility/data/` within the default 5 percent
tolerance. The script prints a diff report to
`reproducibility/data/diff_report.txt`.

## Full multi-backend reproduction (GPU, ~15 min on the reference workstation)

```bash
pip install -e ".[jax,torch,cupy,mpmath,viz,dev]"
ENABLE_GPU=1 bash reproducibility/reproduce.sh
```

This re-runs the 15-rep median+IQR campaign (Table 2), the
multi-backend wall-clock benchmark (Figure 3), and the JAX
CPU-vs-CUDA subprocess sidecar (Section 5.1). Absolute wall-clock
numbers will only reproduce within tolerance on an NVIDIA RTX 5060
under CUDA 13; on other GPUs the relative ordering of backends is
preserved but the absolute multipliers shift, as documented in the
manuscript's Section 7.5.

## CodeOcean capsule (full GPU reproduction)

1. Create a new CodeOcean capsule from the GitHub repository
   `gogipav14/spectral-scalpel`.
2. Select a GPU machine type (NVIDIA T4 or A10; both expose CUDA 13
   driver compatibility).
3. Point the capsule build at `reproducibility/Dockerfile`. The
   Docker image installs CUDA 13 runtime, Python 3.12, the
   spectral-scalpel package, and the full multi-backend extras (JAX
   CUDA, PyTorch, CuPy, mpmath, viz, dev).
4. Set the capsule's run script to `bash reproducibility/reproduce.sh`.
   The Dockerfile's CMD does this by default if the capsule UI does
   not override it.
5. Click "Reproducible Run".

The capsule will publish:

- `reproducibility/figures/*.png`: regenerated manuscript figures.
- `reproducibility/data/benchmark_repeated_long.csv` and `..._summary.csv`: fresh 15-rep median+IQR campaign data.
- `reproducibility/data/diff_report.txt`: per-row CSV diff against the archived reference data.

Expected wall-time on a CodeOcean T4 instance: 8 to 18 minutes,
dominated by the FTCS+L1 baseline sweeps. The CuPy CUDA backend will
record different absolute numbers than the manuscript's RTX 5060
reference (T4 is roughly 2 to 3 times slower on this workload), so
the diff report will flag the GPU wall-clock rows as exceeding the
5 percent tolerance. The relative ordering of backends, the accuracy
claims (Figure 1, Figure 2, Figure 4, Section 6.4), and the
algorithmic speedup over time-marching references are all
hardware-independent and should pass.

## Tree

```
reproducibility/
├── REPRODUCE.md                this file
├── reproduce.sh                CPU-default driver, ENABLE_GPU=1 opts in to GPU
├── Dockerfile                  CUDA-13 + Python 3.12 + multi-backend extras
├── acm_artifact_checklist.md   TOMS RCR checklist mapping claim → script → reference
├── data/                       archived CSVs (manuscript reference) + .pkl sweep dumps
├── baselines/                  reference solver wrappers (Yee FDTD, FTCS+L1, de Hoog, fixed Talbot)
├── scripts/                    figure regeneration + benchmark drivers + csv_diff
└── figures/                    regenerated figure outputs (gitignored)
```

## Tolerance and what counts as "reproduces"

The default tolerance is 5 percent relative on every numeric CSV
cell. This is set in the manuscript's Section 7.5 reproducibility
scope discussion. Accuracy claims (`fractional_mpmath_reference.csv`,
`recoverability_class_comparison.csv`,
`lambda_plus_verification.csv`,
`transform_domain_nilt_comparison.csv`) reproduce to well within
that tolerance on any modern x86 host running CUDA 13. Wall-clock
claims (`benchmark_multi_backend.csv`, `benchmark_repeated_*.csv`)
are hardware-specific: the reference workstation is an Intel Core
Ultra 5 225F (10-core, 32 GB DDR5) with an NVIDIA RTX 5060 (8 GB
VRAM) running CUDA 13 under WSL2.

## Citing the artifact

If you use the reproducibility tree, please cite both the manuscript
and the Zenodo archival snapshot:

```
@misc{pavlov2026recoverability_archive,
  author = {Pavlov, Gorgi},
  title  = {{scalpel}: A multi-backend FFT-NILT library for spectral slab PDE solvers (reproducibility archive, v0.2.0)},
  year   = {2026},
  doi    = {10.5281/zenodo.20682437},
}
```
