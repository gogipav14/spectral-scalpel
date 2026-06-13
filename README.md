# scalpel

`scalpel` (spectral-scalpel) is a multi-backend Python library providing a
batched FFT-accelerated numerical inverse Laplace transform (NILT)
primitive for spectral slab PDE solvers, with class-dependent
finite-precision parameter auto-tuning.

Runs across NumPy, PyTorch, JAX, and (via reference wrapper) CuPy and
Julia on CPU and GPU. See `paper_toms/article.pdf` for the manuscript
that accompanies this release.

## Install

```bash
pip install -e .                          # NumPy only
pip install -e ".[torch]"                 # adds PyTorch backend
pip install -e ".[jax]"                   # adds JAX backend
pip install -e ".[mpmath]"                # adds the 50-digit reference path
pip install -e ".[all]"                   # everything except julia/CUDA extras
```

## Quick start

```python
import scalpel as sc

field, t = sc.propagate_maxwell(
    source, depth=0.5,
    material=sc.MaxwellParams(sigma=1e-3, epsilon_r=4.0),
    grid=grid, nilt=nilt,
)
```

For power users wanting auto-tuner introspection,
parameter sweeps, or matched-precision validation against the
mpmath 50-digit Talbot reference, see `scalpel.diagnose`,
`scalpel.sweep`, and `scalpel.validate` respectively (manuscript
Section 2 for the public API tour).

## Reproducing the manuscript figures

```bash
bash paper_toms/reproducibility/reproduce.sh
# CPU-only by default; enable GPU paths with ENABLE_GPU=1.
```

The script regenerates the CSVs in `reports/repro_run/` and diffs
them against the archived copies in `paper_toms/data/`. Measured
end-to-end wall time: 21 s CPU-only on Intel Core Ultra 5 225F /
32 GB / WSL2 / Python 3.12.3.

## License

Apache-2.0. See `LICENSE`.

## Citation

If you use this software or refer to the manuscript, please cite via
`CITATION.cff` (the GitHub "Cite this repository" button picks it up
automatically). The reproducibility archive is at
[doi:10.5281/zenodo.20682437](https://doi.org/10.5281/zenodo.20682437).
