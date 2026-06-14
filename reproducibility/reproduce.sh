#!/usr/bin/env bash
# Reproduce the quantitative claims of the scalpel TOMS manuscript.
#
# Source of truth: this script + the repository at
#   https://github.com/gogipav14/spectral-scalpel
# CodeOcean / RCR re-execution path: clone the repo and run
#   bash reproducibility/reproduce.sh
# from the repository root.
#
# Default mode is CPU-only verification: regenerate what the CPU stack
# can produce, diff against the archived CSVs in reproducibility/data/
# at the tolerance specified by $TOLERANCE, exit 0 if clean. Pass
# ENABLE_GPU=1 to additionally re-run the multi-backend wall-clock
# benchmark (PyTorch CUDA, JAX CUDA, CuPy CUDA), which is the full
# Replicated Computational Results (RCR) target for a GPU-equipped
# CodeOcean capsule.
#
# Usage:    bash reproducibility/reproduce.sh
# Env:      ENABLE_GPU=1     enable PyTorch/JAX/CuPy CUDA paths
#                              (default: CPU only)
#           ENABLE_JULIA=1   enable Julia / Julia-CUDA reference path
#                              (default: skipped if PyJulia not installed)
#           TOLERANCE=0.05   relative tolerance for CSV diff checks
#                              (default: 5%)
#
# Outputs:
#   reproducibility/figures/*.png     regenerated figures
#   reproducibility/data/diff_report.txt   CSV diff vs archive
#
# Exit code:
#   0  all claims reproduce within tolerance
#   1  diff report shows out-of-tolerance rows; inspect diff_report.txt
#   2  setup failure (missing dependency, etc.)

set -uo pipefail

TOL="${TOLERANCE:-0.05}"
ENABLE_GPU="${ENABLE_GPU:-0}"
ENABLE_JULIA="${ENABLE_JULIA:-0}"

REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${REPRO_ROOT}/.." && pwd)"
DATA="${REPRO_ROOT}/data"
SCRIPTS="${REPRO_ROOT}/scripts"
BASELINES="${REPRO_ROOT}/baselines"
FIG="${REPRO_ROOT}/figures"
DIFF_OUT="${DATA}/diff_report.txt"
mkdir -p "${FIG}"

log()  { printf '[reproduce] %s\n' "$*"; }
warn() { printf '[reproduce] WARN: %s\n' "$*" >&2; }
fail() { printf '[reproduce] FAIL: %s\n' "$*" >&2; exit "${2:-2}"; }

# JAX VRAM hygiene (avoids preallocating the entire GPU when multiple
# CUDA-aware backends share a single consumer GPU).
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"

# --- 0. Sanity check + auto-install of scalpel from the mounted tree --------
# On CodeOcean and other build-from-repo platforms the Dockerfile installs
# the dependency stack at image-build time, but the repo (containing
# scalpel/) is only mounted at runtime under /code. This block does an
# editable install on first run so reproduce.sh works whether scalpel is
# already importable (local dev) or only the source tree is present
# (CodeOcean, fresh Docker container).
log "Python: $(command -v python3 || fail 'python3 not on PATH')"
if ! python3 -c 'import scalpel' 2>/dev/null; then
  log "scalpel not yet importable; installing from ${ROOT} in editable mode..."
  python3 -m pip install --no-cache-dir -e "${ROOT}" \
    || fail "pip install -e ${ROOT} failed; check pyproject.toml and base image"
fi
python3 -c 'import scalpel; print("scalpel", scalpel.__version__)' \
  || fail "scalpel still not importable after install; check pyproject.toml"

# --- 1. Class-dependent feasibility (Figure 2) -------------------------------
log "Class-dependent feasibility sweeps (Figure 2)..."
python3 "${SCRIPTS}/recoverability_class_comparison.py" \
  || warn "feasibility sweep step failed (continuing)"

# --- 2. Multi-backend wall-clock (Figure 3) ----------------------------------
log "Multi-backend wall-clock benchmark (Figure 3): re-render from archived CSV..."
python3 "${SCRIPTS}/render_benchmark_fig_v2.py" \
  || warn "fig3 render step failed (continuing)"

if [[ "${ENABLE_GPU}" == "1" ]]; then
  log "ENABLE_GPU=1: re-running the full multi-backend benchmark"
  log "  (NumPy / PyTorch CPU+CUDA / JAX CPU+CUDA / CuPy CUDA; depends on installed extras)"
  python3 "${SCRIPTS}/benchmark_fractional_heat_3d_all_backends.py" \
    || warn "multi-backend benchmark failed (continuing)"
  log "ENABLE_GPU=1: 15-rep median+IQR campaign (Table 2)"
  python3 "${SCRIPTS}/benchmark_repeated.py" \
    || warn "15-rep campaign failed (continuing)"
  log "ENABLE_GPU=1: JAX CPU subprocess sidecar (Section 5.1)"
  JAX_PLATFORMS=cpu python3 "${SCRIPTS}/benchmark_fractional_heat_3d_jax_cpu_sidecar.py" \
    || warn "JAX CPU sidecar failed (continuing)"
fi

# --- 3. mpmath ground truth (anchor, Section 6.4) ----------------------------
log "mpmath 50-digit ground-truth (accuracy anchor)..."
python3 "${SCRIPTS}/fractional_mpmath_reference.py" \
  || warn "mpmath reference step failed (continuing)"

# --- 4. Spectral convergence (Figure 1) --------------------------------------
log "Spectral-convergence check (Figure 1)..."
python3 "${SCRIPTS}/verify_convergence.py" \
  || warn "convergence check failed (continuing)"

# --- 5. Figure 4 multipanel (transform-domain head-to-head) ------------------
log "Transform-domain head-to-head figure (Figure 4)..."
python3 "${SCRIPTS}/make_fig4_multipanel.py" \
  || warn "fig4 multipanel render failed (continuing)"

# --- 6. Diff regenerated CSVs against archive --------------------------------
log "Diffing regenerated CSVs against archive (tolerance=${TOL})..."
python3 "${SCRIPTS}/csv_diff.py" \
  --archive "${DATA}" \
  --regen   "${DATA}" \
  --tolerance "${TOL}" \
  --skip-missing \
  -v \
  > "${DIFF_OUT}"
DIFF_EXIT=$?

# --- 7. Summary --------------------------------------------------------------
echo
cat "${DIFF_OUT}"
echo

if [[ "${DIFF_EXIT}" == "0" ]]; then
  log "All quantitative claims reproduce within ${TOL} tolerance."
  log "Figures:    ${FIG}/"
  log "Diff log:   ${DIFF_OUT}"
  exit 0
else
  log "Some rows exceed tolerance; see ${DIFF_OUT}"
  exit 1
fi
