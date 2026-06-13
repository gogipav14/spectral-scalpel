"""Spectral Scalpel: Conservation-law spectral factorization for PDE systems."""

__version__ = "0.2.0"

# Top-level public API surface. Importing here makes
# ``import scalpel as sc; sc.<x>`` work for the common entry points
# while keeping per-module imports available for power users.

from . import api
from . import chunked
from . import config
from . import diagnose
from . import sweep
from . import validate
from .core import dispersion

from .api import (
    MaxwellParams,
    AcousticParams,
    ChromatographyParams,
    ElastodynamicsParams,
    propagate_maxwell,
    propagate_acoustic,
    propagate_chromatography,
    propagate_chromatography_hankel,
    propagate_elastodynamics,
)
from .chunked import batched_forward_chunked, plan_tiles, ChunkPlan
from .config import RunConfig, save_run_config, load_run_config, diff_configs
from .diagnose import (
    FeasibilityReport,
    AccuracyReport,
    RunReport,
    run_and_report,
    margin_db,
)
from .sweep import SweepResult, parameter_sweep
from .validate import ValidationReport, against_mpmath, against_rk45

__all__ = [
    "__version__",
    # Submodules
    "api",
    "chunked",
    "config",
    "diagnose",
    "dispersion",
    "sweep",
    "validate",
    # Physics + core propagation
    "MaxwellParams",
    "AcousticParams",
    "ChromatographyParams",
    "ElastodynamicsParams",
    "propagate_maxwell",
    "propagate_acoustic",
    "propagate_chromatography",
    "propagate_chromatography_hankel",
    "propagate_elastodynamics",
    # Memory-bounded dispatch
    "ChunkPlan",
    "plan_tiles",
    "batched_forward_chunked",
    # Diagnostics
    "FeasibilityReport",
    "AccuracyReport",
    "RunReport",
    "margin_db",
    "run_and_report",
    # Parameter sweep
    "SweepResult",
    "parameter_sweep",
    # Validation
    "ValidationReport",
    "against_mpmath",
    "against_rk45",
    # Configuration save/load
    "RunConfig",
    "save_run_config",
    "load_run_config",
    "diff_configs",
]
