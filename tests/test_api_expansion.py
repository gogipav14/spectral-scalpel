"""Tests for the expanded scalpel public API: chunked, diagnose, sweep,
validate, config.

These tests are intentionally lightweight — they exercise the API
shapes and contracts without depending on heavy backends or external
references. Heavy validation tests live alongside the underlying
modules (e.g., tests/test_feasibility.py for the auto-tuner math).
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# chunked dispatch
# ---------------------------------------------------------------------------

def test_chunked_plan_single_tile_when_fits():
    from scalpel.chunked import plan_tiles

    plan = plan_tiles(n_x=64, n_y=64, n_nilt=512, vram_budget_bytes=int(1e9))
    assert plan.n_tiles == 1
    assert plan.tile_shape == (64, 64)


def test_chunked_plan_tiles_when_oversize():
    from scalpel.chunked import plan_tiles

    plan = plan_tiles(
        n_x=1024, n_y=1024, n_nilt=2048,
        vram_budget_bytes=int(2e8),    # tight budget
        dtype_bytes=16,
    )
    assert plan.n_tiles > 1
    assert plan.tile_shape[0] <= 1024
    assert plan.tile_shape[1] <= 1024
    # Each tile fits within the safety-deflated budget
    assert plan.bytes_per_tile <= int(2e8) * 0.7 + 1


def test_chunked_plan_power_of_two_side():
    from scalpel.chunked import plan_tiles

    plan = plan_tiles(n_x=512, n_y=512, n_nilt=4096,
                      vram_budget_bytes=int(1e8))
    # side rounded down to a power of two
    side = plan.tile_shape[0]
    assert side & (side - 1) == 0


def test_chunked_plan_repr_is_informative():
    from scalpel.chunked import plan_tiles

    plan = plan_tiles(n_x=256, n_y=256, n_nilt=1024,
                      vram_budget_bytes=int(5e8))
    s = repr(plan)
    assert "ChunkPlan" in s
    assert "tile=" in s
    assert "n_tiles=" in s


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

def test_margin_db_positive_when_below_theory():
    from scalpel.diagnose import margin_db

    assert margin_db(10.0, 100.0) > 0           # 20 dB headroom
    assert math.isclose(margin_db(10.0, 100.0), 20.0, rel_tol=1e-9)


def test_margin_db_negative_when_above_theory():
    from scalpel.diagnose import margin_db

    assert margin_db(100.0, 10.0) < 0


def test_margin_db_nan_on_zero():
    from scalpel.diagnose import margin_db

    assert math.isnan(margin_db(0.0, 10.0))
    assert math.isnan(margin_db(10.0, 0.0))


def test_feasibility_report_round_trips_as_dict():
    from scalpel.diagnose import build_feasibility_report

    r = build_feasibility_report(
        dispersion_class="telegrapher",
        a=1e7, T=20e-9, n_nilt=2048,
        kpmax_theory=500.0, kpmax_grid=200.0,
        precision="float64",
        bound_label="Theorem 4.1",
        bound_inputs={"sigma": 1e-3, "epsilon_r": 4.0},
    )
    d = r.as_dict()
    assert d["dispersion_class"] == "telegrapher"
    assert d["bromwich_shift"] == 1e7
    assert d["precision"] == "float64"
    assert d["bound_label"] == "Theorem 4.1"
    assert d["margin_db"] > 0          # grid below theory cutoff
    s = r.summary()
    assert "telegrapher" in s and "Theorem 4.1" in s


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def _fake_forward(depth, **kwargs):
    """Stand-in for a real propagate_* callable; returns (field, t)."""
    import numpy as np
    nx = kwargs.get("nx", 8)
    ny = kwargs.get("ny", 8)
    nt = kwargs.get("nt", 4)
    return np.full((nt, nx, ny), depth, dtype=float), np.arange(nt)


def test_parameter_sweep_columns_match():
    from scalpel.sweep import parameter_sweep

    result = parameter_sweep(
        forward_callable=_fake_forward,
        varying={"depth": [0.1, 0.5, 1.0]},
        fixed={"nx": 4, "ny": 4, "nt": 2},
        reduce="rms",
    )
    assert result.columns == ["depth", "rms"]
    assert len(result) == 3
    # RMS of a constant field equals the constant (positive); each row
    # has depth in column 0 and rms in column 1
    for row in result.rows:
        assert math.isclose(row[1], row[0], rel_tol=1e-9)


def test_parameter_sweep_cartesian_product():
    from scalpel.sweep import parameter_sweep

    result = parameter_sweep(
        forward_callable=_fake_forward,
        varying={"depth": [0.1, 0.2], "extra": [10, 20, 30]},
        fixed={"nx": 4, "ny": 4, "nt": 2},
        reduce="rms_and_peak",
    )
    assert result.columns == ["depth", "extra", "rms", "peak"]
    assert len(result) == 2 * 3
    for row in result.rows:
        assert row[2] >= 0     # rms
        assert row[3] >= 0     # peak


def test_parameter_sweep_csv_round_trip():
    from scalpel.sweep import parameter_sweep
    import csv

    result = parameter_sweep(
        forward_callable=_fake_forward,
        varying={"depth": [0.25, 0.75]},
        fixed={"nx": 4, "ny": 4, "nt": 2},
        reduce="centerline",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        result.to_csv(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["depth", "centerline"]
        assert len(rows) == 3            # header + 2 rows
    finally:
        os.unlink(path)


def test_parameter_sweep_rejects_unknown_reduction():
    from scalpel.sweep import parameter_sweep

    with pytest.raises(ValueError):
        parameter_sweep(
            forward_callable=_fake_forward,
            varying={"depth": [0.1]},
            fixed={},
            reduce="bogus",
        )


# ---------------------------------------------------------------------------
# config save/load
# ---------------------------------------------------------------------------

def test_config_round_trip():
    from scalpel.config import save_run_config, load_run_config

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_run_config(
            path,
            dispersion="maxwell",
            dispersion_params={"sigma": 1e-3, "epsilon_r": 4.0},
            auto_tuner_choices={"a": 1.0, "T": 20e-9, "N": 2048},
            grid={"Nx": 128, "Ny": 128, "dx": 0.1},
            precision="float64",
            backend="jax_gpu",
        )
        cfg = load_run_config(path)
        assert cfg.dispersion == "maxwell"
        assert cfg.dispersion_params["sigma"] == 1e-3
        assert cfg.auto_tuner_choices["N"] == 2048
        assert cfg.grid["Nx"] == 128
        assert cfg.precision == "float64"
        assert cfg.backend == "jax_gpu"
    finally:
        os.unlink(path)


def test_config_schema_mismatch_raises():
    from scalpel.config import load_run_config

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
        json.dump({
            "schema_version": "9.9",
            "dispersion": "x",
            "dispersion_params": {},
            "auto_tuner_choices": {},
            "grid": {},
            "precision": "float64",
            "backend": "numpy",
        }, f)
    try:
        with pytest.raises(ValueError, match="schema"):
            load_run_config(path)
    finally:
        os.unlink(path)


def test_diff_configs_reports_changed_fields():
    from scalpel.config import RunConfig, diff_configs

    a = RunConfig(
        dispersion="maxwell",
        dispersion_params={"sigma": 1e-3, "epsilon_r": 4.0},
        auto_tuner_choices={"a": 1.0, "T": 20e-9, "N": 2048},
        grid={"Nx": 128, "Ny": 128, "dx": 0.1},
        precision="float64",
        backend="jax_gpu",
    )
    b = RunConfig(
        dispersion="maxwell",
        dispersion_params={"sigma": 2e-3, "epsilon_r": 4.0},
        auto_tuner_choices={"a": 1.0, "T": 20e-9, "N": 4096},
        grid={"Nx": 128, "Ny": 128, "dx": 0.1},
        precision="float64",
        backend="jax_gpu",
    )
    d = diff_configs(a, b)
    assert "dispersion_params" in d and "sigma" in d["dispersion_params"]
    assert "auto_tuner_choices" in d and "N" in d["auto_tuner_choices"]


# ---------------------------------------------------------------------------
# new physics propagators surface + parameter dataclasses
# ---------------------------------------------------------------------------

def test_chromatography_params_constructable():
    from scalpel import ChromatographyParams

    p = ChromatographyParams(v=1e-3, Dz=1e-7, Dr=1e-9)
    assert p.v == 1e-3 and p.Dz == 1e-7 and p.Dr == 1e-9


def test_elastodynamics_params_constructable():
    from scalpel import ElastodynamicsParams

    p = ElastodynamicsParams(c_p=5000.0, c_s=2800.0, rho=2400.0)
    assert p.c_p == 5000.0
    assert p.eta_p == 0.0     # Kelvin-Voigt off by default


def test_propagate_chromatography_in_public_api():
    import scalpel as sc

    assert hasattr(sc, "propagate_chromatography")
    assert "propagate_chromatography" in sc.__all__


def test_propagate_elastodynamics_in_public_api():
    import scalpel as sc

    assert hasattr(sc, "propagate_elastodynamics")
    assert "propagate_elastodynamics" in sc.__all__


def test_propagate_elastodynamics_rejects_bad_wave():
    import scalpel as sc
    from scalpel.core.engine import GridParams, NILTParams
    import numpy as np

    src = np.zeros((4, 4), dtype=complex)
    grid = GridParams(Nx=4, Ny=4, dx=0.1, dy=0.1)
    nilt = NILTParams(a=1.0, T=1e-3, N=8)
    material = sc.ElastodynamicsParams(c_p=5000.0, c_s=2800.0, rho=2400.0)
    with pytest.raises(ValueError, match="'p' or 's'"):
        sc.propagate_elastodynamics(src, depth=0.1, material=material,
                                    grid=grid, nilt=nilt, wave="z")


# ---------------------------------------------------------------------------
# reference wrapper (talbot_mpmath.centerline_reference)
# ---------------------------------------------------------------------------

def test_centerline_reference_loads_fractional_csv():
    from scalpel.reference.talbot_mpmath import centerline_reference

    # The repository ships a precomputed fractional reference CSV; this
    # function should load it without invoking mpmath at all.
    u = centerline_reference("fractional")
    assert u.ndim == 1
    assert u.size > 0


def test_centerline_reference_rejects_unknown_physics():
    from scalpel.reference.talbot_mpmath import centerline_reference

    with pytest.raises(NotImplementedError):
        centerline_reference("unknown_physics")


# ---------------------------------------------------------------------------
# validate (smoke; full path requires mpmath)
# ---------------------------------------------------------------------------

def test_validation_report_shape():
    from scalpel.validate import ValidationReport

    r = ValidationReport(
        reference="mpmath_50digit_talbot",
        physics="fractional",
        parameters={"alpha": 0.7, "depth": 0.1},
        relative_l2_error=8e-4,
        relative_linf_error=2e-3,
        tolerance=1e-3,
        passed=True,
    )
    d = r.as_dict()
    assert d["passed"] is True
    assert d["relative_l2_error"] == 8e-4
    assert "mpmath" in r.summary()


def test_validation_report_failed_when_above_tolerance():
    from scalpel.validate import ValidationReport

    r = ValidationReport(
        reference="mpmath_50digit_talbot",
        physics="fractional",
        parameters={},
        relative_l2_error=1e-2,
        relative_linf_error=1e-2,
        tolerance=1e-3,
        passed=False,
    )
    assert "FAILED" in r.summary()
    assert not r.passed
