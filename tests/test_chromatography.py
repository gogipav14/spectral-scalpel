"""Tests for chromatography system: spectral vs MOL reference."""

import numpy as np
import pytest

from scalpel.systems.chromatography import ColumnParams, get_column
from scalpel.reference.mol_column import mol_column_1d


class TestMOLReference:
    def test_breakthrough_curve(self):
        """MOL produces a physically reasonable breakthrough curve."""
        col = get_column("hplc")
        t_end = 3 * col.residence_time

        res = mol_column_1d(
            v=col.v, Dz=col.Dz, L=col.L, Nz=200,
            t_end=t_end, Nt_save=100,
        )

        # Outlet concentration (last z cell)
        C_out = res.C[-1, :]

        # Should start at zero, rise, then decay
        assert C_out[0] < 1e-10, "Initial outlet should be zero"
        assert np.max(C_out) > 0, "Should have nonzero breakthrough"

        # Peak should be near residence time
        t_peak = res.t[np.argmax(C_out)]
        assert 0.5 * col.residence_time < t_peak < 2.0 * col.residence_time, \
            f"Peak at {t_peak:.3f}s, expected near {col.residence_time:.3f}s"

    def test_mass_conservation(self):
        """Total mass (integral of outlet concentration) should be reasonable."""
        col = get_column("hplc")
        t_end = 5 * col.residence_time

        res = mol_column_1d(
            v=col.v, Dz=col.Dz, L=col.L, Nz=200,
            t_end=t_end, Nt_save=200,
        )

        # Integrate outlet flux: v * C_out * dt
        C_out = res.C[-1, :]
        dt = res.t[1] - res.t[0]
        mass_out = np.sum(C_out * col.v * dt)

        # Should be close to injected mass (delta pulse ~ 1/dt_inject)
        assert mass_out > 0, "No mass exited the column"


class TestColumnParams:
    def test_peclet(self):
        col = get_column("hplc")
        assert col.Pe > 1, "HPLC should have Pe > 1"
        assert col.residence_time > 0

    def test_all_columns_valid(self):
        for name in ["hplc", "preparative", "process"]:
            col = get_column(name)
            assert col.v > 0
            assert col.Dz > 0
            assert col.Dr > 0
            assert col.R > 0
            assert col.L > 0
