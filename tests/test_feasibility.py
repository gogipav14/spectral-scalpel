"""Tests for CFL feasibility checking."""

import numpy as np
import pytest

from scalpel.core.feasibility import tune_params, check_feasibility, TunedParams


class TestTuneParams:
    def test_stable_system(self):
        """Stable system (alpha_c < 0) should always be feasible."""
        params = tune_params(t_end=10.0, alpha_c=-1.0)
        assert params.feasible
        assert params.a > 0
        assert params.margin > 0

    def test_marginally_stable(self):
        """alpha_c = 0 (e.g., diffusion) should be feasible for moderate t_end."""
        params = tune_params(t_end=10.0, alpha_c=0.0)
        assert params.feasible
        assert params.a > 0

    def test_infeasible_large_t(self):
        """Very large t_end with positive alpha_c should be infeasible."""
        params = tune_params(t_end=1e10, alpha_c=1.0)
        assert not params.feasible

    def test_N_power_of_two(self):
        params = tune_params(t_end=10.0, alpha_c=-1.0, N_init=500)
        assert params.N & (params.N - 1) == 0, f"N={params.N} not power of 2"

    def test_spectral_heuristic(self):
        """Providing rho should increase N when needed."""
        params_no_rho = tune_params(t_end=10.0, alpha_c=-1.0, N_init=256)
        params_rho = tune_params(t_end=10.0, alpha_c=-1.0, N_init=256, rho=1000.0)
        assert params_rho.N >= params_no_rho.N


class TestCheckFeasibility:
    def test_stable(self):
        ok, lhs, rhs = check_feasibility(alpha_c=-1.0, t_max=20.0)
        assert ok
        assert lhs < rhs

    def test_unstable(self):
        ok, lhs, rhs = check_feasibility(alpha_c=100.0, t_max=20.0)
        assert not ok
        assert lhs > rhs
