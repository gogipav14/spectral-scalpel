"""Tests for the quasi-discrete Hankel transform."""

import numpy as np
import pytest
from scipy.special import j0

from scalpel.core.hankel import HankelTransform


class TestHankelRoundtrip:
    def test_gaussian_roundtrip(self):
        """Hankel transform of a Gaussian is a Gaussian. Roundtrip should recover."""
        R = 0.01  # 1 cm
        N = 64
        ht = HankelTransform(R, N)

        # Gaussian on radial grid
        sigma = 0.003  # 3 mm
        f = np.exp(-ht.r**2 / (2 * sigma**2))

        # Forward + inverse
        F = ht.forward(f)
        f_recovered = ht.inverse(F)

        np.testing.assert_allclose(f_recovered, f, rtol=1e-3,
                                   err_msg="Hankel roundtrip failed for Gaussian")

    def test_j0_eigenfunction(self):
        """J_0(alpha*r) is an eigenfunction of the Hankel transform."""
        R = 0.05
        N = 32
        ht = HankelTransform(R, N)

        # Use the first Bessel zero as the eigenvalue
        alpha = ht.kr[0]
        f = j0(alpha * ht.r)

        F = ht.forward(f)

        # F should be peaked at kr = alpha (the first mode)
        peak_idx = np.argmax(np.abs(F))
        assert peak_idx == 0, f"Peak at index {peak_idx}, expected 0"

    def test_different_sizes(self):
        """Roundtrip works for various N."""
        R = 0.01
        for N in [8, 16, 32, 64]:
            ht = HankelTransform(R, N)
            f = np.exp(-ht.r**2 / (2 * 0.003**2))
            f_rt = ht.inverse(ht.forward(f))
            rel_err = np.sqrt(np.mean((f_rt - f)**2)) / np.sqrt(np.mean(f**2))
            assert rel_err < 0.05, f"N={N}: roundtrip rel error {rel_err:.3f}"


class TestHankelProperties:
    def test_sample_points_in_domain(self):
        R = 0.01
        N = 32
        ht = HankelTransform(R, N)
        assert np.all(ht.r > 0)
        assert np.all(ht.r < R)
        assert np.all(ht.kr > 0)

    def test_transform_matrix_square(self):
        ht = HankelTransform(0.01, 16)
        assert ht.T.shape == (16, 16)

    def test_forward_nonzero(self):
        """Forward transform produces nonzero output."""
        R = 0.01
        N = 32
        ht = HankelTransform(R, N)
        f = np.exp(-ht.r**2 / (2 * 0.003**2))
        F = ht.forward(f)
        assert np.max(np.abs(F)) > 0, "Hankel transform output is all zeros"
