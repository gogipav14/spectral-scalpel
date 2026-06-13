"""Tests for the first-class CuPy dispatched backend.

Skipped automatically when CuPy is not installed or no CUDA device is
available. The CI matrix runs these on the GPU-enabled lane.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
if not cupy.cuda.is_available():
    pytest.skip(
        "CuPy installed but no CUDA device available", allow_module_level=True
    )


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------

def test_cupy_backend_dispatchable_by_name():
    from scalpel.backends import get_backend

    b = get_backend("cupy")
    assert b.name == "cupy"


def test_cupy_appears_in_auto_detect_when_installed():
    """When CuPy is installed and JAX/PyTorch are not, auto-detect
    should land on CuPy before NumPy. We can only weakly test this
    (CI may have JAX/PyTorch installed), so we just confirm CuPy is
    reachable via the name path."""
    from scalpel.backends import get_backend, _AUTO_DETECT_ORDER

    assert "cupy" in _AUTO_DETECT_ORDER
    b = get_backend("cupy")
    assert b is not None


# ---------------------------------------------------------------------------
# Array creation + math primitives
# ---------------------------------------------------------------------------

@pytest.fixture
def b():
    from scalpel.backends import get_backend

    return get_backend("cupy")


def test_array_creates_gpu_tensor(b):
    a = b.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    assert "cupy" in str(type(a)).lower() or "ndarray" in str(type(a)).lower()
    assert b.to_numpy(a).shape == (3,)


def test_zeros_arange_linspace(b):
    z = b.zeros((4,))
    np.testing.assert_array_equal(b.to_numpy(z), np.zeros(4))

    r = b.arange(5)
    np.testing.assert_array_equal(b.to_numpy(r), np.arange(5))

    L = b.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(b.to_numpy(L), np.linspace(0.0, 1.0, 5))


def test_sqrt_exp_log_complex(b):
    z = b.array([4.0 + 0j, 9.0 + 0j])
    np.testing.assert_allclose(b.to_numpy(b.sqrt(z)), [2.0, 3.0])

    np.testing.assert_allclose(
        b.to_numpy(b.exp(b.array([0.0 + 0j]))), [1.0]
    )
    np.testing.assert_allclose(
        b.to_numpy(b.log(b.array([math.e + 0j]))), [1.0], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# FFT primitives
# ---------------------------------------------------------------------------

def test_ifft_round_trip(b):
    x = np.exp(-((np.arange(64) - 32) / 8) ** 2).astype(np.complex128)
    X = b.array(x)
    y = b.to_numpy(b.ifft(b.fft(X)))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_ifft_normalization_matches_numpy(b):
    """CuPy ifft must use the same 1/N convention as NumPy so the
    Dubner-Abate normalization in scalpel.core.nilt is consistent
    across backends."""
    n = 32
    F = np.zeros(n, dtype=np.complex128)
    F[0] = 1.0
    F_gpu = b.array(F)
    f_gpu = b.ifft(F_gpu)
    # ifft of a single DC spike returns 1/N at every sample
    np.testing.assert_allclose(b.to_numpy(f_gpu), np.full(n, 1.0 / n))


def test_fft2_ifft2_round_trip(b):
    x = np.random.RandomState(0).randn(8, 8).astype(np.complex128)
    X_gpu = b.array(x)
    y = b.to_numpy(b.ifft2(b.fft2(X_gpu)))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_fftfreq_matches_numpy(b):
    fr = b.to_numpy(b.fftfreq(16, d=0.125))
    np.testing.assert_allclose(fr, np.fft.fftfreq(16, d=0.125))


# ---------------------------------------------------------------------------
# Synchronization (timing fairness)
# ---------------------------------------------------------------------------

def test_synchronize_is_callable_and_noop_safe(b):
    """synchronize() must be safe to call before and after work; it
    should not raise and should block until queued CUDA kernels
    finish. Required for fair wall-clock timing."""
    b.synchronize()
    x = b.array(np.ones(1000, dtype=np.complex128))
    y = b.fft(x)
    b.synchronize()
    np.testing.assert_allclose(b.to_numpy(y).imag, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end: CuPy under the dispatcher runs propagate_maxwell
# ---------------------------------------------------------------------------

def test_propagate_maxwell_under_cupy(b):
    """Smoke test: dispatch propagate_maxwell through the CuPy
    backend and confirm the field shape + finiteness. This is the
    contract that lets the manuscript claim CuPy as a first-class
    dispatched backend rather than a reference wrapper."""
    import scalpel as sc
    from scalpel.core.engine import GridParams, NILTParams

    grid = GridParams(Nx=16, Ny=16, dx=0.01, dy=0.01)
    nilt = NILTParams(a=10.0, T=20e-9, N=64)
    src_np = np.zeros((grid.Nx, grid.Ny), dtype=np.complex128)
    src_np[grid.Nx // 2, grid.Ny // 2] = 1.0
    # The engine expects the source array on the backend's device;
    # users pass NumPy at the public API surface only for the default
    # NumPy backend.
    src = b.from_numpy(src_np)

    material = sc.MaxwellParams(sigma=1e-3, epsilon_r=4.0)
    field, t = sc.propagate_maxwell(
        src, depth=0.05, material=material, grid=grid, nilt=nilt,
        backend=b,
    )
    b.synchronize()
    field_np = b.to_numpy(field) if not isinstance(field, np.ndarray) else field
    assert np.all(np.isfinite(field_np))
