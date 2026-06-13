"""CuPy backend for spectral scalpel.

CuPy is a NumPy-compatible array library that executes on NVIDIA GPUs
through CUDA. This backend mirrors the PyTorch backend's interface so
the dispatcher in ``scalpel.backends`` can route the FFT-NILT primitive
through it identically.

CuPy is GPU-only by construction (no CPU fallback). For CPU execution,
use the NumPy backend instead; the two have near-identical
NumPy-compatible APIs so user code does not change.

Key implementation choices:
    * ``cupy.fft.ifft`` follows the unnormalized ``1/N`` convention
      compatible with the Dubner-Abate pseudocode in the manuscript
      (Algorithm 1).
    * ``synchronize()`` calls ``cupy.cuda.Stream.null.synchronize()``
      to make wall-clock timing fair (CUDA kernel launches are
      asynchronous by default).
    * ``to_numpy()`` calls ``cp.asnumpy()`` which performs the
      device-to-host transfer.
"""

from __future__ import annotations

import math

import numpy as np


def _import_cupy():
    try:
        import cupy as cp
        return cp
    except ImportError as e:
        raise ImportError(
            "CuPy is required for the CuPy backend. "
            "Install with `pip install cupy-cuda13x` matched to your CUDA "
            "version (or cupy-cuda12x, etc.)."
        ) from e


class CuPyBackend:
    """Thin wrapper around CuPy array operations, GPU-only."""

    name = "cupy"

    def __init__(self):
        self._cp = _import_cupy()
        if not self._cp.cuda.is_available():
            raise RuntimeError(
                "CuPy is installed but no CUDA device is available. "
                "The CuPy backend is GPU-only; use the NumPy backend "
                "for CPU execution."
            )
        # Always device 0 by default; users wanting multi-GPU can set
        # CUDA_VISIBLE_DEVICES or use cupy.cuda.Device explicitly.
        self.device = self._cp.cuda.Device(0)

    # --- Array creation ---

    def array(self, x, dtype=None):
        if dtype is None:
            dtype = self._cp.complex128
        return self._cp.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self._cp.complex128
        return self._cp.zeros(shape, dtype=dtype)

    def arange(self, *args, **kwargs):
        return self._cp.arange(*args, **kwargs)

    def linspace(self, start, stop, num, **kwargs):
        return self._cp.linspace(start, stop, num, **kwargs)

    def from_numpy(self, x):
        return self._cp.asarray(x)

    # --- Math ---

    def sqrt(self, x):
        return self._cp.sqrt(x)

    def exp(self, x):
        return self._cp.exp(x)

    def real(self, x):
        return x.real

    def conj(self, x):
        return self._cp.conj(x)

    def abs(self, x):
        return self._cp.abs(x)

    def log(self, x):
        return self._cp.log(x)

    def pi(self):
        return math.pi

    # --- FFT (cuFFT-backed) ---

    def fft2(self, x):
        return self._cp.fft.fft2(x)

    def ifft2(self, x, axes=None):
        if axes is None:
            return self._cp.fft.ifft2(x)
        return self._cp.fft.ifft2(x, axes=axes)

    def fft(self, x, axis=-1):
        return self._cp.fft.fft(x, axis=axis)

    def ifft(self, x, axis=-1):
        return self._cp.fft.ifft(x, axis=axis)

    def fftfreq(self, n, d=1.0):
        return self._cp.fft.fftfreq(n, d=d).astype(self._cp.float64)

    # --- Grid ---

    def meshgrid(self, *arrays, indexing="ij"):
        return self._cp.meshgrid(*arrays, indexing=indexing)

    # --- Reduction ---

    def max(self, x):
        return self._cp.max(x)

    def mean(self, x):
        return self._cp.mean(x)

    def sum(self, x, axis=None):
        if axis is None:
            return self._cp.sum(x)
        return self._cp.sum(x, axis=axis)

    # --- JIT ---

    def jit(self, fn):
        # CuPy is eager-mode like PyTorch; no JIT transformation needed.
        # Users who want kernel fusion should use cp.fuse() decorator
        # directly on their function rather than via the dispatcher.
        return fn

    # --- Synchronization (for fair timing) ---

    def synchronize(self):
        """Block until all queued CUDA work on the default stream completes.

        Call before reading the clock on either side of a timing window.
        cuFFT kernels are asynchronous; without this call, wall-clock
        measurements undercount actual GPU work.
        """
        self._cp.cuda.Stream.null.synchronize()

    # --- Conversion ---

    def to_numpy(self, x) -> np.ndarray:
        return self._cp.asnumpy(x)
