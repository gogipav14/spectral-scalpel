"""PyTorch backend for spectral scalpel."""

from __future__ import annotations

import math
import torch
import numpy as np


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchBackend:
    """Thin wrapper around PyTorch array operations."""

    name = "torch"

    def __init__(self, device=None):
        self.device = device or _default_device()

    # --- Array creation ---

    def array(self, x, dtype=None):
        if dtype is None:
            dtype = torch.complex128
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype=torch.complex128):
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, device=self.device, **kwargs)

    def linspace(self, start, stop, num, **kwargs):
        return torch.linspace(start, stop, num, device=self.device, **kwargs)

    # --- Math ---

    def sqrt(self, x):
        return torch.sqrt(x)

    def exp(self, x):
        return torch.exp(x)

    def real(self, x):
        return x.real

    def conj(self, x):
        return torch.conj(x)

    def abs(self, x):
        return torch.abs(x)

    def log(self, x):
        return torch.log(x)

    def pi(self):
        return math.pi

    # --- FFT ---

    def fft2(self, x):
        return torch.fft.fft2(x)

    def ifft2(self, x, axes=None):
        if axes is None:
            return torch.fft.ifft2(x)
        return torch.fft.ifft2(x, dim=axes)

    def fft(self, x, axis=-1):
        return torch.fft.fft(x, dim=axis)

    def ifft(self, x, axis=-1):
        return torch.fft.ifft(x, dim=axis)

    def fftfreq(self, n, d=1.0):
        return torch.fft.fftfreq(n, d, device=self.device, dtype=torch.float64)

    # --- Grid ---

    def meshgrid(self, *arrays, indexing="ij"):
        return torch.meshgrid(*arrays, indexing=indexing)

    # --- Reduction ---

    def max(self, x):
        return torch.max(x)

    def mean(self, x):
        return torch.mean(x)

    def sum(self, x, axis=None):
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis)

    # --- JIT ---

    def jit(self, fn):
        # PyTorch eager mode — no-op
        return fn

    # --- Conversion ---

    def to_numpy(self, x) -> np.ndarray:
        return x.detach().cpu().numpy()
