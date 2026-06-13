"""JAX backend for spectral scalpel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision — essential for NILT accuracy
jax.config.update("jax_enable_x64", True)


class JAXBackend:
    """Thin wrapper around JAX array operations."""

    name = "jax"

    # --- Array creation ---

    def array(self, x, dtype=None):
        return jnp.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=jnp.complex128):
        return jnp.zeros(shape, dtype=dtype)

    def arange(self, *args, **kwargs):
        return jnp.arange(*args, **kwargs)

    def linspace(self, start, stop, num, **kwargs):
        return jnp.linspace(start, stop, num, **kwargs)

    # --- Math ---

    def sqrt(self, x):
        return jnp.sqrt(x)

    def exp(self, x):
        return jnp.exp(x)

    def real(self, x):
        return jnp.real(x)

    def conj(self, x):
        return jnp.conj(x)

    def abs(self, x):
        return jnp.abs(x)

    def log(self, x):
        return jnp.log(x)

    def pi(self):
        return jnp.pi

    # --- FFT ---

    def fft2(self, x):
        return jnp.fft.fft2(x)

    def ifft2(self, x, axes=None):
        if axes is None:
            return jnp.fft.ifft2(x)
        return jnp.fft.ifft2(x, axes=axes)

    def fft(self, x, axis=-1):
        return jnp.fft.fft(x, axis=axis)

    def ifft(self, x, axis=-1):
        return jnp.fft.ifft(x, axis=axis)

    def fftfreq(self, n, d=1.0):
        return jnp.fft.fftfreq(n, d)

    # --- Grid ---

    def meshgrid(self, *arrays, indexing="ij"):
        return jnp.meshgrid(*arrays, indexing=indexing)

    # --- Reduction ---

    def max(self, x):
        return jnp.max(x)

    def mean(self, x):
        return jnp.mean(x)

    def sum(self, x, axis=None):
        return jnp.sum(x, axis=axis)

    # --- JIT ---

    def jit(self, fn):
        return jax.jit(fn)

    # --- Conversion ---

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)
