"""Backend dispatch: NumPy default, plus JAX / PyTorch / CuPy if installed.

The dispatcher recognizes four Array-API-conformant frontends as
first-class dispatched backends:

    name="numpy"  - NumPy (CPU only; default fallback)
    name="torch"  - PyTorch (CPU or CUDA)
    name="jax"    - JAX     (CPU or CUDA)
    name="cupy"   - CuPy    (CUDA only; raises if no GPU)

Each backend exposes a common subset of array operations
(``array``, ``zeros``, ``arange``, ``linspace``, ``sqrt``, ``exp``,
``real``, ``conj``, ``abs``, ``log``, ``fft``, ``ifft``, ``fft2``,
``ifft2``, ``fftfreq``, ``meshgrid``, ``to_numpy``, ``synchronize``,
plus a no-op ``jit``) that is sufficient for the FFT-NILT primitive
and its diagnostics. The interfaces deliberately stay narrow rather
than wrapping the Array API in full; broader compatibility is handled
upstream by each library's NumPy-API surface.

Julia / Julia-CUDA paths exist as reference benchmark wrappers
under ``scripts/`` and are not exposed through this dispatcher.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jax_backend import JAXBackend
    from .torch_backend import TorchBackend
    from .cupy_backend import CuPyBackend

    Backend = JAXBackend | TorchBackend | CuPyBackend


_AUTO_DETECT_ORDER = ("jax", "torch", "cupy", "numpy")


def get_backend(name: str | None = None):
    """Get a compute backend by name or auto-detect.

    Parameters
    ----------
    name : str, optional
        One of "numpy", "torch", "jax", "cupy". If None, auto-detect
        in order: JAX, PyTorch, CuPy, NumPy.
    """
    if name == "jax":
        from .jax_backend import JAXBackend
        return JAXBackend()
    if name == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend()
    if name == "cupy":
        from .cupy_backend import CuPyBackend
        return CuPyBackend()
    if name == "numpy":
        # NumPy backend is supplied by the engine's default code path
        # (no wrapper class needed since engine ops are NumPy-native);
        # we return a sentinel that the engine recognizes.
        return _NumpyBackend()
    if name is not None:
        raise ValueError(
            f"Unknown backend: {name!r}. "
            f"Use one of: {_AUTO_DETECT_ORDER}."
        )

    # Auto-detect in priority order
    for candidate in _AUTO_DETECT_ORDER:
        try:
            return get_backend(candidate)
        except (ImportError, RuntimeError):
            continue
    raise ImportError(
        "No backend available. Install at least one of: "
        "JAX (pip install jax[cuda13]), PyTorch (pip install torch), "
        "or CuPy (pip install cupy-cuda13x); NumPy is the universal "
        "fallback."
    )


class _NumpyBackend:
    """Minimal NumPy-native backend (CPU). Used as the universal
    fallback when no JIT/GPU library is installed."""

    name = "numpy"

    def __init__(self):
        import numpy as _np
        import math as _math
        self._np = _np
        self._math = _math

    def array(self, x, dtype=None):
        return self._np.asarray(x, dtype=dtype or self._np.complex128)

    def zeros(self, shape, dtype=None):
        return self._np.zeros(shape, dtype=dtype or self._np.complex128)

    def arange(self, *args, **kwargs):
        return self._np.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return self._np.linspace(*args, **kwargs)

    def from_numpy(self, x):
        return self._np.asarray(x)

    def sqrt(self, x):
        return self._np.sqrt(x)

    def exp(self, x):
        return self._np.exp(x)

    def real(self, x):
        return x.real

    def conj(self, x):
        return self._np.conj(x)

    def abs(self, x):
        return self._np.abs(x)

    def log(self, x):
        return self._np.log(x)

    def pi(self):
        return self._math.pi

    def fft2(self, x):
        return self._np.fft.fft2(x)

    def ifft2(self, x, axes=None):
        return self._np.fft.ifft2(x, axes=axes) if axes is not None else self._np.fft.ifft2(x)

    def fft(self, x, axis=-1):
        return self._np.fft.fft(x, axis=axis)

    def ifft(self, x, axis=-1):
        return self._np.fft.ifft(x, axis=axis)

    def fftfreq(self, n, d=1.0):
        return self._np.fft.fftfreq(n, d=d).astype(self._np.float64)

    def meshgrid(self, *arrays, indexing="ij"):
        return self._np.meshgrid(*arrays, indexing=indexing)

    def max(self, x):
        return self._np.max(x)

    def mean(self, x):
        return self._np.mean(x)

    def sum(self, x, axis=None):
        return self._np.sum(x, axis=axis)

    def jit(self, fn):
        return fn

    def synchronize(self):
        # No-op for NumPy (synchronous on CPU)
        return

    def to_numpy(self, x):
        return self._np.asarray(x)
