"""Backend auto-detection: JAX if available, else PyTorch."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jax_backend import JAXBackend
    from .torch_backend import TorchBackend

    Backend = JAXBackend | TorchBackend


def get_backend(name: str | None = None) -> "Backend":
    """Get a compute backend by name or auto-detect.

    Parameters
    ----------
    name : str, optional
        "jax" or "torch". If None, tries JAX first, then PyTorch.
    """
    if name == "jax":
        from .jax_backend import JAXBackend
        return JAXBackend()
    elif name == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend()
    elif name is not None:
        raise ValueError(f"Unknown backend: {name!r}. Use 'jax' or 'torch'.")

    # Auto-detect
    try:
        from .jax_backend import JAXBackend
        return JAXBackend()
    except ImportError:
        pass
    try:
        from .torch_backend import TorchBackend
        return TorchBackend()
    except ImportError:
        pass
    raise ImportError(
        "No backend available. Install JAX (pip install jax[cuda13]) "
        "or PyTorch (pip install torch)."
    )
