"""
Batched FFT-based Numerical Inverse Laplace Transform.

Implements the Dubner-Abate (1968) / Hsu-Dranoff (1987) method adapted for
GPU-batched evaluation across N^2 independent spectral modes.

For real-valued f(t), the Bromwich integral discretized via FFT (Eq. 3):
    f(t_j) = (e^{a*t_j} / T) * Re[N * IFFT(G)]

where G_k = w_k * F(a + ik*pi/T), w_0 = 1/2, w_k = 1 for k >= 1.

Ported from nilt-cfl/nilt_fft.py and generalized for batched (Nx, Ny, N_brom)
tensor operations on GPU backends.
"""

from __future__ import annotations

import math
import numpy as np


def bromwich_contour(a: float, T: float, N: int, backend):
    """Build Bromwich contour points s = a + j*omega.

    Parameters
    ----------
    a : float
        Real part of contour (Bromwich shift).
    T : float
        Half-period. Frequency spacing delta_omega = pi/T.
    N : int
        Number of contour points.
    backend : Backend
        Compute backend (JAX or PyTorch).

    Returns
    -------
    s : array, shape (N,)
        Complex contour points.
    t : array, shape (N,)
        Corresponding time points t_j = j * 2T/N.
    """
    delta_omega = math.pi / T
    omega = backend.arange(N, dtype=float) * delta_omega
    s = a + 1j * omega
    t = backend.arange(N, dtype=float) * (2 * T / N)
    return s, t


def nilt_inverse(F_spectrum, a: float, T: float, backend):
    """Batched NILT: Bromwich integral via IFFT along last axis.

    This is the core GPU kernel. Input is the transfer function evaluated
    at all Bromwich contour points for all spatial modes simultaneously.

    Parameters
    ----------
    F_spectrum : array, shape (..., N)
        F(s_k) values at Bromwich contour points. Leading dimensions are
        spatial modes (e.g., (Nx, Ny, N) or (N_modes, N)).
        DC component (k=0) should NOT be pre-halved.
    a : float
        Bromwich shift parameter.
    T : float
        Half-period.
    backend : Backend
        Compute backend.

    Returns
    -------
    f : array, shape (..., N)
        Real time-domain values f(t_j) for j = 0, ..., N-1.
    t : array, shape (N,)
        Time points.
    """
    N = F_spectrum.shape[-1]
    delta_t = 2 * T / N

    # Trapezoidal quadrature: half-weight at DC (k=0)
    # Use indexing to avoid mutation (JAX arrays are immutable)
    half_mask = backend.array(
        [0.5] + [1.0] * (N - 1), dtype=float
    ).reshape((1,) * (F_spectrum.ndim - 1) + (N,))
    G = F_spectrum * half_mask

    # f(t_j) = (e^{a*t_j} / T) * Re[N * IFFT(G)]
    z_raw = N * backend.ifft(G, axis=-1)

    # Time grid and exponential correction
    t = backend.arange(N, dtype=float) * delta_t
    correction = backend.exp(a * t) / T
    correction = correction.reshape((1,) * (F_spectrum.ndim - 1) + (N,))

    f = backend.real(z_raw) * correction
    return f, t


def nilt_scalar(F_callable, a: float, T: float, N: int):
    """Scalar NILT matching nilt-cfl API. Pure numpy, for testing.

    Parameters
    ----------
    F_callable : callable
        F(s) -> complex, evaluated pointwise.
    a, T, N : float, float, int
        Bromwich shift, half-period, FFT size.

    Returns
    -------
    f : ndarray, shape (N,)
        Time-domain values.
    t : ndarray, shape (N,)
        Time points.
    z_ifft : ndarray, shape (N,)
        Hermitian IFFT output for eps_im diagnostic.
    """
    delta_omega = np.pi / T
    delta_t = 2 * T / N
    t = np.arange(N) * delta_t

    omega = np.arange(N) * delta_omega
    s = a + 1j * omega
    G = np.array([F_callable(sk) for sk in s], dtype=np.complex128)

    G[0] = G[0] / 2

    z_raw = N * np.fft.ifft(G)
    f = np.exp(a * t) / T * np.real(z_raw)

    # Hermitian diagnostic spectrum
    n_pos = N // 2 + 1
    G_herm = np.zeros(N, dtype=np.complex128)
    G_herm[:n_pos] = G[:n_pos]
    G_herm[n_pos:] = np.conj(G[1:N - n_pos + 1][::-1])
    z_ifft = N * np.fft.ifft(G_herm)

    return f, t, z_ifft


def eps_im(z_ifft) -> float:
    """Imaginary leakage diagnostic (Eq. 12).

    eps_Im = max|Im(f)| / max|Re(f)|

    Should be < 1e-2 for well-tuned parameters.
    """
    real_part = np.abs(np.real(z_ifft))
    imag_part = np.abs(np.imag(z_ifft))

    max_real = np.max(real_part)
    max_imag = np.max(imag_part)

    if max_real < 1e-300:
        return np.inf

    return float(max_imag / max_real)
