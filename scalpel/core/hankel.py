"""
Quasi-discrete Hankel transform (QDHT) for cylindrical geometry.

Based on Guizar-Sicairos & Gutierrez-Vega (2004):
    "Computation of quasi-discrete Hankel transforms of integer order
     for propagating optical wave fields"
    J. Opt. Soc. Am. A, 21(1), 53-58.

The QDHT maps a function f(r) sampled at Bessel zeros to its Hankel
transform F(kr) sampled at scaled Bessel zeros. For axisymmetric
problems (order p=0), the Bessel function is J_0.

The transform is its own inverse (up to normalization), analogous to
how the DFT and IDFT are related.

This replaces the 2D FFT in (x,y) for cylindrical geometry:
  - Cartesian: FFT in x, FFT in y -> (kx, ky) modes
  - Cylindrical: Hankel in r -> kr modes (axisymmetric)
"""

from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, j0, j1


class HankelTransform:
    """Quasi-discrete Hankel transform of order 0.

    Parameters
    ----------
    R : float
        Radial domain extent [m]. f(r) is sampled on [0, R].
    N : int
        Number of radial modes (Bessel zeros to use).

    Attributes
    ----------
    r : ndarray, shape (N,)
        Radial sample points r_n = alpha_n * R / alpha_{N+1}.
    kr : ndarray, shape (N,)
        Radial wavenumber points kr_n = alpha_n / R.
    T : ndarray, shape (N, N)
        Transform matrix. F = T @ f, f = T @ F (self-inverse).
    """

    def __init__(self, R: float, N: int):
        self.R = R
        self.N = N
        self.order = 0

        # Bessel zeros: alpha_1 ... alpha_{N+1} of J_0
        alphas = jn_zeros(0, N + 1)
        alpha_N1 = alphas[N]       # alpha_{N+1}, the boundary zero
        self.alphas = alphas[:N]   # alpha_1 ... alpha_N

        # Sample points (Eq. 9 in Guizar-Sicairos 2004)
        self.r = self.alphas * R / alpha_N1      # radial grid
        self.kr = self.alphas / R                 # wavenumber grid

        # Maximum wavenumber and radius
        self.kr_max = alpha_N1 / R
        self.r_max = R

        # Transform matrix (Eq. 13)
        # T_{mn} = (2/alpha_{N+1}) * J_0(alpha_m * alpha_n / alpha_{N+1})
        #          / |J_1(alpha_m)| / |J_1(alpha_n)|
        j1_vals = np.abs(j1(self.alphas))

        # Build matrix
        T = np.zeros((N, N))
        for m in range(N):
            for n in range(N):
                arg = self.alphas[m] * self.alphas[n] / alpha_N1
                T[m, n] = (2.0 / alpha_N1) * j0(arg) / (j1_vals[m] * j1_vals[n])

        self.T = T

        # Scaling vectors for forward/inverse (Eq. 14-15)
        # f_scaled = f * |J_1(alpha)|, F_scaled = F * |J_1(alpha)| * R^2 / alpha_{N+1}
        self._j1_abs = j1_vals
        self._alpha_N1 = alpha_N1

    def forward(self, f_r: np.ndarray) -> np.ndarray:
        """Hankel transform: f(r) -> F(kr).

        Parameters
        ----------
        f_r : ndarray, shape (N,) or (N, ...)
            Function values at radial sample points self.r.

        Returns
        -------
        F_kr : ndarray, shape (N,) or (N, ...)
            Transform values at wavenumber points self.kr.
        """
        # Scale input
        if f_r.ndim == 1:
            f_scaled = f_r / self._j1_abs
            F_scaled = self.T @ f_scaled
            return F_scaled * self._j1_abs * self.R**2 / self._alpha_N1
        else:
            # Batched: f_r shape (N, ...), transform along axis 0
            f_scaled = f_r / self._j1_abs.reshape(-1, *([1]*(f_r.ndim-1)))
            F_scaled = np.tensordot(self.T, f_scaled, axes=([1], [0]))
            return F_scaled * self._j1_abs.reshape(-1, *([1]*(f_r.ndim-1))) \
                   * self.R**2 / self._alpha_N1

    def inverse(self, F_kr: np.ndarray) -> np.ndarray:
        """Inverse Hankel transform: F(kr) -> f(r).

        For order-0 QDHT, the transform is self-inverse up to scaling.

        Parameters
        ----------
        F_kr : ndarray, shape (N,) or (N, ...)
            Transform values at wavenumber points self.kr.

        Returns
        -------
        f_r : ndarray, shape (N,) or (N, ...)
            Function values at radial sample points self.r.
        """
        # Scale input (inverse scaling)
        if F_kr.ndim == 1:
            F_scaled = F_kr / (self._j1_abs * self.R**2 / self._alpha_N1)
            f_scaled = self.T @ F_scaled
            return f_scaled * self._j1_abs
        else:
            F_scaled = F_kr / (self._j1_abs * self.R**2 / self._alpha_N1).reshape(
                -1, *([1]*(F_kr.ndim-1)))
            f_scaled = np.tensordot(self.T, F_scaled, axes=([1], [0]))
            return f_scaled * self._j1_abs.reshape(-1, *([1]*(F_kr.ndim-1)))
