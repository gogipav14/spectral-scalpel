"""
Spectral factorization engine: 2D FFT + 1D NILT forward model.

The entire solver is: batched FFT -> batched pointwise complex exponential
-> batched IFFT. Every step is a native GPU primitive. No time-stepping loop,
no spatial finite differences, no CFL stability condition on the spatial grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..backends import get_backend
from .nilt import bromwich_contour, nilt_inverse


def _require_complex128(backend, source_xy) -> None:
    """Warn loudly if the backend is not in 64-bit mode.

    The NILT feasibility bound L = ln(DBL_MAX) ~ 709.8 and the ~1e-10 accuracy
    floor reported in the paper both assume double precision. In 32-bit mode
    the dynamic range collapses to L ~ 88 and the spectral convergence plateau
    moves to ~1e-4, producing silently wrong results. JAX in particular
    defaults to float32 unless jax_enable_x64 is set; this check is a cheap
    guard against that common footgun.
    """
    dtype_str = str(getattr(source_xy, "dtype", ""))
    if "complex128" in dtype_str or "float64" in dtype_str:
        return
    if "complex64" in dtype_str or "float32" in dtype_str:
        raise TypeError(
            "SpectralEngine requires 64-bit precision; got "
            f"{dtype_str}. For JAX, set jax.config.update('jax_enable_x64', True) "
            "before creating arrays. The NILT feasibility bound and ~1e-10 "
            "convergence floor both assume float64."
        )


@dataclass
class GridParams:
    """Spatial grid parameters."""
    Nx: int         # Grid points in x
    Ny: int         # Grid points in y
    dx: float       # Grid spacing in x [m]
    dy: float       # Grid spacing in y [m]

    @property
    def Lx(self):
        return self.Nx * self.dx

    @property
    def Ly(self):
        return self.Ny * self.dy


@dataclass
class NILTParams:
    """NILT parameters for the Bromwich contour."""
    a: float        # Bromwich shift
    T: float        # Half-period
    N: int          # Number of Bromwich points (FFT size)

    @property
    def delta_t(self):
        return 2 * self.T / self.N

    @property
    def t_max(self):
        return 2 * self.T


class SpectralEngine:
    """Conservation-law spectral factorization engine.

    Propagates a 2D source field through distance `depth` in a homogeneous
    medium characterized by a dispersion relation.

    The full pipeline:
        1. 2D FFT of source -> S(kx, ky)
        2. Build wavenumber grid (kx, ky)
        3. Evaluate gamma_z(s, kx, ky) at Bromwich contour points
        4. Transfer function H = exp(-gamma_z * depth)
        5. F_spectrum = S * H
        6. NILT inverse (batched IFFT along Bromwich axis)
        7. Inverse 2D FFT at each time step
    """

    def __init__(self, dispersion_fn: Callable, backend=None):
        """
        Parameters
        ----------
        dispersion_fn : callable
            Signature: (s, KX, KY, backend) -> gamma_z
            where s has shape (..., N_brom), KX/KY have shape (Nx, Ny, ...),
            and gamma_z has shape (Nx, Ny, N_brom).
        backend : Backend, optional
            Compute backend. Auto-detected if None.
        """
        self.dispersion = dispersion_fn
        self.backend = backend or get_backend()

    def forward(self, source_xy, depth: float, grid: GridParams,
                nilt: NILTParams):
        """Propagate source field through homogeneous medium.

        Parameters
        ----------
        source_xy : array, shape (Nx, Ny)
            2D source field (spatial domain).
        depth : float
            Propagation distance [m].
        grid : GridParams
            Spatial grid parameters.
        nilt : NILTParams
            NILT parameters.

        Returns
        -------
        field : array, shape (Nx, Ny, Nt)
            Real-valued field at each time step.
        t : array, shape (Nt,)
            Time points.
        """
        b = self.backend
        _require_complex128(b, source_xy)

        # 1. 2D FFT of source
        S = b.fft2(source_xy)  # (Nx, Ny)

        # 2. Wavenumber grid
        kx = b.fftfreq(grid.Nx, grid.dx) * 2 * math.pi  # (Nx,)
        ky = b.fftfreq(grid.Ny, grid.dy) * 2 * math.pi  # (Ny,)
        KX, KY = b.meshgrid(kx, ky)  # (Nx, Ny) each

        # 3. Bromwich contour
        s_contour, t = bromwich_contour(nilt.a, nilt.T, nilt.N, b)  # (N_brom,)

        # 4. Dispersion relation: gamma_z for all modes at all contour points
        # Reshape for broadcasting: (Nx, Ny, 1) x (1, 1, N_brom) -> (Nx, Ny, N_brom)
        gamma_z = self.dispersion(
            s_contour.reshape(1, 1, -1),
            KX[:, :, None],
            KY[:, :, None],
            b,
        )

        # 5. Transfer function
        H = b.exp(-gamma_z * depth)  # (Nx, Ny, N_brom)

        # 6. Apply source spectrum
        F_spectrum = S[:, :, None] * H  # (Nx, Ny, N_brom)

        # 7. NILT inverse (batched IFFT along Bromwich axis)
        field_kt, t = nilt_inverse(F_spectrum, nilt.a, nilt.T, b)  # (Nx, Ny, Nt)

        # 8. Inverse 2D FFT at each time step
        field_xt = b.real(b.ifft2(field_kt, axes=(0, 1)))  # (Nx, Ny, Nt)

        return field_xt, t


class CylindricalEngine:
    """Spectral factorization for cylindrical (axisymmetric) geometry.

    Replaces the 2D FFT with a Hankel transform in r, keeping the same
    1D NILT in the propagation direction z.

    Pipeline:
        1. Hankel transform of source f(r) -> F(kr)
        2. For each radial mode kr: evaluate gamma_z(s, kr)
        3. Transfer function H = exp(-gamma_z * depth)
        4. Multiply by convective phase if needed
        5. NILT inverse (batched IFFT along Bromwich axis)
        6. Inverse Hankel -> f(r, t)
    """

    def __init__(self, dispersion_fn, hankel_transform, backend=None):
        """
        Parameters
        ----------
        dispersion_fn : callable
            Signature: (s, KR, backend) -> gamma_z
            where s has shape (1, N_brom), KR has shape (N_modes, 1).
        hankel_transform : HankelTransform
            Pre-built Hankel transform object.
        backend : Backend, optional
        """
        self.dispersion = dispersion_fn
        self.ht = hankel_transform
        self.backend = backend or get_backend()

    def forward(self, source_r, depth: float, nilt: NILTParams,
                conv_phase: float = 0.0):
        """Propagate axisymmetric source through a column/layer.

        Parameters
        ----------
        source_r : ndarray, shape (N_modes,)
            Source profile at radial sample points (self.ht.r).
            In numpy — the Hankel transform is CPU (matrix multiply).
        depth : float
            Propagation distance (column length) [m].
        nilt : NILTParams
            NILT parameters.
        conv_phase : float, optional
            Convective phase factor v/(2*Dz) for the substitution
            C = C_hat * exp(v*z/(2*Dz)). Default 0 (no convection).

        Returns
        -------
        field : ndarray, shape (N_modes, Nt)
            Real-valued field at radial sample points and each time step.
        t : ndarray, shape (Nt,)
            Time points.
        """
        import numpy as np
        b = self.backend
        _require_complex128(b, source_r)
        N_modes = self.ht.N

        # 1. Hankel transform of source (CPU, numpy)
        S_kr = self.ht.forward(source_r)  # (N_modes,)

        # 2. Bromwich contour
        s_contour, t = bromwich_contour(nilt.a, nilt.T, nilt.N, b)

        # 3. Radial wavenumber grid
        KR = b.array(self.ht.kr).reshape(-1, 1)  # (N_modes, 1)

        # 4. Dispersion
        gamma_z = self.dispersion(
            s_contour.reshape(1, -1),  # (1, N_brom)
            KR,                         # (N_modes, 1)
            b,
        )  # (N_modes, N_brom)

        # 5. Transfer function with convective phase
        # For convection-diffusion, the substitution C = C_hat * exp(v*z/(2*Dz))
        # gives a net exponent -(gamma_z - conv_phase)*depth.
        # Computing exp(-gamma_z*d)*exp(conv*d) separately overflows for high Pe,
        # but the net is O(1).
        if abs(conv_phase) > 0:
            H = b.exp(-(gamma_z - conv_phase) * depth)
        else:
            H = b.exp(-gamma_z * depth)

        # 6. Apply source spectrum
        S_kr_gpu = b.array(S_kr).reshape(-1, 1)
        F_spectrum = S_kr_gpu * H  # (N_modes, N_brom)

        # 7. NILT inverse
        field_kt, t = nilt_inverse(F_spectrum, nilt.a, nilt.T, b)

        # 8. Inverse Hankel (CPU) — per time step
        field_kt_np = b.to_numpy(field_kt)  # (N_modes, Nt)
        t_np = b.to_numpy(t)
        Nt = field_kt_np.shape[1]

        field_rt = np.zeros((N_modes, Nt))
        for i in range(Nt):
            field_rt[:, i] = self.ht.inverse(field_kt_np[:, i])

        return field_rt, t_np


def run_centerline_fractional(
    alpha: float = 0.7,
    depth: float = 0.1,
    t_max: float = 20e-3,
    n_nilt: int = 2048,
    precision: str = "float64",
    D: float = 1e-3,
):
    """Run scalpel for the fractional Caputo centerline mode at k_perp=0.

    This is the entry point that ``scalpel.validate.against_mpmath`` and
    ``scalpel.diagnose.run_and_report`` use to anchor against the mpmath
    50-digit Talbot reference. It bypasses the transverse FFT (k_perp=0
    has no transverse dependence) and runs the NILT primitive on the
    scalar modewise transfer function directly.

    Parameters
    ----------
    alpha : float, default 0.7
        Caputo fractional order in (0, 1).
    depth : float
        Slab thickness in meters.
    t_max : float
        Observation horizon in seconds.
    n_nilt : int
        NILT contour length.
    precision : str
        'float32' or 'float64'; controls the cast applied to the
        single-precision NILT output. The NILT itself runs in
        complex128 to avoid contour-precision losses; the result is
        cast at the end to match the user-supplied precision.
    D : float
        Diffusion coefficient scaling the fractional pencil
        ``gamma_z^2 = D * s^alpha + k_perp^2`` at k_perp = 0.

    Returns
    -------
    np.ndarray, shape (n_nilt,)
        Time-domain centerline trace u(t_n) for n=0,..,n_nilt-1.
    """
    from .nilt import bromwich_contour, nilt_inverse
    from ..backends import get_backend

    backend = get_backend()
    # Bromwich shift chosen for the diffusion-class auto-tuner default.
    a = max(1.0 / t_max, 1e-6)
    T = t_max

    s, t = bromwich_contour(a, T, n_nilt, backend)
    # Centerline (k_perp=0) modewise transfer with unit source S(s)=1/s.
    s_np = backend.to_numpy(s)
    gamma_z = np.sqrt(D * s_np ** alpha + 0.0)
    H = np.exp(-depth * gamma_z) / s_np
    H_b = backend.from_numpy(H)
    u_t = nilt_inverse(H_b, a, T, backend)
    u_np = backend.to_numpy(u_t)
    if precision == "float32":
        u_np = u_np.astype(np.float32)
    return np.asarray(u_np, dtype=np.float64 if precision == "float64" else np.float32)
