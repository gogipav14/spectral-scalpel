"""
Pure-JAX ETDRK integrator for reaction-diffusion.

Everything stays on GPU. The entire time loop is JIT-compiled via
jax.lax.fori_loop, eliminating CPU↔GPU round-trips.

This is the high-performance path — ~100x faster than the
backend-agnostic version for long integrations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)


def build_laplacian_eigenvalues(Nx, Ny, Lx, Ly):
    """Fourier eigenvalues of periodic 2D Laplacian (full spectrum)."""
    dx = Lx / Nx
    dy = Ly / Ny
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    return -(KX**2 + KY**2)


def build_laplacian_eigenvalues_rfft(Nx, Ny, Lx, Ly):
    """Fourier eigenvalues for rfft2: shape (Nx, Ny//2+1)."""
    dx = Lx / Nx
    dy = Ly / Ny
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.rfftfreq(Ny, dy) * 2 * jnp.pi  # only non-negative freqs
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    return -(KX**2 + KY**2)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def integrate_strang_jit(
    u0, v0, lap_eig, params, dt,
    n_steps, save_every, Du, Dv,
):
    """JIT-compiled Strang splitting integration.

    Parameters
    ----------
    u0, v0 : arrays (Nx, Ny)
        Initial conditions.
    lap_eig : array (Nx, Ny)
        Laplacian eigenvalues.
    params : dict
        Reaction parameters.
    dt : float
        Time step.
    n_steps : int (static)
        Total steps.
    save_every : int (static)
        Save interval.
    Du, Dv : float (static)
        Diffusion coefficients.

    Returns
    -------
    u_history, v_history : arrays (n_saves, Nx, Ny)
    """
    exp_half_u = jnp.exp(Du * lap_eig * dt / 2)
    exp_half_v = jnp.exp(Dv * lap_eig * dt / 2)

    n_saves = n_steps // save_every + 1
    Nx, Ny = u0.shape

    def step(carry, _):
        u, v, step_idx = carry

        # Half diffusion
        u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
        v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

        # Full reaction (Heun)
        Ru1, Rv1 = _reaction(u, v, params)
        u_t = u + dt * Ru1
        v_t = v + dt * Rv1
        Ru2, Rv2 = _reaction(u_t, v_t, params)
        u = u + dt / 2 * (Ru1 + Ru2)
        v = v + dt / 2 * (Rv1 + Rv2)

        # Half diffusion
        u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
        v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

        return (u, v, step_idx + 1), (u, v)

    # Run with scan (allows saving intermediate states)
    init = (u0, v0, 0)
    _, (u_all, v_all) = jax.lax.scan(step, init, None, length=n_steps)

    # Subsample saves
    indices = jnp.arange(0, n_steps, save_every)
    u_saves = u_all[indices]
    v_saves = v_all[indices]

    return u_saves, v_saves


def _reaction(u, v, params):
    """Override this per system. Default: Brusselator."""
    a = params["a"]
    b = params["b"]
    uvv = u * u * v
    return (a - (b + 1) * u + uvv,
            b * u - uvv)


# ═══════════════════════════════════════════════════════════════════════
#  System-specific JIT integrators
# ═══════════════════════════════════════════════════════════════════════

def _brusselator_step(u, v, dt, params, exp_half_u, exp_half_v):
    """Single Strang step for Brusselator."""
    a, b = params["a"], params["b"]

    # Half diffusion
    u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
    v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

    # Reaction (Heun)
    uvv = u * u * v
    Ru1 = a - (b + 1) * u + uvv
    Rv1 = b * u - uvv
    u_t = u + dt * Ru1
    v_t = v + dt * Rv1
    uvv2 = u_t * u_t * v_t
    Ru2 = a - (b + 1) * u_t + uvv2
    Rv2 = b * u_t - uvv2
    u = u + dt / 2 * (Ru1 + Ru2)
    v = v + dt / 2 * (Rv1 + Rv2)

    # Half diffusion
    u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
    v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

    return u, v


def _gray_scott_step(u, v, dt, params, exp_half_u, exp_half_v):
    """Single Strang step for Gray-Scott."""
    F, k = params["F"], params["k"]

    u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
    v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

    uvv = u * v * v
    Ru1 = -uvv + F * (1 - u)
    Rv1 = uvv - (F + k) * v
    u_t = u + dt * Ru1
    v_t = v + dt * Rv1
    uvv2 = u_t * v_t * v_t
    Ru2 = -uvv2 + F * (1 - u_t)
    Rv2 = uvv2 - (F + k) * v_t
    u = u + dt / 2 * (Ru1 + Ru2)
    v = v + dt / 2 * (Rv1 + Rv2)

    u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
    v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

    return u, v


def _schnakenberg_step(u, v, dt, params, exp_half_u, exp_half_v):
    """Single Strang step for Schnakenberg."""
    a, b = params["a"], params["b"]

    u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
    v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

    uuv = u * u * v
    Ru1 = a - u + uuv
    Rv1 = b - uuv
    u_t = u + dt * Ru1
    v_t = v + dt * Rv1
    uuv2 = u_t * u_t * v_t
    Ru2 = a - u_t + uuv2
    Rv2 = b - uuv2
    u = u + dt / 2 * (Ru1 + Ru2)
    v = v + dt / 2 * (Rv1 + Rv2)

    u = jnp.real(jnp.fft.ifft2(exp_half_u * jnp.fft.fft2(u)))
    v = jnp.real(jnp.fft.ifft2(exp_half_v * jnp.fft.fft2(v)))

    return u, v


_STEP_FNS = {
    "brusselator": _brusselator_step,
    "gray_scott": _gray_scott_step,
    "schnakenberg": _schnakenberg_step,
}


def integrate_jit(
    system_name: str,
    u0, v0,
    Du: float, Dv: float,
    Lx: float, Ly: float,
    params: dict,
    dt: float,
    n_steps: int,
    save_every: int = 100,
):
    """Fully JIT-compiled integration using jax.lax.scan.

    Parameters
    ----------
    system_name : str
        "brusselator", "gray_scott", or "schnakenberg".
    u0, v0 : arrays (Nx, Ny)
    Du, Dv : float
    Lx, Ly : float
    params : dict
    dt : float
    n_steps : int
    save_every : int

    Returns
    -------
    u_saves, v_saves : arrays (n_saves, Nx, Ny)
    times : array (n_saves,)
    """
    step_fn = _STEP_FNS[system_name]

    lap_eig = build_laplacian_eigenvalues(u0.shape[0], u0.shape[1], Lx, Ly)
    exp_half_u = jnp.exp(Du * lap_eig * dt / 2)
    exp_half_v = jnp.exp(Dv * lap_eig * dt / 2)

    jax_params = {k: jnp.float64(v) for k, v in params.items()}

    # Chunked approach: run save_every steps in fori_loop (no intermediates
    # stored), save one snapshot, repeat. Total memory: O(n_saves * Nx * Ny).
    n_chunks = n_steps // save_every

    def inner_chunk(carry, _):
        u, v = carry

        def body(i, state):
            u_, v_ = state
            return step_fn(u_, v_, dt, jax_params, exp_half_u, exp_half_v)

        u, v = jax.lax.fori_loop(0, save_every, body, (u, v))
        return (u, v), (u, v)

    init = (jnp.asarray(u0), jnp.asarray(v0))
    _, (u_saves, v_saves) = jax.lax.scan(inner_chunk, init, None,
                                          length=n_chunks)

    times = jnp.arange(1, n_chunks + 1) * save_every * dt
    return u_saves, v_saves, times


# ═══════════════════════════════════════════════════════════════════════
#  RFFT variants — exploit real-valued fields for ~2x FFT speedup
# ═══════════════════════════════════════════════════════════════════════

def _diffuse_half_rfft(u, exp_half):
    """Half diffusion step using rfft2/irfft2."""
    return jnp.fft.irfft2(exp_half * jnp.fft.rfft2(u), s=u.shape)


def _gray_scott_step_rfft(u, v, dt, params, exp_half_u, exp_half_v):
    F, k = params["F"], params["k"]

    u = _diffuse_half_rfft(u, exp_half_u)
    v = _diffuse_half_rfft(v, exp_half_v)

    uvv = u * v * v
    Ru1 = -uvv + F * (1 - u)
    Rv1 = uvv - (F + k) * v
    u_t = u + dt * Ru1
    v_t = v + dt * Rv1
    uvv2 = u_t * v_t * v_t
    Ru2 = -uvv2 + F * (1 - u_t)
    Rv2 = uvv2 - (F + k) * v_t
    u = u + dt / 2 * (Ru1 + Ru2)
    v = v + dt / 2 * (Rv1 + Rv2)

    u = _diffuse_half_rfft(u, exp_half_u)
    v = _diffuse_half_rfft(v, exp_half_v)
    return u, v


def _brusselator_step_rfft(u, v, dt, params, exp_half_u, exp_half_v):
    a, b = params["a"], params["b"]

    u = _diffuse_half_rfft(u, exp_half_u)
    v = _diffuse_half_rfft(v, exp_half_v)

    uvv = u * u * v
    Ru1 = a - (b + 1) * u + uvv
    Rv1 = b * u - uvv
    u_t = u + dt * Ru1
    v_t = v + dt * Rv1
    uvv2 = u_t * u_t * v_t
    Ru2 = a - (b + 1) * u_t + uvv2
    Rv2 = b * u_t - uvv2
    u = u + dt / 2 * (Ru1 + Ru2)
    v = v + dt / 2 * (Rv1 + Rv2)

    u = _diffuse_half_rfft(u, exp_half_u)
    v = _diffuse_half_rfft(v, exp_half_v)
    return u, v


def _schnakenberg_step_rfft(u, v, dt, params, exp_half_u, exp_half_v):
    a, b = params["a"], params["b"]

    u = _diffuse_half_rfft(u, exp_half_u)
    v = _diffuse_half_rfft(v, exp_half_v)

    uuv = u * u * v
    Ru1 = a - u + uuv
    Rv1 = b - uuv
    u_t = u + dt * Ru1
    v_t = v + dt * Rv1
    uuv2 = u_t * u_t * v_t
    Ru2 = a - u_t + uuv2
    Rv2 = b - uuv2
    u = u + dt / 2 * (Ru1 + Ru2)
    v = v + dt / 2 * (Rv1 + Rv2)

    u = _diffuse_half_rfft(u, exp_half_u)
    v = _diffuse_half_rfft(v, exp_half_v)
    return u, v


_STEP_FNS_RFFT = {
    "brusselator": _brusselator_step_rfft,
    "gray_scott": _gray_scott_step_rfft,
    "schnakenberg": _schnakenberg_step_rfft,
}


def integrate_jit_rfft(
    system_name: str,
    u0, v0,
    Du: float, Dv: float,
    Lx: float, Ly: float,
    params: dict,
    dt: float,
    n_steps: int,
    save_every: int = 100,
):
    """Same as integrate_jit but using rfft2/irfft2 for real fields."""
    step_fn = _STEP_FNS_RFFT[system_name]
    Nx, Ny = u0.shape

    # Half-spectrum eigenvalues: (Nx, Ny//2+1)
    lap_eig = build_laplacian_eigenvalues_rfft(Nx, Ny, Lx, Ly)
    exp_half_u = jnp.exp(Du * lap_eig * dt / 2)
    exp_half_v = jnp.exp(Dv * lap_eig * dt / 2)

    jax_params = {k: jnp.float64(v) for k, v in params.items()}

    n_chunks = n_steps // save_every

    def inner_chunk(carry, _):
        u, v = carry

        def body(i, state):
            u_, v_ = state
            return step_fn(u_, v_, dt, jax_params, exp_half_u, exp_half_v)

        u, v = jax.lax.fori_loop(0, save_every, body, (u, v))
        return (u, v), (u, v)

    init = (jnp.asarray(u0), jnp.asarray(v0))
    _, (u_saves, v_saves) = jax.lax.scan(inner_chunk, init, None,
                                          length=n_chunks)

    times = jnp.arange(1, n_chunks + 1) * save_every * dt
    return u_saves, v_saves, times
