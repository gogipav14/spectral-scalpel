"""
Memory-bounded chunked dispatch over the transverse Fourier grid.

The batched FFT-NILT primitive of ``scalpel.core.engine`` holds the full
(N_NILT, Nx, Ny) complex tensor in memory. For very large transverse
grids this exceeds available VRAM. This module composes the primitive
over transverse tiles, sized to a user-supplied memory budget.

Example
-------
>>> from scalpel.chunked import batched_forward_chunked
>>> field = batched_forward_chunked(
...     source, depth, material, grid, nilt,
...     vram_budget_bytes=int(7e9),  # leave 1 GB headroom on an 8 GB card
... )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ChunkPlan:
    """How a transverse grid is tiled to fit a memory budget."""
    tile_shape: tuple          # (Nx_tile, Ny_tile)
    n_tiles: int                # total number of tiles
    bytes_per_tile: int         # peak bytes per tile (rough estimate)
    dtype_bytes: int            # bytes per complex element

    def __repr__(self) -> str:
        return (
            f"ChunkPlan(tile={self.tile_shape}, "
            f"n_tiles={self.n_tiles}, "
            f"~{self.bytes_per_tile/1e9:.2f} GB/tile, "
            f"dtype={self.dtype_bytes}B)"
        )


def plan_tiles(
    n_x: int,
    n_y: int,
    n_nilt: int,
    vram_budget_bytes: int,
    dtype_bytes: int = 16,
    safety: float = 0.7,
) -> ChunkPlan:
    """Decide how to tile (Nx, Ny) so each tile fits within the budget.

    Parameters
    ----------
    n_x, n_y : int
        Full transverse grid dimensions.
    n_nilt : int
        NILT contour length.
    vram_budget_bytes : int
        Available memory for the peak intermediate tensor.
    dtype_bytes : int, default 16
        Bytes per complex element (16 for complex128, 8 for complex64).
    safety : float, default 0.7
        Fraction of budget actually used; leaves headroom for FFT
        workspace, gradients, etc.

    Returns
    -------
    ChunkPlan
    """
    if n_x * n_y * n_nilt * dtype_bytes <= vram_budget_bytes * safety:
        # Single tile fits; chunk is trivial.
        return ChunkPlan(
            tile_shape=(n_x, n_y),
            n_tiles=1,
            bytes_per_tile=n_x * n_y * n_nilt * dtype_bytes,
            dtype_bytes=dtype_bytes,
        )

    # Pick the largest power-of-two tile that fits.
    elements_per_tile = int((vram_budget_bytes * safety) / dtype_bytes / n_nilt)
    side = int(math.floor(math.sqrt(elements_per_tile)))
    side = max(8, 1 << int(math.floor(math.log2(side))))  # round down to power of 2
    n_tiles_x = math.ceil(n_x / side)
    n_tiles_y = math.ceil(n_y / side)
    return ChunkPlan(
        tile_shape=(side, side),
        n_tiles=n_tiles_x * n_tiles_y,
        bytes_per_tile=side * side * n_nilt * dtype_bytes,
        dtype_bytes=dtype_bytes,
    )


def batched_forward_chunked(
    forward_callable: Callable,
    source_xy,
    depth: float,
    material,
    grid,
    nilt,
    vram_budget_bytes: int,
    dtype_bytes: int = 16,
    backend=None,
    progress: Optional[Callable[[int, int], None]] = None,
    output_layout: str = "tnxny",
):
    """Tile the (N_NILT, Nx, Ny) workload over transverse axes.

    Stitches per-tile outputs back into the full (Nt, Nx, Ny) result.
    For workloads that already fit in budget this reduces to a single
    call to ``forward_callable``.

    ``output_layout`` is the explicit axis contract for the tile
    output of ``forward_callable``. The two supported layouts are
    ``'tnxny'`` (time-first; the layout used by ``scalpel.api``) and
    ``'nxnyt'`` (time-last; some legacy engines). Specifying this
    explicitly prevents the previously-heuristic axis-order detection
    from misfiring when ``Nt == Nx`` by coincidence.

    Parameters
    ----------
    forward_callable : Callable
        A function with the same signature as ``scalpel.api.propagate_*``.
    source_xy : array, shape (Nx, Ny)
        Source field.
    depth, material, grid, nilt :
        Forwarded to ``forward_callable``.
    vram_budget_bytes : int
        Available VRAM (or RAM) for the peak intermediate tensor.
    dtype_bytes : int, default 16
        Bytes per complex element (16 for complex128, 8 for complex64).
    backend : Backend, optional
    progress : Callable[[done, total], None], optional
        Called after each tile so callers can render a progress bar.

    Returns
    -------
    field : array, shape (Nt, Nx, Ny)
    t : array, shape (Nt,)
    """
    n_x, n_y = source_xy.shape
    plan = plan_tiles(n_x, n_y, nilt.N, vram_budget_bytes, dtype_bytes)

    if plan.n_tiles == 1:
        return forward_callable(
            source_xy, depth, material, grid, nilt, backend=backend
        )

    tile_x, tile_y = plan.tile_shape
    # Allocate output on the host as np.float64 then copy back per tile
    # (the field is real; the intermediate is complex but we only return
    # the real part). For genuinely huge runs the user is expected to
    # stream tiles to disk via a callback wrapper.
    out_t = None
    out_field = None
    tiles_done = 0

    for ix in range(0, n_x, tile_x):
        for iy in range(0, n_y, tile_y):
            tile = source_xy[ix : ix + tile_x, iy : iy + tile_y]
            tile_field, tile_t = forward_callable(
                tile, depth, material, grid, nilt, backend=backend
            )
            tile_array = np.asarray(tile_field)
            if output_layout == "nxnyt":
                # (Nx, Ny, Nt) -> (Nt, Nx, Ny)
                tile_array = np.moveaxis(tile_array, -1, 0)
            elif output_layout != "tnxny":
                raise ValueError(
                    f"output_layout={output_layout!r} not recognized; "
                    f"use 'tnxny' or 'nxnyt'"
                )
            if out_field is None:
                out_t = tile_t
                n_t = tile_array.shape[0]
                out_field = np.empty((n_t, n_x, n_y), dtype=np.float64)
            out_field[:, ix : ix + tile_array.shape[1], iy : iy + tile_array.shape[2]] = tile_array
            tiles_done += 1
            if progress is not None:
                progress(tiles_done, plan.n_tiles)

    return out_field, out_t
