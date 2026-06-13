"""
Turing pattern formation via spectral scalpel ETDRK.

Three reaction-diffusion systems, each showing pattern evolution:
  - Brusselator (spots/stripes)
  - Gray-Scott (mitosis/coral)
  - Schnakenberg (spots)

Uses Strang splitting: exact FFT diffusion + explicit Heun reaction.
Same JAX/PyTorch backend as the spectral scalpel propagation engine.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from scalpel.core.etdrk import (
    brusselator, gray_scott, schnakenberg,
    integrate,
)
from scalpel.backends import get_backend

backend = get_backend()
print(f"Backend: {backend.name}")

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

systems = {
    "Brusselator": dict(
        factory=lambda: brusselator(Nx=128, Ny=128, Lx=64, Ly=64,
                                     Du=1.0, Dv=8.0, a=4.5, b=7.5),
        dt=0.01, t_end=20.0, save_every=200,
        field="u",
    ),
    "Gray-Scott": dict(
        factory=lambda: gray_scott(Nx=128, Ny=128, Lx=2.5, Ly=2.5,
                                    Du=0.16, Dv=0.08, F=0.04, k=0.06),
        dt=1.0, t_end=10000.0, save_every=1000,
        field="v",
    ),
    "Schnakenberg": dict(
        factory=lambda: schnakenberg(Nx=128, Ny=128, Lx=50, Ly=50,
                                      Du=1.0, Dv=40.0, a=0.1, b=0.9),
        dt=0.01, t_end=50.0, save_every=500,
        field="u",
    ),
}

results = {}

for name, cfg in systems.items():
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    system = cfg["factory"]()

    t0 = time.perf_counter()
    res = integrate(
        system, t_end=cfg["t_end"], dt=cfg["dt"],
        method="strang", save_every=cfg["save_every"],
        backend=backend,
    )
    wall = (time.perf_counter() - t0) * 1e3

    n_steps = int(cfg["t_end"] / cfg["dt"])
    print(f"  {n_steps} steps, {len(res['times'])} snapshots")
    print(f"  Wall time: {wall:.0f} ms ({wall/n_steps*1e3:.1f} us/step)")

    results[name] = dict(res=res, cfg=cfg, wall=wall, system=system)


# ═══════════════════════════════════════════════════════════════════════
#  Figure: 3 systems x 4 time snapshots
# ═══════════════════════════════════════════════════════════════════════

n_snaps = min(5, min(len(r["res"]["times"]) for r in results.values()))
snap_indices = np.linspace(0, n_snaps - 1, min(5, n_snaps), dtype=int)
# Use first, 25%, 50%, 75%, last
n_total = min(len(r["res"]["times"]) for r in results.values())
snap_indices = [0]
for frac in [0.25, 0.5, 0.75, 1.0]:
    idx = min(int(frac * (n_total - 1)), n_total - 1)
    if idx not in snap_indices:
        snap_indices.append(idx)

n_cols = len(snap_indices)
fig, axes = plt.subplots(3, n_cols, figsize=(3.5*n_cols, 10))

for row, (name, r) in enumerate(results.items()):
    snaps = r["res"]["snapshots"]
    times = r["res"]["times"]
    field_name = r["cfg"]["field"]

    # Determine color range from final snapshot
    final = snaps[-1][field_name]
    vmin, vmax = np.percentile(final, [2, 98])

    for ci, si in enumerate(snap_indices):
        si = min(si, len(snaps) - 1)
        ax = axes[row, ci]
        data = snaps[si][field_name]
        t_val = times[si]

        im = ax.imshow(data.T, origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax, aspect="equal")

        if row == 0:
            ax.set_title(f"t = {t_val:.1f}", fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"{name}\n({field_name} field)", fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar on last column
    plt.colorbar(im, ax=axes[row, -1], shrink=0.8)

fig.suptitle(
    "Turing pattern formation — spectral scalpel ETDRK\n"
    "Strang splitting: exact FFT diffusion + explicit Heun reaction (128x128)",
    fontsize=12, y=1.01)

plt.tight_layout()
out = "scripts/turing_patterns.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
