"""
Turing patterns — JIT-compiled pure-JAX ETDRK.

Everything on GPU, no CPU↔GPU round-trips. jax.lax.scan compiles
the entire time loop into a single XLA kernel.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from scalpel.core.etdrk import brusselator, gray_scott, schnakenberg
from scalpel.core.etdrk_jax import integrate_jit

# ═══════════════════════════════════════════════════════════════════════
configs = {
    "Brusselator": dict(
        system_name="brusselator",
        factory=lambda: brusselator(Nx=256, Ny=256, Lx=64, Ly=64,
                                     Du=1.0, Dv=8.0, a=4.5, b=7.5),
        dt=0.005, n_steps=5000, save_every=500,
        field="u", Du=1.0, Dv=8.0, Lx=64.0, Ly=64.0,
        params={"a": 4.5, "b": 7.5},
    ),
    "Gray-Scott": dict(
        system_name="gray_scott",
        factory=lambda: gray_scott(Nx=256, Ny=256, Lx=2.5, Ly=2.5,
                                    Du=2e-5, Dv=1e-5, F=0.04, k=0.06),
        dt=1.0, n_steps=5000, save_every=500,
        field="v", Du=2e-5, Dv=1e-5, Lx=2.5, Ly=2.5,
        params={"F": 0.04, "k": 0.06},
    ),
    "Schnakenberg": dict(
        system_name="schnakenberg",
        factory=lambda: schnakenberg(Nx=256, Ny=256, Lx=50, Ly=50,
                                      Du=1.0, Dv=40.0, a=0.1, b=0.9),
        dt=0.01, n_steps=5000, save_every=500,
        field="u", Du=1.0, Dv=40.0, Lx=50.0, Ly=50.0,
        params={"a": 0.1, "b": 0.9},
    ),
}

results = {}

for name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"  {name} (256x256, JIT)")
    print(f"{'='*50}")

    sys = cfg["factory"]()
    u0 = sys.fields["u"].data
    v0 = sys.fields["v"].data

    # Warmup (JIT compile)
    print("  Compiling...", end=" ", flush=True)
    t0 = time.perf_counter()
    u_s, v_s, times = integrate_jit(
        cfg["system_name"], u0, v0,
        cfg["Du"], cfg["Dv"], cfg["Lx"], cfg["Ly"],
        cfg["params"], cfg["dt"],
        n_steps=10, save_every=5,
    )
    u_s.block_until_ready()
    compile_time = (time.perf_counter() - t0) * 1e3
    print(f"compiled in {compile_time:.0f} ms")

    # Timed run
    t0 = time.perf_counter()
    u_saves, v_saves, times = integrate_jit(
        cfg["system_name"], u0, v0,
        cfg["Du"], cfg["Dv"], cfg["Lx"], cfg["Ly"],
        cfg["params"], cfg["dt"],
        n_steps=cfg["n_steps"], save_every=cfg["save_every"],
    )
    u_saves.block_until_ready()
    wall = (time.perf_counter() - t0) * 1e3

    us_per_step = wall / cfg["n_steps"] * 1e3
    print(f"  {cfg['n_steps']} steps in {wall:.0f} ms ({us_per_step:.1f} us/step)")

    results[name] = dict(
        u=np.asarray(u_saves), v=np.asarray(v_saves),
        times=np.asarray(times), cfg=cfg, wall=wall,
    )

# ═══════════════════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════════════════

n_snaps = min(len(r["times"]) for r in results.values())
snap_idx = np.linspace(0, n_snaps - 1, min(6, n_snaps), dtype=int)

fig, axes = plt.subplots(3, len(snap_idx), figsize=(3.2 * len(snap_idx), 10))

for row, (name, r) in enumerate(results.items()):
    cfg = r["cfg"]
    field_key = cfg["field"]
    data = r["u"] if field_key == "u" else r["v"]

    vmin, vmax = np.percentile(data[-1], [2, 98])

    for ci, si in enumerate(snap_idx):
        si = min(si, data.shape[0] - 1)
        ax = axes[row, ci]
        im = ax.imshow(data[si].T, origin="lower", cmap="inferno",
                       vmin=vmin, vmax=vmax, aspect="equal")

        if row == 0:
            ax.set_title(f"t = {r['times'][si]:.0f}", fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"{name}\n({field_key})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(im, ax=axes[row, -1], shrink=0.8)

# Timing summary
timing_text = "  |  ".join(
    f"{name}: {r['wall']:.0f} ms ({r['wall']/r['cfg']['n_steps']*1e3:.0f} us/step)"
    for name, r in results.items()
)

fig.suptitle(
    "Turing patterns — JIT-compiled ETDRK on GPU (256x256)\n"
    + timing_text,
    fontsize=10, y=1.02)

plt.tight_layout()
out = "scripts/turing_jit.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
