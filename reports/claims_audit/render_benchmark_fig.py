"""Render honest benchmark figure from the actual measurements of benchmark_all.py."""
import numpy as np
import matplotlib.pyplot as plt

# Measured 2026-04-23 on NVIDIA RTX 5060 via benchmark_all.py
# Wet-clay Maxwell, 64x64 grid, d=0.5m, N_NILT=2048, Nz=500, Nt=4334, NFE ratio=2116x
DATA = [
    # backend,      scalpel_ms, fdtd_ms, ratio
    ("NumPy CPU",      902.7, 100239,  111),
    ("PyTorch CPU",    276.1,  42316,  153),
    ("CuPy GPU",        19.5,   2354,  121),
    ("PyTorch GPU",     21.1,   2312,  109),
    ("JAX GPU",         36.9,    990,   27),
]

labels = [d[0] for d in DATA]
scalpel = [d[1] for d in DATA]
fdtd    = [d[2] for d in DATA]
ratio   = [d[3] for d in DATA]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(len(labels))
w = 0.38
ax1.bar(x - w/2, scalpel, w, label="Spectral scalpel", color="C0")
ax1.bar(x + w/2, fdtd,    w, label="Yee FDTD",         color="C3")
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=20, ha="right")
ax1.set_ylabel("Wall-clock time (ms)")
ax1.set_title("(A) Scalpel vs Yee FDTD, wet-clay Maxwell\n64$\\times$64 grid, $d=0.5$ m, 13 transit observation window")
ax1.legend()
ax1.grid(True, which="both", axis="y", alpha=0.25)

bars = ax2.bar(x, ratio, color=["C0" if "CPU" in l else "C2" for l in labels])
for rect, r in zip(bars, ratio):
    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height()*1.02,
             f"{r}$\\times$", ha="center", va="bottom", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=20, ha="right")
ax2.set_ylabel("Wall-clock speedup (FDTD / scalpel)")
ax2.set_title("(B) Per-backend speedup\nNFE ratio is a hardware-independent $\\sim$2116$\\times$")
ax2.axhline(2116, color="gray", ls="--", lw=1, label=r"Structural NFE ratio $\approx$ 2116$\times$")
ax2.set_yscale("log")
ax2.legend(loc="upper right")
ax2.grid(True, which="both", axis="y", alpha=0.25)
ax2.set_ylim(10, 3500)

plt.tight_layout()
out = "/home/gogip/github_repos/spectral-scalpel-private/paper_pnas/figures/benchmark_fdtd.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)
