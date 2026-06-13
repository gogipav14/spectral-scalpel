"""
Fig. 3 v2: honest two-domain benchmark.

Panel (a): Wave-type / stiff CFL (lossy Maxwell, 3D Yee baseline). Speedup
           comes from the FDTD CFL stiffness vs the factorization's
           N_NILT-only cost.

Panel (b): Fractional Caputo subdiffusion, α = 0.7 (anomalous heat).
           Baseline is FTCS + L1 fractional time scheme. Speedup comes from
           the L1 history convolution being O(N_t^2 N^3); the factorization
           handles s^α natively in the dispersion relation, so its cost is
           unchanged from the integer-order case.

Source-of-truth data: figure_benchmark_data.csv (one row per bar). Re-render
without re-running benchmarks by editing the CSV.
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "figure_benchmark_data.csv")

# Read CSV grouped by (panel, group); per group we expect side in {scalpel,
# baseline}. The 'annotation' is taken from whichever row carries it.
panels = defaultdict(lambda: {})  # panel -> {group: {scalpel, baseline, ann, problem}}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        p, g = row["panel"], row["group"]
        d = panels[p].setdefault(g, {"scalpel": None, "baseline": None,
                                     "ann": "", "problem": ""})
        d[row["side"]] = float(row["wall_ms"])
        if row["annotation"]:
            d["ann"] = row["annotation"]
        if row["problem"]:
            d["problem"] = row["problem"]


def render_panel(ax, panel_key, baseline_label, color_baseline, title):
    groups = list(panels[panel_key].keys())
    x = np.arange(len(groups))
    w = 0.38
    sc = [panels[panel_key][g]["scalpel"] for g in groups]
    bl = [panels[panel_key][g]["baseline"] for g in groups]
    ann = [panels[panel_key][g]["ann"] for g in groups]

    ax.bar(x - w / 2, sc, w, label="Spectral factorization", color="C0")
    ax.bar(x + w / 2, bl, w, label=baseline_label, color=color_baseline)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    for xi, s, b, a in zip(x, sc, bl, ann):
        if a:
            ax.text(xi, max(s, b) * 1.6, f"{a}", ha="center",
                    fontsize=8.5, color="black")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.2))

render_panel(
    ax1, "a", "3D Yee FDTD", "C3",
    "(a) CFL-stiff: Maxwell wet clay, "
    r"64$\times$64, $d = 0.5$ m",
)
render_panel(
    ax2, "b", "FTCS + L1 (fractional)", "C2",
    "(b) Laplace-natural: "
    r"Caputo $\partial_t^{0.7} u = D\nabla^2 u$, $32^3$",
)

plt.tight_layout()
out = "/home/gogip/github_repos/spectral-scalpel-private/paper_pnas/figures/benchmark_fdtd.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)
