#!/usr/bin/env python3
"""Tower plot of the one-insertion spectrum at the critical point (x=10, theta=pi, m/e=0.334):
scaling dimension Delta = N (E - E0) / (2 pi v), v = 2.00, versus momentum k."""
import csv, math, os, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "stage3_levels_final.csv")
V = 2.00
g = defaultdict(dict)
for r in csv.DictReader(open(path)):
    g[(int(r["N"]), int(r["D"]))][(int(r["k"]), int(r["level"]))] = float(r["gap"])

# Ising (c=1/2) tower on the ring, spins 0 and 1 (multiplicities from the Virasoro characters)
SETS = [((64, 20), "N = 64, D = 20", "#2a78d6", "o", 0.0)]

fig, ax = plt.subplots(figsize=(4.6, 5.4))
for key, lab, col, mk, dx in SETS:
    N, D = key
    for k in (0, 1):
        ds = [N * g[key][(k, l)] / (2 * math.pi * V) for l in range(8)]
        ax.plot([k + dx] * len(ds), ds, ls="none", marker=mk, ms=6, color=col, mfc=col if N == 64 else "white",
                label=lab if k == 0 else None, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(["k = 0", "k = 1"])
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.2, 6.4)
ax.set_ylabel("Δ = N (E − E₀) / (2π v)")
ax.set_title("Critical spectrum: x = 10, θ = π, m/e = 0.334\nN = 64, D = 20", fontsize=10, fontweight="bold", loc="left")
ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(root, "figures", "fig_critical_spectrum.png")
fig.savefig(out, dpi=200); print(out)
