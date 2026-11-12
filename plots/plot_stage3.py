#!/usr/bin/env python3
"""Critical showcase figure (stage 3): scaling dimensions and velocity at m/e = 0.334, theta = pi, x = 10 vs 1/N.
Input: the CSV from export_bloch.jl over bloch/crit_*.jls (default stage3_levels.csv). Rows at the highest D per N
are filled markers and enter the 1/N fits; lower-D rows at the same N are open markers (they show the residual
D-dependence of the higher levels). Output: figures/fig_critical_showcase.png.
"""
import csv, math, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "stage3_levels.csv")
out = os.path.join(root, "figures"); os.makedirs(out, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]            # Delta_1..4, fixed order
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 6,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})

data = defaultdict(dict)
for r in csv.DictReader(open(path)):
    if r["me"] != "0.334": continue   # the CSV also carries the m/e=0.3341 N=96 ladder
    data[(int(r["N"]), int(r["D"]))][(int(r["k"]), int(r["level"]))] = float(r["gap"])
bestD = {}
for N, D in data: bestD[N] = max(bestD.get(N, 0), D)
rows = []
for (N, D), g in sorted(data.items()):
    x0 = [N * g[(0, a)] / (2 * math.pi) for a in range(1, 5)]
    x1 = N * g[(1, 0)] / (2 * math.pi)
    vk1 = x1 - x0[0]; vsig = 8 * x0[0]
    rows.append(dict(N=N, D=D, best=(D == bestD[N]), vk1=vk1, vsig=vsig, delta=[x / vk1 for x in x0]))
def linfit(xs, ys):
    A = np.array([[1.0, x] for x in xs]); return np.linalg.lstsq(A, np.array(ys), rcond=None)[0]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axes[0]
ising = [0.125, 1.0, 2.125, 3.0]; names = ["σ (1/8)", "ε (1)", "∂∂̄σ (2.125)", "∂∂̄ε (3)"]
best = [r for r in rows if r["best"]]; other = [r for r in rows if not r["best"]]
for a in range(4):
    xs = [1 / r["N"] for r in best]; ys = [r["delta"][a] for r in best]
    ax.plot(xs, ys, marker="o", ls="none", color=SER[a], label=f"Δ{a+1}")
    if other: ax.plot([1 / r["N"] for r in other], [r["delta"][a] for r in other], marker="o", ls="none", mfc="white", color=SER[a], zorder=1.5)
    ax.axhline(ising[a], color=SER[a], lw=0.8, ls=":", alpha=0.8)
    ax.text(0.0515, ising[a], names[a], color=SER[a], fontsize=7, va="bottom", ha="right")
    if a < 2:   # only the D-converged levels get a fit line
        c0, c1 = linfit(xs, ys); xx = np.array([0, max(xs)]); ax.plot(xx, c0 + c1 * xx, color=SER[a], lw=0.9, ls="--")
        ax.annotate(f"{c0:.4f}", (0, c0), textcoords="offset points", xytext=(4, 3), fontsize=7, color=INK2)
ax.set_xlim(0, 0.053); ax.set_xlabel("1 / N"); ax.set_ylabel("scaling dimension Δ")
ax.set_title("Lowest k = 0 levels, Δ = N(E − E₀)/(2πv)")
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.1), ncol=4)
ax = axes[1]
for key, lab, col, mk in (("vk1", "v from the k = 1 spacing (∂σ − σ)", "#2a78d6", "o"), ("vsig", "v from Δσ = 1/8", "#eb6834", "s")):
    xs = [1 / r["N"] for r in best]; ys = [r[key] for r in best]
    ax.plot(xs, ys, marker=mk, ls="none", color=col, label=lab)
    if other: ax.plot([1 / r["N"] for r in other], [r[key] for r in other], marker=mk, ls="none", mfc="white", color=col, zorder=1.5)
    c0, c1 = linfit(xs, ys); xx = np.array([0, max(xs)]); ax.plot(xx, c0 + c1 * xx, color=col, lw=0.9, ls="--")
    ax.annotate(f"{c0:.3f}", (0, c0), textcoords="offset points", xytext=(4, 3 if key == "vk1" else -9), fontsize=7, color=INK2)
ax.set_xlim(0, 0.053); ax.set_xlabel("1 / N"); ax.set_ylabel("velocity v"); ax.set_title("Two velocity estimates")
ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Critical showcase: x = 10, θ = π, m/e = 0.334 = m_c   (filled: highest D per N; open: lower D)", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_critical_showcase.png")); plt.close(fig)
print("rows used:", [(r["N"], r["D"], "best" if r["best"] else "lowD") for r in rows])
