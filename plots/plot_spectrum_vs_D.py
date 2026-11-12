#!/usr/bin/env python3
"""Velocity-free scaling dimensions versus bond dimension D at N = 64:
    Delta_n^k = (E_n^k - E_0^{k=0}) / (E_0^{k=1} - E_1^{k=0}),
i.e. every gap in units of the (d sigma - sigma) spacing, which is v * 2 pi / N for the Ising CFT (Delta = 1 apart).
Input: ../bloch_levels.csv (the Oct 3 D(N) ladder, m/e = 0.33, theta = pi, x = 10, hmax = 3).
Horizontal dotted lines: Ising CFT (c = 1/2) dimensions of the matching states."""
import csv, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
N = 64
lad = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(root, "..", "bloch_levels.csv"))):
    if int(r["N"]) == N and r["me"] == "0.33" and r["hmax"] == "3":
        lad[int(r["D"])][(int(r["k"]), int(r["level"]))] = float(r["gap"])
def ratio(g, k, n): return g[(k, n)] / (g[(1, 0)] - g[(0, 1)])

Ds = sorted(lad)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#f0a30a", "#8e5cc7"]
LEVELS = {0: [1, 2, 3, 4, 5], 1: [0, 1, 2, 3, 4]}
# Ising state matched to each plotted level (k=0 level 5 sits in the TT-bar / d2 d2-bar sigma pair at 4, 4.125;
# k=1 level 4 in the L-3 Lbar-2 1 / sigma (3,2) cluster at 5, 5.125)
ISING = {0: [(0.125, "σ"), (1, "ε"), (2.125, "∂∂̄σ"), (3, "∂∂̄ε"), (4, "TT̄")],
         1: [(1.125, "∂σ"), (2, "∂ε"), (3.125, "∂²∂̄σ"), (4, "∂²∂̄ε"), (5, "∂³∂̄²1")]}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
for ax, k in zip(axes, (0, 1)):
    for i, n in enumerate(LEVELS[k]):
        d, lab = ISING[k][i]
        ax.axhline(d, color=COL[i], lw=0.9, ls=":", zorder=1)
        ys = [ratio(lad[D], k, n) for D in Ds]
        ax.plot(Ds, ys, marker="o", ms=5, lw=1.2, color=COL[i], zorder=3)
        ax.text(11.6, ys[0], f"n = {n}", color=COL[i], fontsize=7.5, va="center", ha="right", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xticks(Ds); ax.set_xlim(10.3, 20.8); ax.set_xlabel("bond dimension D")
    ax.set_title(f"k = {k}", fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel(r"$(E_n^{k} - E_0^{k=0})\,/\,(E_0^{k=1} - E_1^{k=0})$")
h = [plt.Line2D([], [], marker="o", color="#555555", label="m/e = 0.33"),
     plt.Line2D([], [], ls=":", color="#555555", label="Ising CFT")]
axes[1].legend(handles=h, fontsize=7.5, loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
fig.suptitle(f"Scaling dimensions vs D: x = 10, θ = π, N = {N}", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
out = os.path.join(root, "figures", "fig_spectrum_vs_D.png"); fig.savefig(out, dpi=200); print(out)
for k in (0, 1):
    for n in LEVELS[k]:
        print(k, n, " ".join(f"{ratio(lad[D], k, n):.4f}" for D in Ds))
