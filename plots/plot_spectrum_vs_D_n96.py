#!/usr/bin/env python3
"""Velocity-free scaling dimensions versus bond dimension D at N = 96 at the critical mass m/e = 0.334
(theta = pi, x = 10, hmax = 3):
    Delta_n^k = (E_n^k - E_0^{k=0}) / (E_0^{k=1} - E_1^{k=0}),
i.e. every gap in units of the (d sigma - sigma) spacing, computed per D, for the five lowest excitations per momentum
sector (k = 0: levels 1-5, k = 1: levels 0-4), D = 12..20. Same normalisation and style as plot_spectrum_vs_D.py (N = 64).
Input: stage3_levels_final.csv (export_bloch.jl over bloch/crit_*.jls). Also printed: N gap / (2 pi v) with v = 2.00.
Horizontal dotted lines: Ising CFT (c = 1/2) dimensions of the matching states."""
import csv, os, sys
from collections import defaultdict
from math import pi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
V = 2.00
ME = sys.argv[1] if len(sys.argv) > 1 else "0.334"   # mass as written in the CSV me column
N = int(sys.argv[2]) if len(sys.argv) > 2 else 96
lad = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(root, "stage3_levels_final.csv"))):
    if int(r["N"]) == N and r["me"] == ME and r["hmax"] == "3":
        lad[int(r["D"])][(int(r["k"]), int(r["level"]))] = float(r["gap"])
def delta(g, k, n): return N * g[(k, n)] / (2 * pi * V)
def ratio(g, k, n): return g[(k, n)] / (g[(1, 0)] - g[(0, 1)])

Ds = sorted(lad)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#f0a30a", "#8e5cc7"]
LEVELS = {0: [1, 2, 3, 4, 5], 1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4]}
# Ising towers, Delta = Delta_primary + n + nbar at spin n - nbar (Virasoro characters: sigma 1,1,1,2,2,...; eps 1,1,1,1,2,...;
# identity 1,0,1,1,2,...). spin 2: T 2, d^2 sigma 2.125, d^2 eps 3, sigma level (3,1) x2 4.125.
# spin 3: dT 3, sigma level (3,0) x2 3.125, d^3 eps 4, sigma level (4,1) x2 5.125.
ISING = {0: [(0.125, "σ"), (1, "ε"), (2.125, "∂∂̄σ"), (3, "∂∂̄ε"), (4, "TT̄")],
         1: [(1.125, "∂σ"), (2, "∂ε"), (3.125, "∂²∂̄σ"), (4, "∂²∂̄ε"), (5, "∂³∂̄²1")],
         2: [(2, "T"), (2.125, "∂²σ"), (3, "∂²ε"), (4.125, "σ (3,1)"), (4.125, "σ (3,1)")],
         3: [(3, "∂T"), (3.125, "σ (3,0)"), (3.125, "σ (3,0)"), (4, "∂³ε"), (5.125, "σ (4,1)")]}
# a momentum gets a panel when it has data; each panel uses the D values at which that momentum was computed
# optional 3rd argument restricts the panels, e.g. "0,1,2"
KSEL = [int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(LEVELS)
KS = [k for k in KSEL if any((k, LEVELS[k][-1]) in lad[D] for D in Ds)]
DK = {k: [D for D in Ds if (k, LEVELS[k][-1]) in lad[D]] for k in KS}

ncol = 2 if len(KS) == 4 else len(KS); nrow = (len(KS) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol + 0.8, 4.4 * nrow + 0.4), sharey=True, squeeze=False)
axes = axes.ravel()
def spread(ys, gap):   # nudge label heights apart (in data units) so that close levels keep readable labels
    order = sorted(range(len(ys)), key=lambda i: ys[i]); out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap: out[b] = out[a] + gap
    return out
for ax, k in zip(axes, KS):
    ylab = spread([ratio(lad[DK[k][0]], k, n) for n in LEVELS[k]], 0.22)
    Ds = DK[k]; Dall = sorted(lad)   # every panel shares the full D axis, so a momentum computed at fewer D lines up
    for i, n in enumerate(LEVELS[k]):
        d, lab = ISING[k][i]
        # degenerate Ising values (k=2: 4.125 twice) get one line, in the colour of the lower level
        if all(dj != d for dj, _ in ISING[k][:i]):
            ax.axhline(d, color=COL[i], lw=0.9, ls=":", zorder=1)
        ys = [ratio(lad[D], k, n) for D in Ds]
        ax.plot(Ds, ys, marker="o", ms=5, lw=1.2, color=COL[i], zorder=3)
        ax.text(Ds[0] - 0.4, ylab[i], f"n = {n}", color=COL[i], fontsize=7.5, va="center", ha="right", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xticks(Dall); ax.set_xlim(Dall[0] - 1.7, Dall[-1] + 0.6); ax.set_xlabel("bond dimension D")
    ax.set_title(f"k = {k}", fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axes[::ncol]: ax.set_ylabel(r"$(E_n^{k} - E_0^{k=0})\,/\,(E_0^{k=1} - E_1^{k=0})$")
h = [plt.Line2D([], [], marker="o", color="#555555", label=f"m/e = {ME}"),
     plt.Line2D([], [], ls=":", color="#555555", label="Ising CFT")]
axes[ncol - 1].legend(handles=h, fontsize=7.5, loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
fig.suptitle(f"Scaling dimensions vs D at the critical point: x = 10, θ = π, m/e = {ME}, N = {N}",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
out = os.path.join(root, "figures", f"fig_spectrum_vs_D_n{N}.png" if ME == "0.334" else f"fig_spectrum_vs_D_n{N}_me{ME}.png"); fig.savefig(out, dpi=200); print(out)
print("k n  ratio(D...)  | N gap/(2 pi v)(D...)  | last D step: dgap, dratio, dDelta_v")
for k in KS:
    Dk = DK[k]; a, b = Dk[-2:] if len(Dk) > 1 else (None, Dk[-1])
    print(f"k={k}: D = " + ",".join(map(str, Dk)) + (f"; last step D{a}->{b}" if a else ""))
    for n in LEVELS[k]:
        ch = (f"| {lad[b][(k, n)] - lad[a][(k, n)]:+.2e} {ratio(lad[b], k, n) - ratio(lad[a], k, n):+.4f}"
              f" {delta(lad[b], k, n) - delta(lad[a], k, n):+.4f}") if a else ""
        print(k, n, " ".join(f"{ratio(lad[D], k, n):.4f}" for D in Dk), "|",
              " ".join(f"{delta(lad[D], k, n):.4f}" for D in Dk), ch)
