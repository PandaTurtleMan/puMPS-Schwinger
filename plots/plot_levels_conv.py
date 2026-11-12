#!/usr/bin/env python3
"""One-insertion puMPS levels per momentum (massless Schwinger, m/e = 0, theta = 0, hmax = 3) against
    M_S(n, k) = sqrt(n^2/pi + (k/e)^2),
the energy of n continuum bosons of mass e/sqrt(pi) at rest relative to each other, carrying total momentum k.
Level n = 1, 2, ... is the n-th lowest excitation in each momentum sector (at k = 0 the vacuum is not counted).
Axes in the colleague's convention: Delta E_n(k) / 2 sqrt(x) = (E_n(k) - E_0(k=0)) / 2 sqrt(x) = (sqrt(x)/2) * gap,
against k/e = 2 pi sqrt(x) j / N with j the integer (block) momentum, j = 0 .. N/4 (zone edge k/e = pi sqrt(x)/2).
Usage: plot_levels_conv.py X [--separate]    X = 4 (N = 384, D = 14), 16 (N = 128, D = 16) or 90 (N = 152, D = 18).
Input: bloch_levels.csv. Output: figures/fig_levels_conv_x{X}_N{N}_D{D}.png (all levels); with --separate one figure
per level, figures/fig_levels_conv_x{X}_N{N}_D{D}_n{n}.png, all on the same axes ranges."""
import csv, os, sys
from math import pi, sqrt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
RUNS = {4.0: (384, 14, "xa_N384_D14"), 16.0: (128, 16, "xa16_N128_D16"), 90.0: (152, 18, "xa90_N152_D18")}   # x -> (N, D, file prefix)
X = float(sys.argv[1]); N, D, PRE = RUNS[X]
def MS(n, p): return np.sqrt(n ** 2 / pi + np.asarray(p) ** 2)

lev = {}
for r in csv.DictReader(open(os.path.join(root, "bloch_levels.csv"))):
    if r["file"].startswith(PRE) and int(r["N"]) == N and int(r["D"]) == D and float(r["x"]) == X:
        j, l = int(r["k"]), int(r["level"])
        if j > N // 4: continue                  # momenta up to j = N/4 only
        n = l if j == 0 else l + 1               # n-th excitation; the k = 0 vacuum (level 0) is dropped
        if n >= 1: lev.setdefault(n, {})[j] = sqrt(X) / 2 * float(r["gap"])
kmax = 2 * pi * sqrt(X) * (N // 4) / N
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#b88000", "#8e5cc7", "#d6457a"]   # validated: light mode, adjacent CVD >= 9
pc = np.linspace(0, kmax, 600)
ms = 2.6 if N // 4 > 60 else 3.6
top = max(max(max(v.values()) for v in lev.values()), float(MS(max(lev), kmax)))
def piticks(kmax):   # multiples of pi/2 up to 2 pi, else of pi
    st = 1 if kmax <= 2 * pi + 0.1 else 2
    lab = lambda h: "0" if h == 0 else ("π/2" if h == 1 else ("3π/2" if h == 3 else (f"{h // 2}π" if h > 2 else "π")) if h % 2 else (f"{h // 2}π" if h > 2 else "π"))
    hs = list(range(0, int(kmax / (pi / 2) + 1e-9) + 1, st))
    return [h * pi / 2 for h in hs], [lab(h) for h in hs]
ticks, tlabs = piticks(kmax)

def draw(levels, title, out):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for n in levels:
        c = COL[(n - 1) % len(COL)]
        ax.plot(pc, MS(n, pc), color=c, lw=1.3, ls="--", zorder=2, label=f"$M_S(n={n},k)$")
        js = sorted(lev[n]); p = np.array([2 * pi * sqrt(X) * j / N for j in js])
        ax.plot(p, [lev[n][j] for j in js], lw=0, marker="o", ms=ms, color=c, zorder=3, label=f"puMPS, n = {n}")
    ax.set_ylim(0, 1.05 * top); ax.set_xlim(-0.03 * kmax, 1.03 * kmax)
    ax.set_xticks(ticks, tlabs)
    ax.set_xlabel(r"$k/e = 2\pi\sqrt{x}\,j/N$")
    ax.set_ylabel(r"$\Delta E_n(k)\,/\,2\sqrt{x} = (E_n(k) - E_0(k{=}0))\,/\,2\sqrt{x}$")
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2 if len(levels) > 1 else 1)
    ax.grid(color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig); print(out)

head = f"Massless Schwinger: x = {X:g}, m/e = 0, θ = 0;  " + r"$M_S(n,k) = \sqrt{n^2/\pi + (k/e)^2}$" + "\n"
tag = f"puMPS one insertion, N = {N}, D = {D}"
base = os.path.join(root, "figures", f"fig_levels_conv_x{X:g}_N{N}_D{D}")
if "--separate" in sys.argv:
    for n in sorted(lev): draw([n], head + tag + f", level n = {n}", f"{base}_n{n}.png")
else:
    draw(sorted(lev), head + tag, base + ".png")
