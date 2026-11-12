#!/usr/bin/env python3
"""Massless Schwinger (m/e = 0, theta = 0, hmax = 3) puMPS dispersion in the colleague's convention:
    y = Delta E_n(k) / (2 sqrt(x)) = (E_n(k) - E_0(k=0)) / (2 sqrt(x)),   x-axis k/e = 2 pi sqrt(x) n / N,
with E in the Hamiltonian H_W = y Hc + mu Hm + x Hxy (ours is H_W / x, so Delta E_W / (2 sqrt x) = (sqrt(x)/2) * gap).
n is the integer momentum index (2-site cell); the lowest level in each momentum sector is plotted (the one-boson
branch; at k = 0 the lowest level is the vacuum, so its first excited level is used).
Usage: plot_dispersion_conv.py X [--continuum] [--to-pi]     X = 4 (N = 384, D = 14), 16 (N = 128, D = 16) or 90 (N = 152, D = 18).
--continuum adds the exact continuum curve sqrt(1/pi + (k/e)^2) and a deviation panel; --to-pi keeps 0 <= k/e <= pi only.
Input: bloch_levels.csv. Output: figures/fig_dispersion_conv_x{X}_N{N}_D{D}[_continuum][_topi].png"""
import csv, os, sys
from math import pi, sqrt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
RUNS = {4.0: (384, 14, "xa_N384_D14"), 16.0: (128, 16, "xa16_N128_D16"), 90.0: (152, 18, "xa90_N152_D18")}   # x -> (N, D, file prefix)
X = float(sys.argv[1]); CONT = "--continuum" in sys.argv; TOPI = "--to-pi" in sys.argv
N, D, PRE = RUNS[X]
def cont(p): return np.sqrt(1 / pi + np.asarray(p) ** 2)

lev = {}
for r in csv.DictReader(open(os.path.join(root, "bloch_levels.csv"))):
    k, l = int(r["k"]), int(r["level"])
    if r["file"].startswith(PRE) and int(r["N"]) == N and int(r["D"]) == D and float(r["x"]) == X and l == (1 if k == 0 else 0):
        lev[k] = sqrt(X) / 2 * float(r["gap"])   # = (E_W(k) - E_W,0(k=0)) / (2 sqrt x)
ns = [n for n in sorted(lev) if not TOPI or 2 * pi * sqrt(X) * n / N <= pi + 1e-9]; kk = np.array([2 * pi * sqrt(X) * n / N for n in ns]); ee = np.array([lev[n] for n in ns])
def piticks(kmax):   # multiples of pi/2 up to 2 pi, else of pi
    st = 1 if kmax <= 2 * pi + 0.1 else 2
    lab = lambda h: "0" if h == 0 else ("π/2" if h == 1 else ("3π/2" if h == 3 else (f"{h // 2}π" if h > 2 else "π")) if h % 2 else (f"{h // 2}π" if h > 2 else "π"))
    hs = list(range(0, int(kmax / (pi / 2) + 1e-9) + 1, st))
    return [h * pi / 2 for h in hs], [lab(h) for h in hs]
if TOPI: kmax = pi; ticks, tlabs = [h * pi / 4 for h in range(5)], ["0", "π/4", "π/2", "3π/4", "π"]
else: kmax = kk[-1]; ticks, tlabs = piticks(kmax)
STY = dict(color="#2a78d6", lw=0, marker="o", ms=3.2 if len(ns) > 60 else 4.0, zorder=3)
title = f"Massless Schwinger dispersion: x = {X:g}, m/e = 0, θ = 0\npuMPS one insertion, N = {N}, D = {D}"
ylab = r"$\Delta E_n(k)\,/\,2\sqrt{x} = (E_n(k) - E_0(k{=}0))\,/\,2\sqrt{x}$"
xlab = r"$k/e = 2\pi\sqrt{x}\,n/N$"
if CONT:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.6, 6.8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    pc = np.linspace(0, kmax, 600)
    ax.plot(pc, cont(pc), color="#555555", lw=1.4, ls="--", zorder=1, label=r"continuum (exact): $\sqrt{1/\pi + (k/e)^2}$")
    ax.plot(kk, ee, label="puMPS", **STY); ax.legend(fontsize=8, frameon=False, loc="upper left")
    bx.axhline(0, color="#555555", lw=1.0, ls="--", zorder=1)
    bx.plot(kk, 100 * (ee / cont(kk) - 1), **STY)
    bx.set_ylabel("puMPS / continuum − 1 (%)"); axes = (ax, bx); last = bx
    ax.set_ylim(0, 1.05 * float(cont(kmax)))
else:
    fig, ax = plt.subplots(figsize=(6.6, 4.6)); ax.plot(kk, ee, **STY); axes = (ax,); last = ax
    ax.set_ylim(0, 1.12 * ee.max())
ax.set_ylabel(ylab); ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
last.set_xlabel(xlab); last.set_xlim(-0.03 * kmax, 1.03 * kmax); last.set_xticks(ticks, tlabs)
for a in axes:
    a.grid(color="#eeeeee", lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(root, "figures", f"fig_dispersion_conv_x{X:g}_N{N}_D{D}{'_continuum' if CONT else ''}{'_topi' if TOPI else ''}.png")
fig.savefig(out, dpi=200); print(out, len(ns), "momenta")
