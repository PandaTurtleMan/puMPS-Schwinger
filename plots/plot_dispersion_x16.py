#!/usr/bin/env python3
"""Massless Schwinger (m/e = 0, theta = 0, hmax = 3) dispersion at two lattice spacings against the exact continuum
result E/e = sqrt(1/pi + (p/e)^2) (free boson of mass e/sqrt(pi)):
  x = 16 (a e = 0.25): one-insertion puMPS, N = 128, D = 16, all 33 reduced momenta (run_x16.sh, files xa16_*);
  x = 4  (a e = 0.50): one-insertion puMPS, N = 384, D = 14, all 97 reduced momenta (files xa_N384_D14*).
Units: E/e = (sqrt(x)/2) * gap; p/e = 2 pi sqrt(x) k / N. Zone edge p/e = pi sqrt(x) / 2 (pi at x=4, 2 pi at x=16).
Bottom panel: relative deviation from the continuum, the O(a^2) lattice artefact; also printed: the D=14 -> 16 change
at x = 16 (one-insertion convergence). Input: bloch_levels.csv. Output: figures/fig_dispersion_x4_x16_vs_continuum.png;
with --x16-only, the x = 16 band alone in figures/fig_dispersion_x16_vs_continuum.png"""
import csv, os, sys
from math import pi, sqrt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
RUNS = {16.0: (128, 16, "xa16_"), 4.0: (384, 14, "xa_N384_D14")}   # x -> (N, D, file prefix)
def cont(p): return np.sqrt(1 / pi + np.asarray(p) ** 2)

band = {x: {} for x in RUNS}; d14 = {}
for r in csv.DictReader(open(os.path.join(root, "bloch_levels.csv"))):
    x, k, lv = float(r["x"]), int(r["k"]), int(r["level"])
    if x not in RUNS or float(r["me"]) != 0 or float(r["theta"]) != 0 or lv != (1 if k == 0 else 0): continue
    N, D, pre = RUNS[x]
    if not r["file"].startswith(pre) or int(r["N"]) != N: continue
    e = sqrt(x) / 2 * float(r["gap"])
    if int(r["D"]) == D: band[x][k] = e
    elif x == 16.0 and int(r["D"]) == 14: d14[k] = e
pts = {}
for x, (N, D, _) in RUNS.items():
    ks = sorted(band[x]); pts[x] = (np.array([2 * pi * sqrt(x) * k / N for k in ks]), np.array([band[x][k] for k in ks]))

ONLY16 = "--x16-only" in sys.argv   # drop the x = 4 band
STY = {16.0: dict(color="#2a78d6", marker="o", ms=4.0), 4.0: dict(color="#eb6834", marker="s", ms=2.8)}
pc = np.linspace(0, 2 * pi, 600)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.8, 6.8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax.plot(pc, cont(pc), color="#555555", lw=1.4, ls="--", zorder=1, label=r"continuum (exact): $\sqrt{1/\pi + (p/e)^2}$")
bx.axhline(0, color="#555555", lw=1.0, ls="--", zorder=1)
for x in ((16.0,) if ONLY16 else (16.0, 4.0)):
    N, D, _ = RUNS[x]; p, e = pts[x]
    ax.plot(p, e, lw=0, zorder=3, label=f"puMPS, x = {x:g} (a e = {1 / sqrt(x):g}), N = {N}, D = {D}", **STY[x])
    bx.plot(p, 100 * (e / cont(p) - 1), lw=0, zorder=3, **STY[x])
ax.set_ylabel("E / e"); ax.set_ylim(0, 6.6); ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("Massless Schwinger dispersion vs continuum: m/e = 0, θ = 0" + ("\nx = 16 (a e = 0.25)" if ONLY16 else ""),
             fontsize=10, fontweight="bold", loc="left")
bx.set_ylabel("puMPS / continuum − 1 (%)"); bx.set_xlabel("p / e"); bx.set_xlim(-0.12, 2 * pi + 0.12)
bx.set_xticks([0, pi / 2, pi, 3 * pi / 2, 2 * pi], ["0", "π/2", "π", "3π/2", "2π"])
for a in (ax, bx):
    a.grid(color="#eeeeee", lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(root, "figures", "fig_dispersion_x16_vs_continuum.png" if ONLY16 else "fig_dispersion_x4_x16_vs_continuum.png"); fig.savefig(out, dpi=200); print(out)
print(" p/e     x=4 dev    x=16 dev   x=16 D14->16")
for k16 in range(0, 33, 4):
    p = 8 * pi * k16 / 128; k4 = round(p * 384 / (4 * pi))
    d4 = f"{100 * (band[4.0][k4] / float(cont(p)) - 1):+7.3f}%" if k4 in band[4.0] and abs(4 * pi * k4 / 384 - p) < 1e-9 else "     -  "
    print(f"{p:6.4f}  {d4}  {100 * (band[16.0][k16] / float(cont(p)) - 1):+7.3f}%  {100 * (band[16.0][k16] / d14[k16] - 1):+.4f}%")
