#!/usr/bin/env python3
"""Massless Schwinger dispersion (point xa: x = 4, m/e = 0, theta = 0, hmax = 3): lowest one-boson branch from the
one-insertion puMPS spectrum at N = 384, D = 14, against
  - the exact continuum result: the massless Schwinger model is a free boson of mass e/sqrt(pi), E/e = sqrt(1/pi + (p/e)^2);
  - exact lattice ED at N = 28 (same x, so same lattice spacing a e = 1/sqrt(x) = 0.5), all momenta k = 0..L/2.
Units: E/e = (sqrt(x)/2) * gap = gap at x = 4; p/e = 2 pi sqrt(x) k / N (k = momentum index of the 2-site cell).
Only the lowest branch is shown: at N = 384 the higher one-insertion levels lie inside the two-boson continuum
(threshold 2/sqrt(pi) at p = 0), which a single insertion cannot resolve on a ring this long.
Bottom panel: relative deviation from the continuum curve, i.e. the O(a^2) lattice artefact at x = 4.
Inputs: bloch_levels.csv (export_bloch.jl), ed/xa_N28/ed_levels.csv. Output: figures/fig_dispersion_xa_N384_D14.png;
with --pumps-only, the puMPS points alone in figures/fig_dispersion_xa_N384_D14_pumps.png; with --vs-ed, the puMPS band
against exact lattice ED at N = 20 and 28 (puMPS spline-interpolated to the ED momenta) in figures/fig_dispersion_xa_N384_D14_vs_ed.png;
with --vs-continuum, the puMPS band against the continuum curve alone in figures/fig_dispersion_xa_N384_D14_vs_continuum.png"""
import csv, os, sys
from math import pi, sqrt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))
X, N, D, NED = 4.0, 384, 14, 28
EU = sqrt(X) / 2                                   # gap (our H = H_W / x) -> E/e
def mom(k, n): return 2 * pi * sqrt(X) * k / n     # p/e
def cont(p): return np.sqrt(1 / pi + np.asarray(p) ** 2)

pm = {}                                            # puMPS lowest branch: k=0 level 1 (level 0 is the vacuum), else level 0
for r in csv.DictReader(open(os.path.join(root, "bloch_levels.csv"))):
    if int(r["N"]) == N and int(r["D"]) == D and float(r["x"]) == X and float(r["me"]) == 0 and float(r["theta"]) == 0:
        k, lv = int(r["k"]), int(r["level"])
        if lv == (1 if k == 0 else 0): pm[k] = EU * float(r["gap"])
ed = {}
for r in csv.DictReader(open(os.path.join(root, "ed", f"xa_N{NED}", "ed_levels.csv"))):
    ed.setdefault(int(r["k"]), {})[int(r["level"])] = float(r["E"])
e0 = ed[0][0]
edg = {k: EU * (ed[k][1 if k == 0 else 0] - e0) for k in ed}

kp, ke = sorted(pm), sorted(edg)
pp, ep = np.array([mom(k, N) for k in kp]), np.array([pm[k] for k in kp])
pe, ee = np.array([mom(k, NED) for k in ke]), np.array([edg[k] for k in ke])
pc = np.linspace(0, pi, 400)
if "--vs-continuum" in sys.argv:   # puMPS band vs the exact continuum free-boson dispersion, no ED
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax.plot(pc, cont(pc), color="#555555", lw=1.4, ls="--", zorder=1, label=r"continuum (exact): $\sqrt{1/\pi + (p/e)^2}$")
    ax.plot(pp, ep, color="#2a78d6", lw=0, marker="o", ms=3.2, zorder=3, label=f"puMPS one insertion, N = {N}, D = {D}")
    bx.axhline(0, color="#555555", lw=1.0, ls="--", zorder=1)
    bx.plot(pp, 100 * (ep / cont(pp) - 1), color="#2a78d6", lw=0, marker="o", ms=3.2, zorder=3)
    ax.set_ylabel("E / e"); ax.set_ylim(0, 3.4); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"Massless Schwinger dispersion vs continuum\nx = 4 (a e = 0.5), m/e = 0, θ = 0",
                 fontsize=10, fontweight="bold", loc="left")
    bx.set_ylabel("puMPS / continuum − 1 (%)"); bx.set_xlabel("p / e"); bx.set_xlim(-0.08, pi + 0.08)
    bx.set_xticks([0, pi / 4, pi / 2, 3 * pi / 4, pi], ["0", "π/4", "π/2", "3π/4", "π"])
    for a in (ax, bx):
        a.grid(color="#eeeeee", lw=0.6); a.set_axisbelow(True)
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    fig.tight_layout()
    out = os.path.join(root, "figures", f"fig_dispersion_xa_N{N}_D{D}_vs_continuum.png"); fig.savefig(out, dpi=200); print(out)
    for k in (0, 6, 12, 24, 36, 48, 60, 72, 84, 96):
        q = mom(k, N); print(f"k={k:2} p/e={q:.4f} puMPS={pm[k]:.6f} continuum={float(cont(q)):.6f} dev={100*(pm[k]/float(cont(q))-1):+.3f}%")
    sys.exit()
if "--vs-ed" in sys.argv:   # puMPS band vs exact lattice ED (N = 20 and 28, same x): no continuum curve
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(pp, ep)   # 97 momenta, spacing 0.033: spline error far below the differences shown
    eds = {}
    for n in (20, 28):
        lv = {}
        for r in csv.DictReader(open(os.path.join(root, "ed", f"xa_N{n}", "ed_levels.csv"))):
            lv.setdefault(int(r["k"]), {})[int(r["level"])] = float(r["E"])
        eds[n] = sorted((mom(k, n), EU * (lv[k][1 if k == 0 else 0] - lv[0][0])) for k in lv if k <= n // 4)  # half zone
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax.plot(pp, ep, color="#2a78d6", lw=0, marker="o", ms=3.2, zorder=3, label=f"puMPS one insertion, N = {N}, D = {D}")
    sty = {20: dict(color="#1baf7a", marker="D", ms=6), 28: dict(color="#eb6834", marker="s", ms=7)}
    for n, pts in eds.items():
        q, e = np.array(pts).T
        ax.plot(q, e, lw=0, mfc="white", mew=1.6, zorder=4, label=f"exact lattice ED, N = {n}", **sty[n])
        bx.plot(q, 100 * (spl(q) / e - 1), lw=1.0, mfc="white", mew=1.6, zorder=3, **sty[n])
    bx.axhline(0, color="#555555", lw=1.0, ls="--", zorder=1)
    ax.set_ylabel("E / e"); ax.set_ylim(0, 2.6); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"Massless Schwinger lattice dispersion\nx = 4 (a e = 0.5), m/e = 0, θ = 0",
                 fontsize=10, fontweight="bold", loc="left")
    bx.set_ylabel("puMPS / ED − 1 (%)"); bx.set_xlabel("p / e"); bx.set_xlim(-0.08, pi + 0.08)
    bx.set_xticks([0, pi / 4, pi / 2, 3 * pi / 4, pi], ["0", "π/4", "π/2", "3π/4", "π"])
    for a in (ax, bx):
        a.grid(color="#eeeeee", lw=0.6); a.set_axisbelow(True)
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    fig.tight_layout()
    out = os.path.join(root, "figures", f"fig_dispersion_xa_N{N}_D{D}_vs_ed.png"); fig.savefig(out, dpi=200); print(out)
    for n, pts in eds.items():
        for q, e in pts: print(f"ED N={n:2} p/e={q:.4f} E/e={e:.6f} puMPS={float(spl(q)):.6f} diff={100*(float(spl(q))/e-1):+.4f}%")
    sys.exit()
if "--pumps-only" in sys.argv:   # just the puMPS points: one panel, no ED, no continuum curve
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(pp, ep, color="#2a78d6", lw=0, marker="o", ms=3.2, zorder=3)
    ax.set_ylabel("E / e"); ax.set_xlabel("p / e"); ax.set_ylim(0, 2.6); ax.set_xlim(-0.08, pi + 0.08)
    ax.set_xticks([0, pi / 4, pi / 2, 3 * pi / 4, pi], ["0", "π/4", "π/2", "3π/4", "π"])
    ax.set_title(f"Massless Schwinger dispersion: x = 4, m/e = 0, θ = 0\npuMPS, N = {N}, D = {D}",
                 fontsize=10, fontweight="bold", loc="left")
    ax.grid(color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.tight_layout()
    out = os.path.join(root, "figures", f"fig_dispersion_xa_N{N}_D{D}_pumps.png"); fig.savefig(out, dpi=200); print(out)
    sys.exit()

fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax.plot(pc, cont(pc), color="#555555", lw=1.4, ls="--", zorder=1, label=r"continuum (exact): $\sqrt{1/\pi + (p/e)^2}$")
ax.plot(pe, ee, color="#eb6834", lw=1.0, marker="s", ms=7, mfc="white", mew=1.6, zorder=2, label=f"lattice ED, N = {NED}")
ax.plot(pp, ep, color="#2a78d6", lw=0, marker="o", ms=6, zorder=3, label=f"puMPS one insertion, N = {N}, D = {D}")
ax.set_ylabel("E / e"); ax.set_ylim(0, 3.4)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title(f"Massless Schwinger dispersion: x = 4 (a e = 0.5), m/e = 0, θ = 0", fontsize=10, fontweight="bold", loc="left")
bx.axhline(0, color="#555555", lw=1.0, ls="--", zorder=1)
bx.plot(pe, 100 * (ee / cont(pe) - 1), color="#eb6834", lw=1.0, marker="s", ms=7, mfc="white", mew=1.6, zorder=2)
bx.plot(pp, 100 * (ep / cont(pp) - 1), color="#2a78d6", lw=0, marker="o", ms=6, zorder=3)
bx.set_ylabel("deviation from\ncontinuum (%)"); bx.set_xlabel("p / e"); bx.set_xlim(-0.08, pi + 0.08)
bx.set_xticks([0, pi / 4, pi / 2, 3 * pi / 4, pi], ["0", "π/4", "π/2", "3π/4", "π"])
for a in (ax, bx):
    a.grid(color="#eeeeee", lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(root, "figures", f"fig_dispersion_xa_N{N}_D{D}.png"); fig.savefig(out, dpi=200); print(out)
print("source      k    p/e     E/e        continuum   dev")
for name, ks, n, g in (("puMPS", kp, N, pm), ("ED", ke, NED, edg)):
    for k in ks:
        p = mom(k, n); print(f"{name:6} {n:4} {k:3} {p:7.4f} {g[k]:.7f}  {float(cont(p)):.7f}  {100 * (g[k] / float(cont(p)) - 1):+7.3f}%")
