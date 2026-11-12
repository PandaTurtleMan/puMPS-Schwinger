#!/usr/bin/env python3
"""Figures for the gapped benchmark (stage 1 + 2 + 2b + pair panel). Run from the capsule mirror after
syncing (bloch_levels.csv from export_bloch.jl; runs/*.log, bloch/*.log, ed/*/ed_levels.csv, pairs/*/branches.csv).
Writes figures/fig*.png. Colors: one fixed hue per point (xa blue, xb orange, xc aqua); N or D never recolor a point.
Cost figures show wall time on the shared 16-thread node (8 physical cores, 4 BLAS threads); rows at N >= 96 ran
partly under CPU sharing (see REPORT.md incidents), so gradient counts and seconds-per-gradient are shown alongside.
"""
import csv, os, re, glob, statistics, math
from collections import defaultdict
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

root = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(root, "figures"); os.makedirs(out, exist_ok=True)
TAGS = {"xa": "x=4, m/e=0, θ=0", "xb": "x=10, m/e=0.30, θ=0", "xc": "x=10, m/e=0.25, θ=π"}
COL = {"xa": "#2a78d6", "xb": "#eb6834", "xc": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 1.6, "lines.markersize": 5, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
MARK = {"xa": "o", "xb": "s", "xc": "^"}   # secondary encoding, never color-alone

# ---------------- data ----------------
spec = defaultdict(dict); ground = {}
for r in csv.DictReader(open(os.path.join(root, "bloch_levels.csv"))):
    m = re.match(r"(x[abc])_N(\d+)_D(\d+)\.jls", r["file"])
    if not m: continue
    key = (m.group(1), int(m.group(2)), int(m.group(3)))
    spec[key][(int(r["k"]), int(r["level"]))] = float(r["gap"]); ground[key] = float(r["ground_E"])
ed = defaultdict(dict)
for f in glob.glob(os.path.join(root, "ed", "x?_N*", "ed_levels.csv")):
    tag = os.path.basename(os.path.dirname(f)).split("_")[0]
    for r in csv.DictReader(open(f)):
        ed[(tag, int(r["N"]), int(r["k"]))][int(r["level"])] = float(r["E"])
def branches(g):
    return {(k, lvl if k == 0 else lvl + 1): gap for (k, lvl), gap in g.items() if 1 <= (lvl if k == 0 else lvl + 1) <= 4}
end_re = re.compile(r"\[lbfgs\] ([^:\n]+): E/N (-?[\d.]+) -> (-?[\d.]+) nGrad=(\d+) nHN=(\d+)")
tot_re = re.compile(r"total=([\d.]+) seconds|seconds=([\d.]+) eta=")
cost = {}
for (t, N, D) in spec:
    name = f"{t}_N{N}_D{D}"; f = os.path.join(root, "runs", name + ".log")
    txt = open(f).read() if os.path.exists(f) else ""
    e = end_re.search(txt); tm = tot_re.search(txt)
    bl = os.path.join(root, "bloch", name + ".log"); bt = open(bl).read() if os.path.exists(bl) else ""
    asm = re.search(r"assembly ([\d.]+) s; solve ([\d.]+) s; elapsed ([\d.]+) s", bt)
    el = re.search(r"elapsed ([\d.]+) s", bt)
    cost[(t, N, D)] = dict(nGrad=int(e.group(4)) if e else None, nHN=int(e.group(5)) if e else None,
                           wall=float(tm.group(1) or tm.group(2)) if tm else None,
                           spec=float(el.group(1)) if el else None,
                           asm=float(asm.group(1)) if asm else None, solve=float(asm.group(2)) if asm else None)
def ed_errors(tag, N, D):
    e0 = ed.get((tag, N, 0), {}).get(0)
    if e0 is None or (tag, N, D) not in spec: return None, []
    errs = []
    for (k, b), gap in branches(spec[(tag, N, D)]).items():
        edk = ed.get((tag, N, k), {}); target = b if k == 0 else b - 1
        if target in edk: errs.append(abs(gap - (edk[target] - e0)) / (edk[target] - e0))
    return (ground[(tag, N, D)] - e0) / N, errs
def Ds(tag, N): return sorted(D for (t, n, D) in spec if t == tag and n == N)
def Ns(tag): return sorted({n for (t, n, D) in spec if t == tag})
def legend_points(ax, loc="best"):
    ax.legend([plt.Line2D([], [], color=COL[t], marker=MARK[t], lw=1.6) for t in TAGS], [TAGS[t] for t in TAGS], loc=loc, fontsize=8)

# ---------------- fig 1: ground-state error vs ED ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, N in zip(axes, (20, 28)):
    for t in TAGS:
        xs, ys = [], []
        for D in Ds(t, N):
            dE, _ = ed_errors(t, N, D)
            if dE is not None: xs.append(D); ys.append(abs(dE))
        ax.plot(xs, ys, marker=MARK[t], color=COL[t], label=TAGS[t])
    ax.set_yscale("log"); ax.set_xlabel("bond dimension D"); ax.set_title(f"N = {N}")
    ax.set_xticks([8, 10, 12, 14])
axes[0].set_ylabel("|E − E_ED| / N")
fig.legend([plt.Line2D([], [], color=COL[t], marker=MARK[t]) for t in TAGS], list(TAGS.values()), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Ground-state energy error per site vs exact diagonalization", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(out, "fig1_ground_error_vs_ED.png")); plt.close(fig)

# ---------------- fig 2: gap errors vs ED ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, N in zip(axes, (20, 28)):
    for t in TAGS:
        xs, med, mx = [], [], []
        for D in Ds(t, N):
            _, errs = ed_errors(t, N, D)
            if errs: xs.append(D); med.append(statistics.median(errs)); mx.append(max(errs))
        ax.plot(xs, med, marker=MARK[t], color=COL[t])
        ax.plot(xs, mx, marker=MARK[t], color=COL[t], ls=":", lw=1.1, mfc="white")
    ax.set_yscale("log"); ax.set_xlabel("bond dimension D"); ax.set_title(f"N = {N}"); ax.set_xticks([8, 10, 12, 14])
    ax.axhline(0.01, color=INK2, lw=0.8, ls="--"); ax.text(14.1, 0.0105, "1 %", color=INK2, fontsize=7, va="bottom", ha="right")
axes[0].set_ylabel("relative gap error vs ED")
h = [plt.Line2D([], [], color=COL[t], marker=MARK[t]) for t in TAGS] + [plt.Line2D([], [], color=INK2, ls="-"), plt.Line2D([], [], color=INK2, ls=":", mfc="white", marker="o")]
fig.legend(h, list(TAGS.values()) + ["median over branches 1–4", "max"], loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Excitation-gap errors vs ED, branches 1–4 at all compared momenta", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(out, "fig2_gap_error_vs_ED.png")); plt.close(fig)

# ---------------- fig 3: D->D+2 proxy per branch vs N ----------------
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.8), sharey=True)
for b, ax in zip((1, 2, 3, 4), axes):
    for t in TAGS:
        xs, ys = [], []
        for N in Ns(t):
            if 12 in Ds(t, N) and 14 in Ds(t, N):
                ga, gb = branches(spec[(t, N, 12)]), branches(spec[(t, N, 14)])
                ch = [abs(gb[kb] - ga[kb]) / ga[kb] for kb in set(ga) & set(gb) if kb[1] == b]
                if ch: xs.append(N); ys.append(max(ch))
        ax.plot(xs, ys, marker=MARK[t], color=COL[t])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"branch {b}")
    ax.set_xticks([20, 28, 40, 64, 96, 128, 192, 256, 384]); ax.set_xticklabels([20, 28, 40, 64, 96, 128, 192, 256, 384], fontsize=6.5, rotation=60)
    ax.xaxis.set_minor_locator(LogLocator(subs=())); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(0.01, color=INK2, lw=0.8, ls="--"); ax.set_xlabel("N")
axes[0].set_ylabel("max |Δgap| / gap, D=12 → 14"); legend_points(axes[3], loc="lower right")
fig.suptitle("D-convergence proxy per branch is N-independent from N = 40 (dashed: 1 % certification line)", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_D_proxy_vs_N.png")); plt.close(fig)

# ---------------- fig 4: N-independence of E/N and the lowest gap ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
for t in TAGS:
    ref = {D: ground[(t, 256, D)] / 256 for D in (12, 14) if (t, 256, D) in ground}
    for D, ls in ((12, "-"), (14, "--")):
        if D not in ref: continue
        xs, ys = [], []
        for N in Ns(t):
            if (t, N, D) in ground and N != 256: xs.append(N); ys.append(abs(ground[(t, N, D)] / N - ref[D]))
        ax.plot(xs, [max(y, 1e-13) for y in ys], marker=MARK[t], color=COL[t], ls=ls, mfc="white" if D == 14 else None)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("|E/N − (E/N at N=256)|  (floor 10⁻¹³)")
ax.set_title("|E/N − E/N(256)|  (filled D=12, open D=14)")
ax.set_xticks([20, 28, 40, 64, 96, 128, 192, 384]); ax.set_xticklabels([20, 28, 40, 64, 96, 128, 192, 384], fontsize=7)
ax.xaxis.set_minor_locator(LogLocator(subs=())); ax.xaxis.set_minor_formatter(NullFormatter())
ax = axes[1]
for t in TAGS:
    xs, ys = [], []
    for N in Ns(t):
        if (t, N, 14) in spec and (0, 1) in spec[(t, N, 14)]: xs.append(N); ys.append(spec[(t, N, 14)][(0, 1)])
    ax.plot(xs, ys, marker=MARK[t], color=COL[t])
    ax.annotate(f"{ys[-1]:.5f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(0, 6), ha="right", fontsize=7, color=INK2)
ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_ylabel("lowest gap Δ₁ at D = 14")
ax.set_title("Lowest gap Δ₁ (k = 0) at D = 14")
ax.set_xticks([20, 28, 40, 64, 96, 128, 192, 256, 384]); ax.set_xticklabels([20, 28, 40, 64, 96, 128, 192, 256, 384], fontsize=7)
ax.xaxis.set_minor_locator(LogLocator(subs=())); ax.xaxis.set_minor_formatter(NullFormatter())
legend_points(ax, loc="center right")
fig.suptitle("Finite-size behaviour: N-independent beyond N = 40 at gapped points", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_N_independence.png")); plt.close(fig)

# ---------------- fig 5: costs ----------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
def nlog(ax, ticks):
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
    ax.xaxis.set_minor_locator(LogLocator(subs=())); ax.xaxis.set_minor_formatter(NullFormatter())
NT = [20, 28, 40, 64, 96, 128, 192, 256, 384, 512]
ax = axes[0, 0]
for t in TAGS:
    xs, ys = [], []
    for N in Ns(t):
        c = cost.get((t, N, 14))
        if c and c["wall"] and c["nHN"]: xs.append(N); ys.append(c["wall"] / c["nHN"])
    ax.plot(xs, ys, marker=MARK[t], color=COL[t])
xr = np.array([20, 512]); ax.plot(xr, 0.9 * xr, color=INK2, lw=0.8, ls="--"); ax.text(22, 0.9 * 22 * 1.5, "∝ N", color=INK2, fontsize=7)
nlog(ax, NT); ax.set_yscale("log"); ax.set_ylabel("seconds per gradient"); ax.set_title("Per-gradient cost vs N (D = 14)")
ax = axes[0, 1]
for t in TAGS:
    xs, ys = [], []
    for D in Ds(t, 64):
        c = cost.get((t, 64, D))
        if c and c["wall"] and c["nHN"]: xs.append(D); ys.append(c["wall"] / c["nHN"])
    ax.plot(xs, ys, marker=MARK[t], color=COL[t])
xr = np.array([8, 16]); ax.plot(xr, 1.4 * (xr / 8) ** 5, color=INK2, lw=0.8, ls="--"); ax.text(13.6, 1.4 * (13.6 / 8) ** 5 * 0.45, "∝ D⁵", color=INK2, fontsize=7)
ax.set_xscale("log"); ax.set_xticks([8, 10, 12, 14, 16]); ax.set_xticklabels([8, 10, 12, 14, 16]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yscale("log"); ax.set_ylabel("seconds per gradient"); ax.set_title("Per-gradient cost vs D (N = 64)")
ax = axes[1, 0]
for t in TAGS:
    xs, ys = [], []
    for N in Ns(t):
        c = cost.get((t, N, 14))
        if c and c["nGrad"]: xs.append(N); ys.append(c["nGrad"])
    ax.plot(xs, ys, marker=MARK[t], color=COL[t])
nlog(ax, NT); ax.set_yscale("log"); ax.set_ylabel("gradient evaluations"); ax.set_title("Gradients to tolerance 10⁻⁶ (D = 14)")
ax = axes[1, 1]
for t in TAGS:
    xs, ys = [], []
    for N in Ns(t):
        c = cost.get((t, N, 14))
        if c and c["spec"]: xs.append(N); ys.append(c["spec"] / 3600)
    ax.plot(xs, ys, marker=MARK[t], color=COL[t])
nlog(ax, NT); ax.set_yscale("log"); ax.set_ylabel("wall time (h)"); ax.set_title("Spectrum wall time (D = 14)")
for ax in axes.flat: ax.set_xlabel("N" if ax is not axes[0, 1] else "D")
legend_points(axes[1, 0], loc="upper right")
fig.suptitle("Cost scaling on the shared node (4 BLAS threads; N ≥ 96 partly under CPU sharing)", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_costs.png")); plt.close(fig)

# ---------------- fig 6: pair panel (levels reached) ----------------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
for ax, t in zip(axes, TAGS):
    f = os.path.join(root, "pairs", f"{t}_N28_D10_q4", "branches.csv")
    if not os.path.exists(f): ax.set_title(TAGS[t] + " (missing)"); continue
    lv = defaultdict(dict)
    for r in csv.DictReader(open(f)): lv[(r["ansatz"], int(r["k"]))][int(r["level"])] = float(r["E"])
    e0 = ed[(t, 28, 0)][0]
    labels, one, pair = [], [], []
    for k in (0, 1):
        for b in (1, 2, 3, 4):
            tgt = b if k == 0 else b - 1
            edgap = ed[(t, 28, k)][tgt] - e0
            for name, store in (("one", one), ("pair", pair)):
                g0 = lv[(name, 0)][0]; gap = lv[(name, k)][b if k == 0 else b - 1] - g0
                store.append(abs(gap - edgap) / edgap)
            labels.append(f"k={k}\nb{b}")
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, [max(v, 1e-5) for v in one], w, color="white", edgecolor=COL[t], hatch="////", lw=1.0, label="one-insertion")
    ax.bar(x + w / 2, [max(v, 1e-5) for v in pair], w, color=COL[t], lw=0, label="local + pair (q = 4)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5); ax.set_title(TAGS[t])
    ax.axhline(0.01, color=INK2, lw=0.8, ls="--"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("relative gap error vs ED"); axes[0].set_ylim(1e-5, 0.5)
axes[2].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Levels reached at N = 28, D = 10: adding the pair basis resolves the slow two-particle-like levels", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_pair_panel.png")); plt.close(fig)

# ---------------- fig 7: dispersion at N = 28 vs ED ----------------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=False)
for ax, t in zip(axes, TAGS):
    e0 = ed[(t, 28, 0)][0]; g = spec[(t, 28, 14)]
    ks = sorted({k for (k, l) in g})
    for k in range(8):
        edk = ed.get((t, 28, k), {})
        ax.plot([k] * len([l for l in edk if 1 <= (l if k == 0 else l + 1) <= 5]),
                [edk[l] - e0 for l in sorted(edk) if 1 <= (l if k == 0 else l + 1) <= 5], ls="none", marker="o", mfc="none", mec=INK, ms=7, mew=0.9)
    for k in ks:
        pts = [(k, gap) for (kk, l), gap in g.items() if kk == k and 1 <= (l if k == 0 else l + 1) <= 5]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=MARK[t], color=COL[t], ms=4)
    ax.set_xlabel("momentum index k (half zone)"); ax.set_title(TAGS[t]); ax.set_xticks(range(8))
axes[0].set_ylabel("E − E₀")
axes[2].legend([plt.Line2D([], [], ls="none", marker="o", mfc="none", mec=INK, ms=7), plt.Line2D([], [], ls="none", marker="o", color=INK2, ms=4)],
               ["ED (levels 1–5)", "one-insertion, D = 14"], fontsize=7.5, loc="lower right")
fig.suptitle("Dispersion at N = 28: one-insertion spectrum at D = 14 against ED", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_dispersion_N28.png")); plt.close(fig)

print("wrote", sorted(os.listdir(out)))

# ---------------- fig 8: calibrated unit costs (idle box) and calibrated production cost ----------------
bg, ba = os.path.join(root, "bench", "ground.csv"), os.path.join(root, "bench", "assembly.csv")
if os.path.exists(bg) and os.path.exists(ba):
    G = [r for r in csv.DictReader(open(bg))]; A = [r for r in csv.DictReader(open(ba))]
    tg = {(r["tag"], int(r["N"]), int(r["D"])): float(r["t_median"]) for r in G}
    ta = {(int(r["N"]), int(r["D"])): float(r["t_median"]) for r in A}
    def powfit(rows):   # log t = a + b log N + c log D, least squares over all rows [(N, D, t)]
        X = np.array([[1.0, math.log(N), math.log(D)] for N, D, _ in rows]); y = np.array([math.log(t) for _, _, t in rows])
        a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]; return a, b, c
    ga, gb, gc_ = powfit([(N, D, t) for (t_, N, D), t in tg.items()])
    aa, ab, ac = powfit([(N, D, t) for (N, D), t in ta.items()])
    tgrad = lambda N, D: math.exp(ga) * N ** gb * D ** gc_
    tasm = lambda N, D: math.exp(aa) * N ** ab * D ** ac
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    NT2 = [20, 28, 40, 64, 96, 128, 192, 256, 384, 512]
    ax = axes[0, 0]
    for t in TAGS:
        pts = sorted((N, v) for (tt, N, D), v in tg.items() if tt == t and D == 14)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[t], color=COL[t], ls="none")
    xr = np.array([20, 512]); ax.plot(xr, [tgrad(x, 14) for x in xr], color=INK2, lw=0.9, ls="--")
    ax.text(20, 60, f"fit ∝ N^{gb:.2f} D^{gc_:.2f}", color=INK2, fontsize=7)
    nlog(ax, NT2); ax.set_yscale("log"); ax.set_ylabel("seconds per gradient"); ax.set_title("Gradient cost vs N, D = 14")
    ax = axes[0, 1]
    for t in TAGS:
        pts = sorted((D, v) for (tt, N, D), v in tg.items() if tt == t and N == 64)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[t], color=COL[t], ls="none")
    xr = np.array([8, 14]); ax.plot(xr, [tgrad(64, d) for d in xr], color=INK2, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_xticks([8, 10, 12, 14]); ax.set_xticklabels([8, 10, 12, 14]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log"); ax.set_ylabel("seconds per gradient"); ax.set_title("Gradient cost vs D, N = 64")
    ax = axes[1, 0]
    ramp = {8: "#9cc3ec", 10: "#5f9ee3", 12: "#2a78d6", 14: "#1a4f93"}
    for D in (8, 10, 12, 14):
        pts = sorted((N, v) for (N, d), v in ta.items() if d == D)
        if pts: ax.plot([p[0] for p in pts], [p[1] / 60 for p in pts], marker="o", color=ramp[D], label=f"D = {D}")
    xr = np.array([20, 96]); ax.plot(xr, [tasm(x, 12) / 60 for x in xr], color=INK2, lw=0.9, ls="--")
    ax.text(20, 14, f"fit ∝ N^{ab:.2f} D^{ac:.2f}", color=INK2, fontsize=7)
    nlog(ax, [20, 28, 40, 64, 96]); ax.set_yscale("log"); ax.set_ylabel("assembly time (min)"); ax.set_title("Spectrum assembly vs N (point-independent)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1, 1]
    for t in TAGS:
        xs, ys = [], []
        for N in Ns(t):
            c = cost.get((t, N, 14))
            if c and c["nHN"] and (t, N, 14) in tg: xs.append(N); ys.append(c["nHN"] * tg[(t, N, 14)] / 3600)
        ax.plot(xs, ys, marker=MARK[t], color=COL[t])
    xr = np.array([20, 512]); ax.plot(xr, [tasm(x, 14) / 3600 for x in xr], color=INK2, lw=0.9, ls=":")
    ax.text(150, tasm(150, 14) / 3600 * 0.35, "spectrum assembly (fit)", color=INK2, fontsize=7)
    nlog(ax, NT2); ax.set_yscale("log"); ax.set_ylabel("hours"); ax.set_title("Ground wall = nHN × t_grad, D = 14")
    for ax in axes.flat: ax.set_xlabel("D" if ax is axes[0, 1] else "N")
    legend_points(axes[0, 1], loc="lower right")
    fig.suptitle("Unit costs measured on an idle box (4 BLAS threads, dedicated physical cores)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig8_costs_calibrated.png")); plt.close(fig)
    print(f"fits: t_grad = {math.exp(ga):.3e} N^{gb:.3f} D^{gc_:.3f} s ; t_assembly = {math.exp(aa):.3e} N^{ab:.3f} D^{ac:.3f} s")

# ---------------- fig 9: levels reached (stage 4, 12-level ED) ----------------
ed12 = defaultdict(dict)
for f in glob.glob(os.path.join(root, "ed12", "x?_N*", "ed_levels.csv")):
    tg_, N_ = re.match(r"(x[abc])_N(\d+)", os.path.basename(os.path.dirname(f))).groups()
    for r in csv.DictReader(open(f)): ed12[(tg_, int(N_), int(r["k"]))][int(r["level"])] = float(r["E"])
one12 = defaultdict(dict); b12 = os.path.join(root, "bloch12_levels.csv")
if os.path.exists(b12):
    for r in csv.DictReader(open(b12)):
        m = re.match(r"(x[abc])_N(\d+)_D(\d+)\.jls", r["file"])
        if m: one12[(m.group(1), int(m.group(2)), int(r["k"]))][int(r["level"])] = float(r["E"])
pair12 = defaultdict(dict)
for f in glob.glob(os.path.join(root, "pairs12", "x?_N28_D10_q4", "branches.csv")):
    tg_ = os.path.basename(os.path.dirname(f))[:2]
    for r in csv.DictReader(open(f)):
        if r["ansatz"] == "pair": pair12[(tg_, 28, int(r["k"]))][int(r["level"])] = float(r["E"])
def lvl_errors(src, tag, N, k):
    e0 = ed12.get((tag, N, 0), {}).get(0); d = src.get((tag, N, k)); g0 = src.get((tag, N, 0), {}).get(0)
    if e0 is None or not d or g0 is None: return None
    edk = ed12[(tag, N, k)]; levels = [j for j in sorted(edk) if (j >= 1 if k == 0 else True)][:11]
    return [abs((d[j] - g0) - (edk[j] - e0)) / (edk[j] - e0) if j in d else float("nan") for j in levels]
if ed12 and one12:
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharey=True)
    kramp = ["#9cc3ec", "#7fb0e6", "#5f9ee3", "#3f8bdc", "#2a78d6", "#1a4f93"]
    x = np.arange(1, 12); w = 0.38
    for j, t in enumerate(TAGS):
        ax = axes[0, j]
        e1 = lvl_errors(one12, t, 28, 0); e2 = lvl_errors(pair12, t, 28, 0)
        if e1: ax.bar(x - w / 2, [max(v, 1e-5) for v in e1], w, color="white", edgecolor=COL[t], hatch="////", lw=1.0, label="one-insertion, D = 14")
        if e2: ax.bar(x + w / 2, [max(v, 1e-5) for v in e2], w, color=COL[t], lw=0, label="local + pair, D = 10, q = 4")
        ax.set_yscale("log"); ax.set_title(f"{TAGS[t]}\nN = 28, k = 0"); ax.grid(axis="x", visible=False)
        ax.axhline(0.01, color=INK2, lw=0.8, ls="--"); ax.axhline(0.001, color=INK2, lw=0.6, ls=":"); ax.set_xticks(x)
        if j == 0: ax.legend(fontsize=7, loc="upper left")
        ax = axes[1, j]
        for i, k in enumerate(range(6)):
            e = lvl_errors(one12, t, 20, k)
            if e: ax.plot(x, [max(v, 1e-5) for v in e], marker="o", ms=3.5, lw=1.1, color=kramp[i], label=f"k = {k}")
        ax.set_yscale("log"); ax.set_title("N = 20, one-insertion D = 14"); ax.set_xticks(x)
        ax.axhline(0.01, color=INK2, lw=0.8, ls="--"); ax.axhline(0.001, color=INK2, lw=0.6, ls=":"); ax.set_xlabel("excited level (ED order)")
        if j == 2: ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    axes[0, 0].set_ylabel("relative gap error vs 12-level ED"); axes[1, 0].set_ylabel("relative gap error vs 12-level ED")
    axes[0, 0].set_ylim(1e-5, 0.5)
    fig.suptitle("Levels reached: the first eleven excitations per sector against a 12-level ED (dashed 1 %, dotted 0.1 %)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig9_levels_reached.png")); plt.close(fig)
    print("wrote fig9_levels_reached.png")
