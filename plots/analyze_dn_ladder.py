#!/usr/bin/env python3
"""D(N) ladder: scaled k=0 gaps N*Delta1, N*Delta2 (and the k=1 gap) versus D at fixed N,
with the relative change per D step as the convergence proxy (calibrated at N=20/28 against ED,
where the D->D+2 change is about the error at the smaller D). Also tabulates ground-run cost.
Input: bloch_levels.csv (export_bloch.jl) and runs/*.log.
"""
import csv, os, re, glob
from collections import defaultdict
root = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(root, 'bloch_levels.csv'))))
spec = defaultdict(dict)   # (N, D, tag) -> {(k,lvl): gap}
for r in rows:
    tag = r['file'].replace('.jls', '')
    spec[(int(r['N']), int(r['D']), tag)][(int(r['k']), int(r['level']))] = float(r['gap'])
print('scaled k=0 gaps vs D (x=10, m/e=.33, theta=pi, hmax=3); dX = relative change from the previous D at this N')
print(f"{'N':>3} {'D':>3} {'file':<22} {'N*D1':>8} {'dD1':>8} {'N*D2':>8} {'dD2':>8} {'N*D3':>8} {'k1 gap':>9} {'dk1':>8}")
byN = defaultdict(list)
for (N, D, tag), g in sorted(spec.items()):
    if 'tight' in tag or '_tr' in tag:  # alternates are listed but not used as the previous-D reference
        pass
    d1 = g.get((0, 1)); d2 = g.get((0, 2)); d3 = g.get((0, 3)); k1 = g.get((1, 0))
    byN[N].append((D, tag, d1, d2, d3, k1))
for N in sorted(byN):
    prev = None
    for D, tag, d1, d2, d3, k1 in sorted(byN[N]):
        def rel(a, b): return f'{(a-b)/b:+.3%}' if (a is not None and b is not None) else ''
        if prev and prev[0] == D:  # alternate route at the same D: compare with the main one
            ref = prev
            print(f"{N:>3} {D:>3} {tag:<22} {N*d1:>8.4f} {rel(d1,ref[2]):>8} {N*d2:>8.4f} {rel(d2,ref[3]):>8} {N*d3:>8.4f} {k1:>9.5f} {rel(k1,ref[5]):>8}   (vs same-D main)")
            continue
        print(f"{N:>3} {D:>3} {tag:<22} {N*d1:>8.4f} {rel(d1,prev[2]) if prev else '':>8} {N*d2:>8.4f} {rel(d2,prev[3]) if prev else '':>8} {N*d3:>8.4f} {k1:>9.5f} {rel(k1,prev[5]) if prev else '':>8}")
        prev = (D, tag, d1, d2, d3, k1)
# ground-run costs
print('\nground runs (L-BFGS continuation / padded growth):')
end_re = re.compile(r'\[lbfgs\] ([^:\n]+): E/N (-?[\d.]+) -> (-?[\d.]+) nGrad=(\d+) nHN=(\d+)')
tot_re = re.compile(r'total=([\d.]+) seconds|seconds=([\d.]+) eta=')
print(f"{'run':<18} {'E/N end':>16} {'nGrad':>6} {'nHN':>6} {'wall':>9}  reason")
for f in sorted(glob.glob(os.path.join(root, 'runs', 'n[2468]*_D1[0246]*.log'))):
    txt = open(f).read(); e = end_re.search(txt); t = tot_re.search(txt)
    if not e: continue
    sec = float(t.group(1) or t.group(2)) if t else float('nan')
    print(f"{os.path.basename(f)[:-4]:<18} {float(e.group(3)):>16.12f} {e.group(4):>6} {e.group(5):>6} {sec/3600:>8.2f}h  {e.group(1)}")

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for N in sorted(byN):
        pts = sorted((D, d1, d2) for D, tag, d1, d2, d3, k1 in byN[N] if 'tight' not in tag and '_tr' not in tag and tag.endswith('h3'))
        axes[0].plot([p[0] for p in pts], [N*p[1] for p in pts], 'o-', label=f'N={N}')
        axes[1].plot([p[0] for p in pts], [N*p[2] for p in pts], 'o-', label=f'N={N}')
    for ax, lab in zip(axes, [r'$N\,\Delta_1$', r'$N\,\Delta_2$']):
        ax.set_xlabel('D'); ax.set_ylabel(lab + ' (k=0)'); ax.grid(alpha=.3); ax.legend()
    # ED references at N=20, 28
    for N, d1, d2 in [(20, 0.079309, 0.498945), (28, 0.057565, 0.390141)]:
        axes[0].axhline(N*d1, ls=':', color='gray', lw=.8); axes[1].axhline(N*d2, ls=':', color='gray', lw=.8)
    fig.suptitle('x=10, m/e=0.33, theta=pi: scaled k=0 gaps vs bond dimension (dotted: ED at N=20, 28)')
    fig.tight_layout(); fig.savefig(os.path.join(root, 'dn_ladder.png'), dpi=130); print('\nwrote dn_ladder.png')
except Exception as e:
    print('plot skipped:', e)
