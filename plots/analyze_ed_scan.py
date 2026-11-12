#!/usr/bin/env python3
"""Critical-point location from the flux-orbit ED mass scan (x=10, theta=pi).

Reads ed/N*_h*_m*/ed_levels.csv, builds the k=0 gaps Delta1=E1-E0, Delta2=E2-E0,
and reports the scaled gap N*Delta1 versus m/e for each (N, hmax). Crossings of
N*Delta1 between consecutive N give m_c estimates (phenomenological RG); the
ratio Delta2/Delta1 at the crossing is an Ising check (epsilon/sigma = 8 in a
periodic Ising CFT if both are at k=0 and descendants are excluded).
"""
import csv, glob, os, sys, math
from collections import defaultdict
root = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in glob.glob(os.path.join(root, 'ed', 'N*_h*_m*', 'ed_levels.csv')):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if k in ('x','me','theta','E','projected_residual','full_residual','seconds') else v) for k, v in r.items()})
if not rows:
    sys.exit('no ED rows found')
lev = defaultdict(dict)   # (N,h,me,k) -> {level: E}
cert = {}
for r in rows:
    key = (int(r['N']), int(r['hmax']), r['me'], int(r['k']))
    lev[key][int(r['level'])] = r['E']
    cert[key] = r['certified'] == 'true'
out = []
for (N, h, me, k), d in sorted(lev.items()):
    if k != 0 or 0 not in d or 1 not in d:
        continue
    E0, E1 = d[0], d[1]; E2 = d.get(2, float('nan')); E3 = d.get(3, float('nan'))
    out.append(dict(N=N, hmax=h, me=me, E0_site=E0/N, D1=E1-E0, D2=E2-E0, D3=E3-E0,
                    ND1=N*(E1-E0), ND2=N*(E2-E0), ratio=(E2-E0)/(E1-E0) if E1 > E0 else float('nan'),
                    certified=cert[(N, h, me, 0)]))
with open(os.path.join(root, 'ed_scan_k0.csv'), 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('k=0 gaps (x=10, theta=pi).  ND1 = N*(E1-E0); ratio = (E2-E0)/(E1-E0)')
print(f"{'N':>3} {'h':>2} {'m/e':>5} {'E0/N':>14} {'D1':>11} {'D2':>11} {'N*D1':>9} {'N*D2':>9} {'ratio':>7} cert")
for o in out:
    print(f"{o['N']:>3} {o['hmax']:>2} {o['me']:>5.2f} {o['E0_site']:>14.9f} {o['D1']:>11.3e} {o['D2']:>11.3e} {o['ND1']:>9.4f} {o['ND2']:>9.4f} {o['ratio']:>7.3f} {'y' if o['certified'] else 'n'}")

def crossings(h):
    byN = defaultdict(list)
    for o in out:
        if o['hmax'] == h:
            byN[o['N']].append((o['me'], o['ND1']))
    Ns = sorted(byN)
    res = []
    for a, b in zip(Ns, Ns[1:]):
        ma = dict(byN[a]); mb = dict(byN[b])
        ms = sorted(set(ma) & set(mb))
        for m1, m2 in zip(ms, ms[1:]):
            f1 = ma[m1] - mb[m1]; f2 = ma[m2] - mb[m2]
            if f1 == 0 or f1 * f2 < 0:
                mc = m1 + (m2 - m1) * f1 / (f1 - f2)
                res.append((a, b, mc))
    return res
for h in sorted({o['hmax'] for o in out}):
    cr = crossings(h)
    print(f'\nhmax={h}: N*Delta1 crossings (m_c estimates between consecutive N):')
    for a, b, mc in cr:
        print(f'  N={a:>2} x N={b:>2}: m/e = {mc:.4f}')
    if not cr:
        print('  none inside the scanned range yet')
    if len(cr) >= 2:
        # pseudo-critical points m*(N_eff), N_eff = geometric mean of the pair; fit m* = m_c + a/N_eff^w
        import numpy as np
        Ne = np.array([math.sqrt(a*b) for a, b, _ in cr]); ms = np.array([mc for _, _, mc in cr])
        for w in (1.0, 2.0):
            A = np.vstack([np.ones_like(Ne), Ne**-w]).T
            (mc, a), *_ = np.linalg.lstsq(A, ms, rcond=None)
            print(f'  extrapolation m* = m_c + a/N^{w:.0f}: m_c = {mc:.4f} (a={a:+.3f})')
        if len(cr) >= 3:
            from scipy.optimize import curve_fit
            try:
                (mc, a, w), _ = curve_fit(lambda n, mc, a, w: mc + a*n**-w, Ne, ms, p0=(0.335, 1.0, 2.0), maxfev=20000)
                print(f'  free-exponent fit: m_c = {mc:.4f}, w = {w:.2f}')
            except Exception as e:
                print('  free-exponent fit failed:', e)
# hmax sensitivity
pair = defaultdict(dict)
for o in out:
    pair[(o['N'], o['me'])][o['hmax']] = o
print('\nhmax=3 vs hmax=5 (same N, m/e): shift of E0/N and of Delta1')
for (N, me), d in sorted(pair.items()):
    if 3 in d and 5 in d:
        print(f"  N={N:>2} m/e={me:.2f}: dE0/N={d[5]['E0_site']-d[3]['E0_site']:+.3e}  dD1={d[5]['D1']-d[3]['D1']:+.3e}  (D1 h5={d[5]['D1']:.4e})")
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    hs = sorted({o['hmax'] for o in out})
    fig, axes = plt.subplots(1, len(hs), figsize=(6*len(hs), 4.5), squeeze=False)
    for ax, h in zip(axes[0], hs):
        for N in sorted({o['N'] for o in out if o['hmax'] == h}):
            pts = sorted((o['me'], o['ND1']) for o in out if o['hmax'] == h and o['N'] == N)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=f'N={N}')
        ax.set_xlabel('m/e'); ax.set_ylabel(r'$N\,\Delta_1$ (k=0)'); ax.set_title(f'x=10, theta=pi, hmax={h}'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(root, 'ed_scaled_gap.png'), dpi=130)
    print('\nwrote ed_scaled_gap.png')
except Exception as e:
    print('plot skipped:', e)
