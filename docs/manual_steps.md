# Running each step by hand

`pumps.jl` (see the [README](../README.md)) runs these steps for you from one config file. This page documents
the underlying scripts, for when you need a step on its own: in a cluster job script, or to move an existing
ground state to a new mass or angle. `julia --project=. pumps.jl --dry-run CONFIG.toml` prints these commands
for a given config.

A puMPS run has three steps:

1. Compute a ground state.
2. Optionally, grow the bond dimension D (or change the mass or angle).
3. Compute the excitation spectrum from a saved ground state.

Each step reads a `.jls` file written by the step before it. All commands below are run from the `puMPS_code/`
folder.

## 0. One-time setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Julia 1.12 is required. Every command below uses `--project=.`, so the packages come from this folder's
`Project.toml`.

## 1. Ground state: `run_seed_comparison.jl` + a TOML config

Write a config file, for example `myrun/ground.toml`:

```toml
# physics
N      = 12        # number of lattice sites (even); the ring has L = N/2 two-site unit cells
x      = 10.0      # x = 1/(a e)^2, so a e = 1/sqrt(x)
me     = 0.3       # m/e
theta  = 0.0       # background angle (use 3.141592653589793 for theta = pi)
hmax   = 3         # holonomy cutoff: 2*hmax+1 sectors (3 is what all production runs use)
# ansatz
D      = 4         # bond dimension
output = "ground_N12_D4.jls"   # paths are relative to this config file
# initial state: an OBC DMRG seed (default)...
obc_D      = 16    # OBC bond dimension, must be >= D (default D)
obc_sweeps = 8     # default 12
# optimizer
tol    = 1e-6      # stop when the gradient norm eta < tol (default 1e-7)
maxit  = 2500      # max L-BFGS iterations (default 800)
blas_threads = 1   # BLAS threads; this overrides OPENBLAS_NUM_THREADS
```

Run it:

```bash
julia --project=. src/run_seed_comparison.jl myrun/ground.toml
```

It prints `saved ... E/N=... eta=...` at the end. It also writes `ground_N12_D4.jls.seed.jls`, a checkpoint of the
starting state.

**Changing N (recommended for N ≥ 28): start from a smaller converged ring.** Replace the `obc_*` lines with
`source`:

```toml
N      = 16
D      = 4
source = "ground_N12_D4.jls"   # converged ground at smaller N
output = "ground_N16_D4.jls"
# x, me, theta, hmax, tol, maxit as before
```

The source must have the same D, x and hmax. N, me and theta may differ. Seeding from a smaller ring converges
about 5x faster than an OBC seed, and it converges in cases where OBC seeds get stuck. So the usual recipe is:
start at small D (8) and small N with an OBC seed, step N up with `source`, then grow D at the final N (step 2).

## 2. Grow D, or change the mass or angle: `prepare_bond_ground_tol.jl`

```bash
julia --project=. src/prepare_bond_ground_tol.jl  CACHE N x me theta hmax D maxit OUTPUT [tol]
# e.g. D = 4 -> 6:
julia --project=. src/prepare_bond_ground_tol.jl myrun/ground_N12_D4.jls 12 10.0 0.3 0.0 3 6 2500 myrun/ground_N12_D6.jls 1e-6
```

- The converged tensor is padded from the cached D up to the new D, then re-optimized. Production steps D by 2
  (8 → 10 → 12 …).
- The same command with the same D but a different `me` or `theta` moves an existing ground to new parameters.
  Near the critical point this is a full re-optimization, not a cheap one.
- N and hmax must match the cache. To use more BLAS threads, set `PUMPS_BLAS=4`.

## 3. Spectrum: `run_local_excitation.jl`

```bash
julia --project=. -t T src/run_local_excitation.jl  CACHE N x me theta hmax MOMENTA NEV OUTPUT
# e.g.
PUMPS_SOLVER=dense julia --project=. -t 4 src/run_local_excitation.jl myrun/ground_N12_D6.jls 12 10.0 0.3 0.0 3 0,1,2,3 4 myrun/spec_N12_D6.jls
```

- The physics arguments must match the ground in `CACHE`. The script checks this and stops if they don't.
- `MOMENTA` is a comma-separated list of integers k = 0 … L−1, with per-site momentum 2πk/N. Since E(k) = E(L−k),
  only the half zone is ever computed.
- `NEV` is the number of levels per momentum.
- Levels are absolute energies. At k = 0 the lowest level is the ground state itself, so the gaps are
  `levels − E0`, which the script prints.
- `PUMPS_SOLVER=dense` is recommended at gapped points, where the default Krylov solver can be about 10x slower.
  Leave it unset near criticality.
- `-t T` sets the Julia threads used for the assembly.

To turn the results into a CSV for `plots/`:

```bash
julia --project=. src/export_bloch.jl myrun/spec_*.jls > myrun/bloch_levels.csv
```

## Exact diagonalization reference (small N only)

```bash
julia --project=. -t 4 src/run_flux_ed.jl  N x me theta hmax MOMENTA NEV OUTDIR
julia --project=. -t 4 src/run_flux_ed.jl 12 10.0 0.3 0.0 3 0,1 4 myrun/ed_N12   # -> myrun/ed_N12/ed_levels.csv
```

ED is practical up to N ≈ 20–28. N = 28 needs about 33 GB of memory.

## What to expect

These are the commands above, run at N=12 with x=10, m/e=0.3 and theta=0 (single core, compile time included):

| Step | Time | Result |
|---|---|---|
| ground, D=4 (OBC seed) | ~1 min | E/N = −0.624784 |
| grow to D=6 | ~1 min | E0 = −7.497977 (ED: −7.498096) |
| spectrum, k=0..3, D=6 | ~40 s | first gap 0.6659 (ED: 0.6643) |

Cost grows quickly. Each optimizer step scales as N·D⁵, and the spectrum as N²·D⁵. Production sizes
(N ≥ 64, D ≥ 12) take hours and belong on a cluster, not a laptop. To check convergence, repeat at D+2 and
compare.

## Gotchas

- **Nothing is overwritten.** Every script stops with `refusing to overwrite` if OUTPUT exists. Delete the file
  or choose a new name.
- **Threads:** for ground states use `-t 1` with BLAS threads (`blas_threads` in the config, `PUMPS_BLAS` for
  step 2). Use no more BLAS threads than physical cores that are free, or runs can slow down 50–100x. For spectra
  use `-t T`.
- **Paths:** in a TOML config, `output` and `source` are relative to the config file. On the command line, paths
  are relative to your working directory.
