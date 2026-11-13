# puMPS for the lattice Schwinger model

Ground states and excitation spectra of the lattice Schwinger model on a finite ring, using periodic uniform
matrix product states (puMPS) with a holonomy (flux) variable.

You describe a run in one TOML file and start it with one command.

## Quickstart

Julia 1.12 or newer is required. Install the packages once, from this folder:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then run the example. It takes about 5 minutes on one laptop core:

```bash
julia --project=. pumps.jl examples/quickstart.toml
```

This computes the ground state of a 12-site ring at bond dimension D=4, grows it to D=6, and computes the lowest
four levels at each momentum. It ends with a summary like this:

```
  Ground states
        N    D               E0           E0/N        eta
       12    4      -7.49740781    -0.62478398    9.7e-07
       12    6      -7.49798222    -0.62483185    9.7e-07

  Spectrum N=12 D=6: energies above the ground state (momentum per site = 2 pi k / N)
    k=0     0.00000   0.66594   0.90318   1.10886
    k=1     1.22916   1.57886   1.89749   2.21256
    k=2     2.01802   2.13938   2.40801   2.50855
    k=3     2.42518   2.43666   2.73317   2.75341
```

## Your own run

Copy an example and edit it:

```bash
cp examples/quickstart.toml myrun.toml
julia --project=. pumps.jl myrun.toml
```

Or start from a file that lists every setting with its explanation:

```bash
julia pumps.jl --template > myrun.toml
```

| Example | What it shows |
|---|---|
| [examples/quickstart.toml](examples/quickstart.toml) | smallest complete run: ground state, D growth, spectrum |
| [examples/n_ladder.toml](examples/n_ladder.toml) | the recipe for larger rings: N = 12 → 16 → 20, then grow D |
| [examples/theta_pi.toml](examples/theta_pi.toml) | a point at θ = π |

### What a run does

`pumps.jl` runs these stages in order. Each one starts from the file written by the one before.

1. **Ground state** at the first `N` and first `D`, starting from an open-chain DMRG state.
2. **Climb in N**, if `N` is a list: each larger ring starts from the converged smaller one.
3. **Grow D**, if `D` is a list: at the final `N`, the tensor is padded to the next `D` and re-optimized.
4. **Spectrum** at the final `N`, from the ground state at the largest `D` (or at every `D` with `at = "all"`).
5. **`levels.csv`**, a table of all levels for plotting.
6. **Exact diagonalization**, if enabled.

### Results

Everything goes to the `[output] dir` folder:

| File | Content |
|---|---|
| `ground_N12_D6.jls` | ground state (Julia serialized): energy, tensors, convergence |
| `spectrum_N12_D6.jls` | levels per momentum, with residuals |
| `levels.csv` | all levels as a table: `N, D, x, me, theta, hmax, k, level, E, gap, ...` |
| `ed/ed_levels.csv` | exact diagonalization levels |
| `logs/` | full output of every stage |
| `config_used.toml` | copy of the config of the latest run |

Energies in `levels.csv` are absolute; the `gap` column is the energy above the ground state. At k=0 the lowest
level is the ground state itself, so its gap is 0.

### Useful options

```bash
julia --project=. pumps.jl --dry-run myrun.toml
```

prints the stages and the exact command for each without running anything. Use it to check a config, or to copy
the commands into a cluster job script.

```bash
julia --project=. pumps.jl --only spectrum myrun.toml
```

runs one part only (`ground`, `spectrum` or `ed`).

## All settings

| Section | Key | Default | Meaning |
|---|---|---|---|
| `[physics]` | `x` | **required** | x = 1/(a e)^2, so the lattice spacing is a e = 1/sqrt(x) |
| `[physics]` | `me` | **required** | fermion mass m/e |
| `[physics]` | `theta` | `0.0` | background angle (use 3.141592653589793 for theta = pi) |
| `[physics]` | `hmax` | `3` | holonomy cutoff: 2*hmax+1 flux sectors (3 in all production runs) |
| `[ground]` | `N` | **required** | number of sites (even). A list like [12, 16, 20] climbs in N, each ring seeded from the one before |
| `[ground]` | `D` | **required** | bond dimension. A list grows D step by step at the final N (steps of 2 work well) |
| `[ground]` | `tol` | `1.0e-7` | converged when the gradient norm eta < tol |
| `[ground]` | `maxit` | `800` | maximum optimizer iterations per stage |
| `[ground]` | `blas_threads` | `1` | BLAS threads; keep at or below the number of free physical cores |
| `[ground]` | `obc_D` |  | bond dimension of the open-chain DMRG seed, >= first D (default: first D) |
| `[ground]` | `obc_sweeps` | `12` | DMRG sweeps for the open-chain seed |
| `[ground]` | `start` |  | start from this saved ground state instead of an open-chain seed (needs the same D, x, hmax) |
| `[ground]` | `seed` | `20260930` | random seed |
| `[spectrum]` | `momenta` | `"all"` | "all" = k = 0..L/2 with L = N/2 (the others follow from E(k) = E(L-k)), or a list like [0, 1, 2] |
| `[spectrum]` | `nev` | `4` | number of levels per momentum |
| `[spectrum]` | `solver` | `"krylov"` | "dense": about 10x faster at gapped points. "krylov": better near criticality |
| `[spectrum]` | `threads` | `1` | Julia threads for the spectrum |
| `[spectrum]` | `at` | `"last"` | "last" = only the largest D. "all" = every D in ground.D, to check convergence in D |
| `[ed]` | `enabled` | `false` | set to true to also run exact diagonalization at the final N |
| `[ed]` | `momenta` |  | default: the same as spectrum.momenta |
| `[ed]` | `nev` |  | default: the same as spectrum.nev |
| `[ed]` | `threads` | `1` | Julia threads for ED |
| `[output]` | `dir` |  | results folder, relative to this config file (default: runs/<config file name>) |

## Layout

| Path | Content |
|---|---|
| `pumps.jl` | the entry point; reads the config and runs the stages |
| `examples/` | ready-to-run configs |
| `src/` | the numerical code and the per-stage scripts `pumps.jl` calls |
| `plots/` | Python plotting scripts for `levels.csv`-style tables |
| `docs/manual_steps.md` | running each stage by hand, without `pumps.jl` |
| `test/runtests.jl` | checks of the config handling, and an end-to-end run |

## Tests

```bash
julia --project=. test/runtests.jl          # config checks only, a few seconds
julia --project=. test/runtests.jl --full   # also runs the quickstart and compares the numbers with ED (about 8 minutes)
```
