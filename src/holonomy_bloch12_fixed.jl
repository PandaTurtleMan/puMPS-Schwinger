# N=12 hmax=5 Bloch spectrum vs ED for the CORRECTED (translation-invariant) holonomy
# Schwinger model -- the same check as holonomy_bloch12.jl, but with the mod-1 artifact
# removed (see [[holonomy-mod1-translation]]):
#     N E_0^2 -> e^2/N      and      2N E_0 -> 2e      in the cross-sector coupling.
include("holonomy_fixed_solve.jl")
include("holonomy_bloch12.jl")     # lowestLevels (Lanczos), occAt, and the mod-1 driver
using LinearAlgebra, SparseArrays, Printf, Random, Serialization

# ---------------------------------------------------------------------------
# corrected sparse neutral-sector ED (analogue of holoSparse in holonomy_bloch12.jl)
function holoSparseFixed(N, gC, lam, hmax; mass=0.0)
    idx = zeroChargeIdx(N); n = length(idx); M = 2hmax + 1
    pos = Dict(idx[a] => a for a in 1:n)
    ev = eInt(N)[idx]
    diagc = gC .* (staggeredKernelDiag(N)[idx] .+ ev .^ 2 ./ N) .+ lam .* diag(denseQtot(N)^2)[idx] .+
            mass .* massDiagSite(N)[idx]
    w = wDiag(N)[idx]
    hm, h2, _ = holonomyMats(hmax)
    I0 = Int[]; J0 = Int[]; V0 = ComplexF64[]
    push3!(i, j, v) = (push!(I0, i); push!(J0, j); push!(V0, v))
    for m in 0:M-1
        φ = (2π * m / M) / N; off = m * n
        for a in 1:n
            lin = idx[a] - 1
            push3!(off + a, off + a, diagc[a])
            for i in 1:N
                j = mod1(i + 1, N)
                oi = occAt(lin, i); oj = occAt(lin, j)
                oi == oj && continue
                nl = lin ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                push3!(off + pos[nl+1], off + a, oj ? exp(im * φ) : exp(-im * φ))
            end
        end
    end
    for m in 0:M-1, mp in 0:M-1
        Δ = 2π * (mp - m) / M
        for a in 1:n
            v = gC * (2 * ev[a] * hm[m+1, mp+1] + N * h2[m+1, mp+1]) * exp(im * Δ * w[a] / N)
            push3!(m * n + a, mp * n + a, v)
        end
    end
    H = sparse(I0, J0, V0, n * M, n * M)
    return (H + H') / 2
end

function validateSparseFixed(; N=6, gC=0.5, lam=1.0, hmaxs=(1, 2, 3), mass=0.0)
    worst = 0.0
    for hmax in hmaxs
        Es = lowestLevels(holoSparseFixed(N, gC, lam, hmax; mass=mass), 1)[1]
        Hd, _ = holoFixedH(N, gC, lam, hmax; gauge=:spread_dressed, mass=mass)
        Ed = minimum(real(eigvals(Hd)))
        @printf("  N=%d hmax=%d mass=%.2f : sparse %.12f  dense %.12f  |diff| %.1e\n",
                N, hmax, mass, Es, Ed, abs(Es - Ed))
        worst = max(worst, abs(Es - Ed))
    end
    return worst
end

# ---------------------------------------------------------------------------
function runFixed12(; Nsites=12, D=4, gC=0.5, lam=1.0, mass=0.0, hmax=5, nev=16,
                     warmSteps=1500, maxit=300, alpha=0.03, seed=4, maxshell=nothing,
                     out="holo_bloch12_fixed_D$D.csv")
    Nb = Nsites ÷ 2; M = 2hmax + 1
    BLAS.set_num_threads(1)
    @printf("CORRECTED holonomy Bloch spectrum : N=%d D=%d gC=%.2f lam=%.1f mass=%.3f hmax=%d (M=%d) threads=%d\n",
            Nsites, D, gC, lam, mass, hmax, M, Threads.nthreads()); flush(stdout)

    t = @elapsed edlev = lowestLevels(holoSparseFixed(Nsites, gC, lam, hmax; mass=mass), nev)
    @printf("ED (neutral, sparse Lanczos): E0 = %.10f  E0/N = %.10f   [%.0f s]\n",
            edlev[1], edlev[1]/Nsites, t); flush(stdout)

    # mass=0 keeps the historical name so the existing tensor caches still hit
    msuf = mass == 0.0 ? "" : @sprintf("_m%.3f", mass)
    cache = @sprintf("levelBfix_N%d_D%d_g%.2f_l%.1f%s_h%d_s%d.jls", Nsites, D, gC, lam, msuf, hmax, seed)
    if isfile(cache)
        EB, As, lambdas = deserialize(cache)
        @printf("Level B ground state        : E/N = %.10f  (gap to ED %.3e/site)  [cached]\n",
                EB/Nsites, (EB - edlev[1])/Nsites); flush(stdout)
    else
        t = @elapsed EB, As, lambdas = levelB_GS_fixed(; Nsites=Nsites, hmax=hmax, D=D, gC=gC,
                                        lam=lam, mass=mass, warmSteps=warmSteps, maxit=maxit,
                                        alpha=alpha, seed=seed, verbose=true)
        serialize(cache, (EB, As, lambdas))
        @printf("Level B ground state        : E/N = %.10f  (gap to ED %.3e/site)  [%.0f s]\n",
                EB/Nsites, (EB - edlev[1])/Nsites, t); flush(stdout)
    end

    shells = collect(0:something(maxshell, Nb ÷ 2))
    slev = Dict{Int,Vector{Float64}}(); done = Int[]
    writeCSV() = open(out, "w") do io
        plev = slev[done[end]]
        println(io, "level,E_ED,E_puMPS,dE_ED,dE_puMPS,abs_err_E,abs_err_dE," *
                    join(["E_shell$s" for s in done], ","))
        for l in 1:nev
            println(io, "$l,$(edlev[l]),$(plev[l]),$(edlev[l]-edlev[1]),$(plev[l]-plev[1])," *
                        "$(abs(plev[l]-edlev[l])),$(abs((plev[l]-plev[1])-(edlev[l]-edlev[1]))),"*
                        join([slev[s][l] for s in done], ","))
        end
    end
    for s in shells
        kset = [k for k in 0:Nb-1 if min(k, Nb - k) <= s]
        t = @elapsed lv, rk = enrichedLevelsFixed(As, lambdas, kset, Nsites, gC, lam, hmax; nev=nev, mass=mass)
        slev[s] = lv; push!(done, s)
        @printf("Bloch/enriched shell |k|<=%d : E0/N = %.10f  rank %d of %d   [%.0f s]\n",
                s, lv[1]/Nsites, rk, M*length(kset)*4D*D, t); flush(stdout)
        writeCSV(); GC.gc()
    end
    plev = slev[done[end]]
    println("\n  level      E_ED           E_puMPS        dE_ED        dE_puMPS      err(dE)")
    for l in 1:nev
        @printf("   %2d    %13.8f  %13.8f  %11.7f  %11.7f   %.2e\n", l, edlev[l], plev[l],
                edlev[l]-edlev[1], plev[l]-plev[1], abs((plev[l]-plev[1])-(edlev[l]-edlev[1])))
    end
    println("\nwrote $out")
    return edlev, plev
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "validate"
        println("=== corrected sparse ED vs dense holoFixedH ===")
        e = 0.0
        for mass in (0.0, 0.5, -0.3); e = max(e, validateSparseFixed(; mass=mass)); end
        println(e < 1e-9 ? "  CORRECTED SPARSE ED OK" : "  MISMATCH ($e)")
    else
        D    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
        hmax = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
        nev  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
        ms   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
        mass = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.0
        runFixed12(; D=D, hmax=hmax, nev=nev, maxshell=ms, mass=mass)
    end
end
