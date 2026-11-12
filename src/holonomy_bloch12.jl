# N=12 BLOCH (tangent-space) SPECTRUM of the HOLONOMY Schwinger model, vs ED.
#
#   H = sum_m |m><m| (x) [ H_XX(theta_m/N per bond) + gC*(Coulomb kernel + N E_0^2)
#                          + lambda (sum_j sigma^z_j)^2 ]
#     + gC sum_{m,m'} |m><m'| (x) [ 2N E_0 h_{mm'} + N h2_{mm'} ] e^{i Delta w / N}
#
# i.e. the spread_dressed gauge of holonomy_ed.jl / enrich.jl -- the SAME operator the
# puMPS ansatz variational space is built for.  Holonomy truncated to |h| <= hmax,
# M = 2 hmax + 1 sectors (hmax=5 => M=11).
#
# WHY THERE IS NO MOMENTUM LABEL HERE (unlike bloch12.jl, the zero-holonomy N=12
# spectrum): with the holonomy restored, block translation is NOT a symmetry.  This
# is the operator-level statement of the "GAP DIAGNOSIS" -- the holonomy-changing
# coupling gC 2N h_mm' <psi_m| E_0 U^dag U' |psi_m'> carries a position-weighted
# phase U = exp(i Delta w / N), w = sum_j j n_j, which injects lattice momentum.
# A scan over the whole family T = P (x) exp(i theta_m beta(c)) (P = block shift;
# beta any integer combination of a constant, n_{N-1}+n_N and n_1+n_2), in BOTH the
# spread_dressed and concentrated gauges, found NO commuting T (best ||[H,T]|| ~ 12
# at N=6, vs ||H|| ~ 30).  `checkNoTranslation` below reproduces that.  So the
# spectrum is compared LEVEL BY LEVEL, not momentum by momentum.
#
# ED side  : neutral (charge-zero) sector, dim C(N,N/2)*M = 924*11 = 10164 at N=12.
#            Sparse + Lanczos with full reorthogonalisation (no ARPACK here, and the
#            dense 10164^2 complex matrix would be 1.7 GB).  The nearest charged
#            sector costs 4*lambda, so with lambda=1 it is far above the window.
# puMPS side: Level-B ground tensors A_m (one puMPS per holonomy sector), then the
#            enriched tangent basis {t_{m,k,nu}} of holonomy_bloch.jl at ALL block
#            momenta k = 0..Nb-1, and the lowest generalized eigenvalues of
#            (H_eff, N_eff).  The ground state lives in the k=0 tangent span, so the
#            lowest level is the enriched (= best) variational ground state and the
#            higher ones are the Bloch/tangent excitations.
include("holonomy_bloch.jl")     # buildEnrichedHN, levelB_GS, holonomy_ed helpers, ...
using LinearAlgebra, SparseArrays, Printf, Random, Serialization

# ===========================================================================
# ED: sparse neutral-sector holonomy Hamiltonian (spread_dressed gauge)
# ===========================================================================
occAt(lin, i) = (((lin >> (i - 1)) & 1) == 0)          # bit 0 = occupied (n_i = 1)

function holoSparse(N, gC, lam, hmax; theta_ext=0.0)
    idx = zeroChargeIdx(N); n = length(idx); M = 2hmax + 1
    pos = Dict(idx[a] => a for a in 1:n)
    diagc = gC .* (staggeredKernelDiag(N)[idx] .+ e0DiagSite(N; theta=theta_ext)[idx]) .+
            lam .* diag(denseQtot(N)^2)[idx]
    e0o = e0OperatorDiag(N; theta=theta_ext)[idx]
    w   = wDiag(N)[idx]
    hm, h2, _ = holonomyMats(hmax)

    I0 = Int[]; J0 = Int[]; V0 = ComplexF64[]
    push3!(i, j, v) = (push!(I0, i); push!(J0, j); push!(V0, v))
    for m in 0:M-1
        φ = (2π * m / M) / N                                  # spread twist per bond
        off = m * n
        for a in 1:n
            lin = idx[a] - 1
            push3!(off + a, off + a, diagc[a])
            for i in 1:N
                j = mod1(i + 1, N)
                oi = occAt(lin, i); oj = occAt(lin, j)
                oi == oj && continue
                # gate = e^{i phi} sp_left sm_right + e^{-i phi} sm_left sp_right,
                # slot 1 = LEFT site (ringHamFromGate convention); sp creates.
                nl = lin ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                amp = oj ? exp(im * φ) : exp(-im * φ)          # j->i hop : i->j hop
                push3!(off + pos[nl+1], off + a, amp)
            end
        end
    end
    # holonomy coupling: config-diagonal, couples all (m,m') incl. the m=m' self-energy
    for m in 0:M-1, mp in 0:M-1
        Δ = 2π * (mp - m) / M
        for a in 1:n
            v = gC * (2N * e0o[a] * hm[m+1, mp+1] + N * h2[m+1, mp+1]) * exp(im * Δ * w[a] / N)
            push3!(m * n + a, mp * n + a, v)
        end
    end
    H = sparse(I0, J0, V0, n * M, n * M)
    return (H + H') / 2
end

# Lanczos with FULL reorthogonalisation; returns the lowest nev Ritz values.
# Deterministic (fixed seed) and checked against dense eigvals at small N.
function lowestLevels(H, nev; maxiter=600, seed=11, tol=1e-11)
    n = size(H, 1); Random.seed!(seed)
    v = randn(ComplexF64, n); v ./= norm(v)
    Vs = [v]; α = Float64[]; β = Float64[]; prev = fill(Inf, nev)
    for it in 1:min(maxiter, n)
        wv = H * Vs[end]
        a = real(dot(Vs[end], wv)); push!(α, a)
        wv .-= a .* Vs[end]
        it > 1 && (wv .-= β[end] .* Vs[end-1])
        for u in Vs; wv .-= dot(u, wv) .* u; end      # full reorthogonalisation
        for u in Vs; wv .-= dot(u, wv) .* u; end
        b = norm(wv)
        if it >= nev + 5
            T = SymTridiagonal(α, β)
            ev = sort(eigvals(T))[1:nev]
            maximum(abs.(ev .- prev)) < tol && return ev
            prev = ev
        end
        b < 1e-12 && break
        push!(β, b); push!(Vs, wv ./ b)
    end
    return sort(eigvals(SymTridiagonal(α, β)))[1:min(nev, length(α))]
end

# ===========================================================================
# puMPS: lowest nev generalized eigenvalues of the enriched (H_eff, N_eff)
# ===========================================================================
function enrichedLevels(As, lambdas, kset, Nsites, gC, lam, hmax; nev=12,
                        theta_ext=0.0, cutoff=1e-10)
    Heff, Neff = buildEnrichedHN(As, lambdas, kset, Nsites, gC, lam, hmax; theta_ext=theta_ext)
    F = eigen(Neff); Neff = nothing
    thresh = cutoff * maximum(F.values)
    keep = findall(>(thresh), F.values)
    # W = U diag(1/sqrt(n)) maps the r-dim non-null subspace into the basis; solving in
    # that subspace (He = W' Heff W, r x r) avoids ever forming the two Ntot x Ntot
    # matrices Nh and Nh*Heff*Nh -- at D=6, all momenta, those are 1.4 GB each.
    W = F.vectors[:, keep] * Diagonal(1.0 ./ sqrt.(F.values[keep]))
    He = W' * (Heff * W)
    ev = sort(real(eigvals(Hermitian((He + He') / 2))))
    return ev[1:min(nev, length(ev))], length(keep)
end

# ===========================================================================
# Validation: sparse ED == the dense holonomy_ed reference, and the no-translation
# statement quoted in the header.
# ===========================================================================
function validateSparseED(; N=6, gC=0.5, lam=1.0, hmaxs=(0, 1, 2))
    ok = 0.0
    for hmax in hmaxs
        Es = lowestLevels(holoSparse(N, gC, lam, hmax), 1; maxiter=400)[1]
        Ed = holonomyED_neutral(N, gC, hmax; lam=lam, mode=:spread_dressed)
        Ec = holonomyED_neutral(N, gC, hmax; lam=lam, mode=:concentrated)
        @printf("  N=%d hmax=%d : sparse %.12f  dense %.12f  |diff| %.1e   (concentrated %.12f)\n",
                N, hmax, Es, Ed, abs(Es - Ed), Ec)
        ok = max(ok, abs(Es - Ed))
    end
    # full spectrum check at the smallest case
    Hs = Matrix(holoSparse(N, gC, lam, 1))
    ev = sort(real(eigvals(Hermitian((Hs + Hs') / 2))))
    lv = lowestLevels(holoSparse(N, gC, lam, 1), 10; maxiter=500)
    @printf("  lowest-10 Lanczos vs dense: max |diff| = %.2e\n", maximum(abs.(lv .- ev[1:10])))
    return ok
end

function checkNoTranslation(; N=6, gC=0.5, lam=1.0, hmax=1, r=2)
    H = Matrix(holoSparse(N, gC, lam, hmax))
    idx = zeroChargeIdx(N); n = length(idx); M = 2hmax + 1
    pos = Dict(idx[a] => a for a in 1:n)
    shift(lin) = (nl = 0; for i in 0:N-1; nl |= (((lin >> i) & 1) << mod(i + r, N)); end; nl)
    pidx = [pos[shift(idx[a] - 1)+1] for a in 1:n]
    nb = [occAt(idx[a]-1, N-1) + occAt(idx[a]-1, N) for a in 1:n]
    nf = [occAt(idx[a]-1, 1)   + occAt(idx[a]-1, 2) for a in 1:n]
    best = (Inf, 0, 0, 0)
    for aa in -M:M, b in -3:3, c2 in -3:3
        T = zeros(ComplexF64, n*M, n*M)
        for m in 0:M-1, col in 1:n
            T[m*n + pidx[col], m*n + col] = exp(im * 2π * m / M * (aa + b*nb[col] + c2*nf[col]))
        end
        cn = norm(H*T - T*H); cn < best[1] && (best = (cn, aa, b, c2))
    end
    @printf("  N=%d hmax=%d: best ||[H,T]|| over T = P (x) e^{i th_m beta} is %.3e  (||H|| = %.1f)\n",
            N, hmax, best[1], norm(H))
    return best[1]
end

# ===========================================================================
# DRIVER
# ===========================================================================
function runHoloBloch12(; Nsites=12, D=4, gC=0.5, lam=1.0, hmax=5, nev=12,
                         warmSteps=1500, maxit=300, alpha=0.03, seed=4,
                         maxshell=nothing, out="holo_bloch12_D$D.csv")
    Nb = Nsites ÷ 2; M = 2hmax + 1
    BLAS.set_num_threads(1)
    @printf("HOLONOMY Bloch spectrum : N=%d (Nb=%d) D=%d gC=%.2f lam=%.1f hmax=%d (M=%d) threads=%d\n",
            Nsites, Nb, D, gC, lam, hmax, M, Threads.nthreads()); flush(stdout)

    t = @elapsed edlev = lowestLevels(holoSparse(Nsites, gC, lam, hmax), nev)
    @printf("ED (neutral, sparse Lanczos): E0 = %.10f  E0/N = %.10f   [%.1f s]\n",
            edlev[1], edlev[1]/Nsites, t); flush(stdout)

    # Level-B tensors are cached: they depend only on (N, D, gC, lam, hmax, seed, steps),
    # not on the tangent-space part, so shell reruns / restarts do not repeat them.
    cache = @sprintf("levelB_N%d_D%d_g%.2f_l%.1f_h%d_s%d.jls", Nsites, D, gC, lam, hmax, seed)
    if isfile(cache)
        EB, As, lambdas = deserialize(cache)
        @printf("Level B ground state        : E/N = %.10f  (gap to ED %.3e/site)  [cached]\n",
                EB/Nsites, (EB - edlev[1])/Nsites); flush(stdout)
    else
        t = @elapsed EB, As, lambdas = levelB_GS(; Nsites=Nsites, hmax=hmax, D=D, gC=gC, lam=lam,
                                                  warmSteps=warmSteps, maxit=maxit, alpha=alpha,
                                                  seed=seed, verbose=true)
        serialize(cache, (EB, As, lambdas))
        @printf("Level B ground state        : E/N = %.10f  (gap to ED %.3e/site)  [%.0f s]\n",
                EB/Nsites, (EB - edlev[1])/Nsites, t); flush(stdout)
    end

    # spectrum per momentum shell (shell s = all |k| <= s); the last one is the full
    # tangent space and is the reported puMPS spectrum.
    # maxshell caps the momentum shells (memory: the dense H_eff is (M |kset| 4D^2)^2)
    # The CSV is rewritten after EVERY shell, not once at the end: the largest shells are
    # the ones that can run out of memory (the dense H_eff is (M |kset| 4D^2)^2 -- 1.0 GB
    # per matrix at D=6, |k|<=2), and losing the whole run to the last shell would throw
    # away hours of Level-B + smaller-shell work.
    shells = collect(0:something(maxshell, Nb ÷ 2))
    slev = Dict{Int,Vector{Float64}}()
    done = Int[]
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
        t = @elapsed lv, rk = enrichedLevels(As, lambdas, kset, Nsites, gC, lam, hmax; nev=nev)
        slev[s] = lv; push!(done, s)
        @printf("Bloch/enriched shell |k|<=%d : E0/N = %.10f  rank %d of %d   [%.0f s]\n",
                s, lv[1]/Nsites, rk, M*length(kset)*4D*D, t); flush(stdout)
        writeCSV()
        GC.gc()
    end
    plev = slev[done[end]]
    println("\n  level      E_ED           E_puMPS        |dE_ED|        |dE_puMPS|     err(dE)")
    for l in 1:nev
        @printf("   %2d    %13.8f  %13.8f  %12.8f  %12.8f   %.2e\n", l, edlev[l], plev[l],
                edlev[l]-edlev[1], plev[l]-plev[1], abs((plev[l]-plev[1])-(edlev[l]-edlev[1])))
    end
    println("\nwrote $out")
    return edlev, plev
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "validate"
        println("=== sparse holonomy ED vs dense reference ===")
        e = validateSparseED()
        println(e < 1e-9 ? "  SPARSE ED OK" : "  SPARSE ED MISMATCH ($e)")
        println("\n=== no block-translation symmetry with holonomy ===")
        checkNoTranslation(; hmax=1); checkNoTranslation(; hmax=2)
    else
        D    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
        hmax = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
        nev  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12
        ms   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
        runHoloBloch12(; D=D, hmax=hmax, nev=nev, maxshell=ms)
    end
end
