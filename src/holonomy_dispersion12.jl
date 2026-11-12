# MOMENTUM-RESOLVED puMPS Bloch spectrum for the CORRECTED holonomy Schwinger model
# -- the Zou/Milsted/Vidal method proper: GD ground state, then a Bloch/tangent ansatz
# giving a generalized eigenproblem SOLVED SEPARATELY IN EACH MOMENTUM SECTOR.
#
# This is only possible for the corrected model.  In the spread-dressed gauge the
# symmetry is the plain block translation P, and the tangent basis states are exact
# P-eigenstates,
#     P |t_{m,k,nu}> = e^{-i theta_k} |t_{m,k,nu}>,   theta_k = 2 pi k / Nb,
# independently of the holonomy sector m and the impurity index nu.  Hence H_eff and
# N_eff are block-diagonal in k and the ZMV eigenequation is recovered.  Measured with
# real Level-B tensors: the low enriched eigenvectors have k-weight 1.000 in the
# corrected model vs 0.70-0.94 in the mod-1 model (where no momentum label exists).
#
# ONE GENERALIZATION vs plain ZMV: each momentum block still couples the M holonomy
# sectors, so a block has dimension M * 4D^2 rather than 4D^2.
#
# The pseudo-inverse cutoff is applied PER MOMENTUM BLOCK, exactly as blochSpectrum does
# in puMPS_bloch.jl -- the gauge null space is a per-momentum object, so this (not the
# joint solve used earlier for the mod-1 model) is the correct treatment.
include("holonomy_bloch12_fixed.jl")
using LinearAlgebra, SparseArrays, Printf, Serialization

# ---------------------------------------------------------------------------
# puMPS side: solve the enriched problem one momentum at a time
# ---------------------------------------------------------------------------
# Basis index inside buildEnrichedHNFixed: (m-1)*bs + (ki-1)*dt + nu, bs = |kset|*dt.
function blochDispersionFixed(As, lambdas, Nsites, gC, lam, hmax; nev=8, cutoff=1e-10, mass=0.0)
    Nb = Nsites ÷ 2; M = 2hmax + 1; D = size(As[1], 1); dt = 4 * D * D
    kset = collect(0:Nb-1); bs = length(kset) * dt
    Heff, Neff = buildEnrichedHNFixed(As, lambdas, kset, Nsites, gC, lam, hmax; mass=mass)
    Hm = Matrix(Heff); Nm = Matrix(Neff)
    levels = Vector{Vector{Float64}}(undef, Nb); ranks = zeros(Int, Nb)
    for (ki, k) in enumerate(kset)
        sel = vcat([(m-1)*bs + (ki-1)*dt .+ (1:dt) for m in 1:M]...)
        Hb = Hm[sel, sel]; Nb2 = Hermitian((Nm[sel, sel] + Nm[sel, sel]') / 2)
        F = eigen(Nb2); thresh = cutoff * maximum(F.values)
        keep = findall(>(thresh), F.values); ranks[ki] = length(keep)
        W = F.vectors[:, keep] * Diagonal(1.0 ./ sqrt.(F.values[keep]))
        He = W' * (Hb * W); He = Hermitian((He + He') / 2)
        ev = sort(real(eigvals(He)))
        levels[ki] = ev[1:min(nev, length(ev))]
    end
    return levels, ranks
end

# LEAN assembly: accumulate straight into the Nb momentum blocks and never materialise
# the full (M |kset| 4D^2)^2 matrix -- at D=8 that would be 16896^2 = 4.6 GB, while the
# per-momentum blocks are M*4D^2 = 2816 a side.  Same arithmetic as blochDispersionFixed
# (validated against it); the Hxx/Hker/Hlift pieces are already k-diagonal by construction
# (fourierTT), only the diagPhasedBlock-derived pieces need slicing.
function blochDispersionLean(As, lambdas, Nsites, gC, lam, hmax; nev=8, cutoff=1e-10, mass=0.0)
    Nb = Nsites ÷ 2; M = 2hmax + 1; D = size(As[1], 1); dt = 4 * D * D
    kset = collect(0:Nb-1)
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    hm, h2, _ = holonomyMats(hmax)
    kterms = kernelTerms(Nsites, gC)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    dim = M * dt
    Hk = [zeros(ComplexF64, dim, dim) for _ in 1:Nb]
    Nk = [zeros(ComplexF64, dim, dim) for _ in 1:Nb]
    ks(ki) = (ki-1)*dt+1 : ki*dt
    for m in 1:M
        A = As[m]; li = lamInvs[m]; theta_m = 2π * (m - 1) / M
        hmat = blockedXXGateTwisted(theta_m / Nsites; mass=mass)
        Nblk  = ttpart(diagPhasedBlock(B -> ONES4, A, A, li, li, kset, kset, Nb))
        Hxx   = fourierTT(sep -> H0k(A, li, hmat, sep, Nb), kset, Nb, dt)
        Hker  = fourierTT(sep -> sum(coef .* mpoTangentBlock(A, li, W, vL, vR, sep, Nb)
                                     for (coef, W, vL, vR) in kterms), kset, Nb, dt)
        Hlift = fourierTT(sep -> lam .* mpoTangentBlock(A, li, Wl, vLl, vRl, sep, Nb), kset, Nb, dt)
        HE    = (gC / Nsites) .* ttpart(e2BlockFused(A, A, li, li, kset, kset, Nb))
        se    = gC * Nsites * real(h2[m, m])
        r = (m-1)*dt+1 : m*dt
        for ki in 1:Nb
            s = ks(ki)
            Hk[ki][r, r] = Hxx[s, s] .+ Hker[s, s] .+ Hlift[s, s] .+ HE[s, s] .+ se .* Nblk[s, s]
            Nk[ki][r, r] = Nblk[s, s]
        end
        GC.gc()
    end
    for m in 1:M, mp in m+1:M                      # Hermiticity: compute m<mp, adjoint for mp<m
        Delta = 2π * (mp - m) / M
        ov = ttpart(crossOverlapBlock(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb))
        ee = ttpart(crossEBlockFused(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb))
        rm = (m-1)*dt+1 : m*dt; rp = (mp-1)*dt+1 : mp*dt
        for ki in 1:Nb
            s = ks(ki)
            blk = gC .* (2 * hm[m, mp] .* ee[s, s] .+ Nsites * h2[m, mp] .* ov[s, s])
            Hk[ki][rm, rp] = blk
            Hk[ki][rp, rm] = blk'
        end
    end
    levels = Vector{Vector{Float64}}(undef, Nb); ranks = zeros(Int, Nb)
    for ki in 1:Nb
        Hb = Hermitian((Hk[ki] + Hk[ki]') / 2); Nb2 = Hermitian((Nk[ki] + Nk[ki]') / 2)
        F = eigen(Nb2); thresh = cutoff * maximum(F.values)
        keep = findall(>(thresh), F.values); ranks[ki] = length(keep)
        W = F.vectors[:, keep] * Diagonal(1.0 ./ sqrt.(F.values[keep]))
        He = W' * (Hb * W); He = Hermitian((He + He') / 2)
        ev = sort(real(eigvals(He))); levels[ki] = ev[1:min(nev, length(ev))]
        Hk[ki] = zeros(ComplexF64, 0, 0); Nk[ki] = zeros(ComplexF64, 0, 0); GC.gc()
    end
    return levels, ranks
end

# MEMORY-LEAN variant: identical result to blochDispersionLean, but the kernels return only
# the Nb momentum-DIAGONAL dt x dt blocks (diagOnly=true) instead of the full (|kset|*dt)^2
# momentum matrix.  H_eff/N_eff are block-diagonal in k, so only the (k,k) blocks are ever
# used -- the full block wastes ~Nb x storage AND ~Nb x assembly churn (a diagPhasedBlock call
# allocated 107 GB to build a 1 GB block at N=40 D=10, of which 51 MB was used).  This drops
# the N=40 peak RSS from ~58 GB to ~15-20 GB.  Same arithmetic (the momentum-free Rraw build is
# untouched), so same energies and same parallelism.
# Generalized Hermitian solve for a dense H and sector-diagonal N. Do not apply
# independent relative cutoffs to the sectors: their absolute norm scales differ.
function sectorNormLevels(H, normblocks; nev=8, cutoff=1e-10)
    nev >= 0 || throw(ArgumentError("nev must be nonnegative"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    dims = [size(b, 1) for b in normblocks]
    size(H) == (sum(dims), sum(dims)) || throw(DimensionMismatch("H and norm blocks disagree"))
    isempty(dims) && return Float64[], 0
    fs = [eigen(Hermitian((b+b')/2)) for b in normblocks]
    threshold = cutoff * maximum(maximum(f.values) for f in fs)
    keep = [findall(>(threshold), f.values) for f in fs]
    ranks = length.(keep); rank = sum(ranks)
    (rank == 0 || nev == 0) && return Float64[], rank
    ws = [f.vectors[:, k] .* reshape(1 ./ sqrt.(f.values[k]), 1, :) for (f,k) in zip(fs,keep)]
    offsets = cumsum([0; dims]); roffsets = cumsum([0; ranks])
    reduced = zeros(ComplexF64, rank, rank)
    for m in eachindex(dims), n in m:length(dims)
        (ranks[m] == 0 || ranks[n] == 0) && continue
        rm = offsets[m]+1:offsets[m+1]; rn = offsets[n]+1:offsets[n+1]
        sm = roffsets[m]+1:roffsets[m+1]; sn = roffsets[n]+1:roffsets[n+1]
        block = (@view(H[rm,rn]) + @view(H[rn,rm])') / 2
        reduced[sm,sn] = ws[m]' * (block * ws[n])
    end
    # LAPACK's selected-eigenvalue solve avoids computing unrequested levels.
    return eigvals!(Hermitian(reduced, :U), 1:min(nev, rank)), rank
end

# `momenta` selects block momenta in 0:Nb-1; levels and ranks follow that order.
# Selection saves assembled matrix storage/solves. Most contraction work is
# shared by all momenta, so time does not scale linearly with their count.
function blochDispersionLeanDiag(As, lambdas, Nsites, gC, lam, hmax; nev=8, cutoff=1e-10, mass=0.0,
                                 verbose=false, solveBLAS=min(Threads.nthreads(), 8),
                                 momenta=0:(Nsites÷2-1))
    Nb = Nsites ÷ 2; M = 2hmax + 1; D = size(As[1], 1); dt = 4 * D * D
    kset = collect(Int, momenta); nk = length(kset)
    nev >= 0 || throw(ArgumentError("nev must be nonnegative"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    all(k -> 0 <= k < Nb, kset) || throw(ArgumentError("momenta must be in 0:Nb-1"))
    length(unique(kset)) == nk || throw(ArgumentError("duplicate momenta"))
    nk == 0 && return Vector{Float64}[], Int[]
    verbose && (@printf("      [bloch] N=%d D=%d M=%d Nb=%d: diagonal sectors...\n", Nsites, D, M, Nb); flush(stdout))
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    hm, h2, _ = holonomyMats(hmax)
    kterms = kernelTermsMerged(Nsites, gC)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    dim = M * dt
    Hk = [zeros(ComplexF64, dim, dim) for _ in kset]
    # Holonomy sectors are orthogonal: store only the M nonzero norm blocks.
    Nk = [Vector{Matrix{ComplexF64}}(undef, M) for _ in kset]
    for m in 1:M
        A = As[m]; li = lamInvs[m]; theta_m = 2π * (m - 1) / M
        hmat = blockedXXGateTwisted(theta_m / Nsites; mass=mass)
        Nblk  = diagPhasedBlock(B -> ONES4, A, A, li, li, kset, kset, Nb; diagOnly=true)
        Hxx   = fourierTT(sep -> H0k(A, li, hmat, sep, Nb), kset, Nb, dt; diagOnly=true)
        # Hker: sum over the Coulomb-kernel MPO terms, each via the hoisted Fourier routine
        # (builds T / Ccap / T-powers / mpoKcomb / mpoJcomb ONCE per term instead of Nb times).
        Hker  = [zeros(ComplexF64, dt, dt) for _ in kset]
        for (coef, W, vL, vR) in kterms
            tf = mpoTangentFourierDiag(A, li, W, vL, vR, kset, Nb)
            for ki in 1:nk; Hker[ki] .+= coef .* tf[ki]; end
        end
        Hlift = mpoTangentFourierDiag(A, li, Wl, vLl, vRl, kset, Nb)
        HE    = e2BlockFused(A, A, li, li, kset, kset, Nb; diagOnly=true)   # (gC/Nsites) applied below
        se    = gC * Nsites * real(h2[m, m]); c2 = gC / Nsites
        r = (m-1)*dt+1 : m*dt
        for ki in 1:nk
            Hk[ki][r, r] = Hxx[ki] .+ Hker[ki] .+ lam .* Hlift[ki] .+ c2 .* HE[ki] .+ se .* Nblk[ki]
            Nk[ki][m] = Nblk[ki]
        end
        GC.gc()
        verbose && (@printf("      [bloch] diagonal sector %d/%d\n", m, M); flush(stdout))
    end
    npair = 0; totpair = M*(M-1)÷2
    verbose && (@printf("      [bloch] cross-sector pairs (%d)...\n", totpair); flush(stdout))
    for m in 1:M, mp in m+1:M                      # Hermiticity: compute m<mp, adjoint for mp<m
        Delta = 2π * (mp - m) / M
        ov = crossOverlapBlock(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb; diagOnly=true)
        ee = crossEBlockFused(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb; diagOnly=true)
        rm = (m-1)*dt+1 : m*dt; rp = (mp-1)*dt+1 : mp*dt
        for ki in 1:nk
            blk = gC .* (2 * hm[m, mp] .* ee[ki] .+ Nsites * h2[m, mp] .* ov[ki])
            Hk[ki][rm, rp] = blk
            Hk[ki][rp, rm] = blk'
        end
        GC.gc()
        npair += 1
        verbose && (@printf("      [bloch] cross pair %d/%d (m=%d,mp=%d)\n", npair, totpair, m, mp); flush(stdout))
    end
    # Whiten each holonomy norm block independently, retaining the SAME global
    # per-momentum cutoff. The Hamiltonian remains coupled across all sectors.
    blas0 = BLAS.get_num_threads()
    levels = Vector{Vector{Float64}}(undef, nk); ranks = zeros(Int, nk)
    try
        BLAS.set_num_threads(solveBLAS)
        verbose && (@printf("      [bloch] assembly done; solving %d momenta (BLAS=%d)...\n", nk, solveBLAS); flush(stdout))
        for ki in 1:nk
            levels[ki], ranks[ki] = sectorNormLevels(Hk[ki], Nk[ki]; nev=nev, cutoff=cutoff)
            Hk[ki] = zeros(ComplexF64, 0, 0); empty!(Nk[ki]); GC.gc()
            verbose && (@printf("      [bloch] solved k=%d (%d/%d, rank %d)\n", kset[ki], ki, nk, ranks[ki]); flush(stdout))
        end
    finally
        BLAS.set_num_threads(blas0)
    end
    return levels, ranks
end

# ---------------------------------------------------------------------------
# ED side: exact symmetry-adapted basis from the orbits of the block translation.
# P acts only on configs (identity on the holonomy sector), so the basis is
# (orbit vector) x (sector), and a length-L orbit contributes to momentum k only when
# k*L = 0 mod Nb.
# ---------------------------------------------------------------------------
function momentumBasisConfig(N, idx)
    Nb = N ÷ 2; n = length(idx)
    pos = Dict(idx[a] => a for a in 1:n)
    shift(lin) = (nl = 0; for i in 0:N-1; nl |= (((lin >> i) & 1) << mod(i + 2, N)); end; nl)
    seen = falses(n); orbits = Vector{Vector{Int}}()
    for a in 1:n
        seen[a] && continue
        orb = Int[]; cur = idx[a] - 1
        while true
            q = pos[cur+1]; push!(orb, q); seen[q] = true
            cur = shift(cur); cur == idx[a] - 1 && break
        end
        push!(orbits, orb)
    end
    Vs = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, Nb)
    for k in 0:Nb-1
        I0 = Int[]; J0 = Int[]; V0 = ComplexF64[]; col = 0
        for orb in orbits
            L = length(orb)
            mod(k * L, Nb) == 0 || continue          # projection vanishes otherwise
            col += 1
            for (r, q) in enumerate(orb)
                push!(I0, q); push!(J0, col)
                push!(V0, exp(-2π * im * k * (r - 1) / Nb) / sqrt(L))
            end
        end
        Vs[k+1] = sparse(I0, J0, V0, n, col)
    end
    return Vs
end

function edDispersionFixed(Nsites, gC, lam, hmax; nev=8, mass=0.0)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    idx = zeroChargeIdx(Nsites); n = length(idx)
    H = holoSparseFixed(Nsites, gC, lam, hmax; mass=mass)
    Vc = momentumBasisConfig(Nsites, idx)
    levels = Vector{Vector{Float64}}(undef, Nb)
    for k in 0:Nb-1
        Vk = Vc[k+1]; dk = size(Vk, 2)
        # full basis = blockdiag over the M sectors (P is the identity on m)
        Vfull = blockdiag([Vk for _ in 1:M]...)
        Hk = Matrix(Vfull' * (H * Vfull))
        ev = sort(real(eigvals(Hermitian((Hk + Hk') / 2))))
        levels[k+1] = ev[1:min(nev, length(ev))]
    end
    return levels
end

# ---------------------------------------------------------------------------
# PORTABLE tensor cache.  Serialization's .jls format is JULIA-VERSION SPECIFIC -- a file
# written by 1.12 fails to deserialize under damini's 1.10.1 -- so the Level-B tensors are
# also written as raw Float64 (M, D, d, E, then re/im of each A_m and each lambda_m),
# which any version reads identically.
function saveTensorsRaw(path, EB, As, lambdas)
    M = length(As); D = size(As[1], 1); d = size(As[1], 3)
    open(path, "w") do io
        write(io, Int64(M), Int64(D), Int64(d), Float64(EB))
        for m in 1:M
            write(io, Array{Float64}(real(As[m])))
            write(io, Array{Float64}(imag(As[m])))
            write(io, Vector{Float64}(lambdas[m]))
        end
    end
end
function loadTensorsRaw(path)
    open(path, "r") do io
        M = read(io, Int64); D = read(io, Int64); d = read(io, Int64); EB = read(io, Float64)
        As = Vector{Array{ComplexF64,3}}(undef, M); lambdas = Vector{Vector{Float64}}(undef, M)
        for m in 1:M
            re = Array{Float64,3}(undef, D, D, d); read!(io, re)
            iv = Array{Float64,3}(undef, D, D, d); read!(io, iv)
            lv = Vector{Float64}(undef, D); read!(io, lv)
            As[m] = complex.(re, iv); lambdas[m] = lv
        end
        return EB, As, lambdas
    end
end

# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
function runDispersion12(; Nsites=12, D=4, gC=0.5, lam=1.0, mass=0.0, hmax=5, nev=8, seed=4,
                          warmSteps=1500, maxit=300, alpha=0.03, out=nothing)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    msuf = mass == 0.0 ? "" : @sprintf("_m%.3f", mass)   # mass=0 keeps the historical names
    out === nothing && (out = "holo_dispersion12_D$D$msuf.csv")
    BLAS.set_num_threads(1)
    @printf("MOMENTUM-RESOLVED (ZMV) holonomy Bloch spectrum: N=%d D=%d gC=%.2f mass=%.3f hmax=%d (M=%d)\n",
            Nsites, D, gC, mass, hmax, M); flush(stdout)

    t = @elapsed edlev = edDispersionFixed(Nsites, gC, lam, hmax; nev=nev, mass=mass)
    Eed0 = minimum(minimum.(edlev))
    @printf("ED (symmetry-adapted, per momentum): E0 = %.10f  E0/N = %.10f  [%.0f s]\n",
            Eed0, Eed0/Nsites, t); flush(stdout)

    base  = @sprintf("levelBfix_N%d_D%d_g%.2f_l%.1f%s_h%d_s%d", Nsites, D, gC, lam, msuf, hmax, seed)
    craw  = base * ".bin"; cache = base * ".jls"
    if isfile(craw)                      # portable raw cache first (cross-Julia-version)
        EB, As, lambdas = loadTensorsRaw(craw)
        @printf("Level B (cached, raw)              : E/N = %.10f\n", EB/Nsites)
    elseif isfile(cache)
        EB, As, lambdas = deserialize(cache)
        saveTensorsRaw(craw, EB, As, lambdas)
        @printf("Level B (cached)                   : E/N = %.10f\n", EB/Nsites)
    else
        t = @elapsed EB, As, lambdas = levelB_GS_fixed(; Nsites=Nsites, hmax=hmax, D=D, gC=gC,
                            lam=lam, mass=mass, warmSteps=warmSteps, maxit=maxit, alpha=alpha,
                            seed=seed, verbose=true)
        serialize(cache, (EB, As, lambdas)); saveTensorsRaw(craw, EB, As, lambdas)
        @printf("Level B                            : E/N = %.10f  [%.0f s]\n", EB/Nsites, t)
    end
    flush(stdout)

    t = @elapsed pmlev, ranks = blochDispersionLeanDiag(As, lambdas, Nsites, gC, lam, hmax; nev=nev, mass=mass)
    Epm0 = minimum(minimum.(pmlev))
    @printf("puMPS Bloch (per momentum)         : E0 = %.10f  E0/N = %.10f  [%.0f s]\n",
            Epm0, Epm0/Nsites, t); flush(stdout)

    # The tangent convention |t_k> = sum_x e^{+i theta_k (x-1)} gives P|t_k> = e^{-i theta_k}|t_k>,
    # while the ED basis has P v_k = e^{+i theta_k} v_k -- so the puMPS label k matches ED's
    # Nb-k.  Decide empirically (the direct and reflected assignments are both tried).
    err(f) = maximum(maximum(abs.((pmlev[f(ki)] .- Epm0) .- (edlev[ki] .- Eed0))) for ki in 1:Nb)
    dir = err(ki -> ki); ref = err(ki -> mod(Nb - (ki-1), Nb) + 1)
    flip = ref < dir
    @printf("momentum convention: direct err %.3e, reflected err %.3e -> using %s\n",
            dir, ref, flip ? "REFLECTED (k_puMPS = Nb - k_ED)" : "direct"); flush(stdout)
    pmap(ki) = flip ? mod(Nb - (ki-1), Nb) + 1 : ki

    open(out, "w") do io
        println(io, "k,p_over_pi,level,dE_ED,dE_puMPS,abs_err")
        for ki in 1:Nb, l in 1:nev
            de = edlev[ki][l] - Eed0; dp = pmlev[pmap(ki)][l] - Epm0
            println(io, "$(ki-1),$(2*(ki-1)/Nb),$l,$de,$dp,$(abs(dp-de))")
        end
    end
    println("\n   k   p/pi    level     dE_ED        dE_puMPS      |err|")
    for ki in 1:Nb
        for l in 1:min(4, nev)
            de = edlev[ki][l] - Eed0; dp = pmlev[pmap(ki)][l] - Epm0
            @printf("   %d   %.3f     %d     %10.6f   %10.6f   %.2e\n",
                    ki-1, 2*(ki-1)/Nb, l, de, dp, abs(dp - de))
        end
        println()
    end
    println("wrote $out")
    return edlev, pmlev
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "small"
        # quick end-to-end check at N=6/8 where Level B is cheap and ED is exact
        mass = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0
        for (N, D, hmax) in ((6, 4, 1), (8, 4, 1))
            runDispersion12(; Nsites=N, D=D, hmax=hmax, nev=6, mass=mass,
                            out="holo_dispersion_N$(N)_D$(D)$(mass == 0.0 ? "" : "_m$mass").csv")
            println(repeat("-", 70))
        end
    else
        D    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
        hmax = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
        nev  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
        mass = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.0
        runDispersion12(; D=D, hmax=hmax, nev=nev, mass=mass)
    end
end
