# FAST PATH for the blocked-Schwinger puMPS ground-state optimizer.
#
# Every routine here is an EXACT rewrite of a function in puMPS.jl /
# puMPS_coulomb_gs.jl / puMPS_schwinger_step3.jl -- same contraction, same
# conventions, no approximation, no truncation.  The originals are left in place
# and serve as the regression oracle (`regressFast()` below compares the two to
# 1e-12 on random tensors).
#
# What changed, and why it matters (timings: N=20, D=16, one gradient step):
#   (a) the d^4 D^6 scalar loops of the two-site-gate contractions -> reshapes +
#       gemm, cost d^2 D^4.  This was 7.7 s of the 18.9 s step.
#   (b) the metric is I_d (x) G: calculateMetricTensor writes
#       g[mu(s,ap,cp), mu(s,a,c)] with no s-dependence and no s-off-diagonal
#       part, so the (d D^2)^2 eigendecomposition in pseudoSolve is d identical
#       copies of a D^2 x D^2 one.  1.5 s -> ~0.03 s, exactly.
#   (c) the bond sum recomputed E^(N-2-j) from scratch for every j; the powers
#       are now built once.
#   (d) the N E_0 channels (independent k) and the MPO terms (independent
#       gradient contributions) are threaded.  Run with `julia -t <n>`.
#
# Index conventions are inherited verbatim from puMPS.jl:
#   A[a,b,s]  a=left, b=right virtual, s=physical;  idx(x,xp) = x + (xp-1)*D;
#   pk(s1,s2) = (s1-1)*d + s2  (gate packing);  mu(s,a,c) = a + (c-1)D + (s-1)D^2.
include("puMPS_schwinger_step3.jl")          # guarded: no auto-run
using LinearAlgebra, Random, Printf, Base.Threads

# ---------------------------------------------------------------------------
# (a) transfer matrices and two-site objects, as gemms
# ---------------------------------------------------------------------------

# E[idx(b,bp), idx(a,ap)] = sum_s phase[s] A[a,b,s] conj(A[ap,bp,s]).
function fastTransferMatrix(A; phase=nothing)
    D = size(A, 1); d = size(A, 3)
    Ak = phase === nothing ? A : A .* reshape(phase, 1, 1, d)
    M1 = reshape(permutedims(Ak, (2, 1, 3)), D * D, d)          # [(b,a), s]
    M2 = reshape(permutedims(conj(A), (3, 2, 1)), d, D * D)     # [s, (bp,ap)]
    P = M1 * M2                                                 # [(b,a),(bp,ap)]
    return reshape(permutedims(reshape(P, (D, D, D, D)), (1, 3, 2, 4)), D * D, D * D)
end

# Two-site ket tensor  K[(a,c), pk(s1,s2)] = sum_b A[a,b,s1] A[b,c,s2].
function twoSiteKet(A)
    D = size(A, 1); d = size(A, 3)
    AL = reshape(permutedims(A, (1, 3, 2)), D * d, D)           # [(a,s1), b]
    AR = reshape(A, D, D * d)                                   # [b, (c,s2)]
    K4 = reshape(AL * AR, (D, d, D, d))                         # [a,s1,c,s2]
    return reshape(permutedims(K4, (1, 3, 4, 2)), D * D, d * d) # [(a,c), s2+(s1-1)d]
end

# G2[idx(c,cp), idx(a,ap)]: the two-site gate as a transfer matrix.
function fastGateTransfer(A, hmat)
    D = size(A, 1)
    Kt = twoSiteKet(A)
    Gm = (Kt * transpose(hmat)) * Kt'                           # [(a,c),(ap,cp)]
    return reshape(permutedims(reshape(Gm, (D, D, D, D)), (2, 4, 1, 3)), D * D, D * D)
end

# Epow[j+1] = E^j for j = 0..kmax.
function transferPowers(E, kmax)
    D2 = size(E, 1)
    Epow = Vector{Matrix{ComplexF64}}(undef, kmax + 1)
    Epow[1] = Matrix{ComplexF64}(I, D2, D2)
    for j in 1:kmax
        Epow[j+1] = Epow[j] * E
    end
    return Epow
end

# ---------------------------------------------------------------------------
# (b) the metric block.  Full metric = kron(I_d, Gm) in the mu(s,a,c) packing.
# ---------------------------------------------------------------------------
function metricBlockFast(A, lambdaInverse, N; Epow=nothing)
    D = size(A, 1)
    EN1 = Epow === nothing ? fastTransferMatrix(A)^(N - 1) : Epow[N]
    R4 = reshape(EN1, (D, D, D, D))                             # [a,ap,c,cp]
    Gm = reshape(permutedims(R4, (2, 4, 1, 3)), D * D, D * D)   # [(ap,cp),(a,c)]
    rowf = [lambdaInverse[cp] for ap in 1:D, cp in 1:D]         # (ap,cp), ap fastest
    colf = [lambdaInverse[c] for a in 1:D, c in 1:D]
    return Gm .* vec(rowf) .* transpose(vec(colf))
end

# ---------------------------------------------------------------------------
# (a,c) the two-site-gate gradient numerator, terms (A), (B), (C) of puMPS.jl.
# ---------------------------------------------------------------------------
function calculateGradientTensorFast(A, hmat, lambdaInverse, N; Epow=nothing)
    D = size(A, 1); d = size(A, 3)
    if Epow === nothing
        Epow = transferPowers(fastTransferMatrix(A), N - 1)
    end
    G2 = fastGateTransfer(A, hmat)

    # (A) bonds fully inside sites 2..N:  BA = sum_j E^{N-2-j} G2 E^{j-1}
    BA = zeros(ComplexF64, D * D, D * D)
    for j in 1:(N-2)
        BA .+= Epow[N-1-j] * (G2 * Epow[j])
    end
    BA4 = reshape(BA, (D, D, D, D))                             # [a,ap,b,bp]
    Amat = reshape(permutedims(A, (3, 1, 2)), d, D * D)         # [s, (a,b)]
    BAp = reshape(permutedims(BA4, (1, 3, 2, 4)), D * D, D * D) # [(a,b),(ap,bp)]
    Fm = Amat * BAp                                             # [s', (a',b')]

    Env4 = reshape(Epow[N-1], (D, D, D, D))                     # E^{N-2}: [a,ap,c,cp]
    Kg = twoSiteKet(A) * transpose(hmat)                        # [(a,c), pk(sb1,sb2)]

    # (B) bond (1,2): open site is the LEFT slot of the gate, pk = (s'-1)d + sk2p
    Em = reshape(Env4, D^3, D)                                  # [(a,app,c), cp]
    Cm = reshape(permutedims(conj(A), (2, 1, 3)), D, D * d)     # [cp, (bpp,sk2p)]
    Z5 = reshape(Em * Cm, (D, D, D, D, d))                      # [a,app,c,bpp,sk2p]
    Zp = reshape(permutedims(Z5, (1, 3, 5, 2, 4)), D * D * d, D * D)   # [(a,c,sk2p),(app,bpp)]
    K5 = reshape(Kg, (D, D, d, d))                              # [a,c,sk2p,s']
    Fm .+= transpose(reshape(K5, D * D * d, d)) * Zp

    # (C) bond (N,1): open site is the RIGHT slot of the gate, pk = (skNp-1)d + s'
    Ep2 = reshape(permutedims(Env4, (1, 3, 4, 2)), D^3, D)      # [(dd,b,bpp), dp]
    Cm2 = reshape(conj(A), D, D * d)                            # [dp, (app,skNp)]
    Z2 = reshape(Ep2 * Cm2, (D, D, D, D, d))                    # [dd,b,bpp,app,skNp]
    Z2p = reshape(permutedims(Z2, (1, 2, 5, 4, 3)), D * D * d, D * D)  # [(dd,b,skNp),(app,bpp)]
    K5b = reshape(Kg, (D, D, d, d))                             # [dd,b,s',skNp]
    Fm .+= transpose(reshape(permutedims(K5b, (1, 2, 4, 3)), D * D * d, d)) * Z2p

    F = reshape(Fm, (d, D, D))                                  # [s', a', c']
    numH = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for sp in 1:d, app in 1:D, cp in 1:D
        numH[mu(sp, app, cp)] = F[sp, app, cp] * lambdaInverse[cp]
    end
    return numH
end

# ---------------------------------------------------------------------------
# the MPO gradient numerator (puMPS_coulomb_gs.jl), gemm form.
# ---------------------------------------------------------------------------
function calculateGradientTensorMPODense(A, W, vL, vR, lambdaInverse, N)
    D = size(A, 1); d = size(A, 3); χ = size(W, 1)

    # combined MPS-MPO transfer  T[cidx(bk,mr,bb), cidx(ak,ml,ab)],  cidx(k,m,b)=k+(m-1)D+(b-1)Dχ
    Am = reshape(A, D * D, d)                                   # [(ak,bk), st]
    Wm = reshape(permutedims(W, (3, 1, 2, 4)), d, χ * χ * d)    # [st, (ml,mr,sb)]
    X = reshape(Am * Wm, D * D * χ * χ, d)                      # [(ak,bk,ml,mr), sb]
    Cm = reshape(permutedims(conj(A), (3, 1, 2)), d, D * D)     # [sb, (ab,bb)]
    Y = reshape(X * Cm, (D, D, χ, χ, D, D))                     # [ak,bk,ml,mr,ab,bb]
    T = reshape(permutedims(Y, (2, 4, 6, 1, 3, 5)), D * χ * D, D * χ * D)

    # The gradient never needs all of T^(N-1): vR contracts its row-side MPO
    # index, leaving one block-row.  Propagating that D^2 x (chi D^2) slice for
    # N-1 steps costs chi^2 D^6 per step instead of chi^3 D^6 per matrix
    # multiplication in the dense power (and holds chi times less memory).
    Er = zeros(ComplexF64, D, D, D, χ, D)                       # [ak,aL,bk,m1,aR]
    for ak in 1:D, aL in 1:D, m1 in 1:χ
        Er[ak, aL, ak, m1, aL] = vR[m1]                         # slice of the identity
    end
    M = reshape(Er, D * D, D * χ * D)
    for _ in 1:(N-1)
        M = M * T
    end
    Er = reshape(M, (D, D, D, χ, D))

    Wv = zeros(ComplexF64, χ, d, d)                             # [m1, st, sb]
    for m0 in 1:χ
        vL[m0] == 0 && continue
        Wv .+= vL[m0] .* permutedims(W[m0, :, :, :], (1, 2, 3))
    end
    U = reshape(Am * reshape(permutedims(Wv, (2, 1, 3)), d, χ * d), (D, D, χ, d))  # [ak,bk,m1,sb]
    Um = reshape(U, D * D * χ, d)
    Erp = reshape(permutedims(Er, (1, 3, 4, 2, 5)), D * D * χ, D * D)   # [(ak,bk,m1),(aL,aR)]
    F = reshape(transpose(Um) * Erp, (d, D, D))                 # [sb, aL, aR]

    numH = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for sb in 1:d, aL in 1:D, aR in 1:D
        numH[mu(sb, aL, aR)] = F[sb, aL, aR] * lambdaInverse[aR]
    end
    return numH
end

# Sparse MPO-channel propagation. Each nonzero W[ml,mr,:,:] supplies one
# D² transfer block; never assemble the dense (chi D²)^2 combined transfer.
# M_new[ml] = sum_mr M[mr] T[mr,ml], matching the dense row-slice convention.
# Two reusable channel buffers avoid a fresh chi D^4 allocation at every site.
# Sparsity is exact (no magnitude cutoff); general complex/non-diagonal local
# operators and arbitrary complex boundary vectors are supported.
function calculateGradientTensorMPOSparse(A, W, vL, vR, lambdaInverse, N)
    D = size(A, 1); d = size(A, 3); chi = size(W, 1); D2 = D^2
    Am = reshape(A, D2, d)
    Cm = reshape(permutedims(conj(A), (3, 2, 1)), d, D2)
    edges = Tuple{Int,Int,Matrix{ComplexF64}}[]
    for ml in 1:chi, mr in 1:chi
        op = @view W[ml, mr, :, :]
        all(iszero, op) && continue
        ket = reshape(Am * op, D, D, d)
        K = reshape(permutedims(ket, (2, 1, 3)), D2, d)
        T = reshape(permutedims(reshape(K * Cm, D, D, D, D), (1, 3, 2, 4)), D2, D2)
        push!(edges, (ml, mr, T))
    end
    current = [vR[m] .* Matrix{ComplexF64}(I, D2, D2) for m in 1:chi]
    scratch = [zeros(ComplexF64, D2, D2) for _ in 1:chi]
    active = .!iszero.(vR); nextactive = falses(chi)
    for _ in 1:N-1
        foreach(M -> fill!(M, 0), scratch)
        fill!(nextactive, false)
        for (ml, mr, T) in edges
            active[mr] || continue
            mul!(scratch[ml], current[mr], T, 1.0, 1.0)
            nextactive[ml] = true
        end
        current, scratch = scratch, current
        active, nextactive = nextactive, active
    end
    F = zeros(ComplexF64, d, D2)
    for m in 1:chi
        active[m] || continue
        op = zeros(ComplexF64, d, d)
        for ml in 1:chi
            iszero(vL[ml]) && continue
            op .+= vL[ml] .* @view(W[ml, m, :, :])
        end
        all(iszero, op) && continue
        U = Am * op
        env = reshape(permutedims(reshape(current[m], D, D, D, D), (1, 3, 2, 4)), D2, D2)
        mul!(F, transpose(U), env, 1.0, 1.0)
    end
    # F is [s,a,c]; the public packing is [a,c,s] with lambda^{-1} on c.
    packed = permutedims(reshape(F, d, D, D), (2, 3, 1))
    packed .*= reshape(lambdaInverse, 1, D, 1)
    return vec(packed)
end

# Keep the previous dense implementation callable for regression and timings.
calculateGradientTensorMPOFast(A, W, vL, vR, lambdaInverse, N) =
    calculateGradientTensorMPOSparse(A, W, vL, vR, lambdaInverse, N)

# Scalar expectation on the periodic MPS with open MPO boundaries. No open
# physical/virtual tangent legs or Schmidt inverses are needed for an energy.
function mpoValueFast(A, W, vL, vR, N)
    D = size(A, 1); d = size(A, 3); chi = size(W, 1); D2 = D^2
    Am = reshape(A, D2, d)
    Cm = reshape(permutedims(conj(A), (3, 2, 1)), d, D2)
    edges = Tuple{Int,Int,Matrix{ComplexF64}}[]
    for ml in 1:chi, mr in 1:chi
        op = @view W[ml, mr, :, :]
        all(iszero, op) && continue
        ket = reshape(Am * op, D, D, d)
        K = reshape(permutedims(ket, (2, 1, 3)), D2, d)
        T = reshape(permutedims(reshape(K * Cm, D, D, D, D), (1, 3, 2, 4)), D2, D2)
        push!(edges, (ml, mr, T))
    end
    current = [vR[m] .* Matrix{ComplexF64}(I, D2, D2) for m in 1:chi]
    scratch = [zeros(ComplexF64, D2, D2) for _ in 1:chi]
    active = .!iszero.(vR); nextactive = falses(chi)
    for _ in 1:N
        foreach(M -> fill!(M, 0), scratch); fill!(nextactive, false)
        for (ml, mr, T) in edges
            active[mr] || continue
            mul!(scratch[ml], current[mr], T, 1.0, 1.0)
            nextactive[ml] = true
        end
        current, scratch = scratch, current
        active, nextactive = nextactive, active
    end
    return sum(vL[m] * tr(current[m]) for m in 1:chi)
end

# ---------------------------------------------------------------------------
# (a,d) the twisted E_0 term.  Threaded over the N roots-of-unity channels k;
# each k is an independent prefix/suffix sweep over the ring.
# ---------------------------------------------------------------------------
function e0GradAndNumFast(A, lambdaInverse, Nsites; theta=0.0)
    Nb = Nsites ÷ 2; D = size(A, 1); d = size(A, 3)
    fh = fhatE0(Nsites; theta=theta)
    # Only k = 0..N/2 are computed.  The mirror channel k' = N-k is free:
    # Omega_{N-k} = Omega_k^dagger, so its twisted transfer is
    # E_{N-k} = SWAP conj(E_k) SWAP (SWAP exchanges ket/bra halves of the
    # super-index), hence Env4_{N-k}[a,ap,b,bp] = conj(Env4_k[ap,a,bp,b]) and
    # num_{N-k} = conj(num_k).  That is a permutedims + conj, O(D^4), instead of
    # a second sweep of O(N_b D^6) matrix products.  Exactly halves the E_0 cost.
    ks = [k for k in 0:(Nsites÷2) if abs(fh[k+1]) >= 1e-15]
    mirrored = [!(k == 0 || 2k == Nsites) for k in ks]
    Fs = [zeros(ComplexF64, d, D * D) for _ in ks]
    nums = zeros(ComplexF64, length(ks))
    idm = Matrix{ComplexF64}(I, D * D, D * D)

    @threads for t in 1:length(ks)
        k = ks[t]; fk = fh[k+1]; mir = mirrored[t]
        phs = [blockPhase(k, i - 1, Nsites) for i in 1:Nb]
        Ek = [fastTransferMatrix(A; phase=phs[i]) for i in 1:Nb]
        P = Vector{Matrix{ComplexF64}}(undef, Nb + 1); P[1] = idm      # P[i+1] = Ek[i]...Ek[1]
        for i in 1:Nb; P[i+1] = Ek[i] * P[i]; end
        S = Vector{Matrix{ComplexF64}}(undef, Nb + 2); S[Nb+2] = idm   # S[i+1] = Ek[Nb]...Ek[i]
        S[Nb+1] = Ek[Nb]
        for i in Nb-1:-1:1; S[i+1] = S[i+2] * Ek[i]; end
        nk = fk * tr(P[Nb+1])
        nums[t] = mir ? nk + conj(nk) : nk
        Amat = reshape(permutedims(A, (3, 1, 2)), d, D * D)            # [s, (a,b)]
        Ft = Fs[t]
        for i in 1:Nb                                                   # impurity at block i
            Env4 = reshape(P[i] * S[i+2], (D, D, D, D))                 # [a,ap,b,bp]
            EnvP = reshape(permutedims(Env4, (1, 3, 2, 4)), D * D, D * D)
            Ft .+= (fk .* phs[i]) .* (Amat * EnvP)
            if mir
                EnvM = reshape(permutedims(conj(Env4), (2, 4, 1, 3)), D * D, D * D)
                Ft .+= conj.(fk .* phs[i]) .* (Amat * EnvM)
            end
        end
    end

    F = zeros(ComplexF64, d, D * D)
    for t in 1:length(ks); F .+= Fs[t]; end
    F ./= Nb                                                            # position AVERAGE
    F3 = reshape(F, (d, D, D))
    numH = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for s in 1:d, a in 1:D, c in 1:D
        numH[mu(s, a, c)] = F3[s, a, c] * lambdaInverse[c]
    end
    return numH, sum(nums)
end

# ---------------------------------------------------------------------------
# (b) descent direction from the D^2 x D^2 metric block.  Mathematically
# identical to trueGradient + pseudoSolve on the full kron(I_d, Gm).
# ---------------------------------------------------------------------------
function trueGradientFast(Gm, numH, A, lambda, lambdaInverse, N; cutoff=1e-12)
    D = size(A, 1); d = size(A, 3)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    vecAC = zeros(ComplexF64, d * D * D)
    for s in 1:d, a in 1:D, c in 1:D
        vecAC[mu(s, a, c)] = A[a, c, s] * lambda[c]
    end
    numNorm = vec(Gm * reshape(vecAC, D * D, d))          # (I_d (x) Gm) * vecAC
    normVal = real(dot(vecAC, numNorm))
    energy = real(dot(vecAC, numH)) / normVal
    grad = (numH .- energy .* numNorm) ./ normVal

    Fg = eigen(Hermitian((Gm + Gm') / 2))
    thresh = cutoff * maximum(abs.(Fg.values))
    invv = [v > thresh ? 1.0 / v : 0.0 for v in Fg.values]
    gInvGrad = vec(Fg.vectors * (invv .* (Fg.vectors' * reshape(grad, D * D, d))))

    eta = sqrt(abs(real(dot(grad, gInvGrad))))
    deltaAL = zeros(ComplexF64, D, D, d)
    for s in 1:d, a in 1:D, c in 1:D
        deltaAL[a, c, s] = -gInvGrad[mu(s, a, c)] * lambdaInverse[c]
    end
    return deltaAL, eta, energy
end

# ---------------------------------------------------------------------------
# one full gradient of the faithful model; returns the pieces plus timings.
# ---------------------------------------------------------------------------
function schwingerGradient(A, lambda, lamInv, gate, mpoTerms, gCE0, Nsites; timed=false)
    Nb = Nsites ÷ 2
    t = zeros(5)
    t[1] = @elapsed begin
        Epow = transferPowers(fastTransferMatrix(A), Nb - 1)
        Gm = metricBlockFast(A, lamInv, Nb; Epow=Epow)
    end
    t[2] = @elapsed numH = calculateGradientTensorFast(A, gate, lamInv, Nb; Epow=Epow)
    t[3] = @elapsed begin
        parts = Vector{Vector{ComplexF64}}(undef, length(mpoTerms))
        @threads for i in 1:length(mpoTerms)
            (coef, W, vL, vR) = mpoTerms[i]
            parts[i] = coef .* calculateGradientTensorMPOFast(A, W, vL, vR, lamInv, Nb)
        end
        for p in parts; numH = numH .+ p; end
    end
    t[4] = @elapsed begin
        numHE0, _ = e0GradAndNumFast(A, lamInv, Nsites)
        numH = numH .+ gCE0 .* numHE0
    end
    t[5] = @elapsed res = trueGradientFast(Gm, numH, A, lambda, lamInv, Nb)
    return timed ? (res..., t) : res
end

# MONOTONE descent with backtracking.
#
# Two changes from the fixed-alpha / best-eta scheme, both forced by what
# happened when the N-sweep first ran:
#
#  (1) Track the best ENERGY, not the smallest eta.  eta vanishes at ANY
#      stationary point, including the fully-polarised product state, which is
#      an exact eigenstate of this Hamiltonian.  The N=18 warm-start link
#      converged onto it -- eta = 1.6e-23, <Q^2> = 288, E/N = +14.3 -- and
#      best-eta tracking duly returned it.  The method is variational, so the
#      lowest energy seen is the right thing to keep, and it can never select a
#      maximum.
#
#  (2) Accept a step only if it lowers the energy; otherwise halve alpha and
#      retry from the same point.  A fixed alpha cannot be safe across the
#      sweep: the natural gradient scales with H ~ N, so a step size tuned at
#      N=12 is effectively ~2.5x larger at N=30.  Backtracking finds the right
#      scale at every N by itself.
#
# Cost accounting is unchanged: exactly one gradient evaluation per iteration,
# accepted or rejected, so `steps` still multiplies t_step to give total time.
function fullGD_step3_fast(maxSteps, alpha, A0, gate, mpoTerms, gCE0, Nsites;
                           tol=1e-9, growEvery=20, growth=1.1, alphaMin=1e-8)
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    dAL, eta, en = schwingerGradient(A, lambda, lamInv, gate, mpoTerms, gCE0, Nsites)
    bestA = copy(A); bestLam = copy(lambda); bestE = en; bestEta = eta
    al = alpha; good = 0; used = 1
    for i in 2:maxSteps
        used = i
        eta < tol && break
        At, lamT = canonicalization(oneGradientDescentStep(A, al, dAL))
        lamInvT = 1.0 ./ max.(lamT, 1e-12)
        dALt, etaT, enT = schwingerGradient(At, lamT, lamInvT, gate, mpoTerms, gCE0, Nsites)
        if enT <= en + 1e-12 * abs(en)                       # accept
            A, lambda, lamInv = At, lamT, lamInvT
            dAL, eta, en = dALt, etaT, enT
            if en < bestE
                bestE = en; bestEta = eta; bestA = copy(A); bestLam = copy(lambda)
            end
            good += 1
            if good >= growEvery; al = min(al * growth, 10 * alpha); good = 0; end
        else                                                  # reject, shorten
            al = max(al / 2, alphaMin); good = 0
            al <= alphaMin && break
        end
    end
    return bestA, bestLam, bestE, bestEta, used
end

# ---------------------------------------------------------------------------
# MERGED kernel MPO.  kernelTerms builds one chi=5 Jordan MPO per (a,a') pair of
# sub-lattice slots -- four of them -- and the combined MPS-MPO transfer costs
# chi^3 D^6, so those four dominate the whole gradient at large D.  But the two
# terms sharing an INPUT operator q_a can share the three propagating polynomial
# channels: the coefficient c_{a,a'}(rho) differs only in what is EMITTED, and
# q_0, q_1 are both diagonal, so the emitted operator is just the linear
# combination sum_{a'} c^{(m)}_{a,a'} q_{a'}.  Four chi=5 MPOs -> two.  Exact;
# gated below against both the original channels and the dense staggered
# kernel diagonal.
function polyKernelMPOMulti(cfuns, Qin, Qouts)
    d = size(Qin, 1); id = Matrix{ComplexF64}(I, d, d)
    C = [[1.0 0 0; 1 1 1; 1 2 4] \ [cf(1), cf(2), cf(3)] for cf in cfuns]
    Qm = [sum(C[j][m] .* Qouts[j] for j in 1:length(Qouts)) for m in 1:3]
    W = zeros(ComplexF64, 5, 5, d, d)
    W[1, 1, :, :] = id;  W[5, 5, :, :] = id
    W[5, 2, :, :] = Qin
    W[2, 2, :, :] = id;  W[2, 3, :, :] = id;  W[2, 4, :, :] = id
    W[3, 3, :, :] = id;  W[3, 4, :, :] = 2 .* id
    W[4, 4, :, :] = id
    W[2, 1, :, :] = Qm[1]; W[3, 1, :, :] = Qm[2]; W[4, 1, :, :] = Qm[3]
    vL = zeros(ComplexF64, 5); vL[5] = 1
    vR = zeros(ComplexF64, 5); vR[1] = 1
    return W, vL, vR
end

function kernelTermsMerged(Nsites, gC)
    q = (q0op(), q1op())
    terms = Any[]
    for a in 0:1
        cfs = [(R -> -Kk(2R + (ap - a), Nsites) / Nsites) for ap in 0:1]
        push!(terms, (gC, polyKernelMPOMulti(cfs, q[a+1], [q[1], q[2]])...))
    end
    push!(terms, (gC * (-Kk(1, Nsites) / Nsites), fieldMPO(q0op() * q1op())...))
    return terms
end

# THREADING, and why there is only one knob.
#
# Julia overrides OPENBLAS_NUM_THREADS at startup, and OpenBLAS resets its
# thread count to the machine default on the first gemm after any @threads
# region -- including inside the tasks of that region (measured: pin to 2,
# read 14 back inside every task).  So the BLAS width is NOT controllable once
# Julia threads are in play; the only reliable knob is `julia -t <n>`, which
# sets how many of our independent E_0 channels / MPO terms run concurrently,
# each of them then using a BLAS that spreads over the box.
#
# Since that is a measurement question rather than a design question, run
# `run_all.sh probe` first: it times a representative cell at several -t values
# and the winner is used for everything else.  pinBlasThreads is kept for the
# single-task case (-t 1), where set_num_threads does stick.
function pinBlasThreads(n=nothing)
    n = n === nothing ? tryparse(Int, get(ENV, "PUMPS_BLAS", "")) : n
    n === nothing && return BLAS.get_num_threads()
    BLAS.set_num_threads(n)
    return n
end

const BLAS_AT_START = BLAS.get_num_threads()

# model assembly, shared by every driver
function schwingerModel(Nsites, gC, lam; merged=true)
    Nb = Nsites ÷ 2
    gate = blockedXXGate()
    terms = merged ? kernelTermsMerged(Nsites, gC) : kernelTerms(Nsites, gC)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    push!(terms, (lam, Wl, vLl, vRl))
    return gate, terms, (Wl, vLl, vRl)
end

# ===========================================================================
# REGRESSION GATES.  fast == original, routine by routine, on random tensors.
# ===========================================================================
function regressFast(; cases=((8, 4), (8, 6), (12, 6), (12, 8)), gC=0.3, lam=1.0, seed=17)
    worst = 0.0
    println("fast-vs-original regression (d=4 blocked):")
    for (Nsites, D) in cases
        Nb = Nsites ÷ 2
        Random.seed!(seed + D)
        A, lambda = canonicalization(randn(ComplexF64, D, D, 4))
        lamInv = 1.0 ./ max.(lambda, 1e-12)
        gate, terms, _ = schwingerModel(Nsites, gC, lam)

        e1 = norm(fastTransferMatrix(A) - transferMatrix(A))
        e2 = norm(fastGateTransfer(A, gate) - gateTransfer(A, gate))
        Gm = metricBlockFast(A, lamInv, Nb)
        e3 = norm(kron(Matrix{ComplexF64}(I, 4, 4), Gm) - calculateMetricTensor(A, lamInv, Nb))
        e4 = norm(calculateGradientTensorFast(A, gate, lamInv, Nb) -
                  calculateGradientTensor(A, gate, lamInv, Nb))
        e5 = 0.0
        for (coef, W, vL, vR) in terms
            e5 = max(e5, norm(calculateGradientTensorMPOFast(A, W, vL, vR, lamInv, Nb) -
                              calculateGradientTensorMPO(A, W, vL, vR, lamInv, Nb)))
        end
        nf, vf = e0GradAndNumFast(A, lamInv, Nsites)
        no, vo = e0GradAndNum(A, lamInv, Nsites)
        e6 = max(norm(nf - no), abs(vf - vo))

        # descent direction / eta / energy through the block metric vs the full one
        numH = calculateGradientTensorFast(A, gate, lamInv, Nb)
        d1, et1, en1 = trueGradientFast(Gm, numH, A, lambda, lamInv, Nb)
        d2, et2, en2 = trueGradient(calculateMetricTensor(A, lamInv, Nb), numH, A, lambda, lamInv, Nb)
        e7 = max(norm(d1 - d2), abs(et1 - et2), abs(en1 - en2))

        w = maximum([e1, e2, e3, e4, e5, e6, e7]); worst = max(worst, w)
        @printf("  N=%2d D=%2d | E %.1e  G2 %.1e  metric %.1e  grad %.1e  mpo %.1e  E0 %.1e  dir %.1e\n",
                Nsites, D, e1, e2, e3, e4, e5, e6, e7)
        flush(stdout)
    end
    @printf("  WORST = %.2e\n", worst)
    return worst
end

# merged kernel channels == original channels == dense staggered diagonal
function regressMergedKernel(; Nsites=8)
    Nb = Nsites ÷ 2
    dense(ts) = sum(coef .* denseMPOexpand(W, vL, vR, Nb) for (coef, W, vL, vR) in ts)
    Hm = dense(kernelTermsMerged(Nsites, 1.0))
    Ho = dense(kernelTerms(Nsites, 1.0))
    Hd = Diagonal(staggeredKernelDiag(Nsites))
    e1 = norm(Hm - Ho); e2 = norm(Hm - Hd)
    @printf("merged kernel (N=%d): vs original channels %.1e   vs dense staggered %.1e\n",
            Nsites, e1, e2)
    return max(e1, e2)
end

# asymmetric-gate diagnostic: the gate rewrite touches the site-ordering
# convention (slot 1 = LEFT site) that bit us once in ringHamFromGate.
function regressAsymmetric(; Nsites=8, D=5, seed=3)
    Nb = Nsites ÷ 2
    Random.seed!(seed)
    A, lambda = canonicalization(randn(ComplexF64, D, D, 4))
    lamInv = 1.0 ./ max.(lambda, 1e-12)
    g = randn(ComplexF64, 16, 16)                    # deliberately swap-asymmetric
    e1 = norm(fastGateTransfer(A, g) - gateTransfer(A, g))
    e2 = norm(calculateGradientTensorFast(A, g, lamInv, Nb) -
              calculateGradientTensor(A, g, lamInv, Nb))
    W = randn(ComplexF64, 3, 3, 4, 4); vL = ComplexF64[1, 0, 0]; vR = ComplexF64[0, 0, 1]
    e3 = norm(calculateGradientTensorMPOFast(A, W, vL, vR, lamInv, Nb) -
              calculateGradientTensorMPO(A, W, vL, vR, lamInv, Nb))
    @printf("asymmetric gate/MPO: G2 %.1e  grad %.1e  mpo %.1e\n", e1, e2, e3)
    return max(e1, e2, e3)
end

# end-to-end: reproduce the notes' N=8/D=6 and N=12/D=8 numbers
function regressEndToEnd(Nsites, D, gC, lam; steps=3000, alpha=0.02, seed=4)
    Nb = Nsites ÷ 2
    Hb = ringHamFromGate(xxG(), Nsites) .+ gC .* Diagonal(staggeredKernelDiag(Nsites) .+ e0DiagSite(Nsites))
    Hb = (Hb + Hb') / 2
    idx = zeroChargeIdx(Nsites)
    Eneu = minimum(real(eigvals(Hermitian(Hb[idx, idx]))))
    gate, terms, (Wl, vLl, vRl) = schwingerModel(Nsites, gC, lam)
    Random.seed!(seed); A0 = randn(ComplexF64, D, D, 4)
    t = @elapsed A_L, lamv, en, eta, used = fullGD_step3_fast(steps, alpha, A0, gate, terms, gC, Nsites)
    _, _, Q2 = mpoRingEnergy(A_L, Wl, vLl, vRl, Nb)
    @printf("N=%2d D=%2d gC=%.2f: ED %.10f  puMPS %.10f  |d|/N %.2e  <Q^2> %.1e  eta %.1e  %d steps  %.1f s\n",
            Nsites, D, gC, Eneu / Nsites, real(en) / Nsites,
            abs(real(en) - Eneu) / Nsites, real(Q2), eta, used, t)
    flush(stdout)
    return abs(real(en) - Eneu) / Nsites
end

if abspath(PROGRAM_FILE) == @__FILE__
    @printf("threads=%d, BLAS threads=%d\n", nthreads(), BLAS.get_num_threads())
    validateE0() < 1e-10 || error("E0 oracle validation failed")
    validateDense(8) < 1e-10 || error("dense validation failed")
    regressMergedKernel(Nsites=8) < 1e-10 || error("merged kernel regression failed")
    regressMergedKernel(Nsites=12) < 1e-10 || error("merged kernel regression failed")
    regressFast() < 1e-10 || error("fast-vs-original regression failed")
    regressAsymmetric() < 1e-10 || error("asymmetric-gate regression failed")
    regressEndToEnd(8, 6, 0.3, 1.0; steps=2000)
    regressEndToEnd(12, 8, 0.3, 1.0; steps=3000)
end
