# FUSED tangent block for a ONE-WEIGHT operator:  O = sum_{B0} w(B0)@B0 * prod_B P(B).
#
# crossEBlock currently calls diagPhasedBlock once per weight position B0 (Nb = 6 calls),
# and each call redoes the whole O(Nb^2) tangent-tangent geometry -- so the dominant
# O(dt^2 dim^2) contraction runs Nb^3 times.  It is the hot spot of the corrected
# assembly (it runs over all M(M-1) sector pairs, vs M sectors for e2Block).
#
# THE WIN: for a fixed impurity pair (x, y) the sum over B0 can be collapsed at the
# dim x dim SEGMENT level, BEFORE the expensive dt-level contraction:
#   - B0 in one of the plain arcs / the middle segment  -> replace that segment by its
#     "one weight somewhere inside" version (seg1), an O(Nb dim^3) recurrence;
#   - B0 on the bra impurity -> use the weighted bra impurity Jg;
#   - B0 on the ket impurity -> use the weighted ket impurity Kg.
# The first three share the SAME ket impurity, so their Gmats add before contracting:
# 2 expensive dt^2 dim^2 products per (x,y) instead of Nb.  And Rraw[x,y] is
# MOMENTUM-FREE (same trick as the fast diagPhasedBlock), so extra momentum shells cost
# only a scalar phase accumulation.
#
# Validated against the (dense-validated) crossEBlock to machine precision.
include("holonomy_fixed.jl")
using LinearAlgebra, Printf, Random

# Left-multiply a dim×dim matrix L into each of the dt slices A[μ,:,:] (the "a" index),
# batched into ONE gemm instead of dt tiny matmuls.  A is (dt,dim,dim)=[μ,a,b];
# returns [μ, a', b] with a' = sum_a L[a',a] A[μ,a,b].  (gemm-batching the Rraw hot loop.)
@inline function leftMulEach(L, A, dt, dim)
    Ap  = permutedims(A, (2, 1, 3))                       # [a, μ, b]
    LAp = L * reshape(Ap, dim, dt * dim)                  # [a', (μ,b)]
    return permutedims(reshape(LAp, dim, dt, dim), (2, 1, 3))   # [μ, a', b]
end

function phasedBlockOneWeight(phaseFun, wfun, Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=false)
    D = size(Am, 1); d = size(Am, 3); dim = D * D; dt = d * D * D
    θ(k) = 2π * k / Nb
    Id = Matrix{ComplexF64}(I, dim, dim); Z = zeros(ComplexF64, dim, dim)

    E  = [mixedTwistedE(Am, Amp, phaseFun(B))                 for B in 1:Nb]
    Eg = [mixedTwistedE(Am, Amp, phaseFun(B) .* wfun(B))      for B in 1:Nb]

    imp(B, P, lamK, isket) = begin
        A3 = zeros(ComplexF64, dt, dim, dim)
        if isket
            for sν in 1:d, aν in 1:D, cν in 1:D, ap in 1:D, bp in 1:D
                A3[aν+(cν-1)*D+(sν-1)*dim, cν+(bp-1)*D, aν+(ap-1)*D] +=
                    lamK[cν] * P[sν] * conj(Am[ap, bp, sν])
            end
        else
            for sμ in 1:d, aμ in 1:D, cμ in 1:D, a in 1:D, b in 1:D
                A3[aμ+(cμ-1)*D+(sμ-1)*dim, b+(cμ-1)*D, a+(aμ-1)*D] +=
                    lamK[cμ] * P[sμ] * Amp[a, b, sμ]
            end
        end
        A3
    end
    K  = [imp(B, phaseFun(B),                lamInvK, true)  for B in 1:Nb]
    Kg = [imp(B, phaseFun(B) .* wfun(B),     lamInvK, true)  for B in 1:Nb]
    J  = [imp(B, phaseFun(B),                lamInvB, false) for B in 1:Nb]
    Jg = [imp(B, phaseFun(B) .* wfun(B),     lamInvB, false) for B in 1:Nb]

    # seg0(i,j) = E[j]..E[i];  seg1(i,j) = sum_{B0=i..j} E[j]..Eg[B0]..E[i]  (empty if j<i)
    S0 = [copy(Id) for _ in 1:Nb+2, _ in 1:Nb+2]
    S1 = [copy(Z)  for _ in 1:Nb+2, _ in 1:Nb+2]
    for i in 1:Nb+1
        S0[i, i] = copy(Id); S1[i, i] = copy(Z)
        for j in i:Nb
            S0[i, j+1] = E[j] * S0[i, j]
            S1[i, j+1] = Eg[j] * S0[i, j] .+ E[j] * S1[i, j]
        end
    end
    seg0(i, j) = j < i ? Id : S0[i, j+1]
    seg1(i, j) = j < i ? Z  : S1[i, j+1]

    nB = length(braMoms); nK = length(ketMoms)
    # diagOnly: skip the (1+nB*dt)^2 Bmat and ground row/col (dropped by ttpart); return
    # only the Nb momentum-diagonal tt blocks below.
    Bmat = diagOnly ? zeros(ComplexF64, 0, 0) : zeros(ComplexF64, 1 + nB * dt, 1 + nK * dt)
    !diagOnly && (Bmat[1, 1] = tr(seg1(1, Nb)))                           # <ground|O|ground>

    # environments excluding one block, with 0 or 1 weight in the rest
    Menv  = [seg0(1, x-1) * seg0(x+1, Nb) for x in 1:Nb]
    Menv1 = [seg1(1, x-1) * seg0(x+1, Nb) .+ seg0(1, x-1) * seg1(x+1, Nb) for x in 1:Nb]

    trKM(Kx, ν, M) = sum(M .* transpose(@view Kx[ν, :, :]))

    # ground(bra) -- ket tangents, and bra tangents -- ground(ket): momentum-free per x
    # (only needed for the ground row/col, which diagOnly drops)
    if !diagOnly
        uK = [ComplexF64[trKM(Kg[x], ν, Menv[x]) + trKM(K[x], ν, Menv1[x]) for ν in 1:dt] for x in 1:Nb]
        uJ = [ComplexF64[trKM(Jg[y], μ, Menv[y]) + trKM(J[y], μ, Menv1[y]) for μ in 1:dt] for y in 1:Nb]
        for (kj, kp) in enumerate(ketMoms)
            v = zeros(ComplexF64, dt)
            for x in 1:Nb; v .+= exp(im * θ(kp) * (x - 1)) .* uK[x]; end
            Bmat[1, 2+(kj-1)*dt : 1+kj*dt] = v
        end
        for (ki, k) in enumerate(braMoms)
            v = zeros(ComplexF64, dt)
            for y in 1:Nb; v .+= exp(-im * θ(k) * (y - 1)) .* uJ[y]; end
            Bmat[2+(ki-1)*dt : 1+ki*dt, 1] = v
        end
    end

    # tangent(bra, y) -- tangent(ket, x): MOMENTUM-FREE Rraw[x,y], phases added after
    # Threaded over the (x,y) pairs: each task writes its OWN Rraw[x,y] and allocates its
    # own Gs/Gb scratch, so there are no shared writes and no threadid dependence.
    Rraw = Matrix{Matrix{ComplexF64}}(undef, Nb, Nb)
    pairs = [(x, y) for y in 1:Nb for x in 1:Nb]
    Threads.@threads for pi in 1:length(pairs)
        x, y = pairs[pi]
        Gs = zeros(ComplexF64, dt, dim * dim); Gb = zeros(ComplexF64, dt, dim * dim)
        if x < y
            L  = seg0(1, x-1) * seg0(y+1, Nb); Mid = seg0(x+1, y-1)
            L1 = seg1(1, x-1) * seg0(y+1, Nb) .+ seg0(1, x-1) * seg1(y+1, Nb)
            Mid1 = seg1(x+1, y-1)
            # batched over μ: right-mults by Mid/Mid1 in one gemm each, then left-mults by L/L1
            Xall = reshape(reshape(J[y], dt*dim, dim) * Mid, dt, dim, dim)               # J[y][μ]*Mid
            Yall = reshape(reshape(J[y], dt*dim, dim) * Mid1 .+ reshape(Jg[y], dt*dim, dim) * Mid,
                           dt, dim, dim)                                                 # J[y][μ]*Mid1 + Jg[y][μ]*Mid
            Gb = reshape(leftMulEach(L, Xall, dt, dim), dt, dim*dim)
            Gs = reshape(leftMulEach(L1, Xall, dt, dim) .+ leftMulEach(L, Yall, dt, dim), dt, dim*dim)
            EK  = zeros(ComplexF64, dim * dim, dt); EKg = zeros(ComplexF64, dim * dim, dt)
            for ν in 1:dt
                EK[:, ν]  = vec(transpose(@view K[x][ν, :, :]))
                EKg[:, ν] = vec(transpose(@view Kg[x][ν, :, :]))
            end
            Rraw[x, y] = Gs * EK .+ Gb * EKg
        elseif x > y
            L  = seg0(1, y-1) * seg0(x+1, Nb); Mid = seg0(y+1, x-1)
            L1 = seg1(1, y-1) * seg0(x+1, Nb) .+ seg0(1, y-1) * seg1(x+1, Nb)
            Mid1 = seg1(y+1, x-1)
            Xall = reshape(reshape(K[x], dt*dim, dim) * Mid, dt, dim, dim)               # K[x][ν]*Mid
            Yall = reshape(reshape(K[x], dt*dim, dim) * Mid1 .+ reshape(Kg[x], dt*dim, dim) * Mid,
                           dt, dim, dim)                                                 # K[x][ν]*Mid1 + Kg[x][ν]*Mid
            Gb = reshape(leftMulEach(L, Xall, dt, dim), dt, dim*dim)
            Gs = reshape(leftMulEach(L1, Xall, dt, dim) .+ leftMulEach(L, Yall, dt, dim), dt, dim*dim)
            EJ  = zeros(ComplexF64, dim * dim, dt); EJg = zeros(ComplexF64, dim * dim, dt)
            for μ in 1:dt
                EJ[:, μ]  = vec(transpose(@view J[y][μ, :, :]))
                EJg[:, μ] = vec(transpose(@view Jg[y][μ, :, :]))
            end
            Rraw[x, y] = transpose(Gs * EJ .+ Gb * EJg)
        else
            # both impurities on one block: weight on it (Menv) or elsewhere (Menv1)
            blk = zeros(ComplexF64, dt, dt)
            Pw = phaseFun(x) .* wfun(x); Pp = phaseFun(x)
            for (P, Mx) in ((Pw, Menv[x]), (Pp, Menv1[x]))
                for sν in 1:d, aν in 1:D, cν in 1:D, aμ in 1:D, cμ in 1:D
                    μ = aμ + (cμ-1)*D + (sν-1)*dim; ν = aν + (cν-1)*D + (sν-1)*dim
                    blk[μ, ν] += lamInvK[cν] * lamInvB[cμ] * P[sν] * Mx[aν+(aμ-1)*D, cν+(cμ-1)*D]
                end
            end
            Rraw[x, y] = blk
        end
    end
    if diagOnly
        return [begin
            blk = zeros(ComplexF64, dt, dt)
            for y in 1:Nb, x in 1:Nb
                blk .+= exp(im * θ(k) * (x - y)) .* Rraw[x, y]            # kp == k (diagonal)
            end
            blk
        end for k in braMoms]
    end
    for (ki, k) in enumerate(braMoms), (kj, kp) in enumerate(ketMoms)
        blk = zeros(ComplexF64, dt, dt)
        for y in 1:Nb, x in 1:Nb
            blk .+= exp(-im * θ(k) * (y - 1) + im * θ(kp) * (x - 1)) .* Rraw[x, y]
        end
        Bmat[2+(ki-1)*dt : 1+ki*dt, 2+(kj-1)*dt : 1+kj*dt] = blk
    end
    return Bmat
end

# ===========================================================================
# FUSED TWO-WEIGHT block:  O = sum_{B,B'} w_B w_{B'} * prod_B P(B)  (= e^2 when P=1).
#
# Same collapse as the one-weight case, one order up.  Per region R keep three versions:
#   W0(R) = no weight, W1(R) = one weight anywhere in R,
#   W2(R) = BOTH weights in R = (one w^2 anywhere) + 2*(w at two distinct positions).
# Combining regions:  total = sum_i W2(R_i) prod_{j!=i} W0  +  2 sum_{i<j} W1(R_i) W1(R_j) ...
# The impurities are regions too (W0,W1,W2) = (K,Kg,Kg2).  Grouping by which KET impurity
# a term uses leaves only THREE distinct right-hand factors, so 3 expensive dt^2 dim^2
# products per (x,y) replace e2Block's 21 diagPhasedBlock calls.
function phasedBlockTwoWeight(phaseFun, wfun, Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=false)
    D = size(Am, 1); d = size(Am, 3); dim = D * D; dt = d * D * D
    θ(k) = 2π * k / Nb
    Id = Matrix{ComplexF64}(I, dim, dim); Z = zeros(ComplexF64, dim, dim)

    E   = [mixedTwistedE(Am, Amp, phaseFun(B))                    for B in 1:Nb]
    Eg  = [mixedTwistedE(Am, Amp, phaseFun(B) .* wfun(B))         for B in 1:Nb]
    Eq  = [mixedTwistedE(Am, Amp, phaseFun(B) .* wfun(B) .^ 2)    for B in 1:Nb]

    imp(P, lamK, isket) = begin
        A3 = zeros(ComplexF64, dt, dim, dim)
        if isket
            for sν in 1:d, aν in 1:D, cν in 1:D, ap in 1:D, bp in 1:D
                A3[aν+(cν-1)*D+(sν-1)*dim, cν+(bp-1)*D, aν+(ap-1)*D] +=
                    lamK[cν] * P[sν] * conj(Am[ap, bp, sν])
            end
        else
            for sμ in 1:d, aμ in 1:D, cμ in 1:D, a in 1:D, b in 1:D
                A3[aμ+(cμ-1)*D+(sμ-1)*dim, b+(cμ-1)*D, a+(aμ-1)*D] +=
                    lamK[cμ] * P[sμ] * Amp[a, b, sμ]
            end
        end
        A3
    end
    K  = [imp(phaseFun(B),                     lamInvK, true)  for B in 1:Nb]
    Kg = [imp(phaseFun(B) .* wfun(B),          lamInvK, true)  for B in 1:Nb]
    Kq = [imp(phaseFun(B) .* wfun(B) .^ 2,     lamInvK, true)  for B in 1:Nb]
    J  = [imp(phaseFun(B),                     lamInvB, false) for B in 1:Nb]
    Jg = [imp(phaseFun(B) .* wfun(B),          lamInvB, false) for B in 1:Nb]
    Jq = [imp(phaseFun(B) .* wfun(B) .^ 2,     lamInvB, false) for B in 1:Nb]

    # segment recurrences (appending block j on the left)
    S0 = [copy(Id) for _ in 1:Nb+2, _ in 1:Nb+2]
    S1 = [copy(Z)  for _ in 1:Nb+2, _ in 1:Nb+2]
    Sq = [copy(Z)  for _ in 1:Nb+2, _ in 1:Nb+2]
    S2 = [copy(Z)  for _ in 1:Nb+2, _ in 1:Nb+2]
    for i in 1:Nb+1
        S0[i, i] = copy(Id)
        for j in i:Nb
            S0[i, j+1] = E[j] * S0[i, j]
            S1[i, j+1] = Eg[j] * S0[i, j] .+ E[j] * S1[i, j]
            Sq[i, j+1] = Eq[j] * S0[i, j] .+ E[j] * Sq[i, j]
            S2[i, j+1] = Eg[j] * S1[i, j] .+ E[j] * S2[i, j]
        end
    end
    seg0(i, j) = j < i ? Id : S0[i, j+1]
    seg1(i, j) = j < i ? Z  : S1[i, j+1]
    seg2(i, j) = j < i ? Z  : (Sq[i, j+1] .+ 2 .* S2[i, j+1])   # BOTH weights in [i,j]

    # Combine the two arcs.  ORDER MATTERS: the reference environment is
    # getPf(x-1)*getSf(y+1) = a*b (arc a on the LEFT), not b*a.
    comb0(a0, b0) = a0 * b0
    comb1(a0, a1, b0, b1) = a1 * b0 .+ a0 * b1
    comb2(a0, a1, a2, b0, b1, b2) = a2 * b0 .+ 2 .* (a1 * b1) .+ a0 * b2

    nB = length(braMoms); nK = length(ketMoms)
    Bmat = diagOnly ? zeros(ComplexF64, 0, 0) : zeros(ComplexF64, 1 + nB * dt, 1 + nK * dt)
    !diagOnly && (Bmat[1, 1] = tr(seg2(1, Nb)))

    # arcs around a single excluded position x: a = (1..x-1), b = (x+1..Nb)
    arcs(x) = (seg0(1, x-1), seg1(1, x-1), seg2(1, x-1),
               seg0(x+1, Nb), seg1(x+1, Nb), seg2(x+1, Nb))
    Menv = Vector{NTuple{3,Matrix{ComplexF64}}}(undef, Nb)
    for x in 1:Nb
        a0, a1, a2, b0, b1, b2 = arcs(x)
        Menv[x] = (comb0(a0, b0), comb1(a0, a1, b0, b1), comb2(a0, a1, a2, b0, b1, b2))
    end
    trKM(Kx, ν, M) = sum(M .* transpose(@view Kx[ν, :, :]))

    if !diagOnly
        uK = [ComplexF64[trKM(K[x], ν, Menv[x][3]) + 2 * trKM(Kg[x], ν, Menv[x][2]) +
                         trKM(Kq[x], ν, Menv[x][1]) for ν in 1:dt] for x in 1:Nb]
        uJ = [ComplexF64[trKM(J[y], μ, Menv[y][3]) + 2 * trKM(Jg[y], μ, Menv[y][2]) +
                         trKM(Jq[y], μ, Menv[y][1]) for μ in 1:dt] for y in 1:Nb]
        for (kj, kp) in enumerate(ketMoms)
            v = zeros(ComplexF64, dt)
            for x in 1:Nb; v .+= exp(im * θ(kp) * (x - 1)) .* uK[x]; end
            Bmat[1, 2+(kj-1)*dt : 1+kj*dt] = v
        end
        for (ki, k) in enumerate(braMoms)
            v = zeros(ComplexF64, dt)
            for y in 1:Nb; v .+= exp(-im * θ(k) * (y - 1)) .* uJ[y]; end
            Bmat[2+(ki-1)*dt : 1+ki*dt, 1] = v
        end
    end

    Rraw = Matrix{Matrix{ComplexF64}}(undef, Nb, Nb)
    pairs = [(x, y) for y in 1:Nb for x in 1:Nb]
    Threads.@threads for pi in 1:length(pairs)
        x, y = pairs[pi]
        if x == y
            blk = zeros(ComplexF64, dt, dt)
            P = phaseFun(x); w = wfun(x)
            for (Pw, M) in ((P .* w .^ 2, Menv[x][1]), (2 .* (P .* w), Menv[x][2]), (P, Menv[x][3]))
                for sν in 1:d, aν in 1:D, cν in 1:D, aμ in 1:D, cμ in 1:D
                    μ = aμ + (cμ-1)*D + (sν-1)*dim; ν = aν + (cν-1)*D + (sν-1)*dim
                    blk[μ, ν] += lamInvK[cν] * lamInvB[cμ] * Pw[sν] * M[aν+(aμ-1)*D, cν+(cμ-1)*D]
                end
            end
            Rraw[x, y] = blk
            continue
        end
        lo, hi = min(x, y), max(x, y)
        a0 = seg0(1, lo-1); a1 = seg1(1, lo-1); a2 = seg2(1, lo-1)
        b0 = seg0(hi+1, Nb); b1 = seg1(hi+1, Nb); b2 = seg2(hi+1, Nb)
        L0 = comb0(a0, b0); L1 = comb1(a0, a1, b0, b1); L2 = comb2(a0, a1, a2, b0, b1, b2)
        M0 = seg0(lo+1, hi-1); M1 = seg1(lo+1, hi-1); M2 = seg2(lo+1, hi-1)
        GA = zeros(ComplexF64, dt, dim * dim)   # pairs with the plain impurity
        GB = zeros(ComplexF64, dt, dim * dim)   # pairs with the once-weighted impurity
        GC = zeros(ComplexF64, dt, dim * dim)   # pairs with the twice-weighted impurity
        # inner index runs over the OUTER impurity of tr(L * out * Mid * in)
        Out, Outg, Outq = (x < y) ? (J[y], Jg[y], Jq[y]) : (K[x], Kg[x], Kq[x])
        for m in 1:dt
            O = @view Out[m, :, :]; Og = @view Outg[m, :, :]; Oq = @view Outq[m, :, :]
            XJ = O * M0; XJ1 = O * M1; XJ2 = O * M2
            YJ = Og * M0; YJ1 = Og * M1; ZJ = Oq * M0
            GA[m, :] = vec(L2 * XJ .+ L0 * (ZJ .+ XJ2 .+ 2 .* YJ1) .+ 2 .* (L1 * (YJ .+ XJ1)))
            GB[m, :] = vec(2 .* (L1 * XJ .+ L0 * (YJ .+ XJ1)))
            GC[m, :] = vec(L0 * XJ)
        end
        In, Ing, Inq = (x < y) ? (K[x], Kg[x], Kq[x]) : (J[y], Jg[y], Jq[y])
        EA = zeros(ComplexF64, dim * dim, dt); EB = similar(EA); EC = similar(EA)
        for n in 1:dt
            EA[:, n] = vec(transpose(@view In[n, :, :]))
            EB[:, n] = vec(transpose(@view Ing[n, :, :]))
            EC[:, n] = vec(transpose(@view Inq[n, :, :]))
        end
        R = GA * EA .+ GB * EB .+ GC * EC
        Rraw[x, y] = (x < y) ? R : transpose(R)
    end
    if diagOnly
        return [begin
            blk = zeros(ComplexF64, dt, dt)
            for y in 1:Nb, x in 1:Nb
                blk .+= exp(im * θ(k) * (x - y)) .* Rraw[x, y]            # kp == k (diagonal)
            end
            blk
        end for k in braMoms]
    end
    for (ki, k) in enumerate(braMoms), (kj, kp) in enumerate(ketMoms)
        blk = zeros(ComplexF64, dt, dt)
        for y in 1:Nb, x in 1:Nb
            blk .+= exp(-im * θ(k) * (y - 1) + im * θ(kp) * (x - 1)) .* Rraw[x, y]
        end
        Bmat[2+(ki-1)*dt : 1+ki*dt, 2+(kj-1)*dt : 1+kj*dt] = blk
    end
    return Bmat
end

e2BlockFused(Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=false) =
    phasedBlockTwoWeight(B -> ONES4, B -> eBlk(B - 1),
                         Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=diagOnly)

# drop-in replacement for crossEBlock
crossEBlockFused(Am, Amp, lamInvB, lamInvK, Delta, Nsites, braMoms, ketMoms, Nb; diagOnly=false) = begin
    dpf = dressPhase(Delta, Nsites)
    phasedBlockOneWeight(B -> dpf(B - 1), B -> eBlk(B - 1),
                         Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=diagOnly)
end

function validateFused(; Nsites=8, D=3, hmax=1, m=1, mp=2, seed=37, kset=nothing, reps=2)
    Nb = Nsites ÷ 2; M = 2hmax + 1; kset === nothing && (kset = collect(0:Nb-1))
    Random.seed!(seed)
    Am,  lamB = canonicalization(randn(ComplexF64, D, D, 4)); liB = 1.0 ./ max.(lamB, 1e-12)
    Amp, lamK = canonicalization(randn(ComplexF64, D, D, 4)); liK = 1.0 ./ max.(lamK, 1e-12)
    Delta = 2π * (mp - m) / M
    ref = crossEBlock(Am, Amp, liB, liK, Delta, Nsites, kset, kset, Nb)
    fus = crossEBlockFused(Am, Amp, liB, liK, Delta, Nsites, kset, kset, Nb)
    err = norm(fus - ref) / max(norm(ref), 1e-30)
    t1 = minimum(@elapsed(crossEBlock(Am, Amp, liB, liK, Delta, Nsites, kset, kset, Nb)) for _ in 1:reps)
    t2 = minimum(@elapsed(crossEBlockFused(Am, Amp, liB, liK, Delta, Nsites, kset, kset, Nb)) for _ in 1:reps)
    @printf("  N=%d D=%d |kset|=%d : rel err = %.2e   crossEBlock %.3f s -> fused %.3f s  (%.2fx)\n",
            Nsites, D, length(kset), err, t1, t2, t1 / t2)
    return err
end

function validateFused2(; Nsites=8, D=3, seed=43, kset=nothing, reps=2)
    Nb = Nsites ÷ 2; kset === nothing && (kset = collect(0:Nb-1))
    Random.seed!(seed)
    A, lamv = canonicalization(randn(ComplexF64, D, D, 4)); li = 1.0 ./ max.(lamv, 1e-12)
    ref = e2Block(A, A, li, li, kset, kset, Nb)
    fus = e2BlockFused(A, A, li, li, kset, kset, Nb)
    err = norm(fus - ref) / max(norm(ref), 1e-30)
    t1 = minimum(@elapsed(e2Block(A, A, li, li, kset, kset, Nb)) for _ in 1:reps)
    t2 = minimum(@elapsed(e2BlockFused(A, A, li, li, kset, kset, Nb)) for _ in 1:reps)
    @printf("  N=%d D=%d |kset|=%d : rel err = %.2e   e2Block %.3f s -> fused %.3f s  (%.2fx)\n",
            Nsites, D, length(kset), err, t1, t2, t1 / t2)
    return err
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== fused two-weight block vs e2Block ===")
    e2 = maximum(validateFused2(; Nsites=N, D=D, kset=ks)
                 for (N, D, ks) in ((6,3,nothing), (8,3,nothing), (8,4,nothing),
                                    (8,3,[0,1,7]), (12,4,nothing), (12,6,nothing)))
    println(e2 < 1e-10 ? "  FUSED e^2 BLOCK OK" : "  MISMATCH ($e2)")
    println("\n=== fused one-weight block vs crossEBlock ===")
    e = maximum(validateFused(; Nsites=N, D=D, hmax=h, m=a, mp=b, kset=ks)
                for (N, D, h, a, b, ks) in ((6,3,1,1,2,nothing), (8,3,1,1,2,nothing),
                                            (8,4,2,2,1,nothing), (8,3,2,1,3,[0,1,7]),
                                            (12,4,1,1,2,nothing), (12,6,1,1,2,nothing)))
    println()
    println(e < 1e-10 ? "  FUSED BLOCK OK" : "  MISMATCH ($e)")
end
