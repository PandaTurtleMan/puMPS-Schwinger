# CORRECTED holonomy Schwinger Hamiltonian: translation-invariant reduction.
#
# THE CORRECTION.  The electric zero-mode energy is
#     gC * N * Lbar^2,   Lbar = L_0 + e/N,   e = -sum_i (i-1) q_i  (integer),
# with L_0 in Z the flux on the cut link (the wrap hop c†_N c_1 shifts L_0 by ±1, which
# in the theta basis is the e^{i theta_m} twist).  Expanding,
#     gC * N * Lbar^2 = gC * [ e^2/N + 2 e L_0 + N L_0^2 ].
#
# The existing code instead writes Lbar = E_0 + h with E_0 = mod(e/N, 1), i.e.
#     h = L_0 + floor(e/N),
# a CONFIG-DEPENDENT integer relabelling of the flux.  The wrap hop changes both L_0 and
# floor(e/N), so the relabelled h is not shifted correctly by the hop, and 2-site
# translation invariance is destroyed.  See [[holonomy-mod1-translation]].
#
# Concretely the fix is two substitutions in the Hamiltonian:
#     N E_0^2  ->  e^2/N            (config-diagonal, but position-WEIGHTED)
#     2N E_0   ->  2e               (in the cross-sector coupling)
# The Coulomb kernel (the translation-invariant fluctuation part sum_j (L_j - Lbar)^2)
# is unchanged.
#
# WHY THIS IS A REBUILD, NOT A PATCH, FOR THE puMPS SIDE.  The current transfer-matrix
# machinery works because E_0 depends only on e mod N, giving an EXACT N-term character
# sum N E_0^2 = sum_k fhat_k omega^{k e} in which each character factorises into per-block
# phases (fhatE0 / blockPhase / diagPhasedBlock).  Unmodded e^2 is polynomial, not
# periodic, so no finite character sum exists.  It does, however, slot into the SAME
# prefix/suffix impurity pattern with polynomial position weights in place of phases
# (e is one-body -> single impurity; e^2 -> two impurities), which is actually CHEAPER
# than the current character sum since the k-loop disappears.  That is the planned build.
#
# This file is the dense reference/oracle for that build.
include("holonomy_bloch.jl")  # transitively: holonomy_gsB.jl (fastMixedTransfer, dressPhase),
                              # holonomy_ed.jl (zeroChargeIdx, holonomyMats, HxxSector, wDiag),
                              # step3 (twistedE, Qbvals, q1vals); and diagPhasedBlock /
                              # denseImpBasis / threadedReduce for the tangent blocks below.
using LinearAlgebra, Printf, Random

# e = -sum_i (i-1) q_i  per config (INTEGER; the unmodded quantity)
function eInt(N)
    out = zeros(2^N)
    for lin in 0:2^N-1
        x = lin; e = 0.0
        for i in 1:N
            s = (x % 2 == 0) ? 1 : -1
            e += (i - 1) * (0.5 * (s + 1) - (isodd(i - 1) ? 1.0 : 0.0))
            x ÷= 2
        end
        out[lin+1] = -e
    end
    return out
end

# corrected replacement for e0DiagSite (which returns N*E_0^2): here e^2/N
e2DiagSite(N) = eInt(N) .^ 2 ./ N

# corrected per-sector diagonal on the 4^Nb == 2^N BLOCKED space (drop-in replacement
# for holonomy_oracle.jl's Hdiag_blocked), for validating the puMPS pieces later.
function Hdiag_blocked_fixed(Nsites, theta_m, gC, lam; mass=0.0)
    Nb = Nsites ÷ 2
    H = ringHamFromGate(blockedXXGateTwisted(theta_m / Nsites; mass=mass), Nb)
    H .+= Diagonal(gC .* (staggeredKernelDiag(Nsites) .+ e2DiagSite(Nsites)) .+
                   lam .* diag(denseQtot(Nsites)^2))
    return (H + H') / 2
end

# ---------------------------------------------------------------------------
# Dense neutral-sector holonomy H, corrected.  gauge = :concentrated (twist e^{i theta_m}
# on the wrap bond only, no dressing) or :spread_dressed (uniform theta_m/N twist per
# bond + the U_m^dag U_m' dressing).  The two are related by a unitary and must agree.
function holoFixedH(N, gC, lam, hmax; gauge=:concentrated, mass=0.0, theta=0.0)
    idx = zeroChargeIdx(N); n = length(idx); M = 2hmax + 1
    e    = eInt(N)[idx] .+ N * theta / (2π)        # external theta: E_0 -> E_0 + theta/2pi
    kerd = staggeredKernelDiag(N)[idx]
    Q2d  = diag(denseQtot(N)^2)[idx]
    w    = wDiag(N)[idx]
    msd  = massDiagSite(N)[idx]
    hm, h2, _ = holonomyMats(hmax)
    H = zeros(ComplexF64, n * M, n * M)
    for m in 0:M-1
        theta_m = 2π * m / M
        Hxx = HxxSector(N, theta_m; spread=(gauge != :concentrated))[idx, idx]
        H[m*n+1:(m+1)*n, m*n+1:(m+1)*n] .+=
            Hxx .+ Diagonal(gC .* (kerd .+ e .^ 2 ./ N) .+ lam .* Q2d .+ mass .* msd)
    end
    for m in 0:M-1, mp in 0:M-1
        dth = 2π * (mp - m) / M
        dress = (gauge == :concentrated) ? ones(ComplexF64, n) : exp.(im .* dth .* w ./ N)
        # 2N E_0 -> 2e   (the correction);  N h2 term unchanged
        cd = gC .* (2 .* e .* hm[m+1, mp+1] .+ N .* h2[m+1, mp+1]) .* dress
        H[m*n+1:(m+1)*n, mp*n+1:(mp+1)*n] .+= Diagonal(cd)
    end
    return Hermitian((H + H') / 2), idx
end

# ---------------------------------------------------------------------------
# The magnetic translation.  T = P (x) S^kappa, P the 2-site shift, S the flux shift
# L_0 -> L_0 + kappa, kappa(c) = (e(Pc) - e(c))/N (an integer in the neutral sector);
# in the theta basis S^kappa is the diagonal phase exp(i theta_m kappa).
# NOTE it is exact only in the UNTRUNCATED theory: at finite hmax the theta basis makes
# S a CYCLIC Z_M shift, which differs from the true shift at the window edge h = ±hmax.
# Low-lying states have negligible weight there, so |<T>| -> 1 exponentially in hmax --
# which is why the global ||[H,T]|| (dominated by the edge sectors) does NOT vanish.
function translationT(N, hmax, idx)
    n = length(idx); M = 2hmax + 1
    pos = Dict(idx[a] => a for a in 1:n)
    shift(lin) = (nl = 0; for i in 0:N-1; nl |= (((lin >> i) & 1) << mod(i + 2, N)); end; nl)
    ev = eInt(N)
    T = zeros(ComplexF64, n * M, n * M)
    for a in 1:n
        lin = idx[a] - 1; plin = shift(lin)
        # kappa is the shift of the FLUX L_0 needed to keep Lbar = L_0 + e/N invariant,
        # i.e. kappa = -(e(Pc)-e(c))/N.  (Sign matters: e = -(w - const), so building
        # kappa straight from e gives the inverse translation and |<T>| ~ 0.66.)
        kap = -Int(round((ev[plin+1] - ev[lin+1]) / N))
        for m in 0:M-1
            T[m*n + pos[plin+1], m*n + a] = exp(im * 2π * m / M * kap)
        end
    end
    return T
end

# ===========================================================================
# POLYNOMIAL-WEIGHT IMPURITY MACHINERY (the rebuild).
#
# The current code needs the character sum because E_0 = mod(e/N,1) is PERIODIC in e.
# The corrected terms are polynomial in e instead, and e is a sum of per-block diagonal
# weights with coefficients linear in the block index -- exactly the same prefix/suffix
# impurity structure, with weights in place of phases, and NO k-loop.  On the blocked
# lattice (step3 conventions, block index B 0-based):
#     e = -sum_B [ 2B*Qb(B) + q1(B) ],   so   e = sum_B eBlk(B)[s_B].
# Because the background carries no phase, the prefix/suffix products degenerate into
# plain powers of the transfer matrix E, so a single Epow table serves every impurity
# position -- cheaper than the phased case, which rebuilds transfers per character.
# External theta angle for the electric field.  E_0 -> E_0 + theta/(2pi), i.e. the total
# integer field e -> e + N*theta/(2pi); spread over the Nb blocks that is +theta/pi per block
# (since N/Nb = 2).  Set process-wide before a run (reset to 0.0 afterwards).  Every puMPS
# e-term routes through eBlk, so this single shift threads theta through the whole optimizer.
const E0_THETA = Ref(0.0)
eBlk(B) = ComplexF64[-(2B * Qbvals[s] + q1vals[s]) + E0_THETA[] / π for s in 1:4]

# UNNORMALISED <psi|e|psi>, <psi|e^2|psi> and <psi|psi> for the puMPS ring state of A.
# Ring trace convention matches e0GradAndNum: block i contributes E_i with the product
# ordered E_Nb...E_1, so a single impurity at i gives tr(Eg_i E^{Nb-1}) and a pair i<j
# (separation r = j-i) gives tr(Eg_j E^{r-1} Eg_i E^{Nb-r-1}).
function eMoments(A, Nsites)
    Nb = Nsites ÷ 2; D = size(A, 1)
    E = twistedE(A, ComplexF64[1, 1, 1, 1])
    Epow = Vector{Matrix{ComplexF64}}(undef, Nb + 1)
    Epow[1] = Matrix{ComplexF64}(I, D * D, D * D)
    for k in 1:Nb; Epow[k+1] = Epow[k] * E; end
    Eg  = [twistedE(A, eBlk(i - 1))       for i in 1:Nb]
    Eg2 = [twistedE(A, eBlk(i - 1) .^ 2)  for i in 1:Nb]
    nrm  = tr(Epow[Nb+1])
    num1 = sum(tr(Eg[i]  * Epow[Nb]) for i in 1:Nb)          # sum_B g_B
    num2 = sum(tr(Eg2[i] * Epow[Nb]) for i in 1:Nb)          # sum_B g_B^2
    for i in 1:Nb, j in i+1:Nb                                # 2 sum_{i<j} g_i g_j
        r = j - i
        num2 += 2 * tr(Eg[j] * Epow[r] * Eg[i] * Epow[Nb-r])
    end
    return num1, num2, nrm
end

# the corrected diagonal electric energy  gC * e^2/N  (normalised expectation)
e2Energy(A, Nsites, gC) = begin
    _, n2, nrm = eMoments(A, Nsites); gC * n2 / (Nsites * nrm)
end

# GRADIENT of the corrected diagonal term -- the drop-in for e0GradAndNum, which returns
# (numH, num) for gC*N*E_0^2; here for e^2 (caller supplies the gC/N).  Same conventions:
# num is UNNORMALISED, and numH is the position-AVERAGED tangent numerator (F ./= Nb)
# packed with lamInv, so it composes with the rest of the gradient exactly as e0 did.
#
# Trick: at each bra position p split e = g_p + e_{!=p}, so
#     e^2 = g_p^2 + 2 g_p e_{!=p} + e_{!=p}^2,
# and block p carries the s-dependent weight while the OTHER Nb-1 blocks form an open
# chain with the operator inserted 0, 1 or 2 times.  Because the background is unphased
# those chains are plain powers of E with insertions, so one Epow table covers everything.
function e2GradAndNum_ref(A, lambdaInverse, Nsites)
    Nb = Nsites ÷ 2; D = size(A, 1); d = size(A, 3); L = Nb - 1
    E = twistedE(A, ComplexF64[1, 1, 1, 1])
    Epow = Vector{Matrix{ComplexF64}}(undef, Nb + 1)
    Epow[1] = Matrix{ComplexF64}(I, D * D, D * D)
    for k in 1:Nb; Epow[k+1] = Epow[k] * E; end
    g   = [eBlk(i - 1) for i in 1:Nb]
    Eg  = [twistedE(A, g[i])        for i in 1:Nb]
    Eg2 = [twistedE(A, g[i] .^ 2)   for i in 1:Nb]
    # chain(p): the other blocks in matrix-product (left-to-right) order, matching
    # e0GradAndNum's Env = P[p]*S[p+2] = Es[p-1]..Es[1] Es[Nb]..Es[p+1].
    chain(p) = vcat(collect(p-1:-1:1), collect(Nb:-1:p+1))
    F = zeros(ComplexF64, d, D, D)
    num = 0.0 + 0.0im
    for p in 1:Nb
        ch = chain(p)
        En0 = Epow[Nb]                                            # nothing inserted
        En1 = zeros(ComplexF64, D * D, D * D)                     # e_{!=p} inserted once
        En2 = zeros(ComplexF64, D * D, D * D)                     # e_{!=p}^2
        for t in 1:L
            En1 .+= Epow[t] * Eg[ch[t]]  * Epow[L-t+1]
            En2 .+= Epow[t] * Eg2[ch[t]] * Epow[L-t+1]
            for tp in t+1:L
                En2 .+= 2 .* (Epow[t] * Eg[ch[t]] * Epow[tp-t] * Eg[ch[tp]] * Epow[L-tp+1])
            end
        end
        E04 = reshape(En0, (D, D, D, D)); E14 = reshape(En1, (D, D, D, D))
        E24 = reshape(En2, (D, D, D, D))
        gp = g[p]
        for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
            F[s, ap, bp] += A[a, b, s] * (gp[s]^2 * E04[a, ap, b, bp] +
                                          2 * gp[s] * E14[a, ap, b, bp] +
                                          E24[a, ap, b, bp])
        end
    end
    # value: any single p reconstructs it (block p carries the s-weight in the trace)
    num = eMoments(A, Nsites)[2]
    F ./= Nb                                                      # position AVERAGE
    numH = zeros(ComplexF64, d * D * D)
    for s in 1:d, a in 1:D, c in 1:D
        numH[a + (c-1)*D + (s-1)*D*D] = F[s, a, c] * lambdaInverse[c]
    end
    return numH, num
end

# Prefix/suffix moments of an ordered transfer product.  P[k+1][i+1] is the
# k-th moment on sites i...1; S[k+1][i] is the moment on sites Nb...i.
# Moments are derivatives (not Taylor coefficients), hence the factor 2 for
# the mixed first moments in the second-moment recurrence.  Keep matrix order:
# these transfers do not commute.  Cost and storage are O(Nb) at fixed D.
function electricMomentEnvironments(Es, Eg, Eg2=nothing)
    Nb = length(Es); D2 = size(Es[1], 1)
    order = Eg2 === nothing ? 1 : 2
    P = [Vector{Matrix{ComplexF64}}(undef, Nb+1) for _ in 0:order]
    S = [Vector{Matrix{ComplexF64}}(undef, Nb+1) for _ in 0:order]
    P[1][1] = Matrix{ComplexF64}(I, D2, D2)
    S[1][Nb+1] = Matrix{ComplexF64}(I, D2, D2)
    for k in 2:order+1
        P[k][1] = zeros(ComplexF64, D2, D2)
        S[k][Nb+1] = zeros(ComplexF64, D2, D2)
    end
    for i in 1:Nb
        P[1][i+1] = Es[i] * P[1][i]
        P[2][i+1] = Es[i] * P[2][i] + Eg[i] * P[1][i]
        if order == 2
            P[3][i+1] = Es[i] * P[3][i] + 2 .* (Eg[i] * P[2][i]) + Eg2[i] * P[1][i]
        end
    end
    for i in Nb:-1:1
        S[1][i] = S[1][i+1] * Es[i]
        S[2][i] = S[2][i+1] * Es[i] + S[1][i+1] * Eg[i]
        if order == 2
            S[3][i] = S[3][i+1] * Es[i] + 2 .* (S[2][i+1] * Eg[i]) + S[1][i+1] * Eg2[i]
        end
    end
    return P, S
end

# Scalar electric-square numerator and norm, with rolling moment environments.
function e2ValueFast(A, Nsites; E=fastTransferMatrix(A))
    D2 = size(E, 1)
    p0 = Matrix{ComplexF64}(I, D2, D2)
    p1 = zeros(ComplexF64, D2, D2); p2 = zero(p1)
    q0 = similar(p0); q1 = similar(p1); q2 = similar(p2)
    for B in 0:Nsites÷2-1
        g = eBlk(B)
        eg = fastTransferMatrix(A; phase=g)
        eg2 = fastTransferMatrix(A; phase=g.^2)
        mul!(q2, E, p2); mul!(q2, eg, p1, 2.0, 1.0); mul!(q2, eg2, p0, 1.0, 1.0)
        mul!(q1, E, p1); mul!(q1, eg, p0, 1.0, 1.0)
        mul!(q0, E, p0)
        p0,q0 = q0,p0; p1,q1 = q1,p1; p2,q2 = q2,p2
    end
    return tr(p2), tr(p0)
end

# Same position-averaged numerator as e2GradAndNum_ref, with all open-chain
# moments assembled from two sweeps instead of enumerating pairs at every p.
function e2GradAndNum(A, lambdaInverse, Nsites)
    Nb = Nsites ÷ 2; D = size(A, 1); d = size(A, 3)
    g = [eBlk(i - 1) for i in 1:Nb]
    Es = fill(fastTransferMatrix(A), Nb)
    Eg = [fastTransferMatrix(A; phase=gp) for gp in g]
    Eg2 = [fastTransferMatrix(A; phase=gp .^ 2) for gp in g]
    P, S = electricMomentEnvironments(Es, Eg, Eg2)
    Amat = reshape(permutedims(A, (3, 1, 2)), d, D * D)
    Fmat = zeros(ComplexF64, d, D * D)
    flat(T) = reshape(permutedims(reshape(T, (D, D, D, D)), (1, 3, 2, 4)), D * D, D * D)
    for p in 1:Nb
        Env0 = P[1][p] * S[1][p+1]
        Env1 = P[2][p] * S[1][p+1] + P[1][p] * S[2][p+1]
        Env2 = P[3][p] * S[1][p+1] + 2 .* (P[2][p] * S[2][p+1]) + P[1][p] * S[3][p+1]
        Fmat .+= (g[p] .^ 2) .* (Amat * flat(Env0)) .+
                 (2 .* g[p]) .* (Amat * flat(Env1)) .+ Amat * flat(Env2)
    end
    return packGrad(reshape(Fmat, (d, D, D)), lambdaInverse, Nb), tr(P[3][Nb+1])
end

# validate e2GradAndNum against the dense position-averaged tangent numerator
# (same pattern as validateE0 in puMPS_schwinger_step3.jl).
function validateE2Grad(; N=8, D=3, seed=13)
    Nb = N ÷ 2
    Random.seed!(seed)
    A, lam = canonicalization(randn(ComplexF64, D, D, 4))
    lamInv = 1.0 ./ max.(lam, 1e-12)
    Od = Diagonal(eInt(N) .^ 2)                       # e^2 (caller supplies gC/N)
    ψ = ringStateWithImpurity(A, A, 1, Nb)
    Eref = (ψ' * Od * ψ) / (ψ' * ψ)
    numH, num = e2GradAndNum(A, lamInv, N)
    nrm = tr(twistedE(A, ComplexF64[1,1,1,1])^Nb)
    d = 4; dim = d * D * D
    numH_d = zeros(ComplexF64, dim); Oψ = Od * ψ
    for i in 1:Nb
        basis = zeros(ComplexF64, 2^N, dim)
        for s in 1:d, a in 1:D, c in 1:D
            M = zeros(ComplexF64, D, D, d); M[a, c, s] = lamInv[c]
            basis[:, a + (c-1)*D + (s-1)*D*D] = ringStateWithImpurity(A, M, i, Nb)
        end
        numH_d .+= basis' * Oψ
    end
    numH_d ./= Nb
    ee = abs(num / nrm - Eref); eg = norm(numH - numH_d) / max(norm(numH_d), 1e-30)
    @printf("  N=%d D=%d : |energy err| = %.2e   ||grad err||/||grad|| = %.2e\n", N, D, ee, eg)
    return max(ee, eg)
end

# CROSS-SECTOR term, corrected:  <psi_m| e U^dag U' |psi_m'>  (ket = Amp, bra = Am).
# This REPLACES crossE0Val's character sum (2N E_0 -> 2e).  The exp(i Delta w/N) dressing
# survives untouched -- it still factorises per block -- so this is the same prefix/suffix
# sweep with ONE polynomial-weight impurity, and the k-loop over N characters disappears
# (an O(N) saving over crossE0Val).
function crossEVal(Am, Amp, Nb, Nsites, Delta)
    D = size(Am, 1)
    ph = dressPhase(Delta, Nsites)
    Es = [fastMixedTransfer(Am, Amp; phase=ph(B))                for B in 0:Nb-1]
    Eg = [fastMixedTransfer(Am, Amp; phase=ph(B) .* eBlk(B))     for B in 0:Nb-1]
    idm = Matrix{ComplexF64}(I, D * D, D * D)
    P = Vector{Matrix{ComplexF64}}(undef, Nb + 1); P[1] = idm
    for i in 1:Nb; P[i+1] = Es[i] * P[i]; end
    S = Vector{Matrix{ComplexF64}}(undef, Nb + 2); S[Nb+2] = idm; S[Nb+1] = Es[Nb]
    for i in Nb-1:-1:1; S[i+1] = S[i+2] * Es[i]; end
    return sum(tr(S[i+2] * Eg[i] * P[i]) for i in 1:Nb)
end

# value + bra-gradient of <psi_m| e U^dag U' |psi_m'> -- the drop-in for crossE0Grad.
# Conventions match phasedTraceGrad: F is NOT divided by Nb (packGrad does that).
# At bra position p the impurity is either AT p (weight ph_p*g_p, plain chain elsewhere)
# or elsewhere (weight ph_p, one weighted impurity in the chain).
function crossEGrad_ref(Am, Amp, Nb, Nsites, Delta)
    D = size(Am, 1); d = size(Am, 3); L = Nb - 1
    ph = dressPhase(Delta, Nsites)
    Es = [fastMixedTransfer(Am, Amp; phase=ph(B))            for B in 0:Nb-1]
    Eg = [fastMixedTransfer(Am, Amp; phase=ph(B) .* eBlk(B)) for B in 0:Nb-1]
    idm = Matrix{ComplexF64}(I, D * D, D * D)
    P = Vector{Matrix{ComplexF64}}(undef, Nb + 1); P[1] = idm
    for i in 1:Nb; P[i+1] = Es[i] * P[i]; end
    S = Vector{Matrix{ComplexF64}}(undef, Nb + 2); S[Nb+2] = idm; S[Nb+1] = Es[Nb]
    for i in Nb-1:-1:1; S[i+1] = S[i+2] * Es[i]; end
    val = sum(tr(S[i+2] * Eg[i] * P[i]) for i in 1:Nb)

    Ampmat = reshape(permutedims(Amp, (3, 1, 2)), d, D * D)
    Fmat = zeros(ComplexF64, d, D * D)
    flat(M) = reshape(permutedims(reshape(M, (D, D, D, D)), (1, 3, 2, 4)), D * D, D * D)
    chain(p) = vcat(collect(p-1:-1:1), collect(Nb:-1:p+1))
    for p in 1:Nb
        Env0 = P[p] * S[p+2]
        ch = chain(p)
        Ms = [Es[q] for q in ch]
        Lf = Vector{Matrix{ComplexF64}}(undef, L + 1); Lf[1] = idm
        for t in 1:L; Lf[t+1] = Lf[t] * Ms[t]; end
        Rf = Vector{Matrix{ComplexF64}}(undef, L + 1); Rf[L+1] = idm
        for t in L:-1:1; Rf[t] = Ms[t] * Rf[t+1]; end
        Env1 = zeros(ComplexF64, D * D, D * D)
        for t in 1:L; Env1 .+= Lf[t] * Eg[ch[t]] * Rf[t+1]; end
        php = ph(p - 1); gp = eBlk(p - 1)
        Fmat .+= (php .* gp) .* (Ampmat * flat(Env0)) .+ php .* (Ampmat * flat(Env1))
    end
    return val, reshape(Fmat, (d, D, D))
end

# First-moment version of the same sweeps.  The reference reconstructs an
# Nb-1-site chain for every open bra position (quadratic work in Nb).
function crossEGrad(Am, Amp, Nb, Nsites, Delta)
    D = size(Am, 1); d = size(Am, 3)
    ph = [dressPhase(Delta, Nsites)(B) for B in 0:Nb-1]
    g = [eBlk(B) for B in 0:Nb-1]
    Es = [fastMixedTransfer(Am, Amp; phase=ph[i]) for i in 1:Nb]
    Eg = [fastMixedTransfer(Am, Amp; phase=ph[i] .* g[i]) for i in 1:Nb]
    P, S = electricMomentEnvironments(Es, Eg)
    Ampmat = reshape(permutedims(Amp, (3, 1, 2)), d, D * D)
    Fmat = zeros(ComplexF64, d, D * D)
    flat(T) = reshape(permutedims(reshape(T, (D, D, D, D)), (1, 3, 2, 4)), D * D, D * D)
    for p in 1:Nb
        Env0 = P[1][p] * S[1][p+1]
        Env1 = P[2][p] * S[1][p+1] + P[1][p] * S[2][p+1]
        Fmat .+= (ph[p] .* g[p]) .* (Ampmat * flat(Env0)) .+ ph[p] .* (Ampmat * flat(Env1))
    end
    return tr(P[2][Nb+1]), reshape(Fmat, (d, D, D))
end

# Joint overlap/electric contraction: both use the same dressed transfers,
# zeroth-moment environments and open-bra contraction. Preserve the separate
# routines above as independent references and for callers needing only e.
function crossOvEGrad(Am, Amp, Nb, Nsites, Delta)
    D = size(Am, 1); d = size(Am, 3)
    phase = dressPhase(Delta, Nsites)
    ph = [phase(B) for B in 0:Nb-1]
    g = [eBlk(B) for B in 0:Nb-1]
    Es = [fastMixedTransfer(Am, Amp; phase=ph[i]) for i in 1:Nb]
    Eg = [fastMixedTransfer(Am, Amp; phase=ph[i] .* g[i]) for i in 1:Nb]
    P, S = electricMomentEnvironments(Es, Eg)
    Ampmat = reshape(permutedims(Amp, (3, 1, 2)), d, D * D)
    Fov = zeros(ComplexF64, d, D * D); Fe = zero(Fov)
    flat(T) = reshape(permutedims(reshape(T, (D, D, D, D)), (1, 3, 2, 4)), D * D, D * D)
    for p in 1:Nb
        Env0 = P[1][p] * S[1][p+1]
        Env1 = P[2][p] * S[1][p+1] + P[1][p] * S[2][p+1]
        open0 = Ampmat * flat(Env0)
        Fov .+= ph[p] .* open0
        Fe .+= (ph[p] .* g[p]) .* open0 .+ ph[p] .* (Ampmat * flat(Env1))
    end
    return tr(P[1][Nb+1]), reshape(Fov, (d, D, D)), tr(P[2][Nb+1]), reshape(Fe, (d, D, D))
end

# Values only for line searches. A rolling first-moment recurrence needs no
# open-site environments, so storage is O(D^4), independent of chain length.
function crossOvEVal(Am, Amp, Nb, Nsites, Delta)
    D2 = size(Am, 1)^2; phase = dressPhase(Delta, Nsites)
    P0 = Matrix{ComplexF64}(I, D2, D2); P1 = zeros(ComplexF64, D2, D2)
    for B in 0:Nb-1
        ph = phase(B)
        E = fastMixedTransfer(Am, Amp; phase=ph)
        Eg = fastMixedTransfer(Am, Amp; phase=ph .* eBlk(B))
        P1 = E * P1 + Eg * P0
        P0 = E * P0
    end
    return tr(P0), tr(P1)
end

# validate the cross-sector polynomial-weight impurity against the dense oracle
function validateCrossE(; Nsites=8, D=4, hmax=1, m=1, mp=2, seed=17)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    Random.seed!(seed)
    Am,  _ = canonicalization(randn(ComplexF64, D, D, 4))
    Amp, _ = canonicalization(randn(ComplexF64, D, D, 4))
    Delta = 2π * (mp - m) / M
    psim  = ringStateWithImpurity(Am,  Am,  1, Nb)
    psimp = ringStateWithImpurity(Amp, Amp, 1, Nb)
    dressv = exp.(im .* Delta .* wDiag(Nsites) ./ Nsites)
    dense = dot(psim, (eInt(Nsites) .* dressv) .* psimp)
    tm = crossEVal(Am, Amp, Nb, Nsites, Delta)
    err = abs(tm - dense) / max(abs(dense), 1e-30)
    # gradient: sum over bra impurity positions of <partial psi_m| e U^dag U' |psi_m'>
    vg, F = crossEGrad(Am, Amp, Nb, Nsites, Delta)
    Od = Diagonal(eInt(Nsites) .* dressv); Oψ = Od * psimp
    dim = 4 * D * D; Fd = zeros(ComplexF64, dim)
    for p in 1:Nb
        basis = zeros(ComplexF64, 2^Nsites, dim)
        for s in 1:4, a in 1:D, c in 1:D
            Mu = zeros(ComplexF64, D, D, 4); Mu[a, c, s] = 1.0
            basis[:, a + (c-1)*D + (s-1)*D*D] = ringStateWithImpurity(Am, Mu, p, Nb)
        end
        Fd .+= basis' * Oψ
    end
    Fv = [F[s, a, c] for s in 1:4 for c in 1:D for a in 1:D]      # mu = a+(c-1)D+(s-1)D^2
    eg = norm(Fv - Fd) / max(norm(Fd), 1e-30)
    @printf("  N=%d D=%d (m,m')=(%d,%d) : <e U^dag U'> rel err = %.2e   grad rel err = %.2e\n",
            Nsites, D, m, mp, err, eg)
    return max(err, eg, abs(vg - dense) / max(abs(dense), 1e-30))
end

# validate the polynomial-weight moments against the dense ring state
function validateEMoments(; Nsites=8, D=4, gC=0.5, seed=13)
    Nb = Nsites ÷ 2
    Random.seed!(seed)
    A, _ = canonicalization(randn(ComplexF64, D, D, 4))
    psi = ringStateWithImpurity(A, A, 1, Nb)
    ev = eInt(Nsites); nrmd = dot(psi, psi)
    d1 = dot(psi, ev .* psi) / nrmd
    d2 = dot(psi, (ev .^ 2) .* psi) / nrmd
    n1, n2, nrm = eMoments(A, Nsites)
    e1 = abs(n1 / nrm - d1); e2 = abs(n2 / nrm - d2)
    ed = abs(e2Energy(A, Nsites, gC) - gC * dot(psi, e2DiagSite(Nsites) .* psi) / nrmd)
    @printf("  N=%d D=%d : |<e> err| = %.2e   |<e^2> err| = %.2e   |gC e^2/N err| = %.2e\n",
            Nsites, D, e1, e2, ed)
    return max(e1, e2, ed)
end

# ===========================================================================
# TANGENT (Bloch) BLOCKS for the corrected model.
#
# KEY POINT: diagPhasedBlock already handles ANY operator whose diagonal factorises
# over blocks as prod_B P(B)[s_B] -- and a LOCALISED polynomial weight is exactly that
# (the weight vector on one block, ones elsewhere).  So the corrected tangent blocks need
# NO new contraction machinery: they are sums of diagPhasedBlock calls with different
# per-block weight functions, and inherit its fast path and threading for free.
#
#   e^2 = sum_B g_B^2 + 2 sum_{B<B'} g_B g_B'   -> Nb + Nb(Nb-1)/2 calls  (21 at Nb=6)
#   e U^dag U'  = sum_B g_B * (dressing)        -> Nb calls               (6 at Nb=6)
# For comparison the mod-1 code needs N character calls (12 at N=12) for EACH, so the
# cross-sector term gets ~2x CHEAPER and the diagonal ~1.75x dearer.
const ONES4 = ComplexF64[1, 1, 1, 1]
blkSize(Am, braMoms, ketMoms) = (dt = 4 * size(Am, 1)^2;
    (1 + length(braMoms) * dt, 1 + length(ketMoms) * dt))

# <basis_bra| e^2 |basis_ket> over {ground} U {tangent_{k,nu}} (same sector, no dressing)
function e2Block(Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb)
    sz = blkSize(Am, braMoms, ketMoms)
    items = Tuple{Int,Int,Float64}[(b, b, 1.0) for b in 1:Nb]
    append!(items, [(b0, b1, 2.0) for b0 in 1:Nb for b1 in b0+1:Nb])
    return threadedReduce(items, sz) do it
        b0, b1, mult = it
        wf = b0 == b1 ? (B -> B == b0 ? eBlk(B - 1) .^ 2 : ONES4) :
                        (B -> (B == b0 || B == b1) ? eBlk(B - 1) : ONES4)
        mult .* diagPhasedBlock(wf, Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb)
    end
end

# <basis_bra| e U^dag U' |basis_ket>  (cross sector; replaces crossE0Block)
function crossEBlock(Am, Amp, lamInvB, lamInvK, Delta, Nsites, braMoms, ketMoms, Nb)
    sz = blkSize(Am, braMoms, ketMoms); dpf = dressPhase(Delta, Nsites)
    return threadedReduce(collect(1:Nb), sz) do b0
        diagPhasedBlock(B -> B == b0 ? dpf(B - 1) .* eBlk(B - 1) : dpf(B - 1),
                        Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb)
    end
end

# validate both tangent blocks against the dense {ground} U {tangent} basis oracle
function validateTangentBlocks(; Nsites=8, D=3, hmax=1, m=1, mp=2, seed=31, kset=nothing)
    Nb = Nsites ÷ 2; M = 2hmax + 1; kset === nothing && (kset = collect(0:Nb-1))
    Random.seed!(seed)
    Am,  lamB = canonicalization(randn(ComplexF64, D, D, 4)); lamInvB = 1.0 ./ max.(lamB, 1e-12)
    Amp, lamK = canonicalization(randn(ComplexF64, D, D, 4)); lamInvK = 1.0 ./ max.(lamK, 1e-12)
    Bbra = denseImpBasis(Am,  lamInvB, kset, Nb)
    Bket = denseImpBasis(Amp, lamInvK, kset, Nb)
    ev = eInt(Nsites)
    # same-sector e^2 uses ONE tensor for both sides (Am), matching buildEnrichedHN
    Bs = denseImpBasis(Am, lamInvB, kset, Nb)
    e2D = Bs' * (Diagonal(ev .^ 2) * Bs)
    e2T = e2Block(Am, Am, lamInvB, lamInvB, kset, kset, Nb)
    Delta = 2π * (mp - m) / M
    dress = exp.(im .* Delta .* wDiag(Nsites) ./ Nsites)
    exD = Bbra' * (Diagonal(ev .* dress) * Bket)
    exT = crossEBlock(Am, Amp, lamInvB, lamInvK, Delta, Nsites, kset, kset, Nb)
    e1 = norm(e2T - e2D) / max(norm(e2D), 1e-14)
    e2 = norm(exT - exD) / max(norm(exD), 1e-14)
    @printf("  N=%d D=%d |kset|=%d : <e^2> block rel err = %.2e   <e U^dag U'> block rel err = %.2e\n",
            Nsites, D, length(kset), e1, e2)
    return max(e1, e2)
end

density(psi, N, idx, M) = begin
    n = length(idx); out = zeros(N)
    for m in 0:M-1, a in 1:n
        p = abs2(psi[m*n + a]); p == 0 && continue
        lin = idx[a] - 1
        for j in 1:N; out[j] += p * (((lin >> (j-1)) & 1) == 0 ? 1.0 : 0.0); end
    end
    out
end

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
function validateFixed(; N=6, gC=0.5, lam=1.0, hmaxs=(2, 3, 4, 5), nlow=6)
    @printf("=== corrected holonomy H : N=%d gC=%.2f lam=%.1f ===\n", N, gC, lam)
    @printf(" hmax  E0/N            gauge diff   dens.spread  degen%%   1-|<T>| (low %d)\n", nlow)
    for hmax in hmaxs
        M = 2hmax + 1
        Hc, idx = holoFixedH(N, gC, lam, hmax; gauge=:concentrated)
        Hs, _   = holoFixedH(N, gC, lam, hmax; gauge=:spread_dressed)
        Fc = eigen(Hc); Fs = eigen(Hs)
        gdiff = maximum(abs.(Fc.values .- Fs.values))         # gauge equivalence
        d = density(Fc.vectors[:, 1], N, idx, M)
        dev = max(maximum(d[1:2:end]) - minimum(d[1:2:end]),
                  maximum(d[2:2:end]) - minimum(d[2:2:end]))
        ev = Fc.values
        ndeg = count(i -> abs(ev[i+1] - ev[i]) < 1e-9, 1:length(ev)-1)
        T = translationT(N, hmax, idx)
        # skip states inside an exactly degenerate multiplet: eigen returns an arbitrary
        # basis there, so a single vector need not be a T eigenvector (that IS the doublet).
        qs = Float64[]
        for l in 1:nlow
            isdeg = (l > 1 && abs(ev[l]-ev[l-1]) < 1e-9) || (l < length(ev) && abs(ev[l+1]-ev[l]) < 1e-9)
            isdeg && continue
            push!(qs, 1 - abs(dot(Fc.vectors[:, l], T * Fc.vectors[:, l])))
        end
        @printf(" %d     %.10f  %.1e      %.1e      %5.1f    %s\n", hmax, ev[1]/N, gdiff, dev,
                100*ndeg/length(ev), isempty(qs) ? "-" : @sprintf("%.2e", maximum(qs)))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== corrected model: gauge equivalence, uniform density, exact translation ===")
    validateFixed(; N=6)
    println()
    validateFixed(; N=8, hmaxs=(2, 3, 4))

    println("\n=== polynomial-weight impurities: <e>, <e^2>, gC e^2/N vs dense ===")
    m1 = maximum(validateEMoments(; Nsites=N, D=D) for (N, D) in ((6,3),(8,4),(10,4),(12,4)))
    println("\n=== corrected diagonal gradient vs dense position-averaged tangent ===")
    m2 = maximum(validateE2Grad(; N=N, D=D) for (N, D) in ((6,3),(8,3),(8,4),(10,3)))
    println("\n=== cross-sector <e U^dag U'> value + gradient vs dense ===")
    m3 = maximum(validateCrossE(; Nsites=N, D=D, hmax=h, m=a, mp=b)
                 for (N, D, h, a, b) in ((6,3,1,1,2),(8,3,1,1,2),(8,4,2,2,1),(8,3,2,1,3)))
    println("\n=== tangent (Bloch) blocks vs dense {ground} U {tangent} oracle ===")
    m4 = maximum(validateTangentBlocks(; Nsites=N, D=D, hmax=h, m=a, mp=b, kset=ks)
                 for (N, D, h, a, b, ks) in ((6,3,1,1,2,nothing), (8,3,1,1,2,nothing),
                                             (8,3,2,1,3,[0,1,7]), (8,4,1,2,1,nothing)))
    println()
    println(max(m1, m2, m3, m4) < 1e-10 ? "  ALL POLYNOMIAL-WEIGHT PRIMITIVES OK" :
                                          "  MISMATCH ($m1, $m2, $m3, $m4)")
end
