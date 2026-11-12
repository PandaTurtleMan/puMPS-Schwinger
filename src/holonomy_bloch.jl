# SCALABLE (polynomial-cost) Bloch/tangent enrichment of the holonomy puMPS ansatz.
#
# Motivation ([[pumps-holonomy]]): Level B saturates the momentum-0-per-sector
# subspace; the residual gap to full ED is nonzero MATTER momentum injected by the
# holonomy-changing coupling.  enrich.jl proved (densely, N<=10) that adding
# first-order tangent (Bloch) vectors at nonzero matter momentum recovers the gap,
# and that the lowest +-p shell captures ~85% of it.  This file builds that
# enrichment at POLYNOMIAL cost via transfer-matrix contractions -- the scalable
# method, validated block-by-block against the dense oracle in enrich.jl.
#
# Enriched ansatz:
#   |Psi> = sum_m [ c_m |ground(A_m)> + sum_{k in shells, nu} b_{m,k,nu} |t_{m,k,nu}> ] (x) |theta_m>
#   |t_{m,k,nu}> = sum_x e^{-i p_k x} ring(A_m; impurity E_nu at block x),  p_k = 2 pi k / Nb.
#
# Structure of the effective matrices (theta_m orthonormal => overlap N is block-
# diagonal in m; the ground puMPS and H_diag are block-translation invariant =>
# within a sector N and the diagonal part of H are block-diagonal in matter momentum
# k; only the NON-translation-invariant cross-sector coupling connects different k).
#
# Same-sector diagonal Hamiltonian = twisted XX bond gate (off-diagonal) + config-
# DIAGONAL kernel + lifting + gC*N*E0^2.  Cross-sector = gC (2N h E0 + N h2) e^{i D w/N}.
include("holonomy_gsB.jl")   # brings the whole chain incl. puMPS_bloch (N0k, H0k, effectiveNorm,
                             # effectiveHam, oracleBlocks, ketImpurity, braImpurity, bidx, gateTransfer)
include("enrich.jl")         # dense oracle: denseHolonomyH, sectorGround, impurityVectors, tangentBlock
using LinearAlgebra, Random, Printf

# ===========================================================================
# FOUNDATION CHECK: the puMPS_bloch norm N0k and bond-gate H0k, reused verbatim on
# the BLOCKED space (d=4, Nb blocks, momentum 2pi k/Nb) with the TWISTED XX gate,
# reproduce their own dense oracle.  Confirms the tangent framework transfers to the
# blocked/twisted setting before we add the new (MPO, E0^2, cross-sector) pieces.
# ===========================================================================
function validateTwistedGateBlocks(; Nsites=8, D=3, hmax=1, m=1, seed=17)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    theta_m = 2π * (m - 1) / M
    hmat = blockedXXGateTwisted(theta_m / Nsites)      # 16x16 blocked bond gate
    Hfull = ringHamFromGate(hmat, Nb)                  # dense twisted-XX ring on 4^Nb = 2^N
    Random.seed!(seed)
    A_L, lambda = canonicalization(randn(ComplexF64, D, D, 4))
    lambdaInv = 1.0 ./ max.(lambda, 1e-12)
    maxN = 0.0; maxH = 0.0
    for k in 0:Nb-1
        Ntm = N0k(A_L, lambdaInv, k, Nb)
        Htm = H0k(A_L, lambdaInv, hmat, k, Nb)
        Nor, Hor = oracleBlocks(A_L, lambdaInv, Hfull, k, Nb)
        maxN = max(maxN, norm(Ntm - Nor) / max(norm(Nor), 1e-14))
        maxH = max(maxH, norm(Htm - Hor) / max(norm(Hor), 1e-14))
    end
    @printf("twisted-gate tangent blocks (N=%d D=%d theta=%.3f): max N err=%.2e  H err=%.2e\n",
            Nsites, D, theta_m, maxN, maxH)
    flush(stdout)
    return max(maxN, maxH)
end

# ===========================================================================
# MPO-tangent contraction: the separation-k effective-H block H_{0k}[mu,nu] for a
# config-diagonal (or any) MPO term (W, vL, vR) -- kernel + lifting.  Single-site
# MPO transfer => impurities are single-site, so ONLY k=0 (both impurities on one
# block) is a special case.  Conventions match oracleBlocks: ket impurity (nu) at
# block 1, bra impurity (mu) at block 1+k, both scaled by lamInv, no momentum phase
# (added later in the Fourier sum).  cidx(ak,m,ab)=ak+(m-1)D+(ab-1)D*chi as in
# mpoRingEnergy; ring cut between block Nb and block 1, capped by vL/vR.
cidxc(ak, m, ab, D, χ) = ak + (m - 1) * D + (ab - 1) * D * χ

# plain combined transfer T[out,in], out=(bk,mr,bb) in=(ak,ml,ab).
function mpoCombinedT(A, W)
    D = size(A, 1); d = size(A, 3); χ = size(W, 1); dim = D * χ * D
    T = zeros(ComplexF64, dim, dim)
    for st in 1:d, sb in 1:d, ml in 1:χ, mr in 1:χ
        w = W[ml, mr, st, sb]; w == 0 && continue
        for ak in 1:D, bk in 1:D, ab in 1:D, bb in 1:D
            T[cidxc(bk, mr, bb, D, χ), cidxc(ak, ml, ab, D, χ)] += A[ak, bk, st] * w * conj(A[ab, bb, sb])
        end
    end
    return T
end

# ket-impurity combined transfer for tangent index nu=(s,a,c) (ket=E_nu, bra=A).
function mpoKcomb(A, lamInv, W, ν, D, d, χ)
    s = (ν - 1) ÷ (D * D) + 1; r = (ν - 1) % (D * D); a = r % D + 1; c = r ÷ D + 1  # ν=a+(c-1)D+(s-1)D²
    dim = D * χ * D; K = zeros(ComplexF64, dim, dim)
    for ml in 1:χ, mr in 1:χ, ab in 1:D, bb in 1:D
        acc = zero(ComplexF64)
        for sb in 1:d; acc += W[ml, mr, s, sb] * conj(A[ab, bb, sb]); end
        K[cidxc(c, mr, bb, D, χ), cidxc(a, ml, ab, D, χ)] += lamInv[c] * acc
    end
    return K
end

# bra-impurity combined transfer for tangent index mu=(s,a,c) (ket=A, bra=E_mu).
function mpoJcomb(A, lamInv, W, μ, D, d, χ)
    s = (μ - 1) ÷ (D * D) + 1; r = (μ - 1) % (D * D); a = r % D + 1; c = r ÷ D + 1
    dim = D * χ * D; J = zeros(ComplexF64, dim, dim)
    for ml in 1:χ, mr in 1:χ, ak in 1:D, bk in 1:D
        acc = zero(ComplexF64)
        for st in 1:d; acc += A[ak, bk, st] * W[ml, mr, st, s]; end
        J[cidxc(bk, mr, c, D, χ), cidxc(ak, ml, a, D, χ)] += lamInv[c] * acc
    end
    return J
end

function mpoTangentBlock(A, lamInv, W, vL, vR, k, Nb)
    D = size(A, 1); d = size(A, 3); χ = size(W, 1); dim = D * χ * D; dt = d * D * D
    T = mpoCombinedT(A, W)
    # cap functional: cap(P) = sum_{a,b,m0,mN} vL[m0] vR[mN] P[cidx(a,mN,b), cidx(a,m0,b)]
    Ccap = zeros(ComplexF64, dim, dim)
    for a in 1:D, b in 1:D, m0 in 1:χ, mN in 1:χ
        Ccap[cidxc(a, mN, b, D, χ), cidxc(a, m0, b, D, χ)] += vL[m0] * vR[mN]
    end
    H = zeros(ComplexF64, dt, dt)
    if k == 0
        Denv = transpose(Ccap) * (T^(Nb - 1))                       # H00[μ,ν]=tr(Denv * KJc[μ,ν])
        for μ in 1:dt
            sμ = (μ - 1) ÷ (D*D) + 1; rμ = (μ - 1) % (D*D); aμ = rμ % D + 1; cμ = rμ ÷ D + 1
            for ν in 1:dt
                sν = (ν - 1) ÷ (D*D) + 1; rν = (ν - 1) % (D*D); aν = rν % D + 1; cν = rν ÷ D + 1
                acc = zero(ComplexF64)   # tr(Denv * KJc): KJc[out,in] => Denv[in,out]
                for ml in 1:χ, mr in 1:χ
                    acc += W[ml, mr, sν, sμ] * Denv[cidxc(aν, ml, aμ, D, χ), cidxc(cν, mr, cμ, D, χ)]
                end
                H[μ, ν] = lamInv[cν] * lamInv[cμ] * acc
            end
        end
        return H
    end
    Tl = T^(k - 1); Tr = T^(Nb - 1 - k)
    # EK[ν] = Tl * Kc[ν]; per μ: Pμ = Ccap^T * (Tr * Jc[μ]); H[μ,ν] = tr(Pμ EK[ν]).
    EKmat = zeros(ComplexF64, dim * dim, dt)                          # columns vec(EK[ν])
    for ν in 1:dt
        EKmat[:, ν] = vec(Tl * mpoKcomb(A, lamInv, W, ν, D, d, χ))
    end
    Gmat = zeros(ComplexF64, dt, dim * dim)                           # rows: Pμ arranged for tr
    for μ in 1:dt
        Pμ = transpose(Ccap) * (Tr * mpoJcomb(A, lamInv, W, μ, D, d, χ))
        Gmat[μ, :] = vec(transpose(Pμ))                              # so Gmat*EKmat gives tr(Pμ EK[ν])
    end
    return Gmat * EKmat
end

# Momentum-DIAGONAL Fourier sum of mpoTangentBlock over all separations, for ONE MPO term,
# HOISTING the (A,W)-only pieces (T = mpoCombinedT, Ccap, the T-power chain, and the dt
# mpoKcomb / mpoJcomb impurities) out of the per-separation loop -- fourierTT rebuilt all of
# these Nb times (once per sep) when they don't depend on sep.  The per-sep arithmetic is
# IDENTICAL to mpoTangentBlock (k=0 and k!=0 branches copied verbatim), so results are exact.
# Returns [ out[ki] = Nb * sum_sep exp(-i θk sep) * mpoTangentBlock(sep) : ki in kset ].
function mpoTangentFourierDiag(A, lamInv, W, vL, vR, kset, Nb)
    D = size(A, 1); d = size(A, 3); χ = size(W, 1); dim = D * χ * D; dt = d * D * D
    θ(k) = 2π * k / Nb
    T = mpoCombinedT(A, W)
    Ccap = zeros(ComplexF64, dim, dim)
    for a in 1:D, b in 1:D, m0 in 1:χ, mN in 1:χ
        Ccap[cidxc(a, mN, b, D, χ), cidxc(a, m0, b, D, χ)] += vL[m0] * vR[mN]
    end
    CtCap = transpose(Ccap)
    Tpow = Vector{Matrix{ComplexF64}}(undef, Nb)                 # Tpow[i] = T^(i-1)
    Tpow[1] = Matrix{ComplexF64}(I, dim, dim)
    for i in 2:Nb; Tpow[i] = T * Tpow[i-1]; end
    Kc = [mpoKcomb(A, lamInv, W, ν, D, d, χ) for ν in 1:dt]      # dt impurities, built ONCE
    Jc = [mpoJcomb(A, lamInv, W, μ, D, d, χ) for μ in 1:dt]
    # per-separation blocks are independent -> compute them THREADED (each writes its own
    # blks[sep+1]; Tpow/Kc/Jc/CtCap are read-only shared), then Fourier-combine serially.
    blks = Vector{Matrix{ComplexF64}}(undef, Nb)
    Threads.@threads for sep in 0:Nb-1
        if sep == 0
            Denv = CtCap * Tpow[Nb]                              # Ccap^T * T^(Nb-1)
            blk = zeros(ComplexF64, dt, dt)
            for μ in 1:dt
                sμ = (μ-1)÷(D*D)+1; rμ = (μ-1)%(D*D); aμ = rμ%D+1; cμ = rμ÷D+1
                for ν in 1:dt
                    sν = (ν-1)÷(D*D)+1; rν = (ν-1)%(D*D); aν = rν%D+1; cν = rν÷D+1
                    acc = zero(ComplexF64)
                    for ml in 1:χ, mr in 1:χ
                        acc += W[ml, mr, sν, sμ] * Denv[cidxc(aν, ml, aμ, D, χ), cidxc(cν, mr, cμ, D, χ)]
                    end
                    blk[μ, ν] = lamInv[cν] * lamInv[cμ] * acc
                end
            end
            blks[1] = blk
        else
            Tl = Tpow[sep]; Tr = Tpow[Nb-sep]                    # T^(sep-1), T^(Nb-1-sep)
            EKmat = zeros(ComplexF64, dim * dim, dt)
            for ν in 1:dt; EKmat[:, ν] = vec(Tl * Kc[ν]); end
            CtTr = CtCap * Tr
            Gmat = zeros(ComplexF64, dt, dim * dim)
            for μ in 1:dt; Gmat[μ, :] = vec(transpose(CtTr * Jc[μ])); end
            blks[sep+1] = Gmat * EKmat
        end
    end
    out = [zeros(ComplexF64, dt, dt) for _ in kset]
    for sep in 0:Nb-1, (ki, k) in enumerate(kset)
        out[ki] .+= (Nb * exp(-im * θ(k) * sep)) .* blks[sep+1]
    end
    return out
end

# validate mpoTangentBlock against oracleBlocks for a single MPO term.
function validateMPOterm(A, lamInv, W, vL, vR, Nb; label="")
    Hfull = denseMPOexpand(W, vL, vR, Nb)
    maxe = 0.0
    for k in 0:Nb-1
        Htm = mpoTangentBlock(A, lamInv, W, vL, vR, k, Nb)
        _, Hor = oracleBlocks(A, lamInv, Hfull, k, Nb)
        maxe = max(maxe, norm(Htm - Hor) / max(norm(Hor), 1e-14))
    end
    @printf("  MPO-tangent %-18s max H err = %.2e\n", label, maxe)
    return maxe
end

function validateMPOtangent(; Nsites=8, D=3, gC=0.5, lam=1.0, seed=19)
    Nb = Nsites ÷ 2
    Random.seed!(seed)
    A, lambda = canonicalization(randn(ComplexF64, D, D, 4))
    lamInv = 1.0 ./ max.(lambda, 1e-12)
    maxe = 0.0
    for (i, (coef, W, vL, vR)) in enumerate(kernelTerms(Nsites, gC))
        maxe = max(maxe, validateMPOterm(A, lamInv, W, vL, vR, Nb; label="kernel[$i]"))
    end
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    maxe = max(maxe, validateMPOterm(A, lamInv, Wl, vLl, vRl, Nb; label="lifting"))
    flush(stdout)
    return maxe
end

# ===========================================================================
# Two-impurity phased-transfer WORKHORSE.  For a config-DIAGONAL operator whose
# diagonal factorizes over blocks as prod_B P(B)[s_B] (P a per-block, per-config
# phase), builds the full effective block <basis_bra| D |basis_ket> over the basis
# {ground} U {tangent_{k,nu} : k in moms}, with bra tensor Am (momenta braMoms) and
# ket tensor Amp (momenta ketMoms).  Impurities are lamInv-scaled (matching N0k/H0k
# and the MPO-tangent).  Tangent convention: |t_k> = sum_x e^{+i th_k (x-1)} imp@x,
# th_k = 2 pi k / Nb; the bra carries the conjugate e^{-i th_k (y-1)}.
#
# This ONE routine (summed over a character index with the right fhat weights and
# phase family) gives: E0^2 (same sector: blockPhase, fhatE0), cross-sector overlap
# (dressPhase, single term) and cross-sector E0-linear (e0Phase, e0CharFhat).
seg_index(i, j) = (i, j)   # readability marker

# REFERENCE (scalar-loop) implementation, kept as the regression oracle for the
# fast path below.  The fast `diagPhasedBlock` reproduces it to machine precision.
function diagPhasedBlock_ref(phaseFun, Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb)
    D = size(Am, 1); d = size(Am, 3); dim = D * D; dt = d * D * D
    θ(k) = 2π * k / Nb
    # per-block plain double transfers and impurity transfers (position-dependent phase)
    E = [mixedTwistedE(Am, Amp, phaseFun(B)) for B in 1:Nb]                # [out,in]=[b+bp D, a+ap D]
    ketImp(B) = begin  # K[ν][out,in], ket tensor -> E_ν (ν=aν+(cν-1)D+(sν-1)D²)
        P = phaseFun(B); K = zeros(ComplexF64, dt, dim, dim)
        for sν in 1:d, aν in 1:D, cν in 1:D, ap in 1:D, bp in 1:D
            K[aν+(cν-1)*D+(sν-1)*dim, cν+(bp-1)*D, aν+(ap-1)*D] += lamInvK[cν] * P[sν] * conj(Am[ap, bp, sν])
        end
        K
    end
    braImp(B) = begin  # J[μ][out,in], bra tensor -> E_μ
        P = phaseFun(B); J = zeros(ComplexF64, dt, dim, dim)
        for sμ in 1:d, aμ in 1:D, cμ in 1:D, a in 1:D, b in 1:D
            J[aμ+(cμ-1)*D+(sμ-1)*dim, b+(cμ-1)*D, a+(aμ-1)*D] += lamInvB[cμ] * P[sμ] * Amp[a, b, sμ]
        end
        J
    end
    # prefix Pf[i]=E[i]..E[1] (Pf[1]=I -> index 0), suffix Sf[i]=E[Nb]..E[i] (Sf[Nb+1]=I),
    # segment Seg[i][j]=E[j]..E[i] for i<=j.
    Id = Matrix{ComplexF64}(I, dim, dim)
    Pf = Vector{Matrix{ComplexF64}}(undef, Nb + 1); Pf[1] = Id           # Pf[i+1] = E[i]..E[1]
    for i in 1:Nb; Pf[i+1] = E[i] * Pf[i]; end
    Sf = Vector{Matrix{ComplexF64}}(undef, Nb + 2); Sf[Nb+1] = Id        # Sf[i] = E[Nb]..E[i]
    Sf[Nb+2] = Id
    for i in Nb:-1:1; Sf[i] = Sf[i+1] * E[i]; end
    Seg = [Matrix{ComplexF64}(I, dim, dim) for _ in 1:Nb, _ in 1:Nb]     # Seg[i,j]=E[j]..E[i], i<=j
    for i in 1:Nb
        acc = Id
        for j in i:Nb; acc = (j == i) ? E[i] : E[j] * acc; Seg[i, j] = acc; end
    end
    getPf(i) = Pf[i+1]          # E[i]..E[1]  (i>=0; getPf(0)=I)
    getSf(i) = Sf[i]            # E[Nb]..E[i] (i<=Nb+1; getSf(Nb+1)=I)
    getSeg(i, j) = i > j ? Id : Seg[i, j]

    nB = length(braMoms); nK = length(ketMoms)
    B = zeros(ComplexF64, 1 + nB * dt, 1 + nK * dt)
    B[1, 1] = tr(getPf(Nb))                                              # <ground|D|ground>

    # ground (bra) -- ket tangents:  sum_x e^{i th_{k'}(x-1)} tr(Sf[x+1] Kx[ν] Pf[x-1])
    for (kj, kp) in enumerate(ketMoms)
        v = zeros(ComplexF64, dt)
        for x in 1:Nb
            Mx = getPf(x - 1) * getSf(x + 1)                            # tr(Sf K Pf)=tr((Pf Sf) K)
            Kx = ketImp(x); ph = exp(im * θ(kp) * (x - 1))
            for ν in 1:dt; v[ν] += ph * sum(@view(Mx[:, :]) .* transpose(@view Kx[ν, :, :])); end
        end
        B[1, 2+(kj-1)*dt : 1+kj*dt] = v
    end
    # bra tangents -- ground (ket):  sum_y e^{-i th_k(y-1)} tr(Sf[y+1] Jy[μ] Pf[y-1])
    for (ki, k) in enumerate(braMoms)
        v = zeros(ComplexF64, dt)
        for y in 1:Nb
            My = getPf(y - 1) * getSf(y + 1)
            Jy = braImp(y); ph = exp(-im * θ(k) * (y - 1))
            for μ in 1:dt; v[μ] += ph * sum(@view(My[:, :]) .* transpose(@view Jy[μ, :, :])); end
        end
        B[2+(ki-1)*dt : 1+ki*dt, 1] = v
    end
    # bra tangent_k (imp μ at y) -- ket tangent_{k'} (imp ν at x)
    for (ki, k) in enumerate(braMoms), (kj, kp) in enumerate(ketMoms)
        blk = zeros(ComplexF64, dt, dt)
        for y in 1:Nb, x in 1:Nb
            ph = exp(-im * θ(k) * (y - 1) + im * θ(kp) * (x - 1))
            Kx = ketImp(x); Jy = braImp(y)
            if x < y
                L = getPf(x - 1) * getSf(y + 1); Mid = getSeg(x + 1, y - 1)  # tr(L Jy Mid Kx)
                Gmat = zeros(ComplexF64, dt, dim * dim)
                for μ in 1:dt; Gmat[μ, :] = vec(L * @view(Jy[μ, :, :]) * Mid); end
                EK = zeros(ComplexF64, dim * dim, dt)
                for ν in 1:dt; EK[:, ν] = vec(transpose(@view Kx[ν, :, :])); end
                blk .+= ph .* (Gmat * EK)
            elseif x > y
                L = getPf(y - 1) * getSf(x + 1); Mid = getSeg(y + 1, x - 1)  # tr(L Kx Mid Jy)
                Gmat = zeros(ComplexF64, dt, dim * dim)                       # index by ν then transpose
                for ν in 1:dt; Gmat[ν, :] = vec(L * @view(Kx[ν, :, :]) * Mid); end
                EJ = zeros(ComplexF64, dim * dim, dt)
                for μ in 1:dt; EJ[:, μ] = vec(transpose(@view Jy[μ, :, :])); end
                blk .+= ph .* transpose(Gmat * EJ)
            else  # x == y : both impurities on one block
                Mx = getPf(x - 1) * getSf(x + 1); P = phaseFun(x)
                for sν in 1:d, aν in 1:D, cν in 1:D, aμ in 1:D, cμ in 1:D
                    μ = aμ + (cμ-1)*D + (sν-1)*dim; ν = aν + (cν-1)*D + (sν-1)*dim   # sμ==sν
                    blk[μ, ν] += ph * lamInvK[cν] * lamInvB[cμ] * P[sν] * Mx[aν+(aμ-1)*D, cν+(cμ-1)*D]
                end
            end
        end
        B[2+(ki-1)*dt : 1+ki*dt, 2+(kj-1)*dt : 1+kj*dt] = blk
    end
    return B
end

# FAST path (default).  Same result as diagPhasedBlock_ref, but the momentum
# structure is factored out of the heavy linear algebra:
#   * the impurity transfers K_x / J_y are built ONCE per block (not rebuilt for
#     every (k,k') momentum pair as in the ref);
#   * the tangent-tangent geometry Rraw[x,y] and the ground<->tangent per-position
#     vectors w[x] / u[y] are MOMENTUM-INDEPENDENT -- computed once per (x,y) / x,
#     then combined into every momentum block by a scalar phase weight.
# So the O(dt^2 dim^2) matmuls run Nb^2 times instead of nBra*nKet*Nb^2 times:
# adding a momentum shell now costs only the cheap phase-weighted accumulation,
# which is exactly what makes "many shells -> ED" affordable.  Arithmetic is
# identical to the ref (verified to machine precision by regressDiagPhasedFast).
function diagPhasedBlock(phaseFun, Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=false)
    D = size(Am, 1); d = size(Am, 3); dim = D * D; dt = d * D * D
    θ(k) = 2π * k / Nb
    E = [fastMixedTransfer(Am, Amp; phase=phaseFun(B)) for B in 1:Nb]     # [out,in]=[b+bp D, a+ap D]
    # impurity transfers, HOISTED (identical to the ref's ketImp/braImp)
    Kall = Vector{Array{ComplexF64,3}}(undef, Nb)                          # K_x[ν][out,in], ket -> E_ν
    Jall = Vector{Array{ComplexF64,3}}(undef, Nb)                          # J_y[μ][out,in], bra -> E_μ
    for Bx in 1:Nb
        P = phaseFun(Bx)
        K = zeros(ComplexF64, dt, dim, dim)
        for sν in 1:d, aν in 1:D, cν in 1:D, ap in 1:D, bp in 1:D
            K[aν+(cν-1)*D+(sν-1)*dim, cν+(bp-1)*D, aν+(ap-1)*D] += lamInvK[cν] * P[sν] * conj(Am[ap, bp, sν])
        end
        Kall[Bx] = K
        J = zeros(ComplexF64, dt, dim, dim)
        for sμ in 1:d, aμ in 1:D, cμ in 1:D, a in 1:D, b in 1:D
            J[aμ+(cμ-1)*D+(sμ-1)*dim, b+(cμ-1)*D, a+(aμ-1)*D] += lamInvB[cμ] * P[sμ] * Amp[a, b, sμ]
        end
        Jall[Bx] = J
    end
    Id = Matrix{ComplexF64}(I, dim, dim)
    Pf = Vector{Matrix{ComplexF64}}(undef, Nb + 1); Pf[1] = Id            # Pf[i+1] = E[i]..E[1]
    for i in 1:Nb; Pf[i+1] = E[i] * Pf[i]; end
    Sf = Vector{Matrix{ComplexF64}}(undef, Nb + 2); Sf[Nb+1] = Id         # Sf[i] = E[Nb]..E[i]
    Sf[Nb+2] = Id
    for i in Nb:-1:1; Sf[i] = Sf[i+1] * E[i]; end
    Seg = [Matrix{ComplexF64}(I, dim, dim) for _ in 1:Nb, _ in 1:Nb]      # Seg[i,j]=E[j]..E[i], i<=j
    for i in 1:Nb
        acc = Id
        for j in i:Nb; acc = (j == i) ? E[i] : E[j] * acc; Seg[i, j] = acc; end
    end
    getPf(i) = Pf[i+1]; getSf(i) = Sf[i]; getSeg(i, j) = i > j ? Id : Seg[i, j]

    nB = length(braMoms); nK = length(ketMoms)
    kph = [ComplexF64[exp(im * θ(kp) * (x - 1)) for x in 1:Nb] for kp in ketMoms]   # ket momentum
    bcj = [ComplexF64[exp(-im * θ(k) * (y - 1)) for y in 1:Nb] for k in braMoms]    # bra (conj) momentum
    # diagOnly: skip the (1+nB*dt)^2 Bout and the ground row/col (dropped by ttpart anyway);
    # only the Nb momentum-diagonal tt blocks are returned below.
    Bout = diagOnly ? zeros(ComplexF64, 0, 0) : zeros(ComplexF64, 1 + nB * dt, 1 + nK * dt)
    if !diagOnly
        Bout[1, 1] = tr(getPf(Nb))                                        # <ground|D|ground>
        # ground(bra)--ket tangents: w[ν,x] = tr(Sf[x+1] K_x[ν] Pf[x-1]) (momentum-free)
        wcol = zeros(ComplexF64, dt, Nb)
        for x in 1:Nb
            Mx = getPf(x - 1) * getSf(x + 1); Kx = Kall[x]
            for ν in 1:dt; wcol[ν, x] = sum(Mx .* transpose(@view Kx[ν, :, :])); end
        end
        for kj in 1:nK
            v = zeros(ComplexF64, dt)
            for x in 1:Nb; v .+= kph[kj][x] .* @view wcol[:, x]; end
            Bout[1, 2+(kj-1)*dt : 1+kj*dt] = v
        end
        # bra tangents--ground(ket): u[μ,y] = tr(Sf[y+1] J_y[μ] Pf[y-1]) (momentum-free)
        ucol = zeros(ComplexF64, dt, Nb)
        for y in 1:Nb
            My = getPf(y - 1) * getSf(y + 1); Jy = Jall[y]
            for μ in 1:dt; ucol[μ, y] = sum(My .* transpose(@view Jy[μ, :, :])); end
        end
        for ki in 1:nB
            v = zeros(ComplexF64, dt)
            for y in 1:Nb; v .+= bcj[ki][y] .* @view ucol[:, y]; end
            Bout[2+(ki-1)*dt : 1+ki*dt, 1] = v
        end
    end

    # tangent(bra,imp μ at y)--tangent(ket,imp ν at x): raw dt x dt block Rraw[x,y]
    # is momentum-free; the (k,k') block is sum_{x,y} bcj_k[y] kph_k'[x] Rraw[x,y].
    rawTT = function (x, y)
        if x < y
            L = getPf(x - 1) * getSf(y + 1); Mid = getSeg(x + 1, y - 1)
            Kx = Kall[x]; Jy = Jall[y]
            Gmat = zeros(ComplexF64, dt, dim * dim)
            for μ in 1:dt; Gmat[μ, :] = vec(L * @view(Jy[μ, :, :]) * Mid); end   # tr(L Jy Mid Kx)
            EK = zeros(ComplexF64, dim * dim, dt)
            for ν in 1:dt; EK[:, ν] = vec(transpose(@view Kx[ν, :, :])); end
            return Gmat * EK
        elseif x > y
            L = getPf(y - 1) * getSf(x + 1); Mid = getSeg(y + 1, x - 1)
            Kx = Kall[x]; Jy = Jall[y]
            Gmat = zeros(ComplexF64, dt, dim * dim)
            for ν in 1:dt; Gmat[ν, :] = vec(L * @view(Kx[ν, :, :]) * Mid); end   # tr(L Kx Mid Jy)
            EJ = zeros(ComplexF64, dim * dim, dt)
            for μ in 1:dt; EJ[:, μ] = vec(transpose(@view Jy[μ, :, :])); end
            return Matrix(transpose(Gmat * EJ))
        else  # x == y : both impurities on one block
            Mx = getPf(x - 1) * getSf(x + 1); P = phaseFun(x)
            R = zeros(ComplexF64, dt, dt)
            for sν in 1:d, aν in 1:D, cν in 1:D, aμ in 1:D, cμ in 1:D
                μ = aμ + (cμ-1)*D + (sν-1)*dim; ν = aν + (cν-1)*D + (sν-1)*dim   # sμ==sν
                R[μ, ν] += lamInvK[cν] * lamInvB[cμ] * P[sν] * Mx[aν+(aμ-1)*D, cν+(cμ-1)*D]
            end
            return R
        end
    end
    if diagOnly
        out = [zeros(ComplexF64, dt, dt) for _ in 1:nB]                   # Nb momentum-diagonal tt blocks
        for x in 1:Nb
            Rrow = [rawTT(x, y) for y in 1:Nb]
            for ki in 1:nB
                Ry = zeros(ComplexF64, dt, dt)
                for y in 1:Nb; Ry .+= bcj[ki][y] .* Rrow[y]; end
                out[ki] .+= kph[ki][x] .* Ry                              # kj == ki (diagonal only)
            end
        end
        return out
    end
    for x in 1:Nb
        Rrow = [rawTT(x, y) for y in 1:Nb]                                # Nb raw dt x dt blocks
        for ki in 1:nB
            Ry = zeros(ComplexF64, dt, dt)
            for y in 1:Nb; Ry .+= bcj[ki][y] .* Rrow[y]; end             # bra-momentum sum over y
            for kj in 1:nK
                vb = @view Bout[2+(ki-1)*dt : 1+ki*dt, 2+(kj-1)*dt : 1+kj*dt]
                vb .+= kph[kj][x] .* Ry                                   # ket-momentum weight
            end
        end
    end
    return Bout
end

# ---- dense oracle for any config-diagonal operator Dvec (length 2^N) ----
function denseImpBasis(A, lamInv, moms, Nb)
    D = size(A, 1); d = size(A, 3); dt = d * D * D; n = d^Nb
    B = zeros(ComplexF64, n, 1 + length(moms) * dt)
    B[:, 1] = ringStateWithImpurity(A, A, 1, Nb)
    for (kj, k) in enumerate(moms), s in 1:d, a in 1:D, c in 1:D
        ν = a + (c - 1) * D + (s - 1) * D * D              # column ordering MUST match mu_pack
        M = zeros(ComplexF64, D, D, d); M[a, c, s] = lamInv[c]
        v = zeros(ComplexF64, n)
        for x in 1:Nb; v .+= exp(2π * im * k * (x - 1) / Nb) .* ringStateWithImpurity(A, M, x, Nb); end
        B[:, 1 + (kj - 1) * dt + ν] = v
    end
    return B
end

function validateDiagPhased(; Nsites=8, D=3, seed=23, kset=nothing)
    Nb = Nsites ÷ 2; kset === nothing && (kset = collect(0:Nb-1))
    Random.seed!(seed)
    A, lambda = canonicalization(randn(ComplexF64, D, D, 4)); lamInv = 1.0 ./ max.(lambda, 1e-12)
    # E0^2 operator: N E0^2 = sum_k fhatE0[k] Omega_k  (diagonal e0DiagSite on blocked configs)
    Bt = zeros(ComplexF64, 1 + length(kset)*d0(D), 1 + length(kset)*d0(D))
    fh = fhatE0(Nsites)
    for κ in 0:Nsites-1
        abs(fh[κ+1]) < 1e-15 && continue
        Bt .+= fh[κ+1] .* diagPhasedBlock(B -> blockPhase(κ, B - 1, Nsites), A, A, lamInv, lamInv, kset, kset, Nb)
    end
    Basis = denseImpBasis(A, lamInv, kset, Nb)
    Bdense = Basis' * (Diagonal(e0DiagSite(Nsites)) * Basis)
    err = norm(Bt - Bdense) / max(norm(Bdense), 1e-14)
    @printf("  E0^2 two-impurity block (N=%d D=%d): rel err = %.2e\n", Nsites, D, err)
    flush(stdout)
    return err
end
d0(D) = 4 * D * D   # dt for d=4

# regression + speedup: fast diagPhasedBlock == diagPhasedBlock_ref to machine
# precision, over the E0^2 character family (same phase family as the hot path),
# for the full shell set (worst case for the momentum-loop cost).
function regressDiagPhasedFast(; Nsites=8, D=4, seed=23, reps=1)
    Nb = Nsites ÷ 2; kset = collect(0:Nb-1)
    Random.seed!(seed)
    Am, lB = canonicalization(randn(ComplexF64, D, D, 4)); liB = 1.0 ./ max.(lB, 1e-12)
    Amp, lK = canonicalization(randn(ComplexF64, D, D, 4)); liK = 1.0 ./ max.(lK, 1e-12)
    κs = [κ for κ in 0:Nsites-1 if κ == 0 || κ == 1]   # a couple of representative characters
    maxerr = 0.0
    for κ in κs
        pf = B -> blockPhase(κ, B - 1, Nsites)
        Bref  = diagPhasedBlock_ref(pf, Am, Amp, liB, liK, kset, kset, Nb)
        Bfast = diagPhasedBlock(pf, Am, Amp, liB, liK, kset, kset, Nb)
        maxerr = max(maxerr, norm(Bfast - Bref) / max(norm(Bref), 1e-14))
    end
    # timing over the full character sum (E0^2-like workload)
    pfs = [B -> blockPhase(κ, B - 1, Nsites) for κ in 0:Nsites-1]
    tref = @elapsed for _ in 1:reps, pf in pfs
        diagPhasedBlock_ref(pf, Am, Amp, liB, liK, kset, kset, Nb)
    end
    tfast = @elapsed for _ in 1:reps, pf in pfs
        diagPhasedBlock(pf, Am, Amp, liB, liK, kset, kset, Nb)
    end
    @printf("regressDiagPhasedFast N=%d D=%d shells=%d : rel err=%.2e  ref=%.3fs fast=%.3fs speedup=%.1fx\n",
            Nsites, D, length(kset), maxerr, tref, tfast, tref / tfast)
    flush(stdout)
    return maxerr, tref / tfast
end

# Sum f(item) over items with one @spawn task per thread-chunk.  Schedule-agnostic
# (no threadid, no shared writes: each task owns its accumulator, partials reduced
# after fetch), so it is race-free under any scheduler and degrades to a plain
# serial sum at nthreads()==1.  Run `julia -t n` to parallelise the E0 character
# sums below; the inner transfers are small BLAS calls, so consider
# BLAS.set_num_threads(1) to avoid oversubscription when using many Julia threads.
function threadedReduce(f, items, sz)
    n = length(items)
    n == 0 && return zeros(ComplexF64, sz...)
    nt = max(1, min(Threads.nthreads(), n))
    step = cld(n, nt)
    tasks = map(1:step:n) do lo
        chunk = items[lo:min(lo + step - 1, n)]
        Threads.@spawn begin
            acc = zeros(ComplexF64, sz...)
            for it in chunk; acc .+= f(it); end
            acc
        end
    end
    out = zeros(ComplexF64, sz...)
    for t in tasks; out .+= fetch(t); end
    return out
end

# ---- cross-sector (m != m') dressed overlap + E0-linear, vs dense oracle ----
# Coupling operators (config-diagonal): overlap dressing exp(i Delta w/N) and
# E0-linear E0 * exp(i Delta w/N).  Reuse diagPhasedBlock with the dressPhase /
# e0Phase families (from holonomy_gsB.jl); bra tensor A_m, ket tensor A_{m'}.
function crossOverlapBlock(Am, Amp, lamInvB, lamInvK, Delta, Nsites, braMoms, ketMoms, Nb; diagOnly=false)
    dpf = dressPhase(Delta, Nsites)
    return diagPhasedBlock(B -> dpf(B - 1), Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb; diagOnly=diagOnly)
end
# E0-linear character sum -- threaded over the (independent) characters κ.
function crossE0Block(Am, Amp, lamInvB, lamInvK, Delta, Nsites, braMoms, ketMoms, Nb; theta=0.0)
    fh = e0CharFhat(Nsites; theta=theta); dt = d0(size(Am, 1))
    sz = (1 + length(braMoms) * dt, 1 + length(ketMoms) * dt)
    κs = [κ for κ in 0:Nsites-1 if abs(fh[κ+1]) >= 1e-15]
    return threadedReduce(κs, sz) do κ
        epf = e0Phase(κ, Delta, Nsites)
        fh[κ+1] .* diagPhasedBlock(B -> epf(B - 1), Am, Amp, lamInvB, lamInvK, braMoms, ketMoms, Nb)
    end
end

function validateCrossPhased(; Nsites=8, D=3, hmax=1, m=1, mp=2, seed=29, kset=nothing)
    Nb = Nsites ÷ 2; M = 2hmax + 1; kset === nothing && (kset = collect(0:Nb-1))
    Delta = 2π * (mp - m) / M
    Random.seed!(seed)
    Am, lamB = canonicalization(randn(ComplexF64, D, D, 4)); lamInvB = 1.0 ./ max.(lamB, 1e-12)
    Amp, lamK = canonicalization(randn(ComplexF64, D, D, 4)); lamInvK = 1.0 ./ max.(lamK, 1e-12)
    Bbra = denseImpBasis(Am, lamInvB, kset, Nb); Bket = denseImpBasis(Amp, lamInvK, kset, Nb)
    w = wDiag(Nsites); dress = exp.(im .* Delta .* w ./ Nsites)
    ovD = Bbra' * (Diagonal(dress) * Bket)
    e0D = Bbra' * (Diagonal(e0OperatorDiag(Nsites) .* dress) * Bket)
    ovT = crossOverlapBlock(Am, Amp, lamInvB, lamInvK, Delta, Nsites, kset, kset, Nb)
    e0T = crossE0Block(Am, Amp, lamInvB, lamInvK, Delta, Nsites, kset, kset, Nb)
    eo = norm(ovT - ovD) / max(norm(ovD), 1e-14); ee = norm(e0T - e0D) / max(norm(e0D), 1e-14)
    @printf("  cross m=%d->m'=%d (N=%d D=%d): overlap rel err=%.2e  E0-linear rel err=%.2e\n",
            m, mp, Nsites, D, eo, ee)
    flush(stdout)
    return max(eo, ee)
end

# ===========================================================================
# ASSEMBLY: full enriched effective H_eff / N_eff over the polynomial-size basis
# {tangent_{m,k,nu} : m in 1..M, k in kset, nu} (kset MUST contain 0 -- the ground
# lives in the span of the k=0 tangents, so no separate ground vector is needed;
# the span equals enrich.jl's {ground} U {tangents}, hence the same variational
# energy).  All blocks are tangent-tangent, momentum-resolved:
#   - N (overlap), E0^2, cross-sector: diagPhasedBlock (drop the ground row/col).
#   - XX gate, kernel/lifting MPO: Fourier-sum the validated separation-k blocks
#     H0k / mpoTangentBlock with the |t_k>=sum_x e^{+i th_k(x-1)} convention:
#         block_k = Nb * sum_{sep} e^{-i th_k sep} sepBlock(sep)   (block-diag in k).
θk(k, Nb) = 2π * k / Nb
ttpart(Bfull) = Bfull[2:end, 2:end]              # drop ground row/col

function fourierTT(sepfun, kset, Nb, dt; diagOnly=false)
    # HOIST: the separation blocks do not depend on the momentum, but the original loop
    # called sepfun(sep) inside the k loop -- |kset|*Nb evaluations instead of Nb, i.e. a
    # 6x waste at Nb=6 with the full kset.  sepfun is the expensive part (H0k allocates
    # 0.81 GB per call at D=6), so this is the single biggest cost in the assembly.
    # Identical arithmetic; only the number of sepfun evaluations changes.
    blocks = [sepfun(sep) for sep in 0:Nb-1]
    diagblk(k) = (acc = zeros(ComplexF64, dt, dt);
                  for sep in 0:Nb-1; acc .+= exp(-im * θk(k, Nb) * sep) .* blocks[sep+1]; end;
                  Nb .* acc)
    # diagOnly: this block is already momentum-diagonal (translation invariant), so return
    # just the Nb diagonal dt x dt blocks instead of the (|kset|*dt)^2 mostly-zero matrix.
    diagOnly && return [diagblk(k) for k in kset]
    bs = length(kset) * dt; B = zeros(ComplexF64, bs, bs)
    for (ki, k) in enumerate(kset)
        B[(ki-1)*dt+1:ki*dt, (ki-1)*dt+1:ki*dt] = diagblk(k)
    end
    return B
end

function buildEnrichedHN(As, lambdas, kset, Nsites, gC, lam, hmax; theta_ext=0.0)
    Nb = Nsites ÷ 2; M = 2hmax + 1; D = size(As[1], 1); dt = 4 * D * D
    bs = length(kset) * dt; Ntot = M * bs
    @assert 0 in kset "kset must contain 0 (ground lives in the k=0 tangent span)"
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    hm, h2, _ = holonomyMats(hmax)
    kterms = kernelTerms(Nsites, gC)                     # (coef,W,vL,vR), coef already has gC
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    fhE0 = fhatE0(Nsites; theta=theta_ext)
    ones4 = ComplexF64[1, 1, 1, 1]
    Heff = zeros(ComplexF64, Ntot, Ntot); Neff = zeros(ComplexF64, Ntot, Ntot)
    for m in 1:M
        A = As[m]; li = lamInvs[m]; theta_m = 2π * (m - 1) / M
        hmat = blockedXXGateTwisted(theta_m / Nsites)
        Nblk = ttpart(diagPhasedBlock(B -> ones4, A, A, li, li, kset, kset, Nb))
        Hxx  = fourierTT(sep -> H0k(A, li, hmat, sep, Nb), kset, Nb, dt)
        Hker = fourierTT(sep -> sum(coef .* mpoTangentBlock(A, li, W, vL, vR, sep, Nb)
                                    for (coef, W, vL, vR) in kterms), kset, Nb, dt)
        Hlift = fourierTT(sep -> lam .* mpoTangentBlock(A, li, Wl, vLl, vRl, sep, Nb), kset, Nb, dt)
        κsE0 = [κ for κ in 0:Nsites-1 if abs(fhE0[κ+1]) >= 1e-15]
        HE0 = threadedReduce(κsE0, (bs, bs)) do κ
            (gC * fhE0[κ+1]) .* ttpart(diagPhasedBlock(B -> blockPhase(κ, B - 1, Nsites), A, A, li, li, kset, kset, Nb))
        end
        selfE = (gC * Nsites * real(h2[m, m])) .* Nblk
        rng = (m-1)*bs+1 : m*bs
        Heff[rng, rng] = Hxx .+ Hker .+ Hlift .+ HE0 .+ selfE
        Neff[rng, rng] = Nblk
    end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Delta = 2π * (mp - m) / M
        ov = ttpart(crossOverlapBlock(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb))
        e0 = ttpart(crossE0Block(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb; theta=theta_ext))
        Heff[(m-1)*bs+1:m*bs, (mp-1)*bs+1:mp*bs] = gC .* (2Nsites * hm[m, mp] .* e0 .+ Nsites * h2[m, mp] .* ov)
    end
    return Hermitian((Heff + Heff') / 2), Hermitian((Neff + Neff') / 2)
end

# lowest generalized eigenvalue (pseudo-inverse of Neff drops gauge/null modes).
function enrichedEnergyTM(As, lambdas, kset, Nsites, gC, lam, hmax; theta_ext=0.0, cutoff=1e-10)
    Heff, Neff = buildEnrichedHN(As, lambdas, kset, Nsites, gC, lam, hmax; theta_ext=theta_ext)
    F = eigen(Neff); thresh = cutoff * maximum(F.values)
    keep = findall(>(thresh), F.values); U = F.vectors[:, keep]
    Nh = U * Diagonal(1.0 ./ sqrt.(F.values[keep])) * U'
    He = Hermitian((Nh * Heff * Nh + (Nh * Heff * Nh)') / 2)
    return minimum(real(eigvals(He))), length(keep)
end

# END-TO-END VALIDATION: scalable transfer-matrix enriched energy == dense enrich.jl
# projection (same A_m, same shells), per momentum shell.  Also report vs full ED.
function validateEnrichedVsDense(; Nsites=6, D=4, gC=0.5, lam=1.0, hmax=1,
                                  warmSteps=1500, maxit=600, alpha=0.05, seed=4)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    EB, As, lambdas = levelB_GS(; Nsites=Nsites, hmax=hmax, D=D, gC=gC, lam=lam,
                                 warmSteps=warmSteps, maxit=maxit, alpha=alpha, seed=seed, verbose=false)
    H = denseHolonomyH(Nsites, gC, lam, hmax); Efull = minimum(real(eigvals(H)))
    Vs = [impurityVectors(As[m], Nb) for m in 1:M]
    @printf("N=%d D=%d gC=%.2f hmax=%d : Level B E/N=%.8f  full ED E/N=%.8f\n",
            Nsites, D, gC, hmax, EB/Nsites, Efull/Nsites)
    @printf("  shell(|k|<=)  E_TM/N         E_dense/N      |diff|/site   gap-closed  (rank)\n")
    Eg, _ = enrichedEnergy(H, As, Vs, Nsites, M, Int[])           # ground-only baseline
    maxdiff = 0.0
    for shell in 0:(Nb ÷ 2)
        kset = [k for k in 0:Nb-1 if min(k, Nb - k) <= shell]
        Etm, rk = enrichedEnergyTM(As, lambdas, kset, Nsites, gC, lam, hmax)
        Eds, _  = enrichedEnergy(H, As, Vs, Nsites, M, kset)
        closed = (Eg - Efull) > 1e-13 ? (Eg - Etm) / (Eg - Efull) * 100 : 0.0
        @printf("  %-2d            %.9f  %.9f  %.2e     %6.2f%%   (%d)\n",
                shell, Etm/Nsites, Eds/Nsites, abs(Etm - Eds)/Nsites, closed, rk)
        maxdiff = max(maxdiff, abs(Etm - Eds)/Nsites)
        flush(stdout)
    end
    return maxdiff
end

# Scalable driver (NO dense H): optimize Level B, then the enriched energy per
# momentum shell purely by transfer matrices.  Usable at any N (polynomial cost).
function runEnrichedTM(N, D, gC, hmax; lam=1.0, warmSteps=1500, maxit=400, alpha=0.05, seed=4, verbose=true)
    Nb = N ÷ 2
    t0 = time()
    EB, As, lambdas = levelB_GS(; Nsites=N, hmax=hmax, D=D, gC=gC, lam=lam,
                                 warmSteps=warmSteps, maxit=maxit, alpha=alpha, seed=seed, verbose=false)
    tB = time() - t0
    verbose && @printf("N=%d D=%d gC=%.2f hmax=%d : Level B E/N=%.9f  (%.0fs)\n", N, D, gC, hmax, EB/N, tB)
    verbose && @printf("  shell(|k|<=)   E_enriched/N    lowered vs B   time\n")
    res = Tuple{Int,Float64}[]
    for shell in 0:(Nb ÷ 2)
        kset = [k for k in 0:Nb-1 if min(k, Nb - k) <= shell]
        t1 = time(); E, _ = enrichedEnergyTM(As, lambdas, kset, N, gC, lam, hmax); dt = time() - t1
        push!(res, (shell, E))
        verbose && @printf("  %-2d             %.9f   %.3e     %.0fs\n", shell, E/N, (EB - E)/N, dt)
        flush(stdout)
    end
    return EB, res
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== FOUNDATION: norm + twisted-XX-gate tangent blocks on the blocked space ===")
    err = maximum(validateTwistedGateBlocks(; Nsites=8, D=3, hmax=1, m=m) for m in 1:3)
    println(err < 1e-10 ? "  FOUNDATION OK" : "  FOUNDATION FAILED (err=$err)")

    println("\n=== MPO-tangent contraction (kernel + lifting) vs dense oracle ===")
    errm = validateMPOtangent()
    println(errm < 1e-10 ? "  MPO-TANGENT OK" : "  MPO-TANGENT FAILED (err=$errm)")

    println("\n=== E0^2 two-impurity phased-transfer block vs dense oracle ===")
    erre = validateDiagPhased(; Nsites=8, D=3)
    println(erre < 1e-10 ? "  E0^2 TANGENT OK" : "  E0^2 TANGENT FAILED (err=$erre)")

    println("\n=== cross-sector dressed overlap + E0-linear tangent blocks vs dense oracle ===")
    errc = maximum(validateCrossPhased(; Nsites=8, D=3, m=mm, mp=mmp)
                   for (mm, mmp) in ((1, 2), (2, 1), (1, 3)))
    println(errc < 1e-10 ? "  CROSS TANGENT OK" : "  CROSS TANGENT FAILED (err=$errc)")

    println("\n=== END-TO-END: scalable transfer-matrix enriched energy vs dense enrich.jl ===")
    errE = validateEnrichedVsDense(; Nsites=6, D=4, warmSteps=1500, maxit=600)
    println(errE < 1e-8 ? "  ENRICHED SOLVER OK (matches dense projection)" :
                          "  ENRICHED SOLVER MISMATCH (max |diff|/site = $errE)")
end
