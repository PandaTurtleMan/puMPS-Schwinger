# CORRECTED holonomy solver: Level A / Level B ground state and the enriched (Bloch)
# spectrum, wired onto the polynomial-weight primitives of holonomy_fixed.jl.
#
# Exactly three substitutions relative to the mod-1 path, everything else is verbatim:
#   diagonal      gC * N E_0^2                  ->  (gC/N) * e^2
#                 e0GradAndNumFast              ->  e2GradAndNum
#                 HE0 character sum             ->  e2Block
#   cross sector  gC * 2N h_mm' * <E_0 U^dag U'> ->  gC * 2 h_mm' * <e U^dag U'>
#                 crossE0Val / crossE0Grad      ->  crossEVal / crossEGrad
#                 crossE0Block                  ->  crossEBlock
# The N h2_mm' self-energy, the Coulomb kernel, the lifting term and the twisted XX gate
# are UNCHANGED (the kernel is the translation-invariant fluctuation part of sum_j L_j^2).
include("holonomy_fixed_fused.jl")   # -> holonomy_fixed.jl, plus crossEBlockFused
using LinearAlgebra, Printf, Random

# ---------------------------------------------------------------------------
# LEVEL A: per-sector independent optimisation in the twisted diagonal Hamiltonian
# ---------------------------------------------------------------------------
function schwingerGradientFixed(A, lambda, lamInv, gate, mpoTerms, gC, Nsites)
    Nb = Nsites ÷ 2
    Epow = transferPowers(fastTransferMatrix(A), Nb - 1)
    Gm = metricBlockFast(A, lamInv, Nb; Epow=Epow)
    numH = calculateGradientTensorFast(A, gate, lamInv, Nb; Epow=Epow)
    for (coef, W, vL, vR) in mpoTerms
        numH = numH .+ coef .* calculateGradientTensorMPOFast(A, W, vL, vR, lamInv, Nb)
    end
    numHE, _ = e2GradAndNum(A, lamInv, Nsites)
    numH = numH .+ (gC / Nsites) .* numHE
    return trueGradientFast(Gm, numH, A, lambda, lamInv, Nb)
end

# monotone descent with backtracking (mirrors fullGD_step3_fast; energy-tracked, never
# best-eta -- see [[pumps-fast-optimizer]]: best-eta selects the polarised product state)
function fullGD_step3_fixed(maxSteps, alpha, A0, gate, mpoTerms, gC, Nsites;
                            tol=1e-9, growEvery=20, growth=1.1, alphaMin=1e-8)
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    dAL, eta, en = schwingerGradientFixed(A, lambda, lamInv, gate, mpoTerms, gC, Nsites)
    bestA = copy(A); bestLam = copy(lambda); bestE = en; bestEta = eta
    al = alpha; good = 0; used = 1
    for i in 2:maxSteps
        used = i
        eta < tol && break
        At, lamT = canonicalization(oneGradientDescentStep(A, al, dAL))
        lamInvT = 1.0 ./ max.(lamT, 1e-12)
        dALt, etaT, enT = schwingerGradientFixed(At, lamT, lamInvT, gate, mpoTerms, gC, Nsites)
        if enT <= en + 1e-12 * abs(en)
            A, lambda, lamInv = At, lamT, lamInvT
            dAL, eta, en = dALt, etaT, enT
            if en < bestE; bestE = en; bestEta = eta; bestA = copy(A); bestLam = copy(lambda); end
            good += 1
            good >= growEvery && (al = min(al * growth, 10 * alpha); good = 0)
        else
            al = max(al / 2, alphaMin); good = 0
            al <= alphaMin && break
        end
    end
    return bestA, bestLam, bestE, bestEta, used
end

# momentum-accelerated natural-gradient descent (heavy-ball with restart-on-rejection).
# Same monotone-energy backtracking guard as fullGD_step3_fixed, plus a velocity term
# beta*V that accelerates convergence in the slow large-N regime where plain natural-GD
# stalls (eta ~ 1e-5 at N=40 vs 1e-8 at N=6).  Historical experiment: restarting on
# rejection does not repair the missing gauge transport.  Use fullGD_step3_tmom instead.
function fullGD_step3_accel(maxSteps, alpha, A0, gate, mpoTerms, gC, Nsites;
                            tol=1e-9, beta=0.85, growEvery=20, growth=1.1, alphaMin=1e-8)
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    dAL, eta, en = schwingerGradientFixed(A, lambda, lamInv, gate, mpoTerms, gC, Nsites)
    V = copy(dAL)
    bestA = copy(A); bestLam = copy(lambda); bestE = en; bestEta = eta
    al = alpha; good = 0; used = 1
    for i in 2:maxSteps
        used = i
        eta < tol && break
        Vtry = beta .* V .+ dAL
        At, lamT = canonicalization(oneGradientDescentStep(A, al, Vtry))
        lamInvT = 1.0 ./ max.(lamT, 1e-12)
        dALt, etaT, enT = schwingerGradientFixed(At, lamT, lamInvT, gate, mpoTerms, gC, Nsites)
        if enT <= en + 1e-12 * abs(en)
            A, lambda, lamInv = At, lamT, lamInvT
            dAL, eta, en = dALt, etaT, enT
            V = Vtry
            if en < bestE; bestE = en; bestEta = eta; bestA = copy(A); bestLam = copy(lambda); end
            good += 1
            good >= growEvery && (al = min(al * growth, 10 * alpha); good = 0)
        else
            al = max(al / 2, alphaMin); good = 0
            V = zero(dAL)                       # restart momentum -> next try is pure natural-GD
            al <= alphaMin && break
        end
    end
    return bestA, bestLam, bestE, bestEta, used
end

# Per-sector warm loop with GAUGE-TRANSPORTED heavy-ball momentum.  This is fullGD_step3_accel
# with the one fix that makes momentum work at all here: canonicalisation is the linear gauge
# map AL^s = ML A^s MR, so the velocity must be carried across it as V^s -> ML V^s MR.
# `fullGD_step3_accel` above does NOT transport, and that is not a small effect -- it is the
# difference between converging and diverging.
#
# MEASURED (warm_bench.jl, N=20 D=10 hmax=3 x=4, 1500 steps/sector, 7 sectors, vs the
# `fullGD_step3_fixed` baseline EA/N = -0.5565169456 at max eta 2.85e-4):
#     untransported beta=0.9   EA/N -0.5328593594   max eta 2.63e-1   <- COLLAPSES to alphaMin
#     transported   beta=0.70  EA/N -0.5565161585   max eta 4.22e-3   <- worse than baseline
#     transported   beta=0.90  EA/N -0.5565185837   max eta 9.80e-5
#     transported   beta=0.95  EA/N -0.5565208513   max eta 5.24e-5   <- 5.4x, and 2459 s vs 2556
# The response is NOT monotone in beta: below ~0.9 momentum is worse than no momentum at all.
#
# `capMul` is the step-growth cap as a multiple of alpha.  fullGD_step3_fixed hard-codes 10;
# with momentum on top of an already-10x-grown step the line search just thrashes, so the
# default here is 1 (measured: capMul 10 -> max eta 1.78e-4 vs capMul 1 -> 9.80e-5 at beta=0.9).
# beta=0 reproduces fullGD_step3_fixed exactly apart from that cap.
function fullGD_step3_tmom(maxSteps, alpha, A0, gate, mpoTerms, gC, Nsites;
                           tol=1e-9, beta=0.95, capMul=1.0,
                           growEvery=20, growth=1.1, alphaMin=1e-8)
    usemom = beta > 0; alCap = capMul * alpha
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    dAL, eta, en = schwingerGradientFixed(A, lambda, lamInv, gate, mpoTerms, gC, Nsites)
    V = zero(dAL)
    bestA = copy(A); bestLam = copy(lambda); bestE = en; bestEta = eta
    al = alpha; good = 0; used = 1
    for i in 2:maxSteps
        used = i
        eta < tol && break
        step = usemom ? beta .* V .+ dAL : dAL
        At, lamT, ML, MR = canonicalizationG(oneGradientDescentStep(A, al, step))
        lamInvT = 1.0 ./ max.(lamT, 1e-12)
        dALt, etaT, enT = schwingerGradientFixed(At, lamT, lamInvT, gate, mpoTerms, gC, Nsites)
        if enT <= en + 1e-12 * abs(en)
            A, lambda, lamInv = At, lamT, lamInvT
            dAL, eta, en = dALt, etaT, enT
            usemom && (V = transportV(step, ML, MR))   # into the NEW gauge, or it is noise
            if en < bestE; bestE = en; bestEta = eta; bestA = copy(A); bestLam = copy(lambda); end
            good += 1
            good >= growEvery && (al = min(al * growth, alCap); good = 0)
        else
            al = max(al / 2, alphaMin); good = 0
            V = zero(dAL)                       # restart momentum -> next try is pure natural-GD
            al <= alphaMin && break
        end
    end
    return bestA, bestLam, bestE, bestEta, used
end

# ---------------------------------------------------------------------------
# LEVEL B: coupled sectors
# ---------------------------------------------------------------------------
function diagNumFixed(A, lamInv, gate, mpoTerms, gC, Nsites; Epow=nothing)
    Nb = Nsites ÷ 2
    Epow === nothing && (Epow = transferPowers(fastTransferMatrix(A), Nb - 1))
    numH = calculateGradientTensorFast(A, gate, lamInv, Nb; Epow=Epow)
    for (coef, W, vL, vR) in mpoTerms
        numH = numH .+ coef .* calculateGradientTensorMPOFast(A, W, vL, vR, lamInv, Nb)
    end
    numHE, _ = e2GradAndNum(A, lamInv, Nsites)
    return numH .+ (gC / Nsites) .* numHE
end

function buildHNFixedGradientReference(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2)
    M = length(As); Nb = Nsites ÷ 2
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    vecACs = [packAC(As[m], lambdas[m]) for m in 1:M]
    Heff = zeros(ComplexF64, M, M); Neff = zeros(ComplexF64, M, M)
    for m in 1:M
        Epow = transferPowers(fastTransferMatrix(As[m]), Nb - 1)
        g = metricBlockFast(As[m], lamInvs[m], Nb; Epow=Epow)
        nd = diagNumFixed(As[m], lamInvs[m], gates[m], mpoTerms, gC, Nsites; Epow=Epow)
        nn = vec(g * reshape(vecACs[m], size(g, 1), :))
        Neff[m, m] = real(dot(vecACs[m], nn))
        Heff[m, m] = real(dot(vecACs[m], nd)) + gC * Nsites * real(h2[m, m]) * Neff[m, m]
    end
    for m in 1:M, mp in m+1:M                      # Hermiticity: compute m<mp, conjugate for mp<m
        Delta = 2π * (mp - m) / M
        ov, ee = crossOvEVal(As[m], As[mp], Nb, Nsites, Delta)
        val = gC * (2 * hm[m, mp] * ee + Nsites * h2[m, mp] * ov)   # 2, not 2N
        Heff[m, mp] = val
        Heff[mp, m] = conj(val)
    end
    return (Heff + Heff') / 2, Hermitian(Neff)
end

# Exact scalar trial energy. Preserve the historical Schmidt-floor semantics
# for exceptional rank-deficient inputs by retaining the reference fallback.
function buildHNFixed(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2)
    any(l -> any(x -> x < 1e-12, l), lambdas) &&
        return buildHNFixedGradientReference(As,lambdas,gates,mpoTerms,gC,Nsites,hm,h2)
    M = length(As); Nb = Nsites÷2
    H = zeros(ComplexF64,M,M); N = zeros(ComplexF64,M,M)
    for m in 1:M
        A = As[m]; E = fastTransferMatrix(A)
        electric, nrm = e2ValueFast(A,Nsites;E=E)
        gate = Nb * tr(fastGateTransfer(A,gates[m]) * E^(Nb-2))
        mpo = sum(coef * mpoValueFast(A,W,vL,vR,Nb) for (coef,W,vL,vR) in mpoTerms; init=0.0im)
        N[m,m] = real(nrm)
        H[m,m] = real(gate + mpo + (gC/Nsites)*electric) + gC*Nsites*real(h2[m,m])*real(nrm)
    end
    for m in 1:M, mp in m+1:M
        ov,ee = crossOvEVal(As[m],As[mp],Nb,Nsites,2pi*(mp-m)/M)
        H[m,mp] = gC * (2hm[m,mp]*ee + Nsites*h2[m,mp]*ov)
        H[mp,m] = conj(H[m,mp])
    end
    return Hermitian(H), Hermitian(N)
end

# blockmetric=true keeps just Gm from I_d ⊗ Gm (16x less metric storage for
# d=4). The default retains the full-matrix API used by diagnostic scripts.
function gradientBFixed(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2; blockmetric=false)
    M = length(As); Nb = Nsites ÷ 2
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    vecACs = [packAC(As[m], lambdas[m]) for m in 1:M]
    gs = Vector{Matrix{ComplexF64}}(undef, M)
    numHdiag = Vector{Vector{ComplexF64}}(undef, M)
    normGrads = Vector{Vector{ComplexF64}}(undef, M)
    for m in 1:M
        Epow = transferPowers(fastTransferMatrix(As[m]), Nb - 1)
        gm = metricBlockFast(As[m], lamInvs[m], Nb; Epow=Epow)
        d = size(As[m], 3)
        gs[m] = blockmetric ? gm : kron(Matrix{ComplexF64}(I, d, d), gm)
        normGrads[m] = vec(gm * reshape(vecACs[m], size(gm, 1), d))
        numHdiag[m] = diagNumFixed(As[m], lamInvs[m], gates[m], mpoTerms, gC, Nsites; Epow=Epow)
    end
    Draw = [real(dot(vecACs[m], numHdiag[m])) for m in 1:M]
    nrm = [real(dot(vecACs[m], normGrads[m])) for m in 1:M]
    ovV = zeros(ComplexF64, M, M); eeV = zeros(ComplexF64, M, M)
    ovF = Matrix{Array{ComplexF64,3}}(undef, M, M); eeF = Matrix{Array{ComplexF64,3}}(undef, M, M)
    for m in 1:M, mp in 1:M
        m == mp && continue # diagonal electric cross term is unused; norm gradient is known
        Delta = 2π * (mp - m) / M
        ovV[m, mp], ovF[m, mp], eeV[m, mp], eeF[m, mp] =
            crossOvEGrad(As[m], As[mp], Nb, Nsites, Delta)
    end
    Heff = zeros(ComplexF64, M, M); Neff = Diagonal(ComplexF64.(nrm))
    for m in 1:M; Heff[m, m] = Draw[m] + gC * Nsites * real(h2[m, m]) * nrm[m]; end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Heff[m, mp] = gC * (2 * hm[m, mp] * eeV[m, mp] + Nsites * h2[m, mp] * ovV[m, mp])
    end
    Heff = (Heff + Heff') / 2
    E, c = solveC(Heff, Matrix(Neff))
    Den = real(dot(c, Neff * c))
    grads = Vector{Vector{ComplexF64}}(undef, M)
    for m in 1:M
        acc = c[m] .* numHdiag[m]
        for mp in 1:M
            if mp == m
                acc = acc .+ c[m] .* (gC * Nsites * real(h2[m, m])) .* normGrads[m]
            else
                acc = acc .+ c[mp] .* (gC .* (2 * hm[m, mp] .* packGrad(eeF[m, mp], lamInvs[m], Nb)
                                             .+ Nsites * h2[m, mp] .* packGrad(ovF[m, mp], lamInvs[m], Nb)))
            end
        end
        grads[m] = (conj(c[m]) .* acc .- E .* (abs2(c[m]) .* normGrads[m])) ./ Den
    end
    return E, c, grads, gs, vecACs, lamInvs
end

# Pad a D0-bond tensor into a D-bond tensor (top-left block + tiny noise on the new bonds so
# canonicalization doesn't see an exactly-singular block).  For a D-ladder warm start.
function padTensor(A, D)
    D0 = size(A, 1); d = size(A, 3)
    D0 == D && return copy(A)
    D0 < D || error("padTensor: target D=$D smaller than source D0=$D0")
    B = zeros(ComplexF64, D, D, d)
    B[1:D0, 1:D0, :] .= A
    B[D0+1:D, :, :] .+= 1e-6 .* randn(ComplexF64, D-D0, D, d)
    B[1:D0, D0+1:D, :] .+= 1e-6 .* randn(ComplexF64, D0, D-D0, d)
    return B
end

# canonicalisation that ALSO returns the gauge map (AL^s = ML A^s MR).  Needed to transport a
# momentum velocity across the re-gauging that canonicalisation performs every step: an
# untransported velocity points in a gauge-rotated (wrong) direction, which is why the earlier
# `fullGD_step3_accel` heavy-ball failed.  Same algorithm as `canonicalization`, extra outputs.
function canonicalizationG(A)
    D=size(A,1); d=size(A,3)
    Lmap=zeros(ComplexF64,D*D,D*D)
    for s in 1:d,a in 1:D,b in 1:D,ap in 1:D,bp in 1:D; Lmap[b+(bp-1)*D,a+(ap-1)*D]+=conj(A[a,b,s])*A[ap,bp,s]; end
    Fl=eigen(Lmap); kl=argmax(abs.(Fl.values)); eta=real(Fl.values[kl])
    l=reshape(Fl.vectors[:,kl],(D,D)); l=(l+l')/2; real(tr(l))<0 && (l=-l)
    el=eigen(Hermitian(l)); dl=max.(real(el.values),0.0)
    X=Diagonal(sqrt.(dl))*el.vectors'; Xi=el.vectors*Diagonal(1.0./sqrt.(dl.+eps()))
    Atil=similar(A,ComplexF64); for s in 1:d; Atil[:,:,s]=X*A[:,:,s]*Xi/sqrt(eta); end
    Rmap=zeros(ComplexF64,D*D,D*D)
    for s in 1:d,a in 1:D,b in 1:D,ap in 1:D,bp in 1:D; Rmap[a+(ap-1)*D,b+(bp-1)*D]+=Atil[a,b,s]*conj(Atil[ap,bp,s]); end
    Fr=eigen(Rmap); kr=argmax(abs.(Fr.values)); r=reshape(Fr.vectors[:,kr],(D,D)); r=(r+r')/2; real(tr(r))<0 && (r=-r)
    er=eigen(Hermitian(r)); order=sortperm(real(er.values);rev=true); dr=max.(real(er.values[order]),0.0); W=er.vectors[:,order]
    AL=similar(A,ComplexF64); for s in 1:d; AL[:,:,s]=W'*Atil[:,:,s]*W; end
    lambda=sqrt.(dr); lambda./=norm(lambda)
    return AL, lambda, (W'*X)./sqrt(eta), Xi*W
end
transportV(V,ML,MR)=(Vt=similar(V); for s in 1:size(V,3); Vt[:,:,s]=ML*V[:,:,s]*MR; end; Vt)

# Each Level-B sector metric is exactly I_d ⊗ Gm.  Solve the D² block
# against all physical slices at once, and reuse that solve for eta.  This
# keeps the same pseudo-inverse cutoff as naturalStep; no regularisation change.
function naturalStepsBFixed(gs, grads, lamInvs, D)
    steps = Vector{Array{ComplexF64,3}}(undef, length(gs))
    eta2 = 0.0
    for m in eachindex(gs)
        d = length(grads[m]) ÷ D^2
        gblock = @view gs[m][1:D^2, 1:D^2]
        invgrad = pseudoSolve(gblock, reshape(grads[m], D^2, d))
        eta2 += real(dot(grads[m], vec(invgrad)))
        steps[m] = -reshape(invgrad, D, D, d) .* reshape(lamInvs[m], 1, D, 1)
    end
    return steps, sqrt(abs(eta2))
end

# `betaB` = gauge-transported heavy-ball momentum on the Level-B natural-gradient step.
# MEASURED (opt_bench.jl, N=20 D=10 hmax=3 x=4, 300 iters, vs exact ED -11.365990777):
#   betaB=0    E-E_ED = 6.59e-04     betaB=0.9  E-E_ED = 1.81e-05   (36x, and CHEAPER --
# it barely trips the line search, so it costs FEWER buildHNFixed calls than plain).
# betaB=0 reproduces the original loop exactly.  NOTE: raising the `al` growth cap was also
# tested and does NOTHING at N=20 (it helped only at N=12), so the cap below is left alone.
function levelB_GS_fixed(; Nsites=6, hmax=1, D=4, gC=0.5, lam=1.0, mass=0.0, warmSteps=2000,
                          maxit=400, alpha=0.03, seed=4, tol=1e-9, verbose=true,
                          seedAs=nothing, seed0=nothing, betaB=0.9, betaW=0.95)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    # The two output-sublattice channels share the same input operator and
    # polynomial propagation. Merge them: two chi=5 MPOs instead of four.
    mpoTerms = vcat(kernelTermsMerged(Nsites, gC), [(lam, Wl, vLl, vRl)])
    gates = [blockedXXGateTwisted(2π * (m - 1) / M / Nsites; mass=mass) for m in 1:M]
    hm, h2, _ = holonomyMats(hmax)

    As = Vector{Array{ComplexF64,3}}(undef, M); lambdas = Vector{Vector{Float64}}(undef, M)
    # CASCADE warm start: converge sector 0 from a random seed (or the caller-supplied seedAs
    # for a D-ladder), then seed each next sector from the previous converged tensor.  Adjacent
    # holonomy gates differ only by a twist 2pi/(M*Nsites), so all but the first sector hit the
    # fullGD_step3_fixed gradient-norm criterion (eta<tol) in far fewer steps than from random.
    # sector-0 seed priority: full per-sector seedAs (D-ladder) > seed0 (N-ladder: a converged
    # small-N untwisted D×D×4 tensor) > random.
    if seedAs !== nothing
        prevA = seedAs[1]
    elseif seed0 !== nothing
        prevA = seed0
    else
        Random.seed!(seed); prevA = randn(ComplexF64, D, D, 4)
    end
    for mm in 0:M-1
        A0 = seedAs === nothing ? prevA : seedAs[mm+1]
        # betaW = 0 falls back to the original plain-GD warm loop (fullGD_step3_fixed).
        A_L, _, _, eta_w, used_w = betaW > 0 ?
            fullGD_step3_tmom(warmSteps, alpha, A0, gates[mm+1], mpoTerms, gC, Nsites; beta=betaW) :
            fullGD_step3_fixed(warmSteps, alpha, A0, gates[mm+1], mpoTerms, gC, Nsites)
        As[mm+1], lambdas[mm+1] = canonicalization(A_L)
        verbose && (@printf("      warm sector %2d/%d: %4d steps, eta=%.2e\n", mm, M-1, used_w, eta_w); flush(stdout))
        prevA = A_L
    end
    HeffA, NeffA = buildHNFixed(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2)
    EA, _ = solveC(HeffA, Matrix(NeffA))
    verbose && (@printf("    Level-A warm start  E/N = %.10f  -> starting Level-B (maxit=%d)\n", EA / Nsites, maxit); flush(stdout))

    E, c, grads, gs, vecACs, lamInvs = gradientBFixed(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2; blockmetric=true)
    dALs, eta = naturalStepsBFixed(gs, grads, lamInvs, D)
    bestE = E; bestAs = deepcopy(As); bestLams = deepcopy(lambdas)
    al = alpha
    Vmom = [zero(As[m]) for m in 1:M]
    stopReason = "iteration limit reached"
    acceptedSteps = 0
    bestEta = eta
    for it in 1:maxit
        if eta < tol
            stopReason = "gradient tolerance reached"
            break
        end
        # MOMENTUM-COLLAPSE GUARD.  The momentum direction can become unusable at larger N/D:
        # every trial step is rejected, `al` halves ~24 times to 1e-9, the loop exits with
        # accepted=false and Level-B BREAKS -- silently reporting "converged" at a badly wrong
        # energy (measured: N=32 D=10 broke at it~25 with eta=0.21, E/N 9.1e-4/site off trend;
        # N=28 and N=32 at D=12 likewise, landing above their D=10 results because the
        # optimisation was incomplete).  Fix: if the momentum step exhausts the line search, retry
        # ONCE with the plain natural gradient at the original step size before giving up.
        # Worst case this degrades to plain GD, which is exactly the betaB=0 behaviour that was
        # measured to work; it can only help.
        accepted = false
        nattempt = betaB > 0 ? 2 : 1
        for attempt in 1:nattempt
            useMom = (betaB > 0 && attempt == 1)
            step = useMom ? [betaB .* Vmom[m] .+ dALs[m] for m in 1:M] : dALs
            altry = attempt == 1 ? al : alpha          # reset the step size for the fallback
            while altry > 1e-9
                trialAs = Vector{Array{ComplexF64,3}}(undef, M); trialLams = Vector{Vector{Float64}}(undef, M)
                MLs = Vector{Any}(undef, M); MRs = Vector{Any}(undef, M)
                for m in 1:M
                    if useMom
                        trialAs[m], trialLams[m], MLs[m], MRs[m] = canonicalizationG(As[m] .+ altry .* step[m])
                    else
                        trialAs[m], trialLams[m] = canonicalization(As[m] .+ altry .* step[m])
                    end
                end
                Heff, Neff = buildHNFixed(trialAs, trialLams, gates, mpoTerms, gC, Nsites, hm, h2)
                Etrial, _ = solveC(Heff, Matrix(Neff))
                if Etrial <= E + 1e-12
                    As = trialAs; lambdas = trialLams; E = Etrial; accepted = true; al = altry
                    Vmom = useMom ? [transportV(step[m], MLs[m], MRs[m]) for m in 1:M] :
                                    [zero(As[m]) for m in 1:M]
                    break
                end
                altry *= 0.5
            end
            accepted && break
            # momentum path exhausted: drop the velocity and let attempt 2 try plain GD
            Vmom = [zero(As[m]) for m in 1:M]
            verbose && attempt < nattempt && (@printf("      [guard] it %d: momentum step exhausted the line search; falling back to plain GD\n", it); flush(stdout))
        end
        if !accepted
            stopReason = "line search exhausted"
            break
        end
        acceptedSteps += 1
        E, c, grads, gs, vecACs, lamInvs = gradientBFixed(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2; blockmetric=true)
        dALs, eta = naturalStepsBFixed(gs, grads, lamInvs, D)
        if E < bestE; bestE = E; bestAs = deepcopy(As); bestLams = deepcopy(lambdas); bestEta = eta; end
        al = min(al * 1.2, alpha)
        verbose && (it % 25 == 0 || it <= 3) && (@printf("      Level-B it %3d/%d: E/N=%.10f  eta=%.2e  al=%.2e\n", it, maxit, E/Nsites, eta, al); flush(stdout))
    end
    if verbose
        @printf("    Level-B stopped: %s after %d accepted steps\n", stopReason, acceptedSteps)
        @printf("      best E/N = %.10f  (lowered %.2e/site), eta=%.2e\n",
                bestE / Nsites, (EA - bestE) / Nsites, bestEta)
        flush(stdout)
    end
    return bestE, bestAs, bestLams
end

# ---------------------------------------------------------------------------
# ENRICHED (Bloch) assembly
# ---------------------------------------------------------------------------
function buildEnrichedHNFixed(As, lambdas, kset, Nsites, gC, lam, hmax; mass=0.0)
    Nb = Nsites ÷ 2; M = 2hmax + 1; D = size(As[1], 1); dt = 4 * D * D
    bs = length(kset) * dt; Ntot = M * bs
    @assert 0 in kset "kset must contain 0 (ground lives in the k=0 tangent span)"
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    hm, h2, _ = holonomyMats(hmax)
    kterms = kernelTerms(Nsites, gC)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    Heff = zeros(ComplexF64, Ntot, Ntot); Neff = zeros(ComplexF64, Ntot, Ntot)
    for m in 1:M
        A = As[m]; li = lamInvs[m]; theta_m = 2π * (m - 1) / M
        hmat = blockedXXGateTwisted(theta_m / Nsites; mass=mass)
        Nblk  = ttpart(diagPhasedBlock(B -> ONES4, A, A, li, li, kset, kset, Nb))
        Hxx   = fourierTT(sep -> H0k(A, li, hmat, sep, Nb), kset, Nb, dt)
        Hker  = fourierTT(sep -> sum(coef .* mpoTangentBlock(A, li, W, vL, vR, sep, Nb)
                                     for (coef, W, vL, vR) in kterms), kset, Nb, dt)
        Hlift = fourierTT(sep -> lam .* mpoTangentBlock(A, li, Wl, vLl, vRl, sep, Nb), kset, Nb, dt)
        HE    = (gC / Nsites) .* ttpart(e2BlockFused(A, A, li, li, kset, kset, Nb)) # CORRECTED
        selfE = (gC * Nsites * real(h2[m, m])) .* Nblk
        rng = (m-1)*bs+1 : m*bs
        Heff[rng, rng] = Hxx .+ Hker .+ Hlift .+ HE .+ selfE
        Neff[rng, rng] = Nblk
    end
    for m in 1:M, mp in m+1:M                      # Hermiticity: compute m<mp, adjoint for mp<m
        Delta = 2π * (mp - m) / M
        ov = ttpart(crossOverlapBlock(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb))
        # fused one-weight block: the Nb separate diagPhasedBlock calls of crossEBlock
        # collapse into one sweep (~2x), and this runs over the M(M-1)/2 upper sector pairs
        ee = ttpart(crossEBlockFused(As[m], As[mp], lamInvs[m], lamInvs[mp], Delta, Nsites, kset, kset, Nb))
        blk = gC .* (2 * hm[m, mp] .* ee .+ Nsites * h2[m, mp] .* ov)              # 2, not 2N
        Heff[(m-1)*bs+1:m*bs, (mp-1)*bs+1:mp*bs] = blk
        Heff[(mp-1)*bs+1:mp*bs, (m-1)*bs+1:m*bs] = blk'
    end
    return Hermitian((Heff + Heff') / 2), Hermitian((Neff + Neff') / 2)
end

function enrichedLevelsFixed(As, lambdas, kset, Nsites, gC, lam, hmax; nev=1, cutoff=1e-10, mass=0.0)
    Heff, Neff = buildEnrichedHNFixed(As, lambdas, kset, Nsites, gC, lam, hmax; mass=mass)
    F = eigen(Neff); thresh = cutoff * maximum(F.values)
    keep = findall(>(thresh), F.values)
    W = F.vectors[:, keep] * Diagonal(1.0 ./ sqrt.(F.values[keep]))
    He = W' * (Heff * W)
    ev = sort(real(eigvals(Hermitian((He + He') / 2))))
    return ev[1:min(nev, length(ev))], length(keep)
end

# ---------------------------------------------------------------------------
# DENSE ORACLE for the corrected model on the FULL 2^N * M space (analogue of
# enrich.jl's denseHolonomyH), so the whole assembly can be checked end to end.
function denseHolonomyHFixed(Nsites, gC, lam, hmax; mass=0.0)
    M = 2hmax + 1; n = 2^Nsites
    ev = eInt(Nsites); w = wDiag(Nsites)
    hm, h2, _ = holonomyMats(hmax)
    H = zeros(ComplexF64, n * M, n * M)
    for m in 0:M-1
        H[m*n+1:(m+1)*n, m*n+1:(m+1)*n] .+= Hdiag_blocked_fixed(Nsites, 2π * m / M, gC, lam; mass=mass)
    end
    for m in 0:M-1, mp in 0:M-1
        Delta = 2π * (mp - m) / M
        ph = exp.(im .* Delta .* w ./ Nsites)
        cd = gC .* (2 .* ev .* hm[m+1, mp+1] .+ Nsites .* h2[m+1, mp+1]) .* ph
        H[m*n+1:(m+1)*n, mp*n+1:(mp+1)*n] .+= Diagonal(cd)
    end
    return Hermitian((H + H') / 2)
end

# END-TO-END: the transfer-matrix enriched energy must equal the dense projection of
# denseHolonomyHFixed onto span{ground_m} U {tangents}, per momentum shell.
function validateFixedEnriched(; Nsites=6, D=4, gC=0.5, lam=1.0, hmax=1, mass=0.0,
                                warmSteps=1500, maxit=400, alpha=0.05, seed=4)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    EB, As, lambdas = levelB_GS_fixed(; Nsites=Nsites, hmax=hmax, D=D, gC=gC, lam=lam,
                                       mass=mass, warmSteps=warmSteps, maxit=maxit, alpha=alpha,
                                       seed=seed, verbose=false)
    H = denseHolonomyHFixed(Nsites, gC, lam, hmax; mass=mass)
    Efull = minimum(real(eigvals(H)))
    Vs = [impurityVectors(As[m], Nb) for m in 1:M]
    @printf("N=%d D=%d gC=%.2f hmax=%d mass=%.2f : Level B E/N=%.9f   full ED E/N=%.9f\n",
            Nsites, D, gC, hmax, mass, EB/Nsites, Efull/Nsites)
    @printf("  shell  E_TM/N          E_dense/N       |diff|/site\n")
    worst = 0.0
    for shell in 0:(Nb ÷ 2)
        kset = [k for k in 0:Nb-1 if min(k, Nb - k) <= shell]
        Etm, _ = enrichedLevelsFixed(As, lambdas, kset, Nsites, gC, lam, hmax; mass=mass)
        Eds, _ = enrichedEnergy(H, As, Vs, Nsites, M, kset)
        @printf("  %-2d     %.10f  %.10f  %.2e\n", shell, Etm[1]/Nsites, Eds/Nsites,
                abs(Etm[1] - Eds)/Nsites)
        worst = max(worst, abs(Etm[1] - Eds)/Nsites); flush(stdout)
    end
    # Level B must equal the ground-only projection, and be variational above full ED
    Eg, _ = enrichedEnergy(H, As, Vs, Nsites, M, Int[])
    @printf("  Level B vs ground-only projection: %.2e   (variational margin %.3e/site)\n",
            abs(EB - Eg)/Nsites, (EB - Efull)/Nsites)
    return max(worst, abs(EB - Eg)/Nsites)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== corrected Level B + enriched assembly vs dense projection ===")
    mass = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.0
    e = validateFixedEnriched(; Nsites=6, D=4, hmax=1, mass=mass)
    println(e < 1e-8 ? "  CORRECTED SOLVER OK" : "  MISMATCH ($e)")
end
