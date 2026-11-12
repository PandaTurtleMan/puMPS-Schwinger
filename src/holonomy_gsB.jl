# LEVEL B: fully variational holonomy ground state.
#
# Level A froze each sector tensor A_m at its own-sector optimum, then diagonalized
# the M x M problem for the amplitudes c_m -- leaving a ~4.5% gap to ED because the
# A_m never relax to feel the m<->m' coupling.  Level B closes that gap: it takes the
# gradient of the FULL coupled energy w.r.t. every A_m and does natural-gradient
# descent, re-solving the small c_m eigenproblem after each A-step (block coordinate
# descent, monotone with backtracking).  Warm-started from Level A.
#
# The one genuinely new object is the BRA-gradient of the mixed cross-overlaps
#   d/dconj(A_m) <psi_m| {1, E0} U^dag U' |psi_m'>
# built by the SAME prefix/suffix + impurity-average pattern as e0GradAndNum
# ([[pumps-implementation]] / puMPS_schwinger_step3.jl), only with the mixed
# transfer mixedTwistedE(A_m, A_m', .) (bra=A_m, ket=A_m') in place of twistedE(A,.).
#
# The DEFAULT optimizer (levelB_GS) is the all-gemm FAST path: the diagonal terms go
# through puMPS_fast.jl's BLAS rewrites and the new cross terms through
# fastMixedTransfer / phasedTraceGradFast (the scalar impurity loop -> one gemm).
# The slow scalar-loop implementations are kept below with a `_ref` suffix as the
# readable reference and regression oracle (regressFastB checks fast == _ref).
include("holonomy_gs.jl")   # optimizeSector, holonomyGS, blockedXXGateTwisted,
                            # crossOverlap/crossE0lin (values), mixedTwistedE,
                            # Nsvals/nOvals/Qbvals/q1vals, e0CharFhat, GD machinery
include("puMPS_fast.jl")    # fastTransferMatrix, transferPowers, *Fast diag routines,
                            # fullGD_step3_fast   (re-includes step3: redefinition warnings OK)
using LinearAlgebra, Random, Printf

# ===========================================================================
# packing helpers (shared by fast and reference paths)
# ===========================================================================
mu_pack(s, a, c, D) = a + (c - 1) * D + (s - 1) * D * D
packAC(A, lambda) = begin
    D = size(A, 1); d = size(A, 3); v = zeros(ComplexF64, d * D * D)
    for s in 1:d, a in 1:D, c in 1:D; v[mu_pack(s, a, c, D)] = A[a, c, s] * lambda[c]; end
    v
end
# raw impurity tensor F[s,a,c] -> A_C-gradient packed (1/Nb average + lambda^{-1})
packGrad(F, lamInv, Nb) = begin
    d, D, _ = size(F); v = zeros(ComplexF64, d * D * D)
    for s in 1:d, a in 1:D, c in 1:D; v[mu_pack(s, a, c, D)] = F[s, a, c] * lamInv[c] / Nb; end
    v
end

# per-block phase functions for the cross terms (length-4 vectors over the block basis)
dressPhase(Delta, Nsites) = B -> ComplexF64[exp(im * Delta * (2B * Nsvals[s] + nOvals[s]) / Nsites) for s in 1:4]
e0Phase(k, Delta, Nsites) = B -> ComplexF64[exp(-2π * im * k * (2B * Qbvals[s] + q1vals[s]) / Nsites +
                                                im * Delta * (2B * Nsvals[s] + nOvals[s]) / Nsites) for s in 1:4]

# ===========================================================================
# FAST default path (all gemm)
# ===========================================================================
# mixed transfer as a gemm:  E[idx(b,bp),idx(a,ap)] = sum_s phase[s] Amp[a,b,s] conj(Am[ap,bp,s])
# (ket = Amp, bra = Am).  Same as fastTransferMatrix but with distinct ket/bra tensors.
function fastMixedTransfer(Am, Amp; phase=nothing)
    D = size(Am, 1); d = size(Am, 3)
    Ak = phase === nothing ? Amp : Amp .* reshape(phase, 1, 1, d)
    M1 = reshape(permutedims(Ak, (2, 1, 3)), D * D, d)          # [(b,a), s]   ket
    M2 = reshape(permutedims(conj(Am), (3, 2, 1)), d, D * D)    # [s, (bp,ap)] bra
    P = M1 * M2
    return reshape(permutedims(reshape(P, (D, D, D, D)), (1, 3, 2, 4)), D * D, D * D)
end

# value + bra-gradient of tr prod_B fastMixedTransfer(Am,Amp,phaseFun(B)); the
# impurity loop `for s,a,b,ap,bp` of phasedTraceGrad_ref becomes one gemm Ampmat*EnvP.
function phasedTraceGrad(Am, Amp, Nb, phaseFun)
    D = size(Am, 1); d = size(Am, 3)
    Es = [fastMixedTransfer(Am, Amp; phase=phaseFun(B)) for B in 0:Nb-1]
    idm = Matrix{ComplexF64}(I, D * D, D * D)
    P = Vector{Matrix{ComplexF64}}(undef, Nb + 1); P[1] = idm
    for i in 1:Nb; P[i+1] = Es[i] * P[i]; end
    S = Vector{Matrix{ComplexF64}}(undef, Nb + 2); S[Nb+2] = idm; S[Nb+1] = Es[Nb]
    for i in Nb-1:-1:1; S[i+1] = S[i+2] * Es[i]; end
    val = tr(P[Nb+1])
    Ampmat = reshape(permutedims(Amp, (3, 1, 2)), d, D * D)      # [s, (a,b)]  ket
    Fmat = zeros(ComplexF64, d, D * D)
    for i in 1:Nb
        Env4 = reshape(P[i] * S[i+2], (D, D, D, D))              # [a,ap,b,bp]
        EnvP = reshape(permutedims(Env4, (1, 3, 2, 4)), D * D, D * D)  # [(a,b),(ap,bp)]
        Fmat .+= phaseFun(i - 1) .* (Ampmat * EnvP)
    end
    return val, reshape(Fmat, (d, D, D))
end

# value-only (buildHN / backtracking): just the trace, no impurity sweep.
function phasedTraceVal(Am, Amp, Nb, phaseFun)
    T = fastMixedTransfer(Am, Amp; phase=phaseFun(0))
    for B in 1:Nb-1; T = fastMixedTransfer(Am, Amp; phase=phaseFun(B)) * T; end
    return tr(T)
end

# <psi_m| U^dag U' |psi_m'>  and its bra-gradient (dressing only)
crossOvGrad(Am, Amp, Nb, Nsites, Delta) = phasedTraceGrad(Am, Amp, Nb, dressPhase(Delta, Nsites))
crossOvVal(Am, Amp, Nb, Nsites, Delta) = phasedTraceVal(Am, Amp, Nb, dressPhase(Delta, Nsites))

# <psi_m| E0 U^dag U' |psi_m'>  and its bra-gradient  (character sum over k)
function crossE0Grad(Am, Amp, Nb, Nsites, Delta; theta=0.0)
    fh = e0CharFhat(Nsites; theta=theta); D = size(Am, 1)
    val = 0.0 + 0.0im; F = zeros(ComplexF64, 4, D, D)
    for k in 0:Nsites-1
        abs(fh[k+1]) < 1e-15 && continue
        v, Fk = phasedTraceGrad(Am, Amp, Nb, e0Phase(k, Delta, Nsites))
        val += fh[k+1] * v; F .+= fh[k+1] .* Fk
    end
    return val, F
end
function crossE0Val(Am, Amp, Nb, Nsites, Delta; theta=0.0)
    fh = e0CharFhat(Nsites; theta=theta); val = 0.0 + 0.0im
    for k in 0:Nsites-1
        abs(fh[k+1]) < 1e-15 && continue
        val += fh[k+1] * phasedTraceVal(Am, Amp, Nb, e0Phase(k, Delta, Nsites))
    end
    return val
end

# matter diagonal numerator (d/dconj(A_C) of <psi|H_matter(theta_m)|psi>): twisted
# gate + kernel/lifting MPOs + gC*N E0^2, via the puMPS_fast.jl BLAS routines.
function diagNum(A, lamInv, gate, mpoTerms, gCE0, Nsites, theta_ext; Epow=nothing)
    Nb = Nsites ÷ 2
    Epow === nothing && (Epow = transferPowers(fastTransferMatrix(A), Nb - 1))
    numH = calculateGradientTensorFast(A, gate, lamInv, Nb; Epow=Epow)
    for (coef, W, vL, vR) in mpoTerms
        numH = numH .+ coef .* calculateGradientTensorMPOFast(A, W, vL, vR, lamInv, Nb)
    end
    numHE0, _ = e0GradAndNumFast(A, lamInv, Nsites; theta=theta_ext)
    return numH .+ gCE0 .* numHE0
end

# ---------------------------------------------------------------------------
# Level-B assembly.  A "config" is (As, lambdas): sector tensors A_L_m and their
# Schmidt vectors, evaluated at FIXED gauge (given lambdas, no recanonicalisation).
solveC(Heff, Neff) = begin
    F = eigen(Hermitian(Heff), Neff); i = argmin(real(F.values))
    return real(F.values[i]), F.vectors[:, i]
end

# Build M x M H_eff, N_eff (values only) at fixed gauge.
function buildHN(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    M = length(As); Nb = Nsites ÷ 2
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    vecACs = [packAC(As[m], lambdas[m]) for m in 1:M]
    Heff = zeros(ComplexF64, M, M); Neff = zeros(ComplexF64, M, M)
    for m in 1:M
        Epow = transferPowers(fastTransferMatrix(As[m]), Nb - 1)
        g = metricBlockFast(As[m], lamInvs[m], Nb; Epow=Epow)     # D^2 x D^2 block
        nd = diagNum(As[m], lamInvs[m], gates[m], mpoTerms, gC, Nsites, theta_ext; Epow=Epow)
        nn = vec(g * reshape(vecACs[m], size(g, 1), :))           # (I_d (x) g) vecAC
        Neff[m, m] = real(dot(vecACs[m], nn))
        Heff[m, m] = real(dot(vecACs[m], nd)) + gC * Nsites * real(h2[m, m]) * Neff[m, m]
    end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Delta = 2π * (mp - m) / M
        e0 = crossE0Val(As[m], As[mp], Nb, Nsites, Delta; theta=theta_ext)
        ov = crossOvVal(As[m], As[mp], Nb, Nsites, Delta)
        Heff[m, mp] = gC * (2Nsites * hm[m, mp] * e0 + Nsites * h2[m, mp] * ov)
    end
    return (Heff + Heff') / 2, Hermitian(Neff)
end

energyAt(As, lambdas, c, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext) = begin
    Heff, Neff = buildHN(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    real(dot(c, Heff * c)) / real(dot(c, Neff * c))
end

# Full Level-B evaluation: energy E, amplitudes c, and per-sector Euclidean gradient
# grad_m = dE/dconj(A_C_m) (packed).  Also returns the metrics for the natural step.
function gradientB(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    M = length(As); Nb = Nsites ÷ 2
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    vecACs = [packAC(As[m], lambdas[m]) for m in 1:M]
    gs = [calculateMetricTensor(As[m], lamInvs[m], Nb) for m in 1:M]   # full metric (cheap)
    numHdiag = [diagNum(As[m], lamInvs[m], gates[m], mpoTerms, gC, Nsites, theta_ext) for m in 1:M]
    Draw = [real(dot(vecACs[m], numHdiag[m])) for m in 1:M]
    nrm = [real(dot(vecACs[m], gs[m] * vecACs[m])) for m in 1:M]
    ovV = zeros(ComplexF64, M, M); e0V = zeros(ComplexF64, M, M)
    ovF = Matrix{Array{ComplexF64,3}}(undef, M, M); e0F = Matrix{Array{ComplexF64,3}}(undef, M, M)
    for m in 1:M, mp in 1:M
        Delta = 2π * (mp - m) / M
        ovV[m, mp], ovF[m, mp] = crossOvGrad(As[m], As[mp], Nb, Nsites, Delta)
        e0V[m, mp], e0F[m, mp] = crossE0Grad(As[m], As[mp], Nb, Nsites, Delta; theta=theta_ext)
    end
    Heff = zeros(ComplexF64, M, M); Neff = Diagonal(ComplexF64.(nrm))
    for m in 1:M
        Heff[m, m] = Draw[m] + gC * Nsites * real(h2[m, m]) * nrm[m]
    end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Heff[m, mp] = gC * (2Nsites * hm[m, mp] * e0V[m, mp] + Nsites * h2[m, mp] * ovV[m, mp])
    end
    Heff = (Heff + Heff') / 2
    E, c = solveC(Heff, Matrix(Neff))
    Den = real(dot(c, Neff * c))

    grads = Vector{Vector{ComplexF64}}(undef, M)
    for m in 1:M
        acc = c[m] .* numHdiag[m]                                   # matter diagonal (mp=m)
        for mp in 1:M
            if mp == m
                acc = acc .+ c[m] .* (gC * Nsites * real(h2[m, m])) .* packGrad(ovF[m, m], lamInvs[m], Nb)
            else
                acc = acc .+ c[mp] .* (gC .* (2Nsites * hm[m, mp] .* packGrad(e0F[m, mp], lamInvs[m], Nb)
                                             .+ Nsites * h2[m, mp] .* packGrad(ovF[m, mp], lamInvs[m], Nb)))
            end
        end
        dNum = conj(c[m]) .* acc
        dDen = abs2(c[m]) .* (gs[m] * vecACs[m])
        grads[m] = (dNum .- E .* dDen) ./ Den
    end
    return E, c, grads, gs, vecACs, lamInvs
end

# ===========================================================================
# SLOW reference implementations (regression oracle for the fast path above)
# ===========================================================================
# value + RAW bra-gradient of tr prod_B mixedTwistedE(Am,Amp,phaseFun(B)) via the
# explicit scalar impurity loop.  Index reading Env4[a,ap,b,bp] is the validated one
# from e0GradAndNum (only the ket tensor A -> Amp changes).
function phasedTraceGrad_ref(Am, Amp, Nb, phaseFun)
    D = size(Am, 1); d = size(Am, 3)
    Es = [mixedTwistedE(Am, Amp, phaseFun(B)) for B in 0:Nb-1]
    idm = Matrix{ComplexF64}(I, D * D, D * D)
    P = Vector{Matrix{ComplexF64}}(undef, Nb + 1); P[1] = idm
    for i in 1:Nb; P[i+1] = Es[i] * P[i]; end
    S = Vector{Matrix{ComplexF64}}(undef, Nb + 2); S[Nb+2] = idm; S[Nb+1] = Es[Nb]
    for i in Nb-1:-1:1; S[i+1] = S[i+2] * Es[i]; end
    val = tr(P[Nb+1])
    F = zeros(ComplexF64, d, D, D)
    for i in 1:Nb
        Env4 = reshape(P[i] * S[i+2], (D, D, D, D))
        ph = phaseFun(i - 1)
        for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
            F[s, ap, bp] += ph[s] * Amp[a, b, s] * Env4[a, ap, b, bp]
        end
    end
    return val, F
end
crossOvGrad_ref(Am, Amp, Nb, Nsites, Delta) = phasedTraceGrad_ref(Am, Amp, Nb, dressPhase(Delta, Nsites))
function crossE0Grad_ref(Am, Amp, Nb, Nsites, Delta; theta=0.0)
    fh = e0CharFhat(Nsites; theta=theta); D = size(Am, 1)
    val = 0.0 + 0.0im; F = zeros(ComplexF64, 4, D, D)
    for k in 0:Nsites-1
        abs(fh[k+1]) < 1e-15 && continue
        v, Fk = phasedTraceGrad_ref(Am, Amp, Nb, e0Phase(k, Delta, Nsites))
        val += fh[k+1] * v; F .+= fh[k+1] .* Fk
    end
    return val, F
end
# slow diagonal numerator (scalar-loop puMPS.jl routines)
function diagNum_ref(A, lamInv, gate, mpoTerms, gCE0, Nsites, theta_ext)
    Nb = Nsites ÷ 2
    numH = calculateGradientTensor(A, gate, lamInv, Nb)
    for (coef, W, vL, vR) in mpoTerms
        numH = numH .+ coef .* calculateGradientTensorMPO(A, W, vL, vR, lamInv, Nb)
    end
    numHE0, _ = e0GradAndNum(A, lamInv, Nsites; theta=theta_ext)
    return numH .+ gCE0 .* numHE0
end
function gradientB_ref(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    M = length(As); Nb = Nsites ÷ 2
    lamInvs = [1.0 ./ max.(l, 1e-12) for l in lambdas]
    vecACs = [packAC(As[m], lambdas[m]) for m in 1:M]
    gs = [calculateMetricTensor(As[m], lamInvs[m], Nb) for m in 1:M]
    numHdiag = [diagNum_ref(As[m], lamInvs[m], gates[m], mpoTerms, gC, Nsites, theta_ext) for m in 1:M]
    Draw = [real(dot(vecACs[m], numHdiag[m])) for m in 1:M]
    nrm = [real(dot(vecACs[m], gs[m] * vecACs[m])) for m in 1:M]
    ovV = zeros(ComplexF64, M, M); e0V = zeros(ComplexF64, M, M)
    ovF = Matrix{Array{ComplexF64,3}}(undef, M, M); e0F = Matrix{Array{ComplexF64,3}}(undef, M, M)
    for m in 1:M, mp in 1:M
        Delta = 2π * (mp - m) / M
        ovV[m, mp], ovF[m, mp] = crossOvGrad_ref(As[m], As[mp], Nb, Nsites, Delta)
        e0V[m, mp], e0F[m, mp] = crossE0Grad_ref(As[m], As[mp], Nb, Nsites, Delta; theta=theta_ext)
    end
    Heff = zeros(ComplexF64, M, M); Neff = Diagonal(ComplexF64.(nrm))
    for m in 1:M; Heff[m, m] = Draw[m] + gC * Nsites * real(h2[m, m]) * nrm[m]; end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Heff[m, mp] = gC * (2Nsites * hm[m, mp] * e0V[m, mp] + Nsites * h2[m, mp] * ovV[m, mp])
    end
    Heff = (Heff + Heff') / 2
    E, c = solveC(Heff, Matrix(Neff)); Den = real(dot(c, Neff * c))
    grads = Vector{Vector{ComplexF64}}(undef, M)
    for m in 1:M
        acc = c[m] .* numHdiag[m]
        for mp in 1:M
            if mp == m
                acc = acc .+ c[m] .* (gC * Nsites * real(h2[m, m])) .* packGrad(ovF[m, m], lamInvs[m], Nb)
            else
                acc = acc .+ c[mp] .* (gC .* (2Nsites * hm[m, mp] .* packGrad(e0F[m, mp], lamInvs[m], Nb)
                                             .+ Nsites * h2[m, mp] .* packGrad(ovF[m, mp], lamInvs[m], Nb)))
            end
        end
        grads[m] = (conj(c[m]) .* acc .- E .* (abs2(c[m]) .* (gs[m] * vecACs[m]))) ./ Den
    end
    return E, c, grads, gs, vecACs, lamInvs
end

# ===========================================================================
# validation
# ===========================================================================
# cross-gradients (fast, default) vs finite differences of the ED-validated values.
function validateCrossGrad(; Nsites=6, D=3, seed=11, eps=1e-6)
    Nb = Nsites ÷ 2
    Random.seed!(seed)
    Am = randn(ComplexF64, D, D, 4); Amp = randn(ComplexF64, D, D, 4)
    Delta = 2π * 1 / 5
    maxerr = 0.0
    for (name, gradf, valf) in (
            ("overlap", (a, b) -> crossOvGrad(a, b, Nb, Nsites, Delta),
                        (a, b) -> crossOverlap(a, b, Nb, Nsites, Delta)),
            ("E0",      (a, b) -> crossE0Grad(a, b, Nb, Nsites, Delta),
                        (a, b) -> crossE0lin(a, b, Nb, Nsites, Delta)))
        val, F = gradf(Am, Amp)
        @assert abs(val - valf(Am, Amp)) < 1e-10 "value mismatch $name"
        err = 0.0
        for s in 1:4, ap in 1:D, bp in 1:D
            Axp = copy(Am); Axp[ap, bp, s] += eps
            Axm = copy(Am); Axm[ap, bp, s] -= eps
            Ayp = copy(Am); Ayp[ap, bp, s] += im * eps
            Aym = copy(Am); Aym[ap, bp, s] -= im * eps
            dvdx = (valf(Axp, Amp) - valf(Axm, Amp)) / (2eps)
            dvdy = (valf(Ayp, Amp) - valf(Aym, Amp)) / (2eps)
            fd = 0.5 * (dvdx + im * dvdy)      # d val / d conj z
            err = max(err, abs(fd - F[s, ap, bp]))
        end
        relerr = err / max(norm(F), 1e-12)
        @printf("  cross-grad %-8s max|analytic - FD| = %.2e  (rel %.2e)\n", name, err, relerr)
        maxerr = max(maxerr, relerr)
    end
    flush(stdout)
    return maxerr
end

# full Level-B gradient (fast, default) vs finite differences of E (fixed optimal c).
function validateBGrad(; Nsites=6, hmax=1, D=3, gC=0.5, lam=1.0, seed=3, eps=1e-6)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    mpoTerms = vcat(kernelTerms(Nsites, gC), [(lam, Wl, vLl, vRl)])
    gates = [blockedXXGateTwisted(2π * (m - 1) / M / Nsites) for m in 1:M]
    hm, h2, _ = holonomyMats(hmax)
    Random.seed!(seed)
    As = [randn(ComplexF64, D, D, 4) for _ in 1:M]
    lambdas = [normalize(abs.(randn(D)) .+ 0.5) for _ in 1:M]
    E, c, grads, gs, vecACs, lamInvs = gradientB(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, 0.0)
    maxrel = 0.0
    for m in 1:M
        gerr = 0.0; gnrm = 0.0
        for s in 1:4, a in 1:D, cc in 1:D
            Axp = deepcopy(As); Axp[m] = copy(As[m]); Axp[m][a, cc, s] += eps
            Axm = deepcopy(As); Axm[m] = copy(As[m]); Axm[m][a, cc, s] -= eps
            Ayp = deepcopy(As); Ayp[m] = copy(As[m]); Ayp[m][a, cc, s] += im * eps
            Aym = deepcopy(As); Aym[m] = copy(As[m]); Aym[m][a, cc, s] -= im * eps
            Ex = (energyAt(Axp, lambdas, c, gates, mpoTerms, gC, Nsites, hm, h2, 0.0) -
                  energyAt(Axm, lambdas, c, gates, mpoTerms, gC, Nsites, hm, h2, 0.0)) / (2eps)
            Ey = (energyAt(Ayp, lambdas, c, gates, mpoTerms, gC, Nsites, hm, h2, 0.0) -
                  energyAt(Aym, lambdas, c, gates, mpoTerms, gC, Nsites, hm, h2, 0.0)) / (2eps)
            fd = 0.5 * (Ex + im * Ey)                          # dE/dconj(A_L[a,c,s])
            # grads are (1/Nb)-scaled (Euler convention); restore Euclidean grad with *Nb.
            an = grads[m][mu_pack(s, a, cc, D)] * lambdas[m][cc] * Nb
            gerr = max(gerr, abs(fd - an)); gnrm = max(gnrm, abs(an))
        end
        rel = gerr / max(gnrm, 1e-12)
        @printf("  sector m=%d  E=%.6f  max|dE analytic - FD| = %.2e  (rel %.2e)\n", m, E, gerr, rel)
        maxrel = max(maxrel, rel)
    end
    flush(stdout)
    return maxrel
end

# fast default path == slow _ref path on random tensors.
function regressFastB(; Nsites=8, hmax=1, D=6, gC=0.5, lam=1.0, seed=5)
    Nb = Nsites ÷ 2; M = 2hmax + 1; theta_ext = 0.0
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    mpoTerms = vcat(kernelTerms(Nsites, gC), [(lam, Wl, vLl, vRl)])
    gates = [blockedXXGateTwisted(2π * (m - 1) / M / Nsites) for m in 1:M]
    hm, h2, _ = holonomyMats(hmax)
    Random.seed!(seed)
    As = [randn(ComplexF64, D, D, 4) for _ in 1:M]
    lambdas = [normalize(abs.(randn(D)) .+ 0.5) for _ in 1:M]
    e1 = norm(fastMixedTransfer(As[1], As[2]; phase=dressPhase(0.7, Nsites)(1)) -
              mixedTwistedE(As[1], As[2], dressPhase(0.7, Nsites)(1)))
    v1, F1 = crossE0Grad(As[1], As[2], Nb, Nsites, 2π/M)
    v0, F0 = crossE0Grad_ref(As[1], As[2], Nb, Nsites, 2π/M)
    e2 = max(abs(v1 - v0), norm(F1 .- F0))
    Ef, _, gf, _, _, _ = gradientB(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    Es, _, gsl, _, _, _ = gradientB_ref(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    e3 = abs(Ef - Es); e4 = maximum(norm(gf[m] .- gsl[m]) for m in 1:M)
    @printf("regressFastB: |mixedT|=%.1e  |crossE0(val,F)|=%.1e  |E|=%.1e  |grads|=%.1e\n", e1, e2, e3, e4)
    return max(e1, e2, e3, e4)
end

# ===========================================================================
# optimizer (default = fast)
# ===========================================================================
# natural-gradient descent direction for one sector: dA_L = -(g^{-1} grad) lambda^{-1}
function naturalStep(g, grad, lamInv, D, Nb)
    dAC = -pseudoSolve(g, grad)
    dAL = zeros(ComplexF64, D, D, 4)
    for s in 1:4, a in 1:D, c in 1:D
        dAL[a, c, s] = dAC[mu_pack(s, a, c, D)] * lamInv[c]
    end
    return dAL
end

# Level-B ground state: fast Level-A warm start (fullGD_step3_fast per sector) + fast
# block coordinate descent (joint natural-gradient A-step + exact c re-solve, monotone
# with backtracking on the total energy).
function levelB_GS(; Nsites=6, hmax=1, D=4, gC=0.5, lam=1.0, theta_ext=0.0,
                    warmSteps=2000, maxit=400, alpha=0.03, seed=4, tol=1e-9, verbose=true)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    mpoTerms = vcat(kernelTerms(Nsites, gC), [(lam, Wl, vLl, vRl)])
    gates = [blockedXXGateTwisted(2π * (m - 1) / M / Nsites) for m in 1:M]
    hm, h2, _ = holonomyMats(hmax)

    # --- warm start: Level-A independent sector optimization (fast, monotone) ---
    As = Vector{Array{ComplexF64,3}}(undef, M); lambdas = Vector{Vector{Float64}}(undef, M)
    for mm in 0:M-1
        Random.seed!(seed + mm); A0 = randn(ComplexF64, D, D, 4)
        A_L, _, _, _, _ = fullGD_step3_fast(warmSteps, alpha, A0, gates[mm+1], mpoTerms, gC, Nsites)
        As[mm+1], lambdas[mm+1] = canonicalization(A_L)
    end
    HeffA, NeffA = buildHN(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    EA, _ = solveC(HeffA, Matrix(NeffA))
    verbose && @printf("    Level-A warm start  E/N = %.10f\n", EA / Nsites)

    # --- Level-B block coordinate descent ---
    E, c, grads, gs, vecACs, lamInvs = gradientB(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
    bestE = E; bestAs = deepcopy(As); bestLams = deepcopy(lambdas)
    al = alpha
    for it in 1:maxit
        dALs = [naturalStep(gs[m], grads[m], lamInvs[m], D, Nb) for m in 1:M]
        eta = sqrt(abs(sum(real(dot(grads[m], pseudoSolve(gs[m], grads[m]))) for m in 1:M)))
        eta < tol && break
        accepted = false
        while al > 1e-9
            trialAs = Vector{Array{ComplexF64,3}}(undef, M); trialLams = Vector{Vector{Float64}}(undef, M)
            for m in 1:M
                trialAs[m], trialLams[m] = canonicalization(As[m] .+ al .* dALs[m])
            end
            Heff, Neff = buildHN(trialAs, trialLams, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
            Etrial, _ = solveC(Heff, Matrix(Neff))
            if Etrial <= E + 1e-12
                As = trialAs; lambdas = trialLams; E = Etrial; accepted = true; break
            end
            al *= 0.5
        end
        accepted || break
        E, c, grads, gs, vecACs, lamInvs = gradientB(As, lambdas, gates, mpoTerms, gC, Nsites, hm, h2, theta_ext)
        if E < bestE; bestE = E; bestAs = deepcopy(As); bestLams = deepcopy(lambdas); end
        al = min(al * 1.2, alpha)
    end
    verbose && @printf("    Level-B converged   E/N = %.10f  (lowered %.2e/site)\n",
                       bestE / Nsites, (EA - bestE) / Nsites)
    return bestE, bestAs, bestLams
end

# Compare Level A vs Level B vs exact ED across hmax.
function runLevelB(N, D, gC; hmax_max=2, lam=1.0, warmSteps=2000, maxit=400, alpha=0.03, seed=4)
    @printf("N=%d  D=%d  gC=%.2f  lam=%.1f : Level A vs Level B vs ED (theta_ext=0)\n", N, D, gC, lam)
    @printf("  hmax   E_A/N          E_B/N          E_ED/N         |B-ED|/N\n")
    for hmax in 0:hmax_max
        EA, _, _ = holonomyGS(; Nsites=N, hmax=hmax, D=D, gC=gC, lam=lam, steps=warmSteps, alpha=alpha, seed=seed, verbose=false)
        EB, _, _ = levelB_GS(; Nsites=N, hmax=hmax, D=D, gC=gC, lam=lam, warmSteps=warmSteps, maxit=maxit, alpha=alpha, seed=seed, verbose=false)
        Eed = holonomyED_neutral(N, gC, hmax; lam=lam)
        @printf("  %d      %.10f  %.10f  %.10f  %.2e\n", hmax, EA/N, EB/N, Eed/N, abs(EB-Eed)/N)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Fast == reference regression:")
    errr = regressFastB()
    println(errr < 1e-10 ? "  FAST-B REGRESSION PASSED" : "  FAST-B REGRESSION FAILED (err=$errr)")
    println("Level-B cross-gradient validation:")
    err = validateCrossGrad()
    println(err < 1e-9 ? "  CROSS-GRAD OK" : "  CROSS-GRAD FAILED")
    println("Level-B full-gradient validation (FD of total energy):")
    errg = validateBGrad()
    println(errg < 1e-5 ? "  FULL-GRAD OK" : "  FULL-GRAD FAILED")
    println()
    # Demo: Level A vs Level B vs ED at N=6,D=4.  Level B closes ~60% of the Level-A
    # gap; hmax=0 is exact for both (no cross terms).
    runLevelB(6, 4, 0.50; hmax_max=1, warmSteps=1500, maxit=600, alpha=0.05)
end
