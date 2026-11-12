# Full model: XX kinetic + long-range Coulomb + charge-lifting, all on the ring.
#
#   H = H_XX  +  gC * [ -1/(2N) sum_{j,j'} |j-j'|(N-|j-j'|) Z_j Z_{j'} ]  +  λ (sum_j Z_j)^2
#       \___ gate ___/   \_______________ Coulomb MPO _______________/     \___ lifting MPO ___/
#
# The Coulomb term (coeff < 0 on Z_jZ_{j'}) favours aligned charges and pushes the
# ground state to Q != 0 (polarised).  The lifting term λ Q^2 penalises that and,
# for large enough λ, projects the ground state back into the Q = 0 sector -- which
# also removes the near-rank-1 (λ^{-1}) ill-conditioning we saw at gC=0.3.
#
# Everything is just "gate + a list of MPO terms": the combined MPS-MPO transfer
# gradient sums them.  Validated against ED.
include("puMPS_lifting.jl")     # guarded includes: liftingMPO, chargeMPO, coulombMPO_jordan,
using LinearAlgebra, Random, Printf   # denseCoulomb, denseQ/denseQ2, gsMomentum, fullGD infra

xxG() = (sp = ComplexF64[0 1; 0 0]; sm = ComplexF64[0 0; 1 0]; kron(sp, sm) + kron(sm, sp))

# General best-eta optimizer: a 2-site gate plus any number of MPO terms
# mpoTerms = [(coef, W, vL, vR), ...].  Total gradient numerator is the sum.
function fullGD_general(maxSteps, alpha, A0, gate, mpoTerms, N)
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    bestEta = Inf; bestA = copy(A); bestLam = copy(lambda); bestE = 0.0 + 0im
    for i in 1:maxSteps
        g = calculateMetricTensor(A, lamInv, N)
        numH = calculateGradientTensor(A, gate, lamInv, N)
        for (coef, W, vL, vR) in mpoTerms
            numH = numH .+ coef .* calculateGradientTensorMPO(A, W, vL, vR, lamInv, N)
        end
        dAL, eta, en = trueGradient(g, numH, A, lambda, lamInv, N)
        if eta < bestEta; bestEta = eta; bestA = copy(A); bestLam = copy(lambda); bestE = en; end
        (eta < 1e-9 && i > 1) && break
        A = oneGradientDescentStep(A, alpha, dAL); A, lambda = canonicalization(A)
        lamInv = 1.0 ./ max.(lambda, 1e-12)
    end
    return bestA, bestLam, bestE, bestEta
end

function coulombLiftingDemo(; N=8, D=8, gC=0.3, lambdas=(0.0, 0.2, 0.4, 0.8), seed=4, steps=4000)
    xxg = xxG()
    Wc, vLc, vRc = coulombMPO_jordan(N)     # Coulomb, Q_j = sigma^z
    Wl, vLl, vRl = liftingMPO(N)            # lifting  Q^2, q_j = sigma^z
    Wq, vLq, vRq = chargeMPO(N)             # measure <Q>
    Hc = denseCoulomb(N); Q2d = denseQ2(N); Qd = denseQ(N)
    println("H = H_XX + $gC*Coulomb + λ (Σσ^z)^2    (N=$N, D=$D)\n")
    @printf("    λ      ED E/N    ED<Q>   ED p/pi   puMPS<Q>   puMPS<Q^2>   |E diff|/site\n")
    for lam in lambdas
        Hfull = ringHamFromGate(xxg, N) .+ gC .* Hc .+ lam .* Q2d; Hfull = (Hfull + Hfull') / 2
        Fed = eigen(Hermitian(Hfull)); E0 = Fed.values[1]; v = Fed.vectors[:, 1]
        Qed = real(v' * Qd * v); pmom = 2 .* gsMomentum(Hfull, N) ./ N
        Random.seed!(seed); A0 = randn(ComplexF64, D, D, 2)
        A_L, lamv, en, eta = fullGD_general(steps, 0.02, A0, xxg,
                                            [(gC, Wc, vLc, vRc), (lam, Wl, vLl, vRl)], N)
        _, _, Qexp  = mpoRingEnergy(A_L, Wq, vLq, vRq, N)
        _, _, Q2exp = mpoRingEnergy(A_L, Wl, vLl, vRl, N)
        @printf("  %5.2f  %8.4f  %7.3f   %-8s  %8.3f    %8.3f     %.1e\n",
                lam, E0/N, Qed, string(pmom), real(Qexp), real(Q2exp), abs(real(en)/N - E0/N))
    end
end

# Bloch dispersion (exact dense tangent oracle) for an arbitrary dense H,
# reusing the same optimized ground state.  H includes the lifting term, so the
# effective H(p) = basis' H basis automatically carries lambda*Q^2.
function blochDispersionFull(A_L, lambda, Hfull, N; nev=10)
    D = size(A_L, 1); d = size(A_L, 3); lamInv = 1.0 ./ max.(lambda, 1e-12)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D; dim = d * D * D
    Sn = [zeros(ComplexF64, d^N, dim) for _ in 1:N]
    for s in 1:d, a in 1:D, c in 1:D
        M = zeros(ComplexF64, D, D, d); M[a, c, s] = lamInv[c]
        for n in 1:N; Sn[n][:, mu(s, a, c)] = ringStateWithImpurity(A_L, M, n, N); end
    end
    HSn = [Hfull * Sn[n] for n in 1:N]
    specs = Vector{Vector{Float64}}(undef, N)
    for jm in 0:N-1
        p = 2π * jm / N
        basis = sum(exp(-im * p * n) .* Sn[n] for n in 1:N)
        Hbas  = sum(exp(-im * p * n) .* HSn[n] for n in 1:N)
        specs[jm+1] = tangentSpectrum(basis' * basis, basis' * Hbas; nev=nev)
    end
    return specs
end

# Check the Bloch spectrum of  H_XX + gC*Coulomb + lambda*Q^2  vs ED.  The Q=0
# ground state is lambda-independent (past threshold), so we optimise it once and
# reuse it for several lambda: neutral excitations stay put, charged ones lift.
function coulombLiftingBloch(; N=8, D=8, gC=0.3, lamGS=0.6, lamsDisp=(0.5, 1.0), seed=4, steps=4000)
    xxg = xxG(); Wc, vLc, vRc = coulombMPO_jordan(N); Wl, vLl, vRl = liftingMPO(N)
    Random.seed!(seed); A0 = randn(ComplexF64, D, D, 2)
    A_L, lamv, en, eta = fullGD_general(steps, 0.02, A0, xxg, [(gC, Wc, vLc, vRc), (lamGS, Wl, vLl, vRl)], N)
    _, _, Q2 = mpoRingEnergy(A_L, Wl, vLl, vRl, N)
    @printf("Q=0 ground state:  E/N=%.6f   <Q^2>=%.3e   eta=%.1e\n\n", real(en)/N, real(Q2), eta)
    for lam in lamsDisp
        Hfull = ringHamFromGate(xxg, N) .+ gC .* denseCoulomb(N) .+ lam .* denseQ2(N); Hfull = (Hfull + Hfull') / 2
        edgap = momentumGaps(Hfull, N)
        specs = blochDispersionFull(A_L, lamv, Hfull, N)
        E0b = minimum(specs[1])
        println("λ = $lam :    p/pi     puMPS dE      ED dE       |diff|")
        for j in 0:N-1
            exc = filter(x -> x - E0b > 1e-6, specs[j+1]); bg = isempty(exc) ? NaN : minimum(exc) - E0b
            @printf("            %5.2f   %10.6f  %10.6f    %.1e\n", 2j/N, bg, edgap[j+1], abs(bg - edgap[j+1]))
        end
        println()
    end
end

# Under-complete finite-D Bloch stress test WITH the lifting term.  The lifting
# forces the (otherwise polarised, ill-conditioned) ground state into the
# well-conditioned Q=0 sector; we then watch the Q=0 tower converge with D.
function stressTestLifting(; N=12, Ds=(4, 6, 8), gC=0.3, lam=1.0, seed=4, steps=3000)
    xxg = xxG(); Wc, vLc, vRc = coulombMPO_jordan(N); Wl, vLl, vRl = liftingMPO(N)
    Hfull = ringHamFromGate(xxg, N) .+ gC .* denseCoulomb(N) .+ lam .* denseQ2(N); Hfull = (Hfull + Hfull') / 2
    Fed = eigen(Hermitian(Hfull)); E0 = Fed.values[1]
    Qed = real(Fed.vectors[:, 1]' * denseQ(N) * Fed.vectors[:, 1])
    pm = 2 .* gsMomentum(Hfull, N) ./ N
    edgap = momentumGaps(Hfull, N); sector = round(Int, 2.0^N / N)
    println("Under-complete Bloch stress test WITH lifting:")
    println("  H = H_XX + $gC*Coulomb + $lam*(Σσ^z)^2,   N=$N")
    println("  ED GS/site=$(round(E0/N,digits=6))  <Q>=$(round(Qed,digits=3))  p/pi=$pm  sector~$sector\n")
    @printf("   D   tan_dim   GS err/site   puMPS<Q^2>   max|dE_bloch-dE_ED|   eta\n")
    for D in Ds
        Random.seed!(seed); A0 = randn(ComplexF64, D, D, 2)
        A_L, lamv, en, eta = fullGD_general(steps, 0.02, A0, xxg, [(gC, Wc, vLc, vRc), (lam, Wl, vLl, vRl)], N)
        _, _, Q2exp = mpoRingEnergy(A_L, Wl, vLl, vRl, N)
        specs = blochDispersionFull(A_L, lamv, Hfull, N)
        E0b = minimum(specs[1]); maxerr = 0.0
        for j in 0:N-1
            exc = filter(x -> x - E0b > 1e-6, specs[j+1]); bg = isempty(exc) ? NaN : minimum(exc) - E0b
            isnan(bg) || (maxerr = max(maxerr, abs(bg - edgap[j+1])))
        end
        @printf("  %2d     %4d      %.3e     %.2e      %.3e           %.1e\n",
                D, D^2 + 1, abs(real(en)/N - E0/N), real(Q2exp), maxerr, eta)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__       # only when run directly, not on include
    stressTestLifting()
end
