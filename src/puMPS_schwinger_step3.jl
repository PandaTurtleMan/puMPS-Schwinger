# STEP 3: the N*E0^2 term, exactly, via the discrete-Fourier (roots-of-unity)
# decomposition and position-dependent "twisted" transfer matrices:
#
#   N E0^2 = f(e mod N),  f(r) = N (r/N + theta/2pi)^2,   e = -sum_j j q_j  (integer)
#          = sum_{k=0}^{N-1} fhat_k Omega_k,   Omega_k = omega^{k e} = prod_j omega^{-k j q_j}
#
# Each Omega_k is a product of single-BLOCK diagonal phases (blocked: site j=2B+a):
#   sum_j j q_j = sum_B [ 2B*(q0+q1)(B) + q1(B) ]
# so Omega_k = prod_B P_{k,B},  P_{k,B}[s] = omega^{-k(2B*Qb[s] + q1[s])}.
#
# IMPORTANT: Omega_k is NOT block-translation-invariant as an operator (only the
# full sum is, and only in the neutral sector), so the gradient numerator must be
# the impurity-position AVERAGE over the ring, not Nb x (site-1 term).  Prefix/
# suffix products of the twisted transfers make this O(Nb) matmuls per k.
include("puMPS_schwinger_step2.jl")     # guarded: no auto-run
using LinearAlgebra, Random, Printf

# diagonal values on the block basis (s-1 = bit_even + 2*bit_odd)
const q0vals = [1.0, 0.0, 1.0, 0.0]
const q1vals = [0.0, 0.0, -1.0, -1.0]
const Qbvals = q0vals .+ q1vals                       # block total charge

fhatE0(Nsites; theta=0.0) = begin
    ω = exp(2π * im / Nsites)
    [sum(Nsites * (r / Nsites + theta / (2π))^2 * ω^(-k * r) for r in 0:Nsites-1) / Nsites
     for k in 0:Nsites-1]
end

blockPhase(k, B, Nsites) = begin
    ω = exp(2π * im / Nsites)
    [ω^(-k * (2 * B * Qbvals[s] + q1vals[s])) for s in 1:4]
end

function twistedE(A, phase)
    D = size(A, 1); d = size(A, 3)
    E = zeros(ComplexF64, D * D, D * D)
    for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
        E[b+(bp-1)*D, a+(ap-1)*D] += phase[s] * A[a, b, s] * conj(A[ap, bp, s])
    end
    return E
end

# raw gradient numerator (impurity-position-averaged) and raw energy numerator
# of the E0 term; caller divides energy by the norm.
function e0GradAndNum(A, lambdaInverse, Nsites; theta=0.0)
    Nb = Nsites ÷ 2; D = size(A, 1); d = size(A, 3)
    fh = fhatE0(Nsites; theta=theta)
    F = zeros(ComplexF64, d, D, D)
    num = 0.0 + 0.0im
    idm = Matrix{ComplexF64}(I, D * D, D * D)
    for k in 0:Nsites-1
        fk = fh[k+1]
        abs(fk) < 1e-15 && continue
        Ek = [twistedE(A, blockPhase(k, i - 1, Nsites)) for i in 1:Nb]
        P = Vector{Matrix{ComplexF64}}(undef, Nb + 1); P[1] = idm      # P[i+1] = Ek[i]...Ek[1]
        for i in 1:Nb; P[i+1] = Ek[i] * P[i]; end
        S = Vector{Matrix{ComplexF64}}(undef, Nb + 2); S[Nb+2] = idm   # S[i] = Ek[Nb]...Ek[i]
        S[Nb+1] = Ek[Nb]
        for i in Nb-1:-1:1; S[i+1] = S[i+2] * Ek[i]; end               # S[i+1] holds Ek[Nb]..Ek[i]
        num += fk * tr(P[Nb+1])
        for i in 1:Nb                                                   # impurity at block-site i
            Env = P[i] * S[i+2]                # left-of-i <- ... <- right-of-i
            Env4 = reshape(Env, (D, D, D, D))
            ph = blockPhase(k, i - 1, Nsites)
            for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
                F[s, ap, bp] += fk * ph[s] * A[a, b, s] * Env4[a, ap, b, bp]
            end
        end
    end
    F ./= Nb                                                            # position AVERAGE
    numH = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for s in 1:d, a in 1:D, c in 1:D
        numH[mu(s, a, c)] = F[s, a, c] * lambdaInverse[c]
    end
    return numH, num
end

# ---------------------------------------------------------------------------
# dense E0 diagonal on the site space (warmup2_ed.jl conventions), for oracles
function e0DiagSite(N; theta=0.0)
    dimH = 2^N; out = zeros(dimH); q = zeros(N)
    for lin in 0:dimH-1
        x = lin
        for i in 1:N
            s = (x % 2 == 0) ? 1 : -1
            q[i] = 0.5 * (s + 1) - (isodd(i - 1) ? 1.0 : 0.0)
            x ÷= 2
        end
        E0 = mod(-sum((i - 1) * q[i] for i in 1:N) / N, 1.0)
        out[lin+1] = N * (E0 + theta / (2π))^2
    end
    return out
end

function validateE0(; N=8, D=3, seed=13)
    Nb = N ÷ 2
    Random.seed!(seed)
    A, lam = canonicalization(randn(ComplexF64, D, D, 4))
    lamInv = 1.0 ./ max.(lam, 1e-12)
    Od = Diagonal(e0DiagSite(N))
    ψ = ringStateWithImpurity(A, A, 1, Nb)
    Eref = (ψ' * Od * ψ) / (ψ' * ψ)
    numH, num = e0GradAndNum(A, lamInv, N)
    Etm = transferMatrix(A); nrm = tr(Etm^Nb)
    println("E0 oracle validation (N=$N, D=$D):")
    println("  energy:   |twisted - dense| = ", abs(num / nrm - Eref))
    # position-averaged dense tangent numerator
    d = 4; dim = d * D * D
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    numH_d = zeros(ComplexF64, dim)
    Oψ = Od * ψ
    for i in 1:Nb
        basis = zeros(ComplexF64, 2^N, dim)
        for s in 1:d, a in 1:D, c in 1:D
            M = zeros(ComplexF64, D, D, d); M[a, c, s] = lamInv[c]
            basis[:, mu(s, a, c)] = ringStateWithImpurity(A, M, i, Nb)
        end
        numH_d .+= basis' * Oψ
    end
    numH_d ./= Nb
    println("  gradient: ||twisted - dense|| = ", norm(numH - numH_d))
    flush(stdout)
    return max(abs(num / nrm - Eref), norm(numH - numH_d))
end

# ---------------------------------------------------------------------------
# full faithful optimizer: blocked gate + MPO terms + gC * E0 twisted term
function fullGD_step3(maxSteps, alpha, A0, gate, mpoTerms, gCE0, Nsites; tol=1e-9)
    Nb = Nsites ÷ 2
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    bestEta = Inf; bestA = copy(A); bestLam = copy(lambda); bestE = 0.0 + 0im
    for i in 1:maxSteps
        g = calculateMetricTensor(A, lamInv, Nb)
        numH = calculateGradientTensor(A, gate, lamInv, Nb)
        for (coef, W, vL, vR) in mpoTerms
            numH = numH .+ coef .* calculateGradientTensorMPO(A, W, vL, vR, lamInv, Nb)
        end
        numHE0, _ = e0GradAndNum(A, lamInv, Nsites)
        numH = numH .+ gCE0 .* numHE0
        dAL, eta, en = trueGradient(g, numH, A, lambda, lamInv, Nb)
        if eta < bestEta; bestEta = eta; bestA = copy(A); bestLam = copy(lambda); bestE = en; end
        (eta < tol && i > 1) && break
        A = oneGradientDescentStep(A, alpha, dAL); A, lambda = canonicalization(A)
        lamInv = 1.0 ./ max.(lambda, 1e-12)
    end
    return bestA, bestLam, bestE, bestEta
end

function runStep3(N, D, gC, lam; steps=2500, alpha=0.02, seed=4)
    Nb = N ÷ 2
    # faithful column-C ED reference (neutral sector) + lifting sufficiency check
    Hb = ringHamFromGate(xxG(), N) .+ gC .* Diagonal(staggeredKernelDiag(N) .+ e0DiagSite(N))
    Hb = (Hb + Hb') / 2
    idx = zeroChargeIdx(N)
    Eneu = minimum(real(eigvals(Hermitian(Hb[idx, idx]))))
    Efull = minimum(real(eigvals(Hermitian(Hb .+ lam .* denseQtot(N)^2))))
    @printf("N=%d gC=%.2f lam=%.1f : ED neutral (col C) E/N = %.10f   lift-check = %.1e\n",
            N, gC, lam, Eneu / N, abs(Eneu - Efull)); flush(stdout)
    gate = blockedXXGate()
    terms = kernelTerms(N, gC)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    push!(terms, (lam, Wl, vLl, vRl))
    Random.seed!(seed)
    A0 = randn(ComplexF64, D, D, 4)
    A_L, lamv, en, eta = fullGD_step3(steps, alpha, A0, gate, terms, gC, N)
    _, _, Q2 = mpoRingEnergy(A_L, Wl, vLl, vRl, Nb)
    lamInvf = 1.0 ./ max.(lamv, 1e-12)
    _, e0num = e0GradAndNum(A_L, lamInvf, N)
    e0exp = real(e0num / tr(transferMatrix(A_L)^Nb))
    @printf("  puMPS(D=%d): E/N = %.10f   |diff|/N = %.2e   <Q^2> = %.2e   <NE0^2> = %.4f   eta = %.1e\n",
            D, real(en) / N, abs(real(en) - Eneu) / N, real(Q2), e0exp, eta); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    err = validateE0()
    err < 1e-10 || error("E0 oracle validation failed")
    runStep3(8, 6, 0.3, 1.0; steps=2000)
end
