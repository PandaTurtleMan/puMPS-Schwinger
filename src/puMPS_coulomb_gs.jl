# Ground state + Bloch spectrum for a Hamiltonian that includes the long-range
# Coulomb MPO, validated against exact diagonalisation.
#
# Test model (small N):  H = H_XX + gC * V_Coulomb,
#   H_XX     = sum_j (s+_j s-_{j+1} + s-_j s+_{j+1})        [gate, all N ring bonds]
#   V_Coulomb= -1/(2N) sum_{j,j'} |j-j'|(N-|j-j'|) Z_j Z_{j'}  [Coulomb MPO]
# XX stays a 2-site gate (its gradient already exists, incl. the wrap bond);
# the Coulomb piece uses the new MPO gradient.  The two gradient numerators add.
include("puMPS_coulomb.jl")
using LinearAlgebra, Random, Printf

# ---------------------------------------------------------------------------
# MPO gradient numerator  numH[mu(s,a,c)] = <d_nu psi| H_MPO |psi>  (open MPO).
# Same combined transfer as mpoRingEnergy, but with the BRA tensor removed at
# site 1 (leaving the open index nu = (s,a,c)) and lambda^{-1} on the right leg.
# ---------------------------------------------------------------------------
function calculateGradientTensorMPO(A, W, vL, vR, lambdaInverse, N)
    D = size(A, 1); d = size(A, 3); χ = size(W, 1)
    dim = D * χ * D
    cidx(k, m, b) = k + (m - 1) * D + (b - 1) * D * χ
    T = zeros(ComplexF64, dim, dim)
    for st in 1:d, sb in 1:d, ml in 1:χ, mr in 1:χ
        w = W[ml, mr, st, sb]; w == 0 && continue
        for ak in 1:D, bk in 1:D, ab in 1:D, bb in 1:D
            T[cidx(bk, mr, bb), cidx(ak, ml, ab)] += A[ak, bk, st] * w * conj(A[ab, bb, sb])
        end
    end
    Env = T^(N - 1)                                  # sites 2..N (full combined transfer)
    F = zeros(ComplexF64, d, D, D)                   # F[sb, aLeft, aRight] of the open bra site
    for sb in 1:d, st in 1:d, m0 in 1:χ
        vL[m0] == 0 && continue
        for m1 in 1:χ
            w1 = W[m0, m1, st, sb]; w1 == 0 && continue
            for ak in 1:D, bk in 1:D
                a1 = A[ak, bk, st]; a1 == 0 && continue
                pref = vL[m0] * w1 * a1
                for aL in 1:D, aR in 1:D
                    acc = ComplexF64(0)
                    for mN in 1:χ
                        vR[mN] == 0 && continue
                        acc += vR[mN] * Env[cidx(ak, mN, aL), cidx(bk, m1, aR)]
                    end
                    F[sb, aL, aR] += pref * acc
                end
            end
        end
    end
    numH = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for sb in 1:d, aL in 1:D, aR in 1:D
        numH[mu(sb, aL, aR)] = F[sb, aL, aR] * lambdaInverse[aR]
    end
    return numH
end

# dense tangent basis for the GS gradient (impurity at site 1 only, no momentum sum)
function denseTangentBasisGS(A_L, lambdaInv, N)
    D = size(A_L, 1); d = size(A_L, 3)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    basis = zeros(ComplexF64, d^N, d * D * D)
    for s in 1:d, a in 1:D, c in 1:D
        M = zeros(ComplexF64, D, D, d); M[a, c, s] = lambdaInv[c]
        basis[:, mu(s, a, c)] = ringStateWithImpurity(A_L, M, 1, N)
    end
    return basis
end

# (A) validate the MPO gradient tensor against the dense tangent, random A.
function testGradMPO(; N=6, D=3, seed=11)
    Random.seed!(seed); A = randn(ComplexF64, D, D, 2)
    A, lam = canonicalization(A); lamInv = 1.0 ./ max.(lam, 1e-12)
    W, vL, vR = coulombMPO_jordan(N)
    numH = calculateGradientTensorMPO(A, W, vL, vR, lamInv, N)
    Hopen = denseMPOexpand(W, vL, vR, N)
    psi = ringStateWithImpurity(A, A, 1, N)
    numH_dense = denseTangentBasisGS(A, lamInv, N)' * (Hopen * psi)
    println("MPO gradient vs dense tangent (N=$N,D=$D):  ||numH - numH_dense|| = ",
            norm(numH - numH_dense))
end

# combined XX(gate) + gC*Coulomb(MPO) gradient descent (reuses metric + trueGradient)
function fullGD_XXcoulomb(maxSteps, alpha, tol, A0, xxgate, Wc, vLc, vRc, gC, N; verbose=false, every=25)
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    local energy; eta = Inf
    for i in 1:maxSteps
        g = calculateMetricTensor(A, lamInv, N)
        numH = calculateGradientTensor(A, xxgate, lamInv, N) .+
               gC .* calculateGradientTensorMPO(A, Wc, vLc, vRc, lamInv, N)
        dAL, eta, energy = trueGradient(g, numH, A, lambda, lamInv, N)
        verbose && (i == 1 || i % every == 0) &&
            (println("  step ", lpad(i, 4), "  E/N=", round(real(energy)/N, digits=9),
                     "  eta=", round(eta, sigdigits=4)); flush(stdout))
        (eta < tol && i > 1) && break
        A = oneGradientDescentStep(A, alpha, dAL); A, lambda = canonicalization(A)
        lamInv = 1.0 ./ max.(lambda, 1e-12)
    end
    return A, lambda, energy, eta
end

# tangent-space spectrum from oracle N(p),H(p) (same solve as blochSpectrum)
function tangentSpectrum(Nmat, Hmat; nev=10, cutoff=1e-10)
    Nmat = (Nmat + Nmat') / 2; Hmat = (Hmat + Hmat') / 2
    Fn = eigen(Hermitian(Nmat)); th = cutoff * maximum(Fn.values)
    keep = findall(>(th), Fn.values); U = Fn.vectors[:, keep]
    Nhalf = U * Diagonal(1.0 ./ sqrt.(Fn.values[keep])) * U'
    Heff = Nhalf * Hmat * Nhalf; Heff = (Heff + Heff') / 2
    return sort(real(eigvals(Heff)))[1:min(nev, length(keep))]
end

function validateCoulombGS(; N=8, D=8, gC=0.3, seed=4, steps=2500, alpha=0.03)
    sp = ComplexF64[0 1; 0 0]; sm = ComplexF64[0 0; 1 0]
    xxgate = kron(sp, sm) + kron(sm, sp)
    Wc, vLc, vRc = coulombMPO_jordan(N)

    Hfull = ringHamFromGate(xxgate, N) .+ gC .* denseCoulomb(N)
    Hfull = (Hfull + Hfull') / 2
    Fed = eigen(Hermitian(Hfull)); E0 = Fed.values[1]

    # ED ground-state momentum sector
    d = 2; Top = zeros(ComplexF64, d^N, d^N)
    for idx in 0:d^N-1
        digs = [(idx ÷ d^(i-1)) % d for i in 1:N]
        Top[sum(circshift(digs, 1)[i] * d^(i-1) for i in 1:N)+1, idx+1] = 1
    end
    gset = findall(x -> x - E0 < 1e-8, Fed.values)
    gmom = sort(unique(round.(mod.(angle.(eigvals(Fed.vectors[:, gset]' * Top * Fed.vectors[:, gset])) .* N ./ (2π), N), digits=2)))
    println("XX + $gC*Coulomb  (N=$N):  ED GS/site = ", round(E0/N, digits=8),
            "   GS momentum sector j = ", gmom)

    Random.seed!(seed); A0 = randn(ComplexF64, D, D, 2)
    A_L, lam, en, eta = fullGD_XXcoulomb(steps, alpha, 1e-9, A0, xxgate, Wc, vLc, vRc, gC, N; verbose=true)
    println("GROUND STATE:  puMPS/site = ", round(real(en)/N, digits=8),
            "   |diff/site| = ", abs(real(en)/N - E0/N), "   eta = ", round(eta, sigdigits=3))

    lamInv = 1.0 ./ max.(lam, 1e-12)
    edgap = momentumGaps(Hfull, N)
    Nm0, Hm0 = oracleEffectiveMatrices(A_L, lamInv, 0.0, Hfull, N)
    E0b = minimum(tangentSpectrum(Nm0, Hm0))
    println("\n   p/pi     puMPS dE      ED dE       |diff|")
    for j in 0:N-1
        Nm, Hm = oracleEffectiveMatrices(A_L, lamInv, 2π * j / N, Hfull, N)
        sp_ = tangentSpectrum(Nm, Hm)
        exc = filter(x -> x - E0b > 1e-6, sp_)
        bgap = isempty(exc) ? NaN : minimum(exc) - E0b
        @printf("  %5.2f  %11.6f  %11.6f   %8.2e\n", 2j/N, bgap, edgap[j+1], abs(bgap - edgap[j+1]))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__       # only when run directly, not on include
    testGradMPO()
    validateCoulombGS()
end
