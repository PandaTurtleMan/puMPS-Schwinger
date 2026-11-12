# Lifting term  lambda * Q^2,  Q = sum_j q_j,  used to penalise (lift) all
# sectors with total charge Q != 0 and thereby project the ground state into
# the Q = 0 subspace.  Q^2 = (sum_j q_j)^2 is represented by Ian's chi=3 MPO
#
#           [ I   q   q^2 ]
#   W_j  =  [ 0   I   2q  ] ,   vL = [1 0 0],  vR = [0 0 1]^T
#           [ 0   0   I   ]
#
# which is the standard "square of a sum" finite-state machine (verified below).
# Because we already have the general MPS-MPO ring transfer + MPO gradient, the
# lifting term is just one more MPO added to the Hamiltonian -- no new machinery.
include("puMPS_coulomb_gs.jl")   # mpoRingEnergy, calculateGradientTensorMPO, fullGD_XXcoulomb, opAt, denseMPOexpand, denseTangentBasisGS
using LinearAlgebra, Random, Printf

szop() = ComplexF64[1 0; 0 -1]

# lifting MPO for Q^2 = (sum_j q_j)^2  (chi = 3)
function liftingMPO(N; qop=szop())
    d = size(qop, 1); id = Matrix{ComplexF64}(I, d, d); q2 = qop * qop
    W = zeros(ComplexF64, 3, 3, d, d)
    W[1, 1, :, :] = id;  W[1, 2, :, :] = qop;   W[1, 3, :, :] = q2
    W[2, 2, :, :] = id;  W[2, 3, :, :] = 2 .* qop
    W[3, 3, :, :] = id
    return W, ComplexF64[1, 0, 0], ComplexF64[0, 0, 1]
end

# charge MPO for Q = sum_j q_j  (chi = 2), for measuring <Q>
function chargeMPO(N; qop=szop())
    d = size(qop, 1); id = Matrix{ComplexF64}(I, d, d)
    W = zeros(ComplexF64, 2, 2, d, d)
    W[1, 1, :, :] = id;  W[1, 2, :, :] = qop;  W[2, 2, :, :] = id
    return W, ComplexF64[1, 0], ComplexF64[0, 1]
end

denseQ(N; qop=szop()) = sum(opAt(qop, j, N) for j in 1:N)
denseQ2(N; qop=szop()) = (Q = denseQ(N; qop=qop); Q * Q)

# --- validate the MPO + its gradient against dense references ---------------
function testLiftingMPO(; N=6, D=3, seed=8)
    Random.seed!(seed); A = randn(ComplexF64, D, D, 2)
    A, lam = canonicalization(A); lamInv = 1.0 ./ max.(lam, 1e-12)
    Wl, vL, vR = liftingMPO(N)
    Hexp = denseMPOexpand(Wl, vL, vR, N)
    psi = ringStateWithImpurity(A, A, 1, N)
    _, _, e = mpoRingEnergy(A, Wl, vL, vR, N)
    numH = calculateGradientTensorMPO(A, Wl, vL, vR, lamInv, N)
    numH_d = denseTangentBasisGS(A, lamInv, N)' * (Hexp * psi)
    println("Lifting MPO check (N=$N, D=$D):")
    println("  ||H_MPO - Q^2||               = ", norm(Hexp - denseQ2(N)))
    println("  |<Q^2>_transfer - <psi|Q^2|psi>| = ", abs(e - (psi' * denseQ2(N) * psi) / (psi' * psi)))
    println("  ||grad - grad_dense||         = ", norm(numH - numH_d))
end

# best-eta-tracking combined optimizer (returns the lowest-gradient state seen,
# so a too-large fixed step on the stiff lambda*Q^2 penalty cannot diverge).
function fullGD_lift_besteta(maxSteps, alpha, A0, xxgate, Wl, vL, vR, lam, N)
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    bestEta = Inf; bestA = copy(A); bestLam = copy(lambda); bestE = 0.0 + 0im
    for i in 1:maxSteps
        g = calculateMetricTensor(A, lamInv, N)
        numH = calculateGradientTensor(A, xxgate, lamInv, N) .+
               lam .* calculateGradientTensorMPO(A, Wl, vL, vR, lamInv, N)
        dAL, eta, en = trueGradient(g, numH, A, lambda, lamInv, N)
        if eta < bestEta; bestEta = eta; bestA = copy(A); bestLam = copy(lambda); bestE = en; end
        (eta < 1e-9 && i > 1) && break
        A = oneGradientDescentStep(A, alpha, dAL); A, lambda = canonicalization(A)
        lamInv = 1.0 ./ max.(lambda, 1e-12)
    end
    return bestA, bestLam, bestE, bestEta
end

# ED ground-state momentum sector (j such that T-eigenvalue = e^{i2pi j/N})
function gsMomentum(Hfull, N; d=2)
    F = eigen(Hermitian(Hfull)); E0 = F.values[1]
    T = zeros(ComplexF64, d^N, d^N)
    for idx in 0:d^N-1
        digs = [(idx ÷ d^(i-1)) % d for i in 1:N]
        T[sum(circshift(digs, 1)[i] * d^(i-1) for i in 1:N)+1, idx+1] = 1
    end
    g = findall(x -> x - E0 < 1e-7, F.values)
    return sort(unique(round.(mod.(angle.(eigvals(F.vectors[:, g]' * T * F.vectors[:, g])) .* N ./ (2π), N), digits=1)))
end

# --- demonstrate the projection: sweep lambda, watch <Q> -> 0 ---------------
# H = H_XX - hz sum sigma^z + lambda (sum sigma^z)^2 .  The field pushes the
# unlifted GS to Q != 0; the lifting term forces it back to Q = 0.
function liftingDemo(; N=8, D=8, hz=0.6, lambdas=(0.0, 0.2, 0.4, 0.6), seed=4, steps=4000)
    sp = ComplexF64[0 1; 0 0]; sm = ComplexF64[0 0; 1 0]; id2 = Matrix{ComplexF64}(I, 2, 2)
    xxfield = kron(sp, sm) + kron(sm, sp) .- (hz/2) .* (kron(szop(), id2) + kron(id2, szop()))
    Wl, vLl, vRl = liftingMPO(N)
    Wq, vLq, vRq = chargeMPO(N)
    Q2d = denseQ2(N); Qd = denseQ(N)
    println("\nH = H_XX - $hz Σσ^z + λ (Σσ^z)^2    (N=$N, D=$D);   q_j = σ^z_j,  so Q=0 is S^z=0\n")
    @printf("    λ      ED E/N    ED<Q>   ED p/pi   puMPS<Q>   puMPS<Q^2>   |E diff|/site\n")
    for lam in lambdas
        Hfull = ringHamFromGate(xxfield, N) .+ lam .* Q2d; Hfull = (Hfull + Hfull') / 2
        Fed = eigen(Hermitian(Hfull)); E0 = Fed.values[1]; v = Fed.vectors[:, 1]
        Qed = real(v' * Qd * v); pmom = 2 .* gsMomentum(Hfull, N) ./ N
        Random.seed!(seed); A0 = randn(ComplexF64, D, D, 2)
        A_L, lamv, en, eta = fullGD_lift_besteta(steps, 0.02, A0, xxfield, Wl, vLl, vRl, lam, N)
        _, _, Qexp  = mpoRingEnergy(A_L, Wq, vLq, vRq, N)
        _, _, Q2exp = mpoRingEnergy(A_L, Wl, vLl, vRl, N)
        @printf("  %5.2f  %8.4f  %7.3f   %-8s  %8.3f    %8.3f     %.1e\n",
                lam, E0/N, Qed, string(pmom), real(Qexp), real(Q2exp), abs(real(en)/N - E0/N))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    testLiftingMPO()
    liftingDemo()
end
