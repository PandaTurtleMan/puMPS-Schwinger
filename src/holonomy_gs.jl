# LEVEL A: post-hoc holonomy diagonalization for the full Schwinger model.
#
# Restores the holonomy variable that step 3 fixed to the zero-holonomy sector.
# Ansatz:  |Psi> = sum_m c_m |psi(A_m)> (x) |theta_m>,  theta_m = 2pi m/M, M=2hmax+1.
#
# Level A = optimize each sector's tensor A_m INDEPENDENTLY in its own twisted
# diagonal Hamiltonian H_diag(theta_m) (existing GD, only the XX gate is twisted
# by theta_m/N -- E0^2, kernel, lifting are theta-independent at theta_ext=0),
# THEN solve the small M x M generalized eigenproblem
#         H_eff c = E N_eff c
# for the sector amplitudes c_m, using the ED-validated cross-sector transfer
# contractions (crossOverlap, crossE0lin, holonomyMats from the oracle).
#
# This is variational over {c_m} for FIXED {A_m}: the A_m minimize their OWN
# diagonal energy, they do NOT relax to feel the m<->m' coupling.  So E_holo is
# an UPPER bound that may be loose -- the whole point is to compare against the
# exact holonomyED_neutral and see how much the frozen-A_m approximation costs.
# hmax=0 (M=1, no coupling) must reproduce the step-3 / column-C number exactly.
include("holonomy_oracle.jl")   # -> holonomy_ed.jl (holonomyMats, holonomyED_neutral),
                                #    puMPS_schwinger_step3.jl (e0GradAndNum, ...),
                                #    blockedXXGateTwisted, crossOverlap, crossE0lin,
                                #    kernelTerms, liftingMPO, Zblk, GD machinery
using LinearAlgebra, Random, Printf

# GD for one holonomy sector.  Same faithful Hamiltonian as fullGD_step3 (twisted
# gate + kernel/lifting MPOs + gC*N E0^2 at external theta), but with MONOTONE
# ENERGY tracking + backtracking instead of best-eta.  best-eta is a TRAP here: the
# fully-polarised product state is an exact eigenstate (eta->0) with energy ~lam*N,
# and eta-tracking would select it.  Track energy (variational => a maximum can
# never win) and accept a step only if it lowers the energy, else halve alpha.
function optimizeSector(theta_ext, maxSteps, alpha0, A0, gate, mpoTerms, gCE0, Nsites; tol=1e-9)
    Nb = Nsites ÷ 2
    evalEG(A, lambda, lamInv) = begin
        g = calculateMetricTensor(A, lamInv, Nb)
        numH = calculateGradientTensor(A, gate, lamInv, Nb)
        for (coef, W, vL, vR) in mpoTerms
            numH = numH .+ coef .* calculateGradientTensorMPO(A, W, vL, vR, lamInv, Nb)
        end
        numHE0, _ = e0GradAndNum(A, lamInv, Nsites; theta=theta_ext)
        numH = numH .+ gCE0 .* numHE0
        trueGradient(g, numH, A, lambda, lamInv, Nb)   # -> dAL, eta, en
    end
    A, lambda = canonicalization(A0); lamInv = 1.0 ./ max.(lambda, 1e-12)
    dAL, eta, en = evalEG(A, lambda, lamInv); en = real(en)
    bestA = copy(A); bestE = en; bestEta = eta; alpha = alpha0
    for i in 1:maxSteps
        eta < tol && break
        accepted = false
        while alpha > 1e-8
            At = oneGradientDescentStep(A, alpha, dAL); At, lamt = canonicalization(At)
            lamInvt = 1.0 ./ max.(lamt, 1e-12)
            dALt, etat, ent = evalEG(At, lamt, lamInvt); ent = real(ent)
            if ent <= en + 1e-12
                A, lambda, lamInv, dAL, eta, en = At, lamt, lamInvt, dALt, etat, ent
                accepted = true; break
            end
            alpha *= 0.5
        end
        accepted || break
        if en < bestE; bestE = en; bestA = copy(A); bestEta = eta; end
        alpha = min(alpha * 1.2, alpha0)
    end
    return bestA, bestE, bestEta
end

# Full Level-A ground state at a given hmax.  Returns (E_total, {A_m}, E_diag/sector).
function holonomyGS(; Nsites=8, hmax=1, D=6, gC=0.3, lam=1.0,
                     steps=2000, alpha=0.02, seed=4, theta_ext=0.0, verbose=true)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    termsLift = vcat(kernelTerms(Nsites, gC), [(lam, Wl, vLl, vRl)])
    hm, h2, _ = holonomyMats(hmax)

    As = Vector{Array{ComplexF64,3}}(undef, M)
    Ediag = zeros(M)
    for mm in 0:M-1
        theta_m = 2π * mm / M
        gate_m = blockedXXGateTwisted(theta_m / Nsites)
        Random.seed!(seed + mm)
        A0 = randn(ComplexF64, D, D, 4)
        A_L, en, eta = optimizeSector(theta_ext, steps, alpha, A0, gate_m, termsLift, gC, Nsites)
        As[mm+1] = A_L; Ediag[mm+1] = en
        verbose && @printf("    sector m=%d  theta=%.4f  E_diag/N=%.10f  eta=%.1e\n",
                           mm, theta_m, en / Nsites, eta)
        flush(stdout)
    end

    # M x M effective matrices, all in the RAW (no-lambda) tensor normalization
    # used by crossOverlap/crossE0lin, so H_eff[m,m]/N_eff[m,m] == Ediag[m].
    Heff = zeros(ComplexF64, M, M); Neff = zeros(ComplexF64, M, M)
    for m in 1:M
        nrm = crossOverlap(As[m], As[m], Nb, Nsites, 0.0)   # <psi_m|psi_m>
        Neff[m, m] = nrm
        # diagonal = optimized matter/gauge energy + the holonomy's OWN electric
        # energy gC*N*h2[m,m] = gC*N*hmax(hmax+1)/3 (the m=m' piece of the Eq-72
        # coupling; a per-sector constant, zero at hmax=0).  hm[m,m]=0 so there is
        # no E0-linear diagonal piece.
        Heff[m, m] = Ediag[m] * nrm + gC * Nsites * real(h2[m, m]) * nrm
    end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Delta = 2π * (mp - m) / M
        e0T = crossE0lin(As[m], As[mp], Nb, Nsites, Delta; theta=theta_ext)
        ovT = crossOverlap(As[m], As[mp], Nb, Nsites, Delta)
        Heff[m, mp] = gC * (2Nsites * hm[m, mp] * e0T + Nsites * h2[m, mp] * ovT)
    end
    Heff = (Heff + Heff') / 2

    evals = eigvals(Hermitian(Heff), Hermitian(Neff))   # generalized, Neff diag PD
    return minimum(real(evals)), As, Ediag
end

# Compare Level-A puMPS against exact holonomy ED across hmax.
function runHolonomy(N, D, gC; hmax_max=3, lam=1.0, steps=2000, alpha=0.02, seed=4)
    @printf("N=%d  D=%d  gC=%.2f  lam=%.1f : Level-A post-hoc holonomy vs ED (theta_ext=0)\n",
            N, D, gC, lam)
    @printf("  hmax   E_puMPS/N        E_ED/N           |diff|/N\n")
    for hmax in 0:hmax_max
        Eholo, _, _ = holonomyGS(; Nsites=N, hmax=hmax, D=D, gC=gC, lam=lam,
                                  steps=steps, alpha=alpha, seed=seed, verbose=false)
        Eed = holonomyED_neutral(N, gC, hmax; lam=lam)   # total energy (concentrated gauge)
        @printf("  %d      %.10f   %.10f   %.2e\n",
                hmax, Eholo / N, Eed / N, abs(Eholo - Eed) / N)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Fast validated demo (N=6, D=4).  hmax=0 reproduces the zero-holonomy /
    # column-C number exactly; hmax>=1 is variational-above-ED, with the residual
    # gap = the frozen-A_m limitation of Level A (only Level B can close it).
    runHolonomy(6, 4, 0.50; hmax_max=2, steps=2500, alpha=0.05)
    # N=8, D=6 is physical but slow (~min/sector): runHolonomy(8, 6, 0.30; steps=2000)
end
