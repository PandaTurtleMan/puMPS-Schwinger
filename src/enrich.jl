# DENSE proof-of-concept for FIX (a): Bloch/tangent enrichment of the holonomy ansatz.
#
# Level B (holonomy_gsB.jl) fully relaxes one puMPS A_m per holonomy sector but
# variationally SATURATES the momentum-0-per-sector subspace -- the residual gap to
# full ED is ENTIRELY nonzero matter momentum injected by the holonomy-changing
# coupling gC*2N*h_mm' <psi_m|E0 U^dag U'|psi_m'>  (U = exp(i Delta w/N), a
# position-weighted phase; see [[pumps-holonomy]] "GAP DIAGNOSIS").
#
# This script proves the fix by enriching each sector with first-order tangent
# (Bloch) vectors at nonzero matter momentum and projecting the EXACT dense
# spread_dressed holonomy H onto span{ground_m} U {tangents}, then solving the
# small generalized eigenproblem.  It is an N<~10 oracle (it projects the full
# 2^N * M Hamiltonian), NOT the scalable method -- its job is to (1) confirm the
# tangent enrichment reaches full ED and (2) give the per-momentum-shell breakdown
# that the scalable transfer-matrix version must reproduce.
#
# Momentum k labels a MATTER (block) momentum p = 2*pi*k/Nb, k = 0..Nb-1.  A shell
# is {k, Nb-k} (= +-p).  The scalable version truncates to the lowest one/two shells.
include("holonomy_gsB.jl")   # levelB_GS, Hdiag_blocked, e0OperatorDiag, wDiag,
                             # holonomyMats, ringStateWithImpurity, holonomyED_neutral
using LinearAlgebra, Printf

# ---------------------------------------------------------------------------
# Dense full-space (2^Nsites * M) spread_dressed holonomy H, identical to the
# ED-validated Hbig of oracleHolonomy: block-diagonal Hdiag_blocked(theta_m) per
# sector + off-diagonal (and m=m' self-energy) dressed E0-linear / overlap coupling.
function denseHolonomyH(Nsites, gC, lam, hmax; theta_ext=0.0)
    Nb = Nsites ÷ 2; M = 2hmax + 1; n = 2^Nsites
    e0site = e0OperatorDiag(Nsites; theta=theta_ext)      # E0 value per config
    w = wDiag(Nsites)                                     # sum_j j n_j
    hm, h2, _ = holonomyMats(hmax)
    H = zeros(ComplexF64, n * M, n * M)
    for m in 0:M-1
        H[m*n+1:(m+1)*n, m*n+1:(m+1)*n] .+= Hdiag_blocked(Nsites, 2π * m / M, gC, lam)
    end
    for m in 0:M-1, mp in 0:M-1
        Delta = 2π * (mp - m) / M
        ph = exp.(im .* Delta .* w ./ Nsites)             # U_m^dag U_m'
        cd = gC .* (2Nsites .* e0site .* hm[m+1, mp+1] .+ Nsites .* h2[m+1, mp+1]) .* ph
        H[m*n+1:(m+1)*n, mp*n+1:(mp+1)*n] .+= Diagonal(cd)
    end
    return Hermitian((H + H') / 2)
end

# ---------------------------------------------------------------------------
# Per-sector matter states in the 2^Nsites blocked Hilbert space.
sectorGround(A, Nb) = ringStateWithImpurity(A, A, 1, Nb)   # plain puMPS ring state

# Elementary impurity vectors V[x][:, nu] = ringStateWithImpurity(A, E_nu, x, Nb),
# E_nu the unit tensor at nu=(s,a,c).  Independent of momentum, so build once.
function impurityVectors(A, Nb)
    D = size(A, 1); d = size(A, 3); n = d^Nb; dim = d * D * D
    V = [zeros(ComplexF64, n, dim) for _ in 1:Nb]
    for x in 1:Nb, s in 1:d, a in 1:D, c in 1:D
        M = zeros(ComplexF64, D, D, d); M[a, c, s] = 1.0
        V[x][:, mu_pack(s, a, c, D)] = ringStateWithImpurity(A, M, x, Nb)
    end
    return V
end

# momentum-k tangent block (n x dim): sum_x e^{2pi i k (x-1)/Nb} V[x].
function tangentBlock(V, k, Nb)
    out = zero(V[1])
    for x in 1:Nb
        out .+= exp(2π * im * k * (x - 1) / Nb) .* V[x]
    end
    return out
end

# ---------------------------------------------------------------------------
# Enriched variational energy: embed {ground_m} U {tangent_{m,k}} (k in kset)
# block-per-sector into the M*n space, project H, solve the gen-eig (pseudo-inverse
# of the overlap S drops gauge/parallel null modes -- same as blochSpectrum).
function enrichedEnergy(H, As, Vs, Nsites, M, kset; cutoff=1e-10)
    n = 2^Nsites
    cols = Vector{Vector{ComplexF64}}()
    for m in 1:M
        emb(v) = (u = zeros(ComplexF64, n * M); u[(m-1)*n+1:m*n] = v; u)
        push!(cols, emb(sectorGround(As[m], Nsites ÷ 2)))
        for k in kset
            Tk = tangentBlock(Vs[m], k, Nsites ÷ 2)
            for j in 1:size(Tk, 2); push!(cols, emb(Tk[:, j])); end
        end
    end
    B = reduce(hcat, cols)                       # (M n) x r
    S = Hermitian(B' * B); Hp = Hermitian(B' * (H * B))
    F = eigen(S); thresh = cutoff * maximum(F.values)
    keep = findall(>(thresh), F.values)
    U = F.vectors[:, keep]
    Sh = U * Diagonal(1.0 ./ sqrt.(F.values[keep])) * U'
    He = Hermitian((Sh * Hp * Sh + (Sh * Hp * Sh)') / 2)
    return minimum(real(eigvals(He))), length(keep)
end

# ---------------------------------------------------------------------------
# End-to-end: optimize Level B, build dense H, then report the per-shell gap closure.
function runEnrich(Nsites, D, gC, hmax; lam=1.0, warmSteps=1500, maxit=600, alpha=0.05, seed=4)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    @printf("=== enrichment PoC  N=%d D=%d gC=%.2f lam=%.1f hmax=%d ===\n", Nsites, D, gC, lam, hmax)

    EB, As, _ = levelB_GS(; Nsites=Nsites, hmax=hmax, D=D, gC=gC, lam=lam,
                           warmSteps=warmSteps, maxit=maxit, alpha=alpha, seed=seed, verbose=false)
    @printf("Level B                     E/N = %.10f\n", EB / Nsites)

    H = denseHolonomyH(Nsites, gC, lam, hmax)
    Efull = minimum(real(eigvals(H)))
    @printf("full ED (dense H min)       E/N = %.10f\n", Efull / Nsites)

    Vs = [impurityVectors(As[m], Nb) for m in 1:M]

    Eg, rg = enrichedEnergy(H, As, Vs, Nsites, M, Int[])
    @printf("ground-only projection      E/N = %.10f   (rank %d; == Level B check %.1e)\n",
            Eg / Nsites, rg, abs(Eg - EB) / Nsites)

    gap0 = Eg - Efull
    @printf("gap to full ED              = %.3e/site\n\n", gap0 / Nsites)
    @printf("  shell (|k|<=)   momenta added         E/N            gap closed\n")
    for shell in 0:(Nb ÷ 2)
        kset = [k for k in 0:Nb-1 if min(k, Nb - k) <= shell]
        E, r = enrichedEnergy(H, As, Vs, Nsites, M, kset)
        closed = gap0 > 1e-14 ? (Eg - E) / gap0 * 100 : 0.0
        ps = join([@sprintf("%.2fπ", 2k/Nb) for k in kset], ",")
        @printf("  %-2d              %-22s %.10f   %6.2f%%  (rank %d)\n", shell, ps, E / Nsites, closed, r)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    runEnrich(6, 4, 0.50, 1)
    println()
    runEnrich(8, 4, 0.50, 1; warmSteps=2000, maxit=400)
end
