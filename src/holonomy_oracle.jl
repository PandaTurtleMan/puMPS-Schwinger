# ORACLE for the holonomy puMPS ansatz.  Validates the transfer-matrix
# contractions that couple DIFFERENT holonomy sectors against brute-force dense
# states, for random (non-neutral) blocked tensors A_m.
#
# Ansatz:  |Psi> = sum_m c_m |psi(A_m)> (x) |theta_m>,   theta_m = 2pi m/M, M=2hmax+1.
# Gauge:   SPREAD hopping (twist theta_m/N per site bond, Eq 56) is puMPS-friendly
#          (translation invariant) but the m<->m' couplings must then carry the
#          gauge dressing  U_m^dag U_m' = exp( i (theta_m'-theta_m)/N * sum_j j n_j ).
#          (spread + dressing == concentrated boundary twist, confirmed in holonomy_ed.jl.)
#
# <Psi|H|Psi> = sum_m |c_m|^2 <psi(A_m)|H_diag(theta_m)|psi(A_m)>
#             + gC sum_{m!=m'} conj(c_m) c_m' [ 2N h_{mm'} <psi_m|E0 U^dag U'|psi_m'>
#                                              +  N h2_{mm'} <psi_m|   U^dag U'|psi_m'> ]
# The two cross matrix elements are ordered products of MIXED twisted transfers.
include("puMPS_schwinger_step3.jl")   # blockPhase, fhatE0, twistedE, ringHamFromGate, ...
include("holonomy_ed.jl")             # holonomyMats, e0OperatorDiag, wDiag, HxxSector
using LinearAlgebra, Random, Printf

# blocked occupation values  n_even, n_odd  on s-1 = bit_even + 2 bit_odd (bit0=up)
const nEvals = [1.0, 0.0, 1.0, 0.0]                 # = q0vals
const nOvals = [1.0, 1.0, 0.0, 0.0]                 # = q1vals .+ 1
const Nsvals = nEvals .+ nOvals                     # block occupation
const e0vals = [mod(0.0, 1.0)]                      # (placeholder; e0 uses site diag)

# STAGGERED MASS.  H_m = mass * sum_j (-1)^j n_j = (mass/2) * sum_j (-1)^j sigma^z_j
# (the two agree for even N, since sum_j (-1)^j = 0).  Site j=2B is the "even" site of
# block B and j=2B+1 the "odd" one, so per block the term is mass*(n_even - n_odd) --
# a single-block diagonal, hence strictly block-translation invariant and neutral-sector
# preserving.  Nothing about the holonomy/electric sector is touched.
const massBlkVals = nEvals .- nOvals                # = [0, -1, 1, 0]

# same operator in the SITE basis (bit i-1 of lin, bit 0 = occupied), 0-based j = i-1
function massDiagSite(N)
    out = zeros(2^N)
    for lin in 0:2^N-1
        x = lin; v = 0.0
        for i in 1:N
            v += (isodd(i - 1) ? -1.0 : 1.0) * ((x % 2 == 0) ? 1.0 : 0.0)
            x ÷= 2
        end
        out[lin+1] = v
    end
    return out
end

# twisted blocked XX gate: spread phase e^{i phi} on every forward hop.  The staggered
# mass rides along as an on-block diagonal split 1/2-1/2 onto the two bonds touching a
# block (same trick as h_intra), so every consumer of the gate -- gradient, Level A/B,
# and the H0k tangent blocks -- picks the mass up with no further plumbing.
function blockedXXGateTwisted(phi; mass=0.0)
    z = exp(im * phi)
    h_intra = blkop(sm2, sp2) .* z .+ blkop(sp2, sm2) .* conj(z)      # even(left)->odd(right)
    SpOdd = blkop(sp2, I2b); SmOdd = blkop(sm2, I2b)
    SpEven = blkop(I2b, sp2); SmEven = blkop(I2b, sm2)
    h_inter = kron(SpOdd, SmEven) .* z .+ kron(SmOdd, SpEven) .* conj(z)  # odd(left)->even(right)
    honblk = h_intra .+ mass .* ComplexF64.(Diagonal(massBlkVals))
    return h_inter .+ 0.5 .* (kron(honblk, I4b) .+ kron(I4b, honblk))
end

# mixed twisted transfer:  ket = Amp, bra = Am,  per-block diagonal phase[s]
function mixedTwistedE(Am, Amp, phase)
    D = size(Am, 1); d = size(Am, 3)
    E = zeros(ComplexF64, D * D, D * D)
    for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
        E[b+(bp-1)*D, a+(ap-1)*D] += phase[s] * Amp[a, b, s] * conj(Am[ap, bp, s])
    end
    return E
end

# tr( prod_{B=Nb-1..0} E^{(B)} ) with per-block phase phaseFun(B)::Vector (length 4)
function tracePhasedProduct(Am, Amp, Nb, phaseFun)
    D = size(Am, 1)
    T = Matrix{ComplexF64}(I, D * D, D * D)
    for B in 0:Nb-1
        T = mixedTwistedE(Am, Amp, phaseFun(B)) * T
    end
    return tr(T)
end

# <psi_m| U^dag U' |psi_m'>   (dressing only), Delta = theta_m' - theta_m
function crossOverlap(Am, Amp, Nb, Nsites, Delta)
    dress(B) = ComplexF64[exp(im * Delta * (2B * Nsvals[s] + nOvals[s]) / Nsites) for s in 1:4]
    return tracePhasedProduct(Am, Amp, Nb, dress)
end

# <psi_m| E0 U^dag U' |psi_m'>  =  sum_k fhat_k tr prod_B phi_{k,B}
function crossE0(Am, Amp, Nb, Nsites, Delta; theta=0.0)
    fh = fhatE0(Nsites; theta=theta)
    tot = 0.0 + 0.0im
    for k in 0:Nsites-1
        abs(fh[k+1]) < 1e-15 && continue
        phi(B) = ComplexF64[exp(-2π * im * k * (2B * Qbvals[s] + q1vals[s]) / Nsites +
                                 im * Delta * (2B * Nsvals[s] + nOvals[s]) / Nsites) for s in 1:4]
        tot += fh[k+1] * tracePhasedProduct(Am, Amp, Nb, phi)
    end
    return tot
end
# NOTE: fhatE0 realizes N*E0^2 = sum_k fhat_k Omega_k; the E0-linear cross term needs
# a SEPARATE character sum for E0 itself.  Build it directly instead:
function e0CharFhat(Nsites; theta=0.0)
    ω = exp(2π * im / Nsites)
    [sum((r / Nsites + theta / (2π)) * ω^(-k * r) for r in 0:Nsites-1) / Nsites for k in 0:Nsites-1]
end
function crossE0lin(Am, Amp, Nb, Nsites, Delta; theta=0.0)
    fh = e0CharFhat(Nsites; theta=theta)
    tot = 0.0 + 0.0im
    for k in 0:Nsites-1
        abs(fh[k+1]) < 1e-15 && continue
        phi(B) = ComplexF64[exp(-2π * im * k * (2B * Qbvals[s] + q1vals[s]) / Nsites +
                                 im * Delta * (2B * Nsvals[s] + nOvals[s]) / Nsites) for s in 1:4]
        tot += fh[k+1] * tracePhasedProduct(Am, Amp, Nb, phi)
    end
    return tot
end

# dense per-sector diagonal Hamiltonian on the 4^Nb == 2^N blocked space
function Hdiag_blocked(Nsites, theta_m, gC, lam)
    Nb = Nsites ÷ 2
    H = ringHamFromGate(blockedXXGateTwisted(theta_m / Nsites), Nb)
    H .+= Diagonal(gC .* (staggeredKernelDiag(Nsites) .+ e0DiagSite(Nsites)) .+
                   lam .* diag(denseQtot(Nsites)^2))
    return (H + H') / 2
end

# =====================================================================
function oracleHolonomy(; Nsites=6, hmax=1, D=3, gC=0.4, lam=1.0, seed=7)
    Nb = Nsites ÷ 2; M = 2hmax + 1
    Random.seed!(seed)
    As = [randn(ComplexF64, D, D, 4) for _ in 1:M]
    cs = randn(ComplexF64, M)
    psis = [ringStateWithImpurity(A, A, 1, Nb) for A in As]      # dense 2^N

    # --- primitive checks against dense ---
    e0site = e0OperatorDiag(Nsites)                              # E0 value per config (theta=0)
    w = wDiag(Nsites)
    maxprim = 0.0
    for m in 1:M, mp in 1:M
        Delta = 2π * (mp - m) / M
        ph = exp.(im .* Delta .* w ./ Nsites)
        ovD = dot(psis[m], ph .* psis[mp])                      # <psi_m|U^dag U'|psi_m'>
        e0D = dot(psis[m], (e0site .* ph) .* psis[mp])          # <psi_m|E0 U^dag U'|psi_m'>
        ovT = crossOverlap(As[m], As[mp], Nb, Nsites, Delta)
        e0T = crossE0lin(As[m], As[mp], Nb, Nsites, Delta)
        maxprim = max(maxprim, abs(ovD - ovT), abs(e0D - e0T))
    end

    # --- full energy: dense vs transfer formula ---
    hm, h2, _ = holonomyMats(hmax)
    # dense H on M*2^N
    n = length(psis[1]); Hdim = M * n
    Psi = zeros(ComplexF64, Hdim)
    for m in 1:M; Psi[(m-1)*n+1:m*n] = cs[m] .* psis[m]; end
    Hbig = zeros(ComplexF64, Hdim, Hdim)
    for m in 1:M
        Hbig[(m-1)*n+1:m*n, (m-1)*n+1:m*n] .+= Hdiag_blocked(Nsites, 2π * (m - 1) / M, gC, lam)
    end
    for m in 1:M, mp in 1:M
        Delta = 2π * ((mp - 1) - (m - 1)) / M
        ph = exp.(im .* Delta .* w ./ Nsites)
        cd = gC .* (2Nsites .* e0site .* hm[m, mp] .+ Nsites .* h2[m, mp]) .* ph
        Hbig[(m-1)*n+1:m*n, (mp-1)*n+1:mp*n] .+= Diagonal(cd)
    end
    Hbig = (Hbig + Hbig') / 2
    Edense = real(dot(Psi, Hbig * Psi) / dot(Psi, Psi))

    # transfer formula
    num = 0.0 + 0.0im; den = 0.0 + 0.0im
    for m in 1:M
        den += abs2(cs[m]) * real(dot(psis[m], psis[m]))
        num += abs2(cs[m]) * dot(psis[m], Hdiag_blocked(Nsites, 2π * (m - 1) / M, gC, lam) * psis[m])
    end
    for m in 1:M, mp in 1:M
        m == mp && continue
        Delta = 2π * ((mp - 1) - (m - 1)) / M
        e0T = crossE0lin(As[m], As[mp], Nb, Nsites, Delta)
        ovT = crossOverlap(As[m], As[mp], Nb, Nsites, Delta)
        num += conj(cs[m]) * cs[mp] * gC * (2Nsites * hm[m, mp] * e0T + Nsites * h2[m, mp] * ovT)
    end
    Eformula = real(num / den)

    @printf("holonomy oracle  N=%d hmax=%d D=%d gC=%.2f:\n", Nsites, hmax, D, gC)
    @printf("  cross-term primitives  max|transfer - dense| = %.2e\n", maxprim)
    @printf("  full energy  |E_formula - E_dense|          = %.2e   (E=%.6f)\n",
            abs(Eformula - Edense), Edense)
    flush(stdout)
    return max(maxprim, abs(Eformula - Edense))
end

if abspath(PROGRAM_FILE) == @__FILE__
    err = 0.0
    for (Nn, hh, DD, g) in [(6, 1, 3, 0.4), (6, 2, 3, 0.7), (8, 1, 4, 0.3)]
        err = max(err, oracleHolonomy(; Nsites=Nn, hmax=hh, D=DD, gC=g))
    end
    println(err < 1e-9 ? "HOLONOMY ORACLE PASSED" : "HOLONOMY ORACLE FAILED (err=$err)")
end
