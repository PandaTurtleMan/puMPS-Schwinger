# STEP 2: blocked (d=4) puMPS for the kernel-only faithful Warm-Up Model 2:
#   H_B = H_XX + gC * [ -1/(2N) sum_{jj'} |j-j'|(N-|j-j'|) q_j q_{j'} ]  + lambda (sum_j sigma^z_j)^2
#   q_j = (sigma^z_j+1)/2 - delta_{j,odd}            (staggered charge; NO E0 term yet -> step 3)
# The staggering breaks T down to T^2, so sites are blocked in pairs (2B,2B+1) into
# d=4 super-sites; everything is then block-translation invariant and the existing
# d-generic puMPS machinery applies verbatim.
#
# Blocked pieces (all validated dense below):
#   * XX          -> inter-block 2-site gate + on-block term split 1/2-1/2 onto bonds
#   * kernel      -> 4 Jordan-MPO channels (a,a'), coefficient quadratic in block distance
#                    + on-block q0*q1 term (chi=2 "field" MPO)
#   * lifting     -> existing liftingMPO with block charge Z_B = sz_even + sz_odd
include("puMPS_coulomb_lifting.jl")
using LinearAlgebra, Random, Printf

const I2b = Matrix{ComplexF64}(I, 2, 2)
const I4b = Matrix{ComplexF64}(I, 4, 4)
sp2 = ComplexF64[0 1; 0 0]; sm2 = ComplexF64[0 0; 1 0]; sz2 = ComplexF64[1 0; 0 -1]

# block basis: s-1 = bit_even + 2*bit_odd (even site fast), bit 0 = spin up.
blkop(op_odd, op_even) = kron(op_odd, op_even)

q0op() = blkop(I2b, ComplexF64.(Diagonal([1.0, 0.0])))    # charge at even site: up->1, down->0
q1op() = blkop(ComplexF64.(Diagonal([0.0, -1.0])), I2b)   # charge at odd site:  up->0, down->-1
Zblk() = blkop(I2b, sz2) + blkop(sz2, I2b)                # block total sigma^z
Kk(r, N) = r * (N - r)

# blocked XX gate (d=4 two-block gate, all Nb ring bonds)
function blockedXXGate()
    h_intra = blkop(sm2, sp2) + blkop(sp2, sm2)                       # bond (2B,2B+1) inside a block
    SpOdd = blkop(sp2, I2b); SmOdd = blkop(sm2, I2b)
    SpEven = blkop(I2b, sp2); SmEven = blkop(I2b, sm2)
    h_inter = kron(SpOdd, SmEven) + kron(SmOdd, SpEven)               # bond (2B+1,2B+2) across blocks
    return h_inter .+ 0.5 .* (kron(h_intra, I4b) .+ kron(I4b, h_intra))
end

# generic quadratic-kernel two-body MPO:  sum_{B<B'} cfun(B'-B) * Qin(B) Qout(B')
function polyKernelMPO(cfun, Qin, Qout)
    d = size(Qin, 1); id = Matrix{ComplexF64}(I, d, d)
    c = [1.0 0 0; 1 1 1; 1 2 4] \ [cfun(1), cfun(2), cfun(3)]   # fit c0+c1*rho+c2*rho^2 = cfun(rho+1)
    W = zeros(ComplexF64, 5, 5, d, d)
    W[1, 1, :, :] = id;  W[5, 5, :, :] = id
    W[5, 2, :, :] = Qin
    W[2, 2, :, :] = id;  W[2, 3, :, :] = id;  W[2, 4, :, :] = id
    W[3, 3, :, :] = id;  W[3, 4, :, :] = 2 .* id
    W[4, 4, :, :] = id
    W[2, 1, :, :] = c[1] .* Qout; W[3, 1, :, :] = c[2] .* Qout; W[4, 1, :, :] = c[3] .* Qout
    vL = zeros(ComplexF64, 5); vL[5] = 1
    vR = zeros(ComplexF64, 5); vR[1] = 1
    return W, vL, vR
end

# chi=2 on-block MPO:  sum_B op(B)
function fieldMPO(op)
    d = size(op, 1); id = Matrix{ComplexF64}(I, d, d)
    W = zeros(ComplexF64, 2, 2, d, d)
    W[1, 1, :, :] = id; W[1, 2, :, :] = op; W[2, 2, :, :] = id
    return W, ComplexF64[1, 0], ComplexF64[0, 1]
end

# all Coulomb-kernel MPO terms (with their scalar coefficients), for Nsites sites
function kernelTerms(Nsites, gC)
    q = (q0op(), q1op())
    terms = Any[]
    for a in 0:1, ap in 0:1
        δ = ap - a
        cf = R -> -Kk(2R + δ, Nsites) / Nsites          # -(1/N)*2*K/2 : both orders folded
        push!(terms, (gC, polyKernelMPO(cf, q[a+1], q[ap+1])...))
    end
    push!(terms, (gC * (-Kk(1, Nsites) / Nsites), fieldMPO(q0op() * q1op())...))
    return terms
end

# dense staggered-kernel diagonal (reference; conventions of warmup2_ed.jl)
function staggeredKernelDiag(N)
    dimH = 2^N; out = zeros(dimH); q = zeros(N)
    for lin in 0:dimH-1
        x = lin
        for i in 1:N
            s = (x % 2 == 0) ? 1 : -1
            q[i] = 0.5 * (s + 1) - (isodd(i - 1) ? 1.0 : 0.0)
            x ÷= 2
        end
        v = 0.0
        for j in 1:N, jp in 1:N
            v += Kk(abs(j - jp), N) * q[j] * q[jp]
        end
        out[lin+1] = -v / (2N)
    end
    return out
end

zeroChargeIdx(N) = [lin + 1 for lin in 0:2^N-1 if
    sum((lin >> (i - 1)) & 1 == 0 ? 1 : -1 for i in 1:N) == 0]

denseQtot(N) = Diagonal([sum((lin >> (i - 1)) & 1 == 0 ? 1.0 : -1.0 for i in 1:N) for lin in 0:2^N-1])

# ---------------------------------------------------------------------------
function validateDense(N)
    Nb = N ÷ 2
    Hxx_site = ringHamFromGate(xxG(), N)
    Hxx_blk = ringHamFromGate(blockedXXGate(), Nb)
    e1 = norm(Hxx_blk - Hxx_site)
    Hker = zeros(ComplexF64, 2^N, 2^N)
    for (coef, W, vL, vR) in kernelTerms(N, 1.0)
        Hker .+= coef .* denseMPOexpand(W, vL, vR, Nb)
    end
    e2 = norm(Hker - Diagonal(staggeredKernelDiag(N)))
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    e3 = norm(denseMPOexpand(Wl, vLl, vRl, Nb) - denseQtot(N)^2)
    println("dense validation @ N=$N:")
    println("  blocked XX gate ring == site H_XX      : ", e1)
    println("  kernel MPO channels == staggered diag  : ", e2)
    println("  blocked lifting MPO == (sum sz)^2      : ", e3)
    flush(stdout)
    return max(e1, e2, e3)
end

function edReferenceB(N, gC, lam)
    H = ringHamFromGate(xxG(), N) .+ gC .* Diagonal(staggeredKernelDiag(N))
    H = (H + H') / 2
    idx = zeroChargeIdx(N)
    Eneu = minimum(real(eigvals(Hermitian(H[idx, idx]))))
    Hfull = H .+ lam .* denseQtot(N)^2
    Efull = minimum(real(eigvals(Hermitian((Hfull + Hfull') / 2))))
    return Eneu, Efull
end

function runStep2(N, D, gC, lam; steps=2500, alpha=0.02, seed=4)
    Nb = N ÷ 2
    Eneu, Efull = edReferenceB(N, gC, lam)
    @printf("N=%d gC=%.2f lam=%.1f : ED neutral E/N = %.10f   (full+lift = %.10f, match=%.1e)\n",
            N, gC, lam, Eneu / N, Efull / N, abs(Eneu - Efull)); flush(stdout)
    gate = blockedXXGate()
    terms = kernelTerms(N, gC)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    push!(terms, (lam, Wl, vLl, vRl))
    Random.seed!(seed)
    A0 = randn(ComplexF64, D, D, 4)
    A_L, lamv, en, eta = fullGD_general(steps, alpha, A0, gate, terms, Nb)
    _, _, Q2 = mpoRingEnergy(A_L, Wl, vLl, vRl, Nb)
    @printf("  puMPS(D=%d): E/N = %.10f   |diff|/N = %.2e   <Q^2> = %.2e   eta = %.1e\n",
            D, real(en) / N, abs(real(en) - Eneu) / N, real(Q2), eta); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    err = validateDense(8)
    err < 1e-10 || error("dense validation failed")
    runStep2(8, 6, 0.3, 1.0; steps=2000)
end
