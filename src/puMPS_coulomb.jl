# Long-range (Coulomb) MPO on a puMPS ring.
#
# Two pieces:
#   1. mpoRingEnergy(A, W, vL, vR, N): the combined MPS-MPO-MPS transfer
#      contraction, giving <psi|H|psi> for a puMPS |psi> = Tr(A^{s1}..A^{sN})
#      with H supplied as a (general, possibly long-range) MPO (W, vL, vR).
#      The MPS bonds are traced (ring); the MPO bond is OPEN, capped by vL,vR
#      at the cut.  This is the missing machinery that lets a >2-range MPO be
#      used by the puMPS code (mpoToBondGate only handled range-2).
#   2. Two MPOs for the Coulomb interaction
#         V = -1/(2N) sum_{j,j'} |j-j'|(N-|j-j'|) Q_j Q_{j'}
#      (a) coulombMPO_ON     : the O(N)-bond "shift register" (Ian's idea), and
#      (b) coulombMPO_jordan : the constant chi=5 Pascal/Jordan version.
#
# Q_j is the on-site charge operator; default sigma^z (any Hermitian on-site
# operator works -- pass Qop).  The constant N(E0+theta/2pi)^2 is omitted here
# (it is just a c-number * identity, trivially added).
#
# Validated below against the dense ring Hamiltonian for small N.

include("puMPS_mpo.jl")          # -> puMPS_bloch.jl -> puMPS.jl (ringStateWithImpurity, transferMatrix)
using LinearAlgebra, Random, Printf

coulombKernel(r, N) = abs(r) * (N - abs(r))            # K(r) = |r|(N-|r|), symmetric K(r)=K(N-r)

# ===========================================================================
# (1) Combined MPS-MPO-MPS ring transfer contraction.
# ===========================================================================
function mpoRingEnergy(A, W, vL, vR, N)
    D = size(A, 1); d = size(A, 3); χ = size(W, 1)
    dim = D * χ * D
    cidx(ak, m, ab) = ak + (m - 1) * D + (ab - 1) * D * χ      # (ket bond, mpo bond, bra bond)
    T = zeros(ComplexF64, dim, dim)                            # one-site combined transfer
    for st in 1:d, sb in 1:d, ml in 1:χ, mr in 1:χ
        w = W[ml, mr, st, sb]
        w == 0 && continue
        for ak in 1:D, bk in 1:D, ab in 1:D, bb in 1:D
            T[cidx(bk, mr, bb), cidx(ak, ml, ab)] += A[ak, bk, st] * w * conj(A[ab, bb, sb])
        end
    end
    TN = T^N
    # trace the two MPS bonds (ring); cap the MPO bond with vL (left m0) and vR (right mN)
    H = ComplexF64(0)
    for a in 1:D, b in 1:D, m0 in 1:χ, mN in 1:χ
        H += vL[m0] * TN[cidx(a, mN, b), cidx(a, m0, b)] * vR[mN]
    end
    E = transferMatrix(A)
    nrm = tr(E^N)
    return H, nrm, H / nrm
end

# ===========================================================================
# (2a) O(N)-bond-dimension "shift register" MPO  (the suggested construction).
#
#   states:  done = 1,  mem_1..mem_{N-1} = 2..N,  start = N+1.
#   * start -> mem_1            emit Q          (first charge)
#   * mem_r -> mem_{r+1}        emit I          (carry it one site: distance++)
#   * mem_r -> done            emit -K(r)/N * Q (second charge at distance r)
# Reading the FSM, every contributing path emits exactly two charges at
# separation r and picks up weight -K(r)/N, i.e. V above.  chi = N+1.
#
#   W =  [  I                                            ]
#        [ c_1 Q   .     I                               ]
#        [ c_2 Q   .     .     I                         ]
#        [  ...                      .                   ]
#        [ c_{N-1}Q .    .     .           .             ]      c_r = -K(r)/N
#        [  0      Q     .     .     .     .      I      ]
#   (first column = read-outs, super-diagonal in the mem block = the shift,
#    bottom row = start: inject into mem_1 and stay-in-start identity.)
# ===========================================================================
function coulombMPO_ON(N; Qop=ComplexF64[1 0; 0 -1])
    d = size(Qop, 1); id = Matrix{ComplexF64}(I, d, d)
    χ = N + 1
    W = zeros(ComplexF64, χ, χ, d, d)
    W[1, 1, :, :] = id                       # done -> done
    W[χ, χ, :, :] = id                       # start -> start
    W[χ, 2, :, :] = Qop                      # start -> mem_1 : first charge
    for r in 1:N-2
        W[r+1, r+2, :, :] = id               # mem_r -> mem_{r+1} : carry one site
    end
    for r in 1:N-1
        W[r+1, 1, :, :] = (-coulombKernel(r, N) / N) .* Qop   # mem_r -> done : second charge
    end
    vL = zeros(ComplexF64, χ); vL[χ] = 1
    vR = zeros(ComplexF64, χ); vR[1] = 1
    return W, vL, vR
end

# ===========================================================================
# (2b) constant chi=5 Pascal/Jordan MPO (kernel is a degree-2 polynomial in r).
#   states: done=1, a0=2,a1=3,a2=4, start=5.  Pascal block advances (1,r,r^2).
#   read-out coefficients fixed by  c0+c1*rho+c2*rho^2 = -K(rho+1)/N.
# ===========================================================================
function coulombMPO_jordan(N; Qop=ComplexF64[1 0; 0 -1])
    d = size(Qop, 1); id = Matrix{ComplexF64}(I, d, d)
    W = zeros(ComplexF64, 5, 5, d, d)
    W[1, 1, :, :] = id;  W[5, 5, :, :] = id
    W[5, 2, :, :] = Qop                       # inject first charge into a0
    W[2, 2, :, :] = id;  W[2, 3, :, :] = id;  W[2, 4, :, :] = id     # Pascal
    W[3, 3, :, :] = id;  W[3, 4, :, :] = 2id
    W[4, 4, :, :] = id
    W[2, 1, :, :] = (-(N - 1) / N) .* Qop      # c0
    W[3, 1, :, :] = (-(N - 2) / N) .* Qop      # c1
    W[4, 1, :, :] = (1 / N) .* Qop             # c2
    vL = zeros(ComplexF64, 5); vL[5] = 1
    vR = zeros(ComplexF64, 5); vR[1] = 1
    return W, vL, vR
end

# ===========================================================================
# Validation helpers (dense, small N).
# ===========================================================================
# on-site operator at site j, with site 1 = least significant digit (matches
# ringStateWithImpurity's config ordering).
function opAt(op, j, N)
    d = size(op, 1); id = Matrix{ComplexF64}(I, d, d)
    return reduce(kron, reverse([k == j ? op : id for k in 1:N]))
end

# dense ring Coulomb interaction  -1/(2N) sum_{j,j'} K(|j-j'|) Q_j Q_{j'}.
function denseCoulomb(N; Qop=ComplexF64[1 0; 0 -1])
    d = size(Qop, 1)
    Qs = [opAt(Qop, j, N) for j in 1:N]
    H = zeros(ComplexF64, d^N, d^N)
    for j in 1:N, jp in 1:N
        H .+= (-1 / (2N)) * coulombKernel(j - jp, N) .* (Qs[j] * Qs[jp])
    end
    return H
end

# dense operator from an OPEN-chain MPO:  <s_b|H|s_k> = vL' prod W^{s_k,s_b} vR.
function denseMPOexpand(W, vL, vR, N)
    d = size(W, 3); χ = size(W, 1); dim = d^N
    H = zeros(ComplexF64, dim, dim)
    ck = zeros(Int, N); cb = zeros(Int, N)
    for ik in 0:dim-1, ib in 0:dim-1
        x = ik; for i in 1:N; ck[i] = (x % d) + 1; x ÷= d; end
        x = ib; for i in 1:N; cb[i] = (x % d) + 1; x ÷= d; end
        M = reshape(ComplexF64.(vL), 1, χ)
        for i in 1:N; M = M * W[:, :, ck[i], cb[i]]; end
        H[ib+1, ik+1] = (M * vR)[1]
    end
    return H
end

function testCoulomb(; N=6, D=3, seed=7)
    Random.seed!(seed)
    A = randn(ComplexF64, D, D, 2)
    psi = ringStateWithImpurity(A, A, 1, N)            # plain puMPS state (M = A)
    Hd = denseCoulomb(N)
    edE = (psi' * Hd * psi) / (psi' * psi)
    println("Coulomb-interaction validation  (N=$N, D=$D, Q=sigma^z):")
    println("  reference  <psi|H_C|psi>/<psi|psi> = ", round(real(edE), digits=10))
    for (name, mk) in (("O(N) shift-register", coulombMPO_ON), ("chi=5 Jordan      ", coulombMPO_jordan))
        W, vL, vR = mk(N)
        Hmpo = denseMPOexpand(W, vL, vR, N)
        _, _, eMPO = mpoRingEnergy(A, W, vL, vR, N)
        @printf("  %s  (chi=%2d):  ||H_MPO - H_C|| = %.2e   |E_transfer - ref| = %.2e\n",
                name, size(W, 1), norm(Hmpo - Hd), abs(eMPO - edE))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__       # only when run directly, not on include
    testCoulomb()
end
