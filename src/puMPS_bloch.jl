# puMPS Bloch (tangent-space / excitation) ansatz for the periodic TFI ring.
# Follows Zou/Milsted/Vidal arXiv:1710.05397v2, "Excited state ansatz" (Eqs 23-30),
# and Ian Tice's single-sum derivation (puMPS_bloch.pdf).
#
# The excited states with momentum p are Bloch states / tangent vectors of the
# optimized puMPS ground state A_L:
#     |Phi_p(B_C, A_L)> = sum_n e^{-ipn} sum_s Tr( (B_C^{s_n} lambda^{-1}) prod_{m!=n} A_L^{s_m} ) |s>
# i.e. one site n carries the central tangent tensor B = B_C lambda^{-1}; we sum
# over its position n with momentum phase, p = 2*pi*j/N.
#
# The variational problem is the generalized eigenvalue equation (Eq 28)
#     H^C_{mu nu}(p) B_C^nu = E N^C_{mu nu}(p) B_C^nu,
# with the effective Hamiltonian / norm matrices in the B_C parameter space.
# Using translation invariance these reduce to single Fourier sums (PDF):
#     N^C(p) = sum_k e^{i p k} N_{0k},   H^C(p) = sum_k e^{i p k} H_{0k}
# (the overall factor N is common to both and cancels in the eigenproblem).
# We then pseudo-invert N^C(p) and solve  Ntilde H^C B_C = E B_C.
#
# Index conventions match puMPS.jl: A_L[a,b,s] (D x D x d); virtual super-index
# idx(ket,bra) = ket + (bra-1)*D; B_C tangent index packed as
#     mu(s,a,c) = a + (c-1)*D + (s-1)*D*D     (left a, right c, physical s).

include("puMPS.jl")

bidx(s, a, c, D) = a + (c - 1) * D + (s - 1) * D * D   # B_C parameter packing

# ===========================================================================
# Brute-force ORACLE in the full 2^N Hilbert space (small N only). Used to
# validate the transfer-matrix contractions and as an independent solver.
# ===========================================================================

# Amplitude vector (length d^N) of the periodic MPS with impurity tensor M at
# site n and A_L elsewhere:  sum_s Tr(A^{s1}..M^{sn}..A^{sN}) |s>.
function ringStateWithImpurity(A_L, M, n, N)
    D = size(A_L, 1)
    d = size(A_L, 3)
    vec = zeros(ComplexF64, d^N)
    config = zeros(Int, N)
    for idx in 0:(d^N-1)
        x = idx
        for i in 1:N
            config[i] = (x % d) + 1
            x ÷= d
        end
        T = Matrix{ComplexF64}(I, D, D)
        for i in 1:N
            T = T * (i == n ? M[:, :, config[i]] : A_L[:, :, config[i]])
        end
        vec[idx+1] = tr(T)
    end
    return vec
end

# Tangent basis vector |d_nu Phi(p)> in the full Hilbert space.
function oracleBlochVector(A_L, lambdaInv, p, s_nu, a_nu, c_nu, N)
    D = size(A_L, 1)
    d = size(A_L, 3)
    M = zeros(ComplexF64, D, D, d)                 # M = B_C^nu lambda^{-1}
    M[a_nu, c_nu, s_nu] = lambdaInv[c_nu]
    v = zeros(ComplexF64, d^N)
    for n in 1:N
        v .+= exp(-im * p * n) .* ringStateWithImpurity(A_L, M, n, N)
    end
    return v
end

# Full effective matrices N(p), H(p) from the oracle (dense, exact).
function oracleEffectiveMatrices(A_L, lambdaInv, p, Hfull, N)
    D = size(A_L, 1)
    d = size(A_L, 3)
    dim = d * D * D
    basis = zeros(ComplexF64, d^N, dim)
    for s in 1:d, a in 1:D, c in 1:D
        basis[:, bidx(s, a, c, D)] = oracleBlochVector(A_L, lambdaInv, p, s, a, c, N)
    end
    Nmat = basis' * basis
    Hmat = basis' * (Hfull * basis)
    return Nmat, Hmat
end

# Dense periodic TFI Hamiltonian (2^N x 2^N).
function tfiRingHam(h, J, N)
    sx = ComplexF64[0 1; 1 0]; sz = ComplexF64[1 0; 0 -1]
    id = Matrix{ComplexF64}(I, 2, 2)
    op(o, site) = reduce(kron, [k == site ? o : id for k in 1:N])
    H = zeros(ComplexF64, 2^N, 2^N)
    for i in 1:N
        j = mod1(i + 1, N)
        H .-= J * op(sx, i) * op(sx, j)
        H .-= h * op(sz, i)
    end
    return H
end

# ===========================================================================
# Transfer-matrix construction of N_{0k}  (effective norm, no Hamiltonian).
#
# Geometry: ket tangent B_C at site 1 (open leg nu), bra tangent B_C at site
# q = 1+k (open leg mu), A_L everywhere else. The two open sites are joined by
# the plain double-layer transfer matrix E along the two arcs of the ring.
# ===========================================================================

# Ket impurity K[nu, gR, gL]: open nu=(s,a,c) at site 1.
#   gL = idx(a, bL)  (incoming bond, ket part = a),   gR = idx(c, bR).
#   value = lambdaInv[c] * conj(A_L[bL, bR, s]).
function ketImpurity(A_L, lambdaInv)
    D = size(A_L, 1); d = size(A_L, 3)
    K = zeros(ComplexF64, d * D * D, D * D, D * D)
    for s in 1:d, a in 1:D, c in 1:D, bL in 1:D, bR in 1:D
        K[bidx(s, a, c, D), c+(bR-1)*D, a+(bL-1)*D] += lambdaInv[c] * conj(A_L[bL, bR, s])
    end
    return K
end

# Bra impurity J[mu, w, u]: open mu=(s,a,c) at site q.
#   u = idx(kL, a)  (incoming bond, bra part = a),    w = idx(kR, c).
#   value = lambdaInv[c] * A_L[kL, kR, s].
function braImpurity(A_L, lambdaInv)
    D = size(A_L, 1); d = size(A_L, 3)
    J = zeros(ComplexF64, d * D * D, D * D, D * D)
    for s in 1:d, a in 1:D, c in 1:D, kL in 1:D, kR in 1:D
        J[bidx(s, a, c, D), kR+(c-1)*D, kL+(a-1)*D] += lambdaInv[c] * A_L[kL, kR, s]
    end
    return J
end

# contract_N(Kop, P1, Jop, P2)[mu,nu] = sum Kop[nu,gR,gL] P1[u,gR] Jop[mu,w,u] P2[gL,w]
# Kop: [nu, gR, gL];  Jop: [mu, w, u];  P1,P2: D^2 x D^2 (out,in).
function contractRing(Kop, P1, Jop, P2)
    dim = size(Kop, 1); DD = size(P1, 1)
    Tmp = zeros(ComplexF64, dim, DD, DD)           # Tmp[nu,u,w]
    for nu in 1:dim
        A1 = P1 * Kop[nu, :, :]                     # (u x gL)
        Tmp[nu, :, :] = A1 * P2                     # (u x w)
    end
    # Nout[mu,nu] = sum_{u,w} Jop[mu,w,u] Tmp[nu,u,w], as one BLAS matmul.
    JopR = reshape(permutedims(Jop, (1, 3, 2)), dim, DD * DD)   # [mu,(u,w)] = Jop[mu,w,u]
    TmpR = reshape(Tmp, dim, DD * DD)                            # [nu,(u,w)] = Tmp[nu,u,w]
    return JopR * transpose(TmpR)
end

# N_{0k} for separation k (ket-B at site 1, bra-B at site 1+k).
function N0k(A_L, lambdaInv, k, N)
    D = size(A_L, 1); d = size(A_L, 3)
    E = transferMatrix(A_L)
    K = ketImpurity(A_L, lambdaInv)
    J = braImpurity(A_L, lambdaInv)
    if k == 0
        # both tangents on site 1; ring closed by E^{N-1}.
        Epow = E^(N - 1)
        dim = d * D * D
        Nout = zeros(ComplexF64, dim, dim)
        for snu in 1:d, anu in 1:D, cnu in 1:D, amu in 1:D, cmu in 1:D
            nu = bidx(snu, anu, cnu, D); mu = bidx(snu, amu, cmu, D)
            Nout[mu, nu] += lambdaInv[cnu] * lambdaInv[cmu] *
                            Epow[anu+(amu-1)*D, cnu+(cmu-1)*D]
        end
        return Nout
    end
    return contractRing(K, E^(k - 1), J, E^(N - 1 - k))
end

# ===========================================================================
# Fourier sums and the generalized eigenproblem.
# ===========================================================================
function effectiveNorm(A_L, lambdaInv, p, N)
    D = size(A_L, 1); d = size(A_L, 3)
    Nc = zeros(ComplexF64, d * D * D, d * D * D)
    for k in 0:(N-1)
        Nc .+= exp(im * p * k) .* N0k(A_L, lambdaInv, k, N)
    end
    return Nc
end

# ===========================================================================
# Transfer-matrix construction of H_{0k} (effective Hamiltonian).
#
# H = sum_b h_b over the N nearest-neighbour bonds. With the ket tangent at
# site 1 and the bra tangent at site q = 1+k, each bond contributes a ring
# contraction. Bonds split into: those internal to the two A_L arcs (handled by
# sweeping the gate G2 through the transfer matrices), and those touching a
# tangent site (handled by dedicated "gated impurity" operators below).
# Everything is validated per-k against the brute-force oracle.
# ===========================================================================

# Sum of the bond gate over the internal bonds of a segment of `nsites` normal
# sites:  sum_{j=0}^{nsites-2} E^{nsites-2-j} G2 E^{j}   (zero if nsites < 2).
function gatedSegment(E, G2, nsites)
    DD = size(E, 1)
    out = zeros(ComplexF64, DD, DD)
    for j in 0:(nsites-2)
        out .+= E^(nsites - 2 - j) * G2 * E^j
    end
    return out
end

# --- gated operators with one open tangent leg (gate on tangent + neighbour) ---

# Gate on ket-B (site 1) + normal right neighbour (site 2). [nu, out=bond2, in=gL]
function GKright(A_L, lambdaInv, hmat)
    D = size(A_L, 1); d = size(A_L, 3)
    G = zeros(ComplexF64, d * D * D, D * D, D * D)
    for vp in 1:d, va in 1:D, vc in 1:D, bL in 1:D, β in 1:D, b2R in 1:D, k2R in 1:D,
        sk2 in 1:d, sb1 in 1:d, sb2 in 1:d
        w = hmat[pk(sb1, sb2, d), pk(vp, sk2, d)]
        w == 0 && continue
        G[bidx(vp, va, vc, D), k2R+(b2R-1)*D, va+(bL-1)*D] +=
            lambdaInv[vc] * conj(A_L[bL, β, sb1]) * A_L[vc, k2R, sk2] * conj(A_L[β, b2R, sb2]) * w
    end
    return G
end

# Gate on normal left neighbour (site N) + ket-B (site 1). [nu, out=gR, in=bond_{N-1}]
function GKleft(A_L, lambdaInv, hmat)
    D = size(A_L, 1); d = size(A_L, 3)
    G = zeros(ComplexF64, d * D * D, D * D, D * D)
    for vp in 1:d, va in 1:D, vc in 1:D, kNL in 1:D, bNL in 1:D, mb in 1:D, b1R in 1:D,
        sk1 in 1:d, sb1 in 1:d, sb2 in 1:d
        w = hmat[pk(sb1, sb2, d), pk(sk1, vp, d)]
        w == 0 && continue
        G[bidx(vp, va, vc, D), vc+(b1R-1)*D, kNL+(bNL-1)*D] +=
            lambdaInv[vc] * A_L[kNL, va, sk1] * conj(A_L[bNL, mb, sb1]) * conj(A_L[mb, b1R, sb2]) * w
    end
    return G
end

# Gate on bra-B (site q) + normal right neighbour (site q+1). [mu, out=bond_{q+1}, in=u]
function GJright(A_L, lambdaInv, hmat)
    D = size(A_L, 1); d = size(A_L, 3)
    G = zeros(ComplexF64, d * D * D, D * D, D * D)
    for up in 1:d, ua in 1:D, uc in 1:D, kL in 1:D, mk in 1:D, kR in 1:D, bR in 1:D,
        sk1 in 1:d, sk2 in 1:d, sb2 in 1:d
        w = hmat[pk(up, sb2, d), pk(sk1, sk2, d)]
        w == 0 && continue
        G[bidx(up, ua, uc, D), kR+(bR-1)*D, kL+(ua-1)*D] +=
            lambdaInv[uc] * A_L[kL, mk, sk1] * A_L[mk, kR, sk2] * conj(A_L[uc, bR, sb2]) * w
    end
    return G
end

# Gate on normal left neighbour (site q-1) + bra-B (site q). [mu, out=w, in=bond_{q-2}]
function GJleft(A_L, lambdaInv, hmat)
    D = size(A_L, 1); d = size(A_L, 3)
    G = zeros(ComplexF64, d * D * D, D * D, D * D)
    for up in 1:d, ua in 1:D, uc in 1:D, kL in 1:D, bL in 1:D, mk in 1:D, kWR in 1:D,
        sk1 in 1:d, sk2 in 1:d, sb1 in 1:d
        w = hmat[pk(sb1, up, d), pk(sk1, sk2, d)]
        w == 0 && continue
        G[bidx(up, ua, uc, D), kWR+(uc-1)*D, kL+(bL-1)*D] +=
            lambdaInv[uc] * A_L[kL, mk, sk1] * conj(A_L[bL, ua, sb1]) * A_L[mk, kWR, sk2] * w
    end
    return G
end

# --- gated operators with BOTH tangent legs open (gate joins the two tangents) ---

# Gate on ket-B (site 1) + bra-B (site 2), i.e. k=1. [mu, nu, out=bond_{2,3}, in=gL]
function GKJ12fused(A_L, lambdaInv, hmat, Env)
    D = size(A_L, 1); d = size(A_L, 3); dim = d * D * D
    Hout = zeros(ComplexF64, dim, dim)                    # fold Env in; no dim x dim x D^2 x D^2 array
    for vp in 1:d, va in 1:D, vc in 1:D, up in 1:d, ua in 1:D, uc in 1:D,
        bL in 1:D, k2R in 1:D, sk2 in 1:d, sb1 in 1:d
        w = hmat[pk(sb1, up, d), pk(vp, sk2, d)]
        w == 0 && continue
        Hout[bidx(up, ua, uc, D), bidx(vp, va, vc, D)] +=
            lambdaInv[vc] * lambdaInv[uc] * conj(A_L[bL, ua, sb1]) * A_L[vc, k2R, sk2] * w *
            Env[va+(bL-1)*D, k2R+(uc-1)*D]               # o=k2R+(uc-1)D, i=va+(bL-1)D
    end
    return Hout
end

# Gate on bra-B (site N) + ket-B (site 1), i.e. k=N-1. [mu, nu, out=gR, in=u]
function GKJN1fused(A_L, lambdaInv, hmat, Env)
    D = size(A_L, 1); d = size(A_L, 3); dim = d * D * D
    Hout = zeros(ComplexF64, dim, dim)                    # fold Env in; no dim x dim x D^2 x D^2 array
    for vp in 1:d, va in 1:D, vc in 1:D, up in 1:d, ua in 1:D, uc in 1:D,
        kL in 1:D, b1R in 1:D, sk1 in 1:d, sb2 in 1:d
        w = hmat[pk(up, sb2, d), pk(sk1, vp, d)]
        w == 0 && continue
        Hout[bidx(up, ua, uc, D), bidx(vp, va, vc, D)] +=
            lambdaInv[uc] * lambdaInv[vc] * A_L[kL, va, sk1] * conj(A_L[uc, b1R, sb2]) * w *
            Env[kL+(ua-1)*D, vc+(b1R-1)*D]               # o=vc+(b1R-1)D, i=kL+(ua-1)D
    end
    return Hout
end

# Contract a both-legs-open operator G[mu,nu,out,in] with the closing environment
# Epow[in,out]:  result[mu,nu] = sum_{out,in} G[mu,nu,out,in] Epow[in,out].
function contractDouble(G, Epow)
    dim = size(G, 1); DD = size(Epow, 1)
    out = zeros(ComplexF64, dim, dim)
    for mu in 1:dim, nu in 1:dim
        s = zero(ComplexF64)
        @inbounds for o in 1:DD, i in 1:DD
            s += G[mu, nu, o, i] * Epow[i, o]
        end
        out[mu, nu] = s
    end
    return out
end

# H_{0k}: effective Hamiltonian block for tangent separation k.
function H0k(A_L, lambdaInv, hmat, k, N)
    D = size(A_L, 1); d = size(A_L, 3)
    E = transferMatrix(A_L)
    G2 = gateTransfer(A_L, hmat)
    K = ketImpurity(A_L, lambdaInv)
    J = braImpurity(A_L, lambdaInv)

    if k == 0
        gP0 = gatedSegment(E, G2, N - 1)          # gates within sites 2..N
        Env = E^(N - 2)
        dim = d * D * D
        Hout = zeros(ComplexF64, dim, dim)
        # (a) gate within the A_L arc
        for vp in 1:d, va in 1:D, vc in 1:D, ua in 1:D, uc in 1:D
            nu = bidx(vp, va, vc, D); mu = bidx(vp, ua, uc, D)
            Hout[mu, nu] += lambdaInv[vc] * lambdaInv[uc] * gP0[va+(ua-1)*D, vc+(uc-1)*D]
        end
        # (b) gate on bond(1,2): site1(both tangents) + site2.  Fold Env directly into the
        # accumulation instead of building the dim x dim x D^2 x D^2 = 16 D^8 array M0R and
        # calling contractDouble (out[mu,nu] = sum_{o,i} M0R[mu,nu,o,i] Env[i,o]).  Same
        # arithmetic, no D^8 allocation (25.6 GB at D=10!), and no 16 D^8 contraction.
        for vp in 1:d, va in 1:D, vc in 1:D, up in 1:d, ua in 1:D, uc in 1:D,
            k2R in 1:D, b2R in 1:D, sk2 in 1:d, sb2 in 1:d
            w = hmat[pk(up, sb2, d), pk(vp, sk2, d)]
            w == 0 && continue
            Hout[bidx(up, ua, uc, D), bidx(vp, va, vc, D)] +=
                lambdaInv[vc] * lambdaInv[uc] * A_L[vc, k2R, sk2] * conj(A_L[uc, b2R, sb2]) * w *
                Env[va+(ua-1)*D, k2R+(b2R-1)*D]
        end
        # (c) gate on bond(N,1): siteN + site1(both tangents).  Same Env fold.
        for vp in 1:d, va in 1:D, vc in 1:D, up in 1:d, ua in 1:D, uc in 1:D,
            kL in 1:D, bL in 1:D, sk1 in 1:d, sb1 in 1:d
            w = hmat[pk(sb1, up, d), pk(sk1, vp, d)]
            w == 0 && continue
            Hout[bidx(up, ua, uc, D), bidx(vp, va, vc, D)] +=
                lambdaInv[vc] * lambdaInv[uc] * A_L[kL, va, sk1] * conj(A_L[bL, ua, sb1]) * w *
                Env[kL+(bL-1)*D, vc+(uc-1)*D]
        end
        return Hout
    end

    H = zeros(ComplexF64, d * D * D, d * D * D)
    # gates strictly inside the two A_L arcs (valid for all k)
    H .+= contractRing(K, gatedSegment(E, G2, k - 1), J, E^(N - 1 - k))
    H .+= contractRing(K, E^(k - 1), J, gatedSegment(E, G2, N - 1 - k))

    if k == 1
        H .+= GKJ12fused(A_L, lambdaInv, hmat, E^(N - 2))                 # bond(1,2)
        H .+= contractRing(K, E^0, GJright(A_L, lambdaInv, hmat), E^(N - 3))  # bond(2,3)
        H .+= contractRing(GKleft(A_L, lambdaInv, hmat), E^0, J, E^(N - 3))   # bond(N,1)
    elseif k == N - 1
        H .+= contractRing(GKright(A_L, lambdaInv, hmat), E^(N - 3), J, E^0)  # bond(1,2)
        H .+= contractRing(K, E^(N - 3), GJleft(A_L, lambdaInv, hmat), E^0)   # bond(N-1,N)
        H .+= GKJN1fused(A_L, lambdaInv, hmat, E^(N - 2))                 # bond(N,1)
    else
        H .+= contractRing(GKright(A_L, lambdaInv, hmat), E^(k - 2), J, E^(N - 1 - k))  # bond(1,2)
        H .+= contractRing(GKleft(A_L, lambdaInv, hmat), E^(k - 1), J, E^(N - 2 - k))   # bond(N,1)
        H .+= contractRing(K, E^(k - 1), GJright(A_L, lambdaInv, hmat), E^(N - 2 - k))  # bond(q,q+1)
        H .+= contractRing(K, E^(k - 2), GJleft(A_L, lambdaInv, hmat), E^(N - 1 - k))   # bond(q-1,q)
    end
    return H
end

function effectiveHam(A_L, lambdaInv, hmat, p, N)
    D = size(A_L, 1); d = size(A_L, 3)
    Hc = zeros(ComplexF64, d * D * D, d * D * D)
    for k in 0:(N-1)
        Hc .+= exp(im * p * k) .* H0k(A_L, lambdaInv, hmat, k, N)
    end
    return Hc
end

# ===========================================================================
# Generalized eigenproblem  H^C(p) B = E N^C(p) B, solved via the pseudoinverse
# of the (semi-positive-definite, ill-conditioned) effective norm matrix.
# Returns the lowest `nev` variational energies in the momentum-p sector.
# ===========================================================================
function blochSpectrum(A_L, lambda, hmat, p, N; nev=6, cutoff=1e-10)
    lambdaInv = 1.0 ./ max.(lambda, 1e-12)
    Nc = effectiveNorm(A_L, lambdaInv, p, N)
    Hc = effectiveHam(A_L, lambdaInv, hmat, p, N)
    Nc = (Nc + Nc') / 2
    Hc = (Hc + Hc') / 2
    # Pseudoinverse of Nc projected onto its non-null subspace (drops gauge modes).
    F = eigen(Hermitian(Nc))
    thresh = cutoff * maximum(F.values)
    keep = findall(>(thresh), F.values)
    U = F.vectors[:, keep]
    Nhalf = U * Diagonal(1.0 ./ sqrt.(F.values[keep])) * U'   # Nc^{-1/2} on the range
    Heff = Nhalf * Hc * Nhalf                                   # standard Hermitian eigenproblem
    Heff = (Heff + Heff') / 2
    evals = sort(real(eigvals(Heff)))
    return evals[1:min(nev, length(evals))]
end

# ===========================================================================
# Tests
# ===========================================================================

# Per-k validation of N_{0k} and H_{0k} against the brute-force oracle.
function oracleBlocks(A_L, lambdaInv, Hfull, k, N)
    D = size(A_L, 1); d = size(A_L, 3); dim = d * D * D
    function impVec(site, s, a, c)
        M = zeros(ComplexF64, D, D, d); M[a, c, s] = lambdaInv[c]
        return ringStateWithImpurity(A_L, M, site, N)
    end
    ket = zeros(ComplexF64, d^N, dim)
    bra = zeros(ComplexF64, d^N, dim)
    for s in 1:d, a in 1:D, c in 1:D
        ket[:, bidx(s, a, c, D)] = impVec(1, s, a, c)
        bra[:, bidx(s, a, c, D)] = impVec(1 + k, s, a, c)
    end
    return bra' * ket, bra' * (Hfull * ket)
end

function testBlocks()
    h, J, N, D, d = 1.0, 1.0, 6, 3, 2
    Random.seed!(3)
    hmat = tfiBondGate(h, J)
    A0 = randn(ComplexF64, D, D, d)
    A_L, lambda, _ = fullGradientDescent(1500, 0.03, 1e-6, A0, hmat, N)
    lambdaInv = 1.0 ./ max.(lambda, 1e-12)
    Hfull = tfiRingHam(h, J, N)

    println("Per-k block validation (transfer matrix vs oracle):")
    maxN = 0.0; maxH = 0.0
    for k in 0:(N-1)
        Ntm = N0k(A_L, lambdaInv, k, N)
        Htm = H0k(A_L, lambdaInv, hmat, k, N)
        Nor, Hor = oracleBlocks(A_L, lambdaInv, Hfull, k, N)
        eN = norm(Ntm - Nor) / max(norm(Nor), 1e-14)
        eH = norm(Htm - Hor) / max(norm(Hor), 1e-14)
        maxN = max(maxN, eN); maxH = max(maxH, eH)
        println("  k=$k   N err=$eN   H err=$eH")
    end
    println("max block err:  N=$maxN  H=$maxH")
end

# End-to-end: momentum-resolved exact diagonalization. Because H is highly
# degenerate, we project onto each momentum sector first (H and T commute), then
# diagonalize H within the sector. Sector j is labelled so that T has eigenvalue
# e^{i 2π j/N} there, matching the Bloch state convention T|Phi_p> = e^{ip}|Phi_p>.
function momentumED(h, J, N)
    Hfull = tfiRingHam(h, J, N)
    d = 2
    T = zeros(ComplexF64, d^N, d^N)               # translation: site i -> i+1
    for idx in 0:(d^N-1)
        digits_ = [(idx ÷ d^(i - 1)) % d for i in 1:N]
        shifted = circshift(digits_, 1)
        jdx = sum(shifted[i] * d^(i - 1) for i in 1:N)
        T[jdx+1, idx+1] = 1
    end
    result = Dict{Int,Vector{Float64}}()
    for j in 0:(N-1)
        P = zeros(ComplexF64, d^N, d^N)            # projector onto T-eig e^{i2πj/N}
        Tr = Matrix{ComplexF64}(I, d^N, d^N)
        for r in 0:(N-1)
            P .+= exp(-im * 2π * j * r / N) .* Tr
            Tr = Tr * T
        end
        P ./= N
        F = eigen(Hermitian((P + P') / 2))
        Q = F.vectors[:, findall(>(0.5), real(F.values))]   # orthonormal basis of sector
        Hj = Q' * Hfull * Q
        result[j] = sort(real(eigvals(Hermitian((Hj + Hj') / 2))))
    end
    return result
end

function testSpectrum()
    h, J, N, D, d = 1.0, 1.0, 8, 6, 2
    Random.seed!(3)
    hmat = tfiBondGate(h, J)
    A0 = randn(ComplexF64, D, D, d)
    A_L, lambda, en = fullGradientDescent(1500, 0.03, 1e-6, A0, hmat, N)
    println("ground-state energy/site = ", en / N, "  (ED = ", tfiRingED(h, J, N), ")\n")

    edsec = momentumED(h, J, N)
    println("Bloch spectrum vs exact diagonalization (energies per momentum sector):")
    for j in 0:(N-1)
        p = 2π * j / N
        bloch = blochSpectrum(A_L, lambda, hmat, p, N; nev=4)
        exact = edsec[j][1:min(4, length(edsec[j]))]
        println("  j=$j  Bloch=", round.(bloch, digits=5))
        println("        exact=", round.(exact, digits=5))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    testBlocks()
    println()
    testSpectrum()
end
