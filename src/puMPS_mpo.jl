# MPO interface for the puMPS ground-state and Bloch codes.
#
# A Hamiltonian is supplied as a nearest-neighbour MPO in the standard
# finite-state-machine form (the same convention as TFIDMRGNEW.jl):
#     W[a, b, st, sb]   a = left MPO bond, b = right MPO bond,
#                       st = ket (top) physical, sb = bra (bottom) physical,
#   with boundary vectors vL, vR so that the OPEN-chain Hamiltonian is
#     H_open = vL' W W ... W vR   (an operator on the physical legs).
# vL selects the FSM start state s0, vR the end state s_end.
#
# Why we convert to a bond gate instead of "threading the MPO around the ring":
# a periodic ring Hamiltonian H = sum_i h_i CANNOT be written as a single traced
# MPO -- the auxiliary trace produces products of disjoint terms (or, for an
# absorbing FSM, only the trivial identity loops). The correct ring contraction
# is "cut + wrap": open the MPO at one bond (giving every non-wrapping term) and
# add the terms straddling the cut. For a nearest-neighbour MPO that is exactly
# equivalent to a single translation-invariant 2-site gate summed over all bonds
# -- which is precisely what puMPS.jl / puMPS_bloch.jl already do. So the whole
# pipeline reuses the validated code once we extract the gate.
#
# Extracting the bond gate from an NN MPO:
#   * 2-site interaction terms come from FSM paths s0 -> m -> s_end through an
#     intermediate state m:        sum_m  W[s0,m] (x) W[m,s_end]
#   * 1-site field terms come from the direct path s0 -> s_end, i.e. W[s0,s_end];
#     a single-site operator sits on two adjacent bonds, so it enters each bond
#     gate with weight 1/2 to avoid double counting when summed over the ring.

include("puMPS_bloch.jl")

# ---------------------------------------------------------------------------
# TFI MPO (lower-triangular FSM), matching the convention in TFIDMRGNEW.jl.
#   states: 1 = "term finished" (identity past), 3 = "start" (identity future),
#           2 = "began sigma^x, awaiting partner".
#   H = -J sum sx_i sx_{i+1} - h sum sz_i
# ---------------------------------------------------------------------------
function tfiMPO(h, J)
    sx = ComplexF64[0 1; 1 0]; sz = ComplexF64[1 0; 0 -1]
    id = Matrix{ComplexF64}(I, 2, 2)
    W = zeros(ComplexF64, 3, 3, 2, 2)        # W[a,b,st,sb]
    W[1, 1, :, :] = id
    W[2, 1, :, :] = sx
    W[3, 1, :, :] = -h * sz
    W[3, 2, :, :] = -J * sx
    W[3, 3, :, :] = id
    vL = ComplexF64[0, 0, 1]                 # start state s0 = 3
    vR = ComplexF64[1, 0, 0]                 # end   state s_end = 1
    return W, vL, vR
end

# ---------------------------------------------------------------------------
# Extract the translation-invariant nearest-neighbour bond gate (d^2 x d^2,
# pk-packed as in puMPS.jl) from an NN MPO.
# ---------------------------------------------------------------------------
function mpoToBondGate(W, vL, vR)
    χ, _, d, _ = size(W)
    s0 = argmax(abs.(vL))
    se = argmax(abs.(vR))
    id = Matrix{ComplexF64}(I, d, d)
    site(a, b) = W[a, b, :, :]               # the d x d operator at MPO entry (a,b)

    gate = zeros(ComplexF64, d * d, d * d)
    # 2-site interactions: paths s0 -> m -> s_end
    for m in 1:χ
        (m == s0 || m == se) && continue
        gate .+= kron(site(s0, m), site(m, se))
    end
    # 1-site fields: direct path s0 -> s_end, split 1/2 onto each adjacent bond
    field = site(s0, se)
    gate .+= 0.5 .* (kron(field, id) + kron(id, field))
    return gate
end

# ===========================================================================
# Validation + demonstration
# ===========================================================================
function testMPOgate()
    # (1) The extracted TFI gate must match the hand-written tfiBondGate.
    for (h, J) in [(1.0, 1.0), (1.5, 1.0), (0.7, 1.3)]
        W, vL, vR = tfiMPO(h, J)
        g_mpo = mpoToBondGate(W, vL, vR)
        g_ref = ComplexF64.(tfiBondGate(h, J))
        println("TFI  h=$h J=$J   ||gate_MPO - tfiBondGate|| = ", norm(g_mpo - g_ref))
    end
end

# A genuinely different NN model defined purely by an MPO: the "mixed-field
# Ising" model  H = -J sum sx sx - h sum sz - g sum sx  (longitudinal field g).
function mixedFieldIsingMPO(h, J, g)
    sx = ComplexF64[0 1; 1 0]; sz = ComplexF64[1 0; 0 -1]
    id = Matrix{ComplexF64}(I, 2, 2)
    W = zeros(ComplexF64, 3, 3, 2, 2)
    W[1, 1, :, :] = id
    W[2, 1, :, :] = sx
    W[3, 1, :, :] = -h * sz - g * sx          # both fields on the direct path
    W[3, 2, :, :] = -J * sx
    W[3, 3, :, :] = id
    return W, ComplexF64[0, 0, 1], ComplexF64[1, 0, 0]
end

# Dense ring Hamiltonian for an arbitrary NN bond gate (for ED validation).
function ringHamFromGate(gate, N)
    d = Int(round(sqrt(size(gate, 1))))
    g4 = reshape(gate, d, d, d, d)            # [st1,st2,sb1,sb2] (pk packing: st2 fast)
    dimH = d^N
    H = zeros(ComplexF64, dimH, dimH)
    idx(cfg) = sum((cfg[i] - 1) * d^(i - 1) for i in 1:N) + 1
    cfg = zeros(Int, N)
    for lin in 0:(dimH-1)
        x = lin
        for i in 1:N; cfg[i] = (x % d) + 1; x ÷= d; end
        for i in 1:N
            j = mod1(i + 1, N)
            for st1 in 1:d, st2 in 1:d
                # pk slot 1 = LEFT site (matches gateTransfer): gate[pk(b1,b2),pk(k1,k2)]
                # = g4[b2,b1,k2,k1] (column-major reshape makes pk's 2nd slot the fast
                # index).  The old line g4[st1,st2,cfg[i],cfg[j]] silently swapped the
                # two sites -- a no-op for the swap-symmetric gates used previously,
                # wrong for asymmetric gates (caught by the blocked-XX validation).
                w = g4[st2, st1, cfg[j], cfg[i]]
                w == 0 && continue
                ncfg = copy(cfg); ncfg[i] = st1; ncfg[j] = st2
                H[idx(ncfg), lin+1] += w
            end
        end
    end
    return H
end

# Lowest excitation per momentum sector from a dense Hamiltonian (cluster method,
# robust to degeneracies). Convention matches the Bloch states: sector j has
# T-eigenvalue e^{i 2π j/N}.
function momentumGaps(Hfull, N; d=2, tol=1e-6)
    T = zeros(ComplexF64, d^N, d^N)
    for idx in 0:(d^N-1)
        digs = [(idx ÷ d^(i - 1)) % d for i in 1:N]
        jdx = sum(circshift(digs, 1)[i] * d^(i - 1) for i in 1:N)
        T[jdx+1, idx+1] = 1
    end
    F = eigen(Hermitian(Hfull)); vals = F.values; vecs = F.vectors
    E0 = vals[1]
    sectors = [Float64[] for _ in 0:(N-1)]
    i = 1
    while i <= length(vals)
        je = i
        while je < length(vals) && vals[je+1] - vals[i] < 1e-8; je += 1; end
        for ph in eigvals(vecs[:, i:je]' * T * vecs[:, i:je])
            push!(sectors[mod(round(Int, angle(ph) * N / (2π)), N)+1], vals[i])
        end
        i = je + 1
    end
    gaps = Float64[]
    for j in 0:(N-1)
        exc = filter(x -> x - E0 > tol, sort(sectors[j+1]))
        push!(gaps, isempty(exc) ? NaN : exc[1] - E0)
    end
    return gaps
end

function testMixedField()
    h, J, g, N = 1.0, 1.0, 0.4, 8
    W, vL, vR = mixedFieldIsingMPO(h, J, g)
    gate = mpoToBondGate(W, vL, vR)
    Hfull = ringHamFromGate(gate, N)
    edE = minimum(real(eigvals(Hermitian(Hfull)))) / N

    # Ground state from the extracted gate (existing pipeline, unchanged).
    D = 6; Random.seed!(5)
    A0 = randn(ComplexF64, D, D, 2)
    A_L, lambda, en = fullGradientDescent(2500, 0.03, 1e-6, A0, gate, N)
    println("\nMixed-field Ising  H = -J Σ sx sx - h Σ sz - g Σ sx  (h=$h, J=$J, g=$g, N=$N):")
    println("  ground state:  puMPS/site = ", en / N, "   ED = ", edE,
            "   |diff| = ", abs(en / N - edE))

    # Excitation dispersion: puMPS Bloch vs momentum-resolved ED, same gate.
    edgap = momentumGaps(Hfull, N)
    E0b = minimum(blochSpectrum(A_L, lambda, gate, 0.0, N; nev=6))
    println("  excitation gaps per momentum (puMPS Bloch vs ED):")
    for j in 0:(N-1)
        bl = blochSpectrum(A_L, lambda, gate, 2π * j / N, N; nev=6)
        exc = filter(x -> x - E0b > 1e-6, bl)
        bgap = isempty(exc) ? NaN : minimum(exc) - E0b
        println("    j=$j   puMPS=", round(bgap, digits=5), "   ED=", round(edgap[j+1], digits=5))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    testMPOgate()
    testMixedField()
end
