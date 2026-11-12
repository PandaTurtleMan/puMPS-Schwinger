# This is code for finding the ground state of a Hamiltonian with a periodic
# uniform MPS (puMPS), following the "Gradient descent for puMPS" section of the
# supplemental material of Zou, Milsted & Vidal, arXiv:1710.05397v2.
#
# Conventions
# -----------
# * The puMPS on N sites is  |Psi(A)> = sum_s Tr(A^{s_1} A^{s_2} ... A^{s_N}) |s>.
# * A single tensor A is stored as a D x D x d array  A[a,b,s], where
#       a = left virtual index, b = right virtual index, s = physical index,
#   so A[:,:,s] is the D x D matrix A^s.
# * We work in the LEFT-canonical gauge: A = A_L with
#       sum_s (A_L^s)' A_L^s = I                                 (Eq. 6)
#       sum_s  A_L^s lambda^2 (A_L^s)' = lambda^2                (Eq. 7)
#   lambda is the diagonal matrix of Schmidt coefficients (stored as a vector).
# * The "central tensor" deformation lives on site 1:
#       |Psi_{A_L}(A_C)> = sum_s Tr[(A_C^{s_1} lambda^{-1}) A_L^{s_2} ... A_L^{s_N}] |s>   (Eq. 18)
#   At the expansion point A_C = A_L * lambda the deformed state equals |Psi(A_L)>.
#
# A "super-index" packs a (ket, bra) pair of virtual indices as
#       idx(x, xp) = x + (xp - 1)*D          (ket index x runs fastest).
# With this packing a D^2 x D^2 transfer matrix M reshapes as
#       reshape(M, (D,D,D,D))[x, xp, y, yp] = M[idx(x,xp), idx(y,yp)],
# with (x,xp) the OUTPUT (right) super-index and (y,yp) the INPUT (left) one.

using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Hamiltonian: transverse field Ising model on a ring,
#   H = -J sum_i sx_i sx_{i+1} - h sum_i sz_i   (periodic, s_{N+1} = s_1).
# We express it as a sum over nearest-neighbour bonds. The single-site field is
# split evenly between the two bonds touching a site (h/2 each) so that summing
# the bond gate over all N bonds reproduces H exactly. The gate is returned as a
# d^2 x d^2 matrix with index packing pk(s1,s2) = (s1-1)*d + s2.
# ---------------------------------------------------------------------------
function tfiBondGate(h, J)
    sx = [0.0 1.0; 1.0 0.0]
    sz = [1.0 0.0; 0.0 -1.0]
    id = Matrix{Float64}(I, 2, 2)
    return -J * kron(sx, sx) - (h / 2) * (kron(sz, id) + kron(id, sz))
end

pk(s1, s2, d) = (s1 - 1) * d + s2

# ---------------------------------------------------------------------------
# Transfer matrix  E[idx(b,bp), idx(a,ap)] = sum_s A[a,b,s] conj(A[ap,bp,s]).
# Maps a left super-index (a,ap) to a right super-index (b,bp).
# ---------------------------------------------------------------------------
function transferMatrix(A)
    D = size(A, 1)
    d = size(A, 3)
    E = zeros(ComplexF64, D * D, D * D)
    for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
        E[b+(bp-1)*D, a+(ap-1)*D] += A[a, b, s] * conj(A[ap, bp, s])
    end
    return E
end

# Two-site bond gate as a transfer matrix.
# G2[idx(c,cp), idx(a,ap)] contracts a bond (with the gate hmat) spanning two
# sites: input = left super-index (a,ap) of the first site, output = right
# super-index (c,cp) of the second site.
function gateTransfer(A, hmat)
    D = size(A, 1)
    d = size(A, 3)
    G = zeros(ComplexF64, D * D, D * D)
    for sk1 in 1:d, sk2 in 1:d, sb1 in 1:d, sb2 in 1:d
        w = hmat[pk(sb1, sb2, d), pk(sk1, sk2, d)]
        w == 0 && continue
        for a in 1:D, b in 1:D, c in 1:D, ap in 1:D, bp in 1:D, cp in 1:D
            G[c+(cp-1)*D, a+(ap-1)*D] +=
                w * A[a, b, sk1] * A[b, c, sk2] * conj(A[ap, bp, sb1]) * conj(A[bp, cp, sb2])
        end
    end
    return G
end

# ---------------------------------------------------------------------------
# The local effective norm matrix g_{mu,nu}  (Eq. 19, Fig. 6 top).
#
# The deformed-puMPS norm is <Psi|Psi> = conj(A_C^mu) g_{mu,nu} A_C^nu, with
# mu = (s, a, c) the physical and two virtual indices of A_C. Removing A_C and
# its conjugate from site 1 leaves the ring of A_L tensors on sites 2..N (the
# environment R = E^{N-1}) together with a lambda^{-1} on each of the two open
# bonds, and an identity on the open physical leg:
#
#   g_{(s',a',c'),(s,a,c)} = delta_{s' s} * lambdaInv[c'] * lambdaInv[c] * R4[a,a',c,c']
#
# where R4[a,a',c,c'] is E^{N-1} with output = left-of-site-1 super-index (a,a')
# and input = right-of-site-1 super-index (c,c').
# ---------------------------------------------------------------------------
function calculateMetricTensor(A, lambdaInverse, N)
    D = size(A, 1)
    d = size(A, 3)
    E = transferMatrix(A)
    R4 = reshape(E^(N - 1), (D, D, D, D))   # [a, ap, b, bp], out = left, in = right
    g = zeros(ComplexF64, d * D * D, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for s in 1:d, a in 1:D, c in 1:D, ap in 1:D, cp in 1:D
        g[mu(s, ap, cp), mu(s, a, c)] =
            lambdaInverse[cp] * lambdaInverse[c] * R4[a, ap, c, cp]
    end
    return g
end

# ---------------------------------------------------------------------------
# The (unnormalised, un-shifted) local gradient tensor  <d_nu Psi | H | Psi>
# (numerator of Eq. 13, Fig. 6 bottom), returned together with the metric so
# that the caller can assemble the properly normalised, Hamiltonian-shifted
# gradient  (1/<Psi|Psi>) <d_nu Psi | (H - <H>) | Psi>.
#
# d_nu Psi removes the conjugate central tensor from site 1 of the bra, leaving
# the open legs nu = (s', a', c') (the lambda^{-1} that multiplied A_C stays
# behind and is applied at the end). We build the tensor F[s',a',b'] with b' the
# open bra bond that connects into the ring, then attach lambda^{-1}.
#
# H is a sum over the N bonds of the ring. Grouping the bonds by their position
# relative to the open site 1:
#   (A) the N-2 bonds among sites 2..N (open site only carries the identity),
#   (B) the bond (1,2),
#   (C) the bond (N,1).
# ---------------------------------------------------------------------------
function calculateGradientTensor(A, hmat, lambdaInverse, N)
    D = size(A, 1)
    d = size(A, 3)
    E = transferMatrix(A)
    G2 = gateTransfer(A, hmat)
    Env4 = reshape(E^(N - 2), (D, D, D, D))   # [a, ap, b, bp], out = left, in = right

    # (A) bonds fully inside sites 2..N: B_A = sum_{j=1}^{N-2} E^{N-2-j} G2 E^{j-1}
    BA = zeros(ComplexF64, D * D, D * D)
    for j in 1:(N-2)
        BA .+= E^(N - 2 - j) * G2 * E^(j - 1)
    end
    BA4 = reshape(BA, (D, D, D, D))           # [a, ap, b, bp], out = left, in = right

    F = zeros(ComplexF64, d, D, D)            # F[s', a', b']

    # (A) site 1 carries only ket A and the identity on the physical leg.
    for sp in 1:d, app in 1:D, bpp in 1:D, a in 1:D, b in 1:D
        F[sp, app, bpp] += A[a, b, sp] * BA4[a, app, b, bpp]
    end

    # (B) bond (1,2): gate acts on the open site 1 (ket) and site 2.
    for sp in 1:d, sk1 in 1:d, sk2 in 1:d, sk2p in 1:d
        w = hmat[pk(sp, sk2p, d), pk(sk1, sk2, d)]
        w == 0 && continue
        for app in 1:D, bpp in 1:D, a in 1:D, b in 1:D, c in 1:D, cp in 1:D
            F[sp, app, bpp] +=
                w * A[a, b, sk1] * A[b, c, sk2] * conj(A[bpp, cp, sk2p]) * Env4[a, app, c, cp]
        end
    end

    # (C) bond (N,1): gate acts on site N and the open site 1 (ket).
    for sp in 1:d, skN in 1:d, skNp in 1:d, sk1 in 1:d
        w = hmat[pk(skNp, sp, d), pk(skN, sk1, d)]
        w == 0 && continue
        for app in 1:D, bpp in 1:D, a in 1:D, b in 1:D, dd in 1:D, dp in 1:D
            F[sp, app, bpp] +=
                w * A[a, b, sk1] * A[dd, a, skN] * conj(A[dp, app, skNp]) * Env4[dd, dp, b, bpp]
        end
    end

    # Attach lambda^{-1} to the open bond, turning index b' into the A_C index c'.
    numH = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for sp in 1:d, app in 1:D, cp in 1:D
        numH[mu(sp, app, cp)] = F[sp, app, cp] * lambdaInverse[cp]
    end
    return numH
end

# ---------------------------------------------------------------------------
# Assemble the true (normalised, Hamiltonian-shifted) local gradient and the
# corresponding descent direction in A_L. This is where the metric must be
# INVERTED (Eq. 20): the descent direction is
#       Delta A_C^mu = - g^{mu nu} (dE/d conj(A_C^nu)),
# i.e. g^{-1} times the gradient, NOT g times the gradient. We then map the
# change of the central tensor back to a change of the puMPS tensor via
#       Delta A_L = Delta A_C lambda^{-1}                        (text above Eq. 21).
# ---------------------------------------------------------------------------
# Pseudo-inverse solve  x = g^{-1} b  for a Hermitian PSD g with null modes:
# invert only the eigenvalues above a relative cutoff.
function pseudoSolve(g, b; cutoff=1e-12)
    F = eigen(Hermitian((g + g') / 2))
    vals = F.values
    thresh = cutoff * maximum(abs.(vals))
    inv = [v > thresh ? 1.0 / v : 0.0 for v in vals]
    return F.vectors * (inv .* (F.vectors' * b))
end

function trueGradient(g, numH, A, lambda, lambdaInverse, N)
    D = size(A, 1)
    d = size(A, 3)

    # Central tensor at the expansion point: A_C = A_L * lambda (lambda on the
    # right virtual leg), flattened with the same packing as the gradient.
    vecAC = zeros(ComplexF64, d * D * D)
    mu(s, a, c) = a + (c - 1) * D + (s - 1) * D * D
    for s in 1:d, a in 1:D, c in 1:D
        vecAC[mu(s, a, c)] = A[a, c, s] * lambda[c]
    end

    numNorm = g * vecAC                       # <d_nu Psi | Psi> = (g A_C)_nu
    normVal = real(dot(vecAC, numNorm))       # <Psi|Psi>
    energy = real(dot(vecAC, numH)) / normVal # <Psi|H|Psi> / <Psi|Psi>

    grad = (numH .- energy .* numNorm) ./ normVal   # dE/d conj(A_C) (Eq. 13)

    # g^{-1} . gradient. The local effective norm matrix g is positive
    # semidefinite but has (near-)null gauge directions, so we apply a
    # pseudo-inverse that drops eigenvalues below a relative cutoff. This is the
    # dense analogue of the iterative solve in the range of g used in the paper.
    gInvGrad = pseudoSolve(g, grad)
    deltaACvec = -gInvGrad                     # Delta A_C (Eq. 20)

    # Convergence monitor eta = sqrt( g^{mu nu} dE/dA^mu dE/dA^nu )  (Eq. 22).
    # NOTE: this also uses the INVERSE metric.
    eta = sqrt(abs(real(dot(grad, gInvGrad))))

    # Delta A_L = Delta A_C lambda^{-1}  (lambda^{-1} on the right virtual leg).
    deltaAL = zeros(ComplexF64, D, D, d)
    for s in 1:d, a in 1:D, c in 1:D
        deltaAL[a, c, s] = deltaACvec[mu(s, a, c)] * lambdaInverse[c]
    end
    return deltaAL, eta, energy
end

function oneGradientDescentStep(A, alpha, deltaAL)
    return A .+ alpha .* deltaAL              # A'_L = A_L + alpha Delta A_L (Eq. 21)
end

# ---------------------------------------------------------------------------
# Left canonicalisation of a uniform MPS tensor (standard procedure).
# Returns the left-canonical tensor A_L and the vector of Schmidt coefficients
# lambda (normalised to unit norm). Steps:
#   1. l  = dominant left fixed point of the transfer matrix, eigenvalue eta.
#   2. l  = X' X ;  A~^s = X A^s X^{-1} / sqrt(eta)   -> left canonical.
#   3. r  = dominant right fixed point of A~ ;  r = W diag(lambda^2) W'.
#   4. A_L^s = W' A~^s W   -> right fixed point diagonal, left condition kept.
# ---------------------------------------------------------------------------
function canonicalization(A)
    D = size(A, 1)
    d = size(A, 3)

    # Left fixed point: l_out[b,bp] = sum_s (A^s)' l A^s. Build the linear map.
    Lmap = zeros(ComplexF64, D * D, D * D)
    for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
        Lmap[b+(bp-1)*D, a+(ap-1)*D] += conj(A[a, b, s]) * A[ap, bp, s]
    end
    Fl = eigen(Lmap)
    kl = argmax(abs.(Fl.values))
    eta = real(Fl.values[kl])
    l = reshape(Fl.vectors[:, kl], (D, D))
    l = (l + l') / 2                          # Hermitian part
    real(tr(l)) < 0 && (l = -l)               # fix overall sign to PSD
    el = eigen(Hermitian(l))
    dl = max.(real(el.values), 0.0)
    X = Diagonal(sqrt.(dl)) * el.vectors'     # l = X' X
    Xi = el.vectors * Diagonal(1.0 ./ sqrt.(dl .+ eps()))

    Atil = similar(A, ComplexF64)
    for s in 1:d
        Atil[:, :, s] = X * A[:, :, s] * Xi / sqrt(eta)
    end

    # Right fixed point of A~: r_out[a,ap] = sum_s A~^s r (A~^s)'.
    Rmap = zeros(ComplexF64, D * D, D * D)
    for s in 1:d, a in 1:D, b in 1:D, ap in 1:D, bp in 1:D
        Rmap[a+(ap-1)*D, b+(bp-1)*D] += Atil[a, b, s] * conj(Atil[ap, bp, s])
    end
    Fr = eigen(Rmap)
    kr = argmax(abs.(Fr.values))
    r = reshape(Fr.vectors[:, kr], (D, D))
    r = (r + r') / 2
    real(tr(r)) < 0 && (r = -r)
    er = eigen(Hermitian(r))
    order = sortperm(real(er.values); rev=true)
    dr = max.(real(er.values[order]), 0.0)
    W = er.vectors[:, order]

    AL = similar(A, ComplexF64)
    for s in 1:d
        AL[:, :, s] = W' * Atil[:, :, s] * W
    end

    lambda = sqrt.(dr)
    lambda ./= norm(lambda)                   # normalise the state
    return AL, lambda
end

# ---------------------------------------------------------------------------
# Variational energy of the (normalised) uniform puMPS,  E = <H>/<Psi|Psi>.
# By translation invariance <H> = N * <h_{1,2}>, and with transfer matrices
#   <h_{1,2}> numerator = Tr[ G2 E^{N-2} ],   <Psi|Psi> = Tr[ E^N ].
# ---------------------------------------------------------------------------
function energyPuMPS(A, hmat, N)
    E = transferMatrix(A)
    G2 = gateTransfer(A, hmat)
    return real(N * tr(G2 * E^(N - 2)) / tr(E^N))
end

# ---------------------------------------------------------------------------
# Full gradient-descent optimisation (fixed step size alpha; line search can be
# added later).
# ---------------------------------------------------------------------------
function fullGradientDescent(maxSteps, alpha, convergenceTolerance, initialA, hmat, N;
                             verbose=false, reportEvery=5)
    A, lambda = canonicalization(initialA)
    lambdaInverse = 1.0 ./ max.(lambda, 1e-12)
    local energy
    local eta = Inf
    for i in 1:maxSteps
        g = calculateMetricTensor(A, lambdaInverse, N)
        numH = calculateGradientTensor(A, hmat, lambdaInverse, N)
        deltaAL, eta, energy = trueGradient(g, numH, A, lambda, lambdaInverse, N)
        if verbose && (i == 1 || i % reportEvery == 0)
            println("  step ", lpad(i, 5), "   E/N = ", round(real(energy) / N, digits=10),
                    "   eta = ", round(eta, sigdigits=4)); flush(stdout)
        end
        if eta < convergenceTolerance && i > 1
            verbose && (println("  converged at step ", i, ":  eta = ", round(eta, sigdigits=4)); flush(stdout))
            return A, lambda, energy
        end
        A = oneGradientDescentStep(A, alpha, deltaAL)
        A, lambda = canonicalization(A)
        lambdaInverse = 1.0 ./ max.(lambda, 1e-12)
    end
    verbose && (println("  reached maxSteps=", maxSteps, ":  final eta = ", round(eta, sigdigits=4)); flush(stdout))
    return A, lambda, energy
end

# ===========================================================================
# Self-test: compare against exact diagonalisation of the TFI ring and verify
# the gradient machinery is internally consistent. Runs only when this file is
# executed directly (`julia puMPS.jl`).
# ===========================================================================
function tfiRingED(h, J, N)
    d = 2
    sx = ComplexF64[0 1; 1 0]
    sz = ComplexF64[1 0; 0 -1]
    id = Matrix{ComplexF64}(I, 2, 2)
    op(o, site) = reduce(kron, [k == site ? o : id for k in 1:N])
    H = zeros(ComplexF64, d^N, d^N)
    for i in 1:N
        j = mod1(i + 1, N)
        H .-= J * op(sx, i) * op(sx, j)
        H .-= h * op(sz, i)
    end
    return minimum(real(eigvals(Hermitian(H)))) / N   # energy per site
end

function selftest()
    Random.seed!(3)   # fixed-step GD is initialisation sensitive (see note below)
    h, J, N, D, d = 1.0, 1.0, 8, 8, 2
    hmat = tfiBondGate(h, J)

    # Internal consistency: <Psi|H|Psi> and <Psi|Psi> from the gradient tensors
    # must match the direct transfer-matrix evaluation.
    A0 = randn(ComplexF64, D, D, d)
    A, lambda = canonicalization(A0)
    lambdaInverse = 1.0 ./ max.(lambda, 1e-12)
    g = calculateMetricTensor(A, lambdaInverse, N)
    numH = calculateGradientTensor(A, hmat, lambdaInverse, N)
    vecAC = zeros(ComplexF64, d * D * D)
    for s in 1:d, a in 1:D, c in 1:D
        vecAC[a+(c-1)*D+(s-1)*D*D] = A[a, c, s] * lambda[c]
    end
    normGrad = real(dot(vecAC, g * vecAC))
    E = transferMatrix(A)
    normDirect = real(tr(E^N))
    energyGrad = real(dot(vecAC, numH)) / normGrad
    energyDirect = energyPuMPS(A, hmat, N)
    println("norm   (metric vs direct):  ", normGrad, "  ", normDirect)
    println("energy (gradient vs direct): ", energyGrad, "  ", energyDirect)

    # Optimise and compare to exact diagonalisation.
    # Note: with a fixed step size the optimisation is sensitive to the random
    # starting tensor -- some initialisations get stuck in local minima. A line
    # search and/or preconditioning from a smaller system (as in the paper)
    # makes this robust; here we simply use a fixed seed that converges.
    println("\nOptimising puMPS (N=$N, D=$D)...")
    Aopt, lam, en = fullGradientDescent(1500, 0.03, 1e-6, A0, hmat, N)
    eed = tfiRingED(h, J, N)
    println("puMPS energy/site = ", en / N)
    println("ED    energy/site = ", eed)
    println("error             = ", abs(en / N - eed))
end

if abspath(PROGRAM_FILE) == @__FILE__
    selftest()
end
