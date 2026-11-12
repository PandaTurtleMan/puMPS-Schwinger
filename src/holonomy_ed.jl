# Dense ED reference for the FULL Schwinger model with holonomy restored
# (Schwinger notes Eqs 57-74), in the |theta_m> holonomy basis, neutral spin
# sector.  With external theta=0:
#   H = sum_m |m><m| (x) [ H_XX(theta_m) + gC*H_C1 + lambda Q^2 ]           (diag in m)
#     + gC sum_{m,m'} |m><m'| (x) [ 2N E0 h_{mm'} + N h2_{mm'} ]            (couples m; diag in spin)
# where theta_m = 2pi m/(2hmax+1), H_XX(theta_m) twisted by theta_m/N per bond (Eq 56),
# H_C1 = N E0^2 - 1/(2N) sum K q q, and h_{mm'},h2_{mm'} are Eqs 71-72.
# hmax=0 must reproduce Warm-Up Model 2 (column C).
include("puMPS_schwinger_step3.jl")     # guarded: staggeredKernelDiag, e0DiagSite, denseQtot, zeroChargeIdx, xxG
using LinearAlgebra, Printf

twistedXXgate(phi) = (sp=ComplexF64[0 1;0 0]; sm=ComplexF64[0 0;1 0];
                      kron(sp,sm).*exp(im*phi) .+ kron(sm,sp).*exp(-im*phi))

# E0 value per config (the operator itself, not N E0^2), incl. external theta
function e0OperatorDiag(N; theta=0.0)
    out = zeros(2^N)
    for lin in 0:2^N-1
        x = lin; e = 0.0
        for i in 1:N
            s = (x % 2 == 0) ? 1 : -1
            qi = 0.5 * (s + 1) - (isodd(i - 1) ? 1.0 : 0.0)
            e += (i - 1) * qi; x ÷= 2
        end
        out[lin+1] = mod(-e / N, 1.0) + theta / (2π)
    end
    return out
end

# holonomy matrices in the theta basis, both by direct DFT and by the closed forms (Eqs 71-72)
function holonomyMats(hmax)
    M = 2hmax + 1; hs = collect(-hmax:hmax)
    hm = zeros(ComplexF64, M, M); h2 = zeros(ComplexF64, M, M)
    hmc = similar(hm); h2c = similar(h2)
    for m in 0:M-1, mp in 0:M-1
        th = 2π * (m - mp) / M
        hm[m+1, mp+1]  = sum(h * exp(im * th * h) for h in hs) / M
        h2[m+1, mp+1]  = sum(h^2 * exp(im * th * h) for h in hs) / M
        if m == mp
            hmc[m+1, mp+1] = 0; h2c[m+1, mp+1] = hmax * (hmax + 1) / 3
        else
            d = m - mp
            hmc[m+1, mp+1] = -0.5im * (-1)^d / sin(π * d / M)
            h2c[m+1, mp+1] = 0.5 * (-1)^d * cos(π * d / M) / sin(π * d / M)^2
        end
    end
    return hm, h2, max(norm(hm - hmc), norm(h2 - h2c))
end

# H_XX in sector theta_m: SPREAD gauge (twist theta_m/N on all bonds, Eq 56) or
# CONCENTRATED (twist e^{i theta_m} on the single boundary bond N-1,0 -- the exact
# finite-holonomy form, consistent with the un-gauged h-hat of Eq 71).
function HxxSector(N, theta_m; spread=true)
    if spread
        return ringHamFromGate(twistedXXgate(theta_m / N), N)
    else
        sp = ComplexF64[0 1; 0 0]; sm = ComplexF64[0 0; 1 0]
        H = ringHamFromGate(xxG(), N)                         # untwisted ring
        # replace the wrap bond (site N,site 1) hop phases: 1 -> e^{i theta_m}
        bp = opAt(sp, N, N) * opAt(sm, 1, N)                  # sigma^+_N sigma^-_1
        H .+= (exp(im * theta_m) - 1) .* bp .+ (exp(-im * theta_m) - 1) .* bp'
        return H
    end
end

# occupation moment  w(c) = sum_j j n_j,  n_j = (sigma^z_j+1)/2 in {0,1}
function wDiag(N)
    out = zeros(2^N)
    for lin in 0:2^N-1
        x = lin; w = 0.0
        for i in 1:N; w += (i - 1) * ((x % 2 == 0) ? 1.0 : 0.0); x ÷= 2; end
        out[lin+1] = w
    end
    return out
end

# mode = :spread_bare (Eq 56 as written), :concentrated (correct), :spread_dressed (spread + U_m^dag U_m')
function holonomyED_neutral(N, gC, hmax; lam=1.0, theta_ext=0.0, mode=:concentrated)
    idx = zeroChargeIdx(N); n = length(idx); M = 2hmax + 1
    kerd = staggeredKernelDiag(N)[idx]
    e0d  = e0DiagSite(N; theta=theta_ext)[idx]
    e0o  = e0OperatorDiag(N; theta=theta_ext)[idx]
    Q2d  = diag(denseQtot(N)^2)[idx]
    w    = wDiag(N)[idx]
    hm, h2, _ = holonomyMats(hmax)
    spread = (mode != :concentrated)
    H = zeros(ComplexF64, n * M, n * M)
    for m in 0:M-1
        theta_m = 2π * m / M
        Hxx = HxxSector(N, theta_m; spread=spread)[idx, idx]
        H[m*n+1:(m+1)*n, m*n+1:(m+1)*n] .+= Hxx .+ Diagonal(gC .* (kerd .+ e0d) .+ lam .* Q2d)
    end
    for m in 0:M-1, mp in 0:M-1
        dth = 2π * (mp - m) / M
        dress = (mode == :spread_dressed) ? exp.(im .* dth .* w ./ N) : ones(ComplexF64, n)
        cd = gC .* (2N .* e0o .* hm[m+1, mp+1] .+ N .* h2[m+1, mp+1]) .* dress
        H[m*n+1:(m+1)*n, mp*n+1:(mp+1)*n] .+= Diagonal(cd)
    end
    H = (H + H') / 2
    return minimum(real(eigvals(Hermitian(H))))
end

# column-C reference (hmax=0 zero-holonomy) for comparison
function colC(N, gC)
    idx = zeroChargeIdx(N)
    H = ringHamFromGate(xxG(), N) .+ gC .* Diagonal(staggeredKernelDiag(N) .+ e0DiagSite(N))
    return minimum(real(eigvals(Hermitian((H[idx,idx]+H[idx,idx]')/2))))
end

if abspath(PROGRAM_FILE) == @__FILE__
for N in (8,)
    @printf("N=%d  neutral-sector ED with holonomy   (E0/site; theta_ext=0)\n", N)
    for mode in (:spread_bare, :concentrated, :spread_dressed)
        @printf("  --- mode: %s ---\n", mode)
        @printf("  gC    hmax=0        hmax=1        hmax=2        hmax=3\n")
        for gC in (0.30, 0.75)
            e = [holonomyED_neutral(N, gC, h; mode=mode) / N for h in 0:3]
            @printf("  %4.2f  %.8f   %.8f   %.8f   %.8f\n", gC, e...)
            flush(stdout)
        end
    end
end
end
