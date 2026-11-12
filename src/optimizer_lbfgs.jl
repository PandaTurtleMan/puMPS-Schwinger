# Experimental continuation optimizer. Production Level-B defaults are unchanged.
# Gradients below are covectors in AL coordinates for E/(2Nb). Secant steps are
# vectors: they and the covectors require DIFFERENT gauge transport maps.
include("holonomy_fixed_solve.jl")

sectorDot(a, b) = sum(real(dot(x, y)) for (x, y) in zip(a, b))
sectorAxpy(a, t, b) = [x .+ t .* y for (x, y) in zip(a, b)]
alCovectors(grads, lams, D) =
    [reshape(g, D, D, :) .* reshape(l, 1, D, 1) for (g, l) in zip(grads, lams)]
function transportCovector(q, ML, MR)
    out = similar(q)
    for s in axes(q, 3)
        out[:, :, s] = ML' \ q[:, :, s] / MR'
    end
    return out
end

function naturalInverseAL(q, gs, lis, D)
    gc = [vec(v .* reshape(li, 1, D, 1)) for (v, li) in zip(q, lis)]
    steps, _ = naturalStepsBFixed(gs, gc, lis, D)
    return [-s for s in steps]
end

"""Continue a cached Level-B state with transported, naturally preconditioned L-BFGS.

This is an opt-in experimental solver. Positive-curvature secants only, a descent
check, Armijo backtracking, and a natural-gradient fallback protect each update.
Returned energy, tensors and eta all describe the same best accepted state.
"""
function continueLBFGS(As0, lams0, gates, terms, gC, Nsites, hm, h2;
                       maxit=50, memory=8, alpha=0.03, tol=1e-9, verbose=true)
    memory >= 1 || throw(ArgumentError("memory must be positive"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    D = size(As0[1], 1); M = length(As0); Nb = Nsites ÷ 2
    As = deepcopy(As0); lams = deepcopy(lams0)
    E, _, grads, gs, _, lis = gradientBFixed(As, lams, gates, terms, gC, Nsites, hm, h2; blockmetric=true)
    q = alCovectors(grads, lams, D)
    E0 = E; bestE = E; bestAs = deepcopy(As); bestLams = deepcopy(lams)
    _, eta = naturalStepsBFixed(gs, grads, lis, D); bestEta = eta
    S = Vector{typeof(As)}(); Y = Vector{typeof(q)}()
    nGrad = 1; nHN = 0; resets = 0; skipped = 0
    traj = NamedTuple[]; reason = "iteration limit reached"
    for it in 1:maxit
        if eta < tol
            reason = "gradient tolerance reached"; break
        end
        # Standard two-loop recursion, with the current natural metric as H0.
        v = deepcopy(q); n = length(S); a = zeros(n); rho = zeros(n)
        for j in n:-1:1
            rho[j] = 1 / sectorDot(Y[j], S[j])
            a[j] = rho[j] * sectorDot(S[j], v)
            v = sectorAxpy(v, -a[j], Y[j])
        end
        scale = alpha
        if n > 0
            hy = naturalInverseAL(Y[end], gs, lis, D)
            scale = clamp(sectorDot(S[end], Y[end]) / sectorDot(Y[end], hy), 1e-8, 10.0)
        end
        r = [scale .* x for x in naturalInverseAL(v, gs, lis, D)]
        for j in 1:n
            b = rho[j] * sectorDot(Y[j], r)
            r = sectorAxpy(r, a[j] - b, S[j])
        end
        p = [-x for x in r]
        accepted = false
        for attempt in 1:2
            slope = 2Nb * sectorDot(q, p)
            if attempt == 2 || !isfinite(slope) || slope >= 0
                empty!(S); empty!(Y); resets += 1
                p = [-alpha .* x for x in naturalInverseAL(q, gs, lis, D)]
                slope = 2Nb * sectorDot(q, p)
            end
            t = 1.0
            while t >= 2.0^-24
                trial = [canonicalizationG(As[m] .+ t .* p[m]) for m in 1:M]
                Ats = [x[1] for x in trial]; lts = [x[2] for x in trial]
                H, N = buildHNFixed(Ats, lts, gates, terms, gC, Nsites, hm, h2)
                Et, _ = solveC(H, Matrix(N)); nHN += 1
                if isfinite(Et) && Et <= E + 1e-4*t*slope + 1e-12
                    tv(vv) = [transportV(vv[m], trial[m][3], trial[m][4]) for m in 1:M]
                    tq(qq) = [transportCovector(qq[m], trial[m][3], trial[m][4]) for m in 1:M]
                    S = [tv(s) for s in S]; Y = [tq(y) for y in Y]
                    snew = tv([t .* x for x in p]); qold = tq(q)
                    As = Ats; lams = lts
                    E, _, grads, gs, _, lis = gradientBFixed(As, lams, gates, terms, gC, Nsites, hm, h2; blockmetric=true)
                    nGrad += 1; q = alCovectors(grads, lams, D)
                    ynew = sectorAxpy(q, -1, qold)
                    sy = sectorDot(snew, ynew)
                    if isfinite(sy) && sy > 1e-10 * sqrt(sectorDot(snew, snew)*sectorDot(ynew, ynew))
                        push!(S, snew); push!(Y, ynew)
                        if length(S) > memory; popfirst!(S); popfirst!(Y); end
                    else
                        skipped += 1
                    end
                    _, eta = naturalStepsBFixed(gs, grads, lis, D)
                    if E < bestE
                        bestE = E; bestAs = deepcopy(As); bestLams = deepcopy(lams); bestEta = eta
                    end
                    push!(traj, (it=it, E=E, eta=eta, nGrad=nGrad, nHN=nHN, step=t, history=length(S)))
                    verbose && (it <= 3 || it % 10 == 0) &&
                        (@printf("[lbfgs] it %d E/N=%.12f eta=%.3e nHN=%d step=%.3g history=%d\n", it, E/Nsites, eta, nHN, t, length(S)); flush(stdout))
                    accepted = true; break
                end
                t *= 0.5
            end
            accepted && break
        end
        if !accepted; reason = "line search exhausted"; break; end
    end
    verbose && (@printf("[lbfgs] %s: E/N %.12f -> %.12f nGrad=%d nHN=%d resets=%d skipped=%d\n", reason, E0/Nsites, bestE/Nsites, nGrad, nHN, resets, skipped); flush(stdout))
    return (E0=E0, E=bestE, As=bestAs, lams=bestLams, eta=bestEta,
            nGrad=nGrad, nHN=nHN, resets=resets, skipped=skipped, traj=traj, reason=reason)
end
