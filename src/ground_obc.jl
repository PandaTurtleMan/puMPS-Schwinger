# Converged holonomy puMPS ground state, warm-started from the colleague's OBC
# Schwinger DMRG bulk tensor. Parameterised port of obc_warmstart_levelb.jl
# (which hardcodes N=20, D=8 and the colleague's x=10/theta=pi point).
#
# WARM ONLY: cold is not run. Cold is known to stall every sector at the step cap
# (eta~1e-4..1e-2) at N>=20, so at N=64 it would waste hours to produce a worse seed.
#
# Output is the 3-tuple (E, As, lams) exactly as levelB_GS_fixed returns it, which is
# the established ground-cache format (cf. levelb_massless_x4_N28_D10_h3_m800.jls).
# Provenance goes to OUTPUT.meta.jls so the primary file stays directly loadable by
# the existing readers (`ec,As,ls=deserialize(path)`; prepare_bond_ground's source[2]).
#
# Usage: OPENBLAS_NUM_THREADS=4 julia -t 1 ground_obc.jl N D x m_over_e theta hmax warmSteps maxit OUTPUT
using ITensors, ITensorMPS
using LinearAlgebra, Random, Printf, Serialization
include("OBC_Schwinger_dmrg.jl")    # Schwinger, schwinger_gauge, get_MPO_neutral
include("holonomy_fixed_solve.jl")  # levelB_GS_fixed, E0_THETA, blockedXXGateTwisted, ...
include("obc_seed_alignment.jl")

function obcColleagueDMRG(N, D, x, y, mass_phys, theta, g; nsweeps=12, energy_tol=1e-10, min_sweeps=1, observer_stats=nothing)
    sites = siteinds("S=1/2", N; conserve_qns=false)
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = MPS(sites, state)
    os = OpSum()
    for j in 1:(N-1)
        os += 1, "S+", j, "S-", j+1
        os += 1, "S-", j, "S+", j+1
        os += -4*g, "Sz", j, "Sz", j+1
    end
    Hxy = MPO(os, sites)
    os = OpSum(); for j in 1:N; os += (-1)^(j), "Sz", j; end
    Hm = MPO(os, sites)
    Hc = get_MPO_neutral((ham,n) -> schwinger_gauge(ham,n,N,theta), sites, 3, false)
    E, psi = Schwinger(N, x, y, mass_phys, theta, nsweeps, D, 0, psi0, Hc, Hm, Hxy, 1; energy_tol=energy_tol, min_sweeps=min_sweeps, observer_stats=observer_stats)
    return E[1], psi[1]
end

# Match the independent virtual spaces before closing a central OBC cell on a
# ring. OBC uses one-based staggering, while puMPS cells are even-odd with the
# even bit fast: select OBC sites (2b,2b+1). Keep the old path only for controls.
function extractBulkSeed(psi, N, D, gate, mpoTerms, gC; seed_method=:aligned)
    if seed_method==:aligned
        A,meta=extractAlignedBulkSeed(psi,N,D)
        Ac,lam=canonicalization(A);lamInv=1.0 ./ max.(lam,1e-12)
        _,_,eR=schwingerGradientFixed(Ac,lam,lamInv,gate,mpoTerms,gC,N)
        return A,real(eR)/N,meta.block
    end
    seed_method==:legacy||throw(ArgumentError("seed_method must be :aligned or :legacy"))
    # Compress the entire OBC state in a consistent Schmidt basis before taking
    # a bulk block. Slicing two independently gauged virtual legs is not an SVD
    # truncation and can destroy the seed. Never mutate the caller's DMRG state.
    psi=deepcopy(psi)
    maximum(linkdims(psi))>D && truncate!(psi;maxdim=D,cutoff=0.0)
    Nb = N ÷ 2; center = Nb ÷ 2
    blk(B0) = begin
        se_site=2B0-1; so_site=2B0; orthogonalize!(psi, so_site+1)
        lb=linkind(psi,se_site-1); rb=linkind(psi,so_site); se=siteind(psi,se_site); so=siteind(psi,so_site)
        raw=Array(psi[se_site]*psi[so_site], lb, se, so, rb); D1,_,_,D2=size(raw); Dt=D
        max(D1,D2)<=D || error("OBC compression did not enforce target bond dimension")
        A=zeros(ComplexF64, Dt,Dt,4); for pe in 1:2,po in 1:2; A[1:D1,1:D2,(pe-1)+2*(po-1)+1] .= raw[:,pe,po,:]; end
        (A,D1,D2)
    end
    chosenA = nothing; chosenB = clamp(center,2,Nb-1)
    for B0 in sort(collect(2:(Nb-1)); by=b->abs(b-center))
        A,D1,D2 = blk(B0)
        if D1==D && D2==D; chosenA=A; chosenB=B0; break; end
    end
    chosenA === nothing && ((chosenA,_,_) = blk(chosenB))
    Ac,lam = canonicalization(chosenA); lamInv = 1.0 ./ max.(lam,1e-12)
    _,_,eR = schwingerGradientFixed(Ac,lam,lamInv,gate,mpoTerms,gC,N)
    return chosenA, real(eR)/N, chosenB
end

function groundOBC(args)
    length(args) in (9,11) || error("usage: N D x m_over_e theta hmax warmSteps maxit OUTPUT [OBC_D OBC_SWEEPS]")
    N=parse(Int,args[1]); D=parse(Int,args[2]); x=parse(Float64,args[3]); me=parse(Float64,args[4])
    theta=parse(Float64,args[5]); hmax=parse(Int,args[6])
    warmSteps=parse(Int,args[7]); maxit=parse(Int,args[8]); out=args[9]
    obcD=length(args)==11 ? parse(Int,args[10]) : D
    obcSweeps=length(args)==11 ? parse(Int,args[11]) : 12
    D>0 && obcD>=D && obcSweeps>0 || error("require OBC_D >= D > 0 and positive sweeps")
    ispath(out) && error("refusing to overwrite $out")
    iseven(N) && N>=6 || error("N must be even and >=6")

    # Colleague units -> puMPS holonomy. gC=y/x with y=1; mass=(2 sqrt(x) m/e - 1/4)/x.
    y = 1.0; gC = y/x; massP = (2*sqrt(x)*me - 0.25)/x; lam = 1.0
    Nb = N ÷ 2; M = 2hmax + 1
    E0_THETA[] = theta
    @printf("=== N=%d D=%d hmax=%d  x=%g m/e=%g -> gC=%.6f mass=%.6f theta=%.6f (M=%d) ===\n",
            N, D, hmax, x, me, gC, massP, theta, M)
    @printf("Julia threads=%d BLAS=%d; warmSteps=%d maxit=%d\n",
            Threads.nthreads(), BLAS.get_num_threads(), warmSteps, maxit); flush(stdout)

    gate0 = blockedXXGateTwisted(0.0; mass=massP)
    Wl, vLl, vRl = liftingMPO(Nb; qop=Zblk())
    mpoTerms = vcat(kernelTerms(N, gC), [(lam, Wl, vLl, vRl)])

    tdmrg = @elapsed ((Eobc, psi) = obcColleagueDMRG(N, obcD, x, y, me, theta, 0.0;nsweeps=obcSweeps))
    @printf("OBC DMRG done in %.1f s: E=%.9f, E/site=%.9f (colleague units)\n",
            tdmrg, Eobc, Eobc/N); flush(stdout)
    tseed = @elapsed ((Aseed, eSeed, bestB) = extractBulkSeed(psi, N, D, gate0, mpoTerms, gC))
    @printf("seed extracted in %.1f s: block B0=%d, seed ring E/site=%.9f\n",
            tseed, bestB, eSeed); flush(stdout)

    println("\n--- WARM Level-B (seedAs = OBC bulk in all $M sectors) ---"); flush(stdout)
    tw = @elapsed ((E, As, lams) = levelB_GS_fixed(; Nsites=N, hmax=hmax, D=D, gC=gC, lam=lam,
                                                   mass=massP, seedAs=[copy(Aseed) for _ in 1:M],
                                                   warmSteps=warmSteps, maxit=maxit, verbose=true))
    @printf("\n>>> WARM final E=%.10f  E/site=%.10f  wall=%.1f s\n", E, E/N, tw); flush(stdout)

    serialize(out, (E, As, lams))
    serialize(out*".meta.jls", (physics=(N=N,x=x,me=me,theta=theta,hmax=hmax,lam=lam),D=D,
        gC=gC,mass=massP,E=E,E_per_site=E/N,obc_E=Eobc,obc_E_per_site=Eobc/N,
        seed_ring_E_per_site=eSeed,seed_block=bestB,seed_method=:aligned,seed_physical_sites=(2bestB,2bestB+1),
        seconds=(dmrg=tdmrg,seed=tseed,levelb=tw),warmSteps=warmSteps,maxit=maxit,obc_D=obcD,obc_sweeps=obcSweeps,
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads()))
    @printf("saved %s (E,As,lams) and %s.meta.jls\n", out, out); flush(stdout)
    E0_THETA[] = 0.0
    return E/N
end

abspath(PROGRAM_FILE)==(@__FILE__) && groundOBC(ARGS)
