# One-insertion (full local core, q=0) momentum-resolved spectrum from a saved ground cache,
# using the optimized pair-code assembly (twoParticleMomentumMatrices with no channels) and
# its Ritz solver. This is the same variational space as the dense/matrix-free Bloch solvers
# (dD^2 tensor units per sector; reproduces the historical one-insertion CSVs to ~5e-11) but
# with the D^5 kernels, so it is the right tool at moderate D.
# Levels are ABSOLUTE energies; at k=0 the lowest level is the vacuum itself.
#   julia --project=matrixfree_env -t T run_local_excitation.jl CACHE N x m_over_e theta hmax MOMENTA NEV OUTPUT
include("run_pair_bondcheck.jl")
using Serialization, LinearAlgebra
function solveDuration(s)   # "1h05m" / "4m30s" (kept local: this script must not depend on the assembly file's helpers)
    s=round(Int,max(s,0.));h,r=divrem(s,3600);m,sec=divrem(r,60)
    h>0 ? "$(h)h$(lpad(m,2,'0'))m" : "$(m)m$(lpad(sec,2,'0'))s"
end
function runLocalExcitation(args)
    length(args)==9 || error("CACHE N x m_over_e theta hmax MOMENTA NEV OUTPUT")
    cache=args[1]; N=parse(Int,args[2]); x=parse(Float64,args[3]); me=parse(Float64,args[4])
    theta=parse(Float64,args[5]); h=parse(Int,args[6]); ks=parse.(Int,split(args[7],","))
    nev=parse(Int,args[8]); out=args[9]
    ispath(out) && error("refusing to overwrite $out")
    BLAS.set_num_threads(1); E0_THETA[]=theta
    source=deserialize(cache); result=hasproperty(source,:result) ? source.result : nothing
    E,As,ls=isnothing(result) ? source : (result.E,result.As,result.lams)
    D=size(first(As),1); gC=1/x; mass=(2sqrt(x)*me-.25)/x; M=2h+1
    length(As)==M || error("cache sector count disagrees with hmax")
    hm,h2,_=holonomyMats(h)
    gates=[blockedXXGateTwisted(2pi*(m-1)/M/N;mass=mass) for m in 1:M]
    terms=vcat(kernelTermsMerged(N,gC),[(1.,liftingMPO(N÷2;qop=Zblk())...)])
    HH,NN=buildHNFixed(As,ls,gates,terms,gC,N,hm,h2); check,_=solveC(HH,Matrix(NN))
    isapprox(E,check;atol=1e-8,rtol=1e-10) || error("cache energy disagrees with supplied physics")
    println("N=$N D=$D x=$x m/e=$me theta=$theta h=$h E/N=$(check/N) momenta=$ks (local core, q=0)"); flush(stdout)
    # Reflection symmetry: E(k) = E(L-k) exactly (verified against ED and on the N=20 puMPS tables),
    # so only the reduced momenta min(k, L-k) are assembled and solved; the rest are mirrored.
    L=N÷2; kred=unique([min(k,L-k) for k in ks]); kmap=Dict(k=>findfirst(==(min(k,L-k)),kred) for k in ks)
    length(kred)<length(ks) && println("reduced momenta by reflection: $kred")
    ta=@timed twoParticleMomentumMatrices(As,[Array{ComplexF64,3}[] for _ in As],N,gC,1.,h;
        mass=mass,theta=theta,momenta=kred,separations=Int[],verbose=true)
    base=ta.value
    # Solver choice: the default block-Krylov path is fast when the low spectrum is sparse (critical point)
    # but converges erratically on the dense two-particle continuum of gapped points (10x slower at D>=12).
    # A file named SOLVER_DENSE next to this script, or PUMPS_SOLVER=dense, selects the dense LAPACK
    # subset eigensolver (W'HW formed explicitly, lowest nev by eigen(...,1:nev)) with BLAS threads.
    dense = isfile(joinpath(@__DIR__,"SOLVER_DENSE")) || get(ENV,"PUMPS_SOLVER","")=="dense"
    if dense
        groups=[collect(base.offsets[m]+1:base.offsets[m+1]) for m in 1:length(base.bases)]
        BLAS.set_num_threads(max(1,Threads.nthreads()))
        # One line per solved momentum with an ETA (momenta are solved one after another and cost about the same).
        nsolve=length(kred);t0solve=time()
        ts=@timed map(enumerate(zip(base.H,base.N))) do (i,(H,NN))
            ti=time()
            sol=solveTwoParticleRitz(H,NN;nev=nev,cutoff=1e-10,residualtol=1e-7,normblocks=groups,eigensolve=nothing)
            el=time()-t0solve;eta=el/i*(nsolve-i)
            println("[solve] k=$(kred[i]) ($i/$nsolve) $(round(time()-ti;digits=1)) s; elapsed $(solveDuration(el))",
                    i<nsolve ? "  ETA $(solveDuration(eta))  (finish ~$(Libc.strftime("%Y-%m-%d %H:%M",time()+eta)))" : "  done");flush(stdout)
            sol
        end
        BLAS.set_num_threads(1)
    else
        ts=@timed solveMomentumMatrices(base;nev=nev)
    end
    sols=[ts.value[kmap[k]] for k in ks]   # expand reduced solutions back to the requested momenta
    levels=[Vector{Float64}(s.levels) for s in sols]
    ranks=[s.rank for s in sols]
    residuals=[Vector{Float64}(s.residuals) for s in sols]
    serialize(out,(physics=(N=N,x=x,me=me,theta=theta,hmax=h,lam=1.),D=D,
        settings=(momenta=ks,nev=nev,cutoff=1e-10,residualtol=1e-7,solver=:local_core_q0,eigensolver=dense ? :dense : :krylov,reduced_momenta=kred),
        ground_energy=E,levels=levels,ranks=ranks,residuals=residuals,
        assembly_seconds=ta.time,solve_seconds=ts.time,seconds=ta.time+ts.time))
    println("saved $out; assembly $(ta.time) s; solve $(ts.time) s; elapsed $(ta.time+ts.time) s; peak RSS $(Sys.maxrss()/2^30) GiB; ranks=$ranks")
    for (ki,k) in enumerate(ks); println("k=$k levels-E0: ",round.(levels[ki].-E,digits=8)," maxres=$(maximum(residuals[ki]))"); end
end
abspath(PROGRAM_FILE)==(@__FILE__) && runLocalExcitation(ARGS)
