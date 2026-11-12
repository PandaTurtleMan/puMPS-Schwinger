# julia --project=matrixfree_env -t 2 run_matrixfree_excitation.jl 
#   CACHE N x m_over_e theta hmax MOMENTA NEV OUTPUT [eigtol]
include("holonomy_matrixfree.jl")
using Serialization
function runMatrixFreeExcitation(args)
    length(args) in (9,10) || error("CACHE N x m_over_e theta hmax MOMENTA NEV OUTPUT [eigtol]")
    cache=args[1]; N=parse(Int,args[2]); x=parse(Float64,args[3]); me=parse(Float64,args[4])
    theta=parse(Float64,args[5]); h=parse(Int,args[6]); ks=parse.(Int,split(args[7],","))
    nev=parse(Int,args[8]); out=args[9]; tol=length(args)==10 ? parse(Float64,args[10]) : 1e-8
    ispath(out) && error("refusing to overwrite $out")
    state=deserialize(cache)
    # Accept both historical (E,As,lambdas) caches and L-BFGS continuation results.
    E,As,ls=hasproperty(state,:result) ? (state.result.E,state.result.As,state.result.lams) : state
    M=2h+1; length(As)==M || error("cache sector count disagrees with hmax")
    BLAS.set_num_threads(1); E0_THETA[]=theta
    gC=1/x; mass=(2sqrt(x)*me-.25)/x; lam=1.
    terms=vcat(kernelTermsMerged(N,gC),[(lam,liftingMPO(N÷2;qop=Zblk())...)])
    gates=[blockedXXGateTwisted(2pi*(m-1)/M/N;mass=mass) for m in 1:M]
    hm,h2,_=holonomyMats(h); H,NN=buildHNFixed(As,ls,gates,terms,gC,N,hm,h2)
    check,_=solveC(H,Matrix(NN))
    isapprox(check,E;atol=1e-8,rtol=1e-10) || error("cache energy disagrees with supplied physics")
    println("N=$N D=$(size(As[1],1)) x=$x m/e=$me theta=$theta h=$h E/N=$(check/N) momenta=$ks");flush(stdout)
    stat=@timed blochDispersionMatrixFree(As,ls,N,gC,lam,h;mass=mass,momenta=ks,
                                         nev=nev,eigtol=tol,returninfo=true,verbose=true)
    levels,ranks,diagnostics=stat.value
    serialize(out,(physics=(N=N,x=x,me=me,theta=theta,hmax=h,lam=lam),
                   settings=(momenta=ks,nev=nev,cutoff=1e-10,eigtol=tol,krylovdim=max(160,4nev),maxiter=50,seed=821),
                   ground_energy=E,levels=levels,ranks=ranks,diagnostics=diagnostics,seconds=stat.time))
    println("saved $out; elapsed $(stat.time) s; peak RSS $(Sys.maxrss()/2^30) GiB")
end
abspath(PROGRAM_FILE)==(@__FILE__) && runMatrixFreeExcitation(ARGS)
