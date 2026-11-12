# TOML configuration: one seed experiment per process for comparable peak RSS.
include("ground_obc.jl")
include("optimizer_lbfgs.jl")
include("pair_work_control.jl")
using TOML
function runSeedComparison(configfile)
    study_source=pairSourceDigest()
    cfg=TOML.parsefile(configfile);resolve(p)=abspath(dirname(abspath(configfile)),p)
    out=resolve(cfg["output"]);ispath(out)&&error("refusing to overwrite $out")
    N=Int(cfg["N"]);D=Int(cfg["D"]);x=Float64(cfg["x"]);me=Float64(cfg["me"]);theta=Float64(cfg["theta"]);h=Int(cfg["hmax"])
    BLAS.set_num_threads(Int(get(cfg,"blas_threads",1)));Random.seed!(Int(get(cfg,"seed",20260930)))
    E0_THETA[]=theta;g=1/x;mass=(2sqrt(x)*me-.25)/x;M=2h+1
    gates=[blockedXXGateTwisted(2pi*(m-1)/M/N;mass=mass) for m in 1:M]
    terms=vcat(kernelTermsMerged(N,g),[(1.,liftingMPO(N÷2;qop=Zblk())...)])
    hm,h2,_=holonomyMats(h);start=time_ns();obc_seconds=0.;obc_energy=NaN;block=0;obc_stats=nothing;seed_method=:transport
    if haskey(cfg,"source")
        path=resolve(cfg["source"]);src=deserialize(path)
        # Deliberate size/mass/theta transport; other conventions must match.
        src.physics.x==x && src.physics.hmax==h||error("transport source x/flux mismatch")
        oldAs=src.result.As;length(oldAs)==M||error("transport source sectors")
        all(size(A)==(D,D,4) for A in oldAs)||error("size transport requires same D")
        As=deepcopy(oldAs);pairs=canonicalization.(As);As=first.(pairs);ls=last.(pairs)
    else
        seed_method=Symbol(get(cfg,"obc_seed_method","aligned"))
        seed_method in (:aligned,:legacy)||error("obc_seed_method must be aligned or legacy")
        obcD=Int(get(cfg,"obc_D",D));ns=Int(get(cfg,"obc_sweeps",12));obcD>=D||error("obc_D must cover target D")
        stats=Ref{Any}()
        tt=@timed obcColleagueDMRG(N,obcD,x,1.,me,theta,0.;nsweeps=ns,
            energy_tol=Float64(get(cfg,"obc_energy_tol",1e-10)),
            min_sweeps=Int(get(cfg,"obc_min_sweeps",1)),observer_stats=stats)
        obc_stats=only(stats[])
        obc_energy,psi=tt.value;obc_seconds=tt.time
        A,_,block=extractBulkSeed(psi,N,D,gates[1],terms,g;seed_method=seed_method)
        As=[copy(A) for _ in 1:M];pairs=canonicalization.(As);As=first.(pairs);ls=last.(pairs)
    end
    HH,NN=buildHNFixed(As,ls,gates,terms,g,N,hm,h2);seedenergy,_=solveC(HH,Matrix(NN))
    seed_seconds=(time_ns()-start)/1e9
    # Save the seed before expensive optimization so collection is resumable.
    pairAtomicSave(out*".seed.jls",(physics=(N=N,x=x,me=me,theta=theta,hmax=h),D=D,
        As=As,lams=ls,energy=seedenergy,seconds=seed_seconds,obc_stats=obc_stats,seed_method=seed_method,config=cfg))
    maxit=Int(get(cfg,"maxit",800));tol=Float64(get(cfg,"tol",1e-7))
    tt=@timed continueLBFGS(As,ls,gates,terms,g,N,hm,h2;maxit=maxit,tol=tol)
    result=(physics=(N=N,x=x,me=me,theta=theta,hmax=h),D=D,result=tt.value,config=cfg,
        seed_energy=seedenergy,seed_seconds=seed_seconds,obc_seconds=obc_seconds,obc_energy=obc_energy,seed_block=block,obc_stats=obc_stats,seed_method=seed_method,
        optimization_seconds=tt.time,seconds=(time_ns()-start)/1e9,peak_rss=Sys.maxrss(),source_code_sha256=study_source)
    pairAtomicSave(out,result)
    println("saved $out E/N=$(tt.value.E/N) eta=$(tt.value.eta) total=$(result.seconds) seconds")
    result
end
abspath(PROGRAM_FILE)==(@__FILE__)&&runSeedComparison(only(ARGS))
