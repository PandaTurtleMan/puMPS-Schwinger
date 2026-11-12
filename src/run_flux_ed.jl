# N x m/e theta hmax comma-separated-momenta levels output-directory
include("ed_flux_orbits.jl")
using KrylovKit,Printf
function fluxBlockEigenpairs(action,n,nev;real_start=false,seed=734,tol=1e-9,krylovdim=parse(Int,get(ENV,"ED_KRYLOVDIM","80")),maxiter=200)
    for attempt in 0:2
        rng=Xoshiro(seed+attempt);T=real_start ? Float64 : ComplexF64
        start=KrylovKit.Block([randn(rng,T,n) for _ in 1:min(n,nev)])
        alg=KrylovKit.BlockLanczos(;tol=tol,krylovdim=min(n,max(krylovdim,2nev+1)),maxiter=maxiter,eager=false)
        result=try
            KrylovKit.eigsolve(action,start,nev,:SR,alg)
        catch err
            upstream=any(f->occursin("blocklanczos.jl",String(f.file)),stacktrace(catch_backtrace()))
            if attempt<2 && upstream && err isa DomainError && err.val isa Real && -1e-12<err.val<0
                println("ED block Lanczos retry after internal residual roundoff");flush(stdout)
                GC.gc();continue
            end
            rethrow()
        end
        return result
    end
end
function runFluxED(args)
    length(args)==8 || error("N x m/e theta hmax MOMENTA levels OUTPUT_DIR")
    N=parse(Int,args[1]);x=parse(Float64,args[2]);me=parse(Float64,args[3]);theta=parse(Float64,args[4])
    h=parse(Int,args[5]);ks=parse.(Int,split(args[6],','));nev=parse(Int,args[7]);dir=args[8]
    all(k->0<=k<N÷2,ks) && allunique(ks) && nev>0 && h>=0 && x>0 || error("invalid ED settings")
    mkpath(dir);mass=(2sqrt(x)*me-.25)/x;BLAS.set_num_threads(2)
    settings=(N=N,x=x,me=me,theta=theta,hmax=h,mass=mass)
    println("ED settings=$settings threads=$(Threads.nthreads()), BLAS=2");flush(stdout)
    ctime=@elapsed c=prepareFluxOrbits(N,h;gC=1/x,mass=mass,theta=theta,verbose=true)
    println("ED orbit preparation=$ctime s; context=$(Base.summarysize(c)/2^30) GiB");flush(stdout)
    for k in ks
        outfile=joinpath(dir,"ed_k$(k).jls");ispath(outfile) && error("refusing to overwrite $outfile")
        b=fluxMomentumBasis(c,k);n=length(b.keep);calls=Ref(0);started=time()
        function action(v)
            calls[]+=1
            calls[]%10==0 && (println("ED k=$k product=$(calls[]) elapsed=$(round(time()-started;digits=1)) s");flush(stdout))
            fluxMomentumAction(c,b,v)
        end
        elapsed=@elapsed vals,vecs,info=fluxBlockEigenpairs(action,n,nev;
            real_start=eltype(b.phase)<:Real,seed=734+k)
        info.converged>=nev || error("ED k=$k converged only $(info.converged)/$nev")
        vals=real.(vals[1:nev]);vecs=vecs[1:nev]
        projected=[norm(action(v)-e*v) for (e,v) in zip(vals,vecs)]
        outside=[sqrt(sum(abs2(v[j])*c.variance[id] for (j,id) in enumerate(b.keep))) for v in vecs]
        full=hypot.(projected,outside)
        # The finite flux window breaks translation at its edge. Report both
        # errors; never certify a full-H eigenpair from a projected residual.
        certified=all(<=(1e-7),full)
        println("ED k=$k energies=$vals projected=$projected outside=$outside full=$full certified=$certified seconds=$elapsed");flush(stdout)
        serialize(outfile,(physics=settings,k=k,levels=vals,projected_residuals=projected,
            outside_residuals=outside,full_residuals=full,certified=certified,seconds=elapsed,
            preparation_seconds=ctime,matvecs=calls[],dimension=n,peak_rss=Sys.maxrss(),
            solver=:block_lanczos,block_size=nev))
        open(joinpath(dir,"ed_levels.csv"),isfile(joinpath(dir,"ed_levels.csv")) ? "a" : "w") do io
            position(io)==0 && println(io,"N,x,me,theta,hmax,k,level,E,projected_residual,full_residual,certified,seconds")
            for j in eachindex(vals)
                println(io,"$N,$x,$me,$theta,$h,$k,$(j-1),$(vals[j]),$(projected[j]),$(full[j]),$certified,$elapsed")
            end
        end
        vecs=nothing;GC.gc()
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && runFluxED(ARGS)
