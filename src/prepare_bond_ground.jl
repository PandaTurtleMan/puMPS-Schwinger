# CACHE N x m/e theta hmax D maxit OUTPUT
using LinearAlgebra,Serialization,Random
include("optimizer_lbfgs.jl")
function prepareBondGround(args)
    length(args)==9 || error("CACHE N x m/e theta hmax D maxit OUTPUT")
    cache=args[1];N=parse(Int,args[2]);x=parse(Float64,args[3]);me=parse(Float64,args[4]);theta=parse(Float64,args[5])
    h=parse(Int,args[6]);D=parse(Int,args[7]);maxit=parse(Int,args[8]);out=args[9]
    ispath(out) && error("refusing to overwrite $out")
    source=deserialize(cache);oldAs=hasproperty(source,:result) ? source.result.As : source[2]
    BLAS.set_num_threads(1);E0_THETA[]=theta;rng=Xoshiro(751+D)
    pairs=map(oldAs) do A
        oldD=size(A,1);B=zeros(ComplexF64,D,D,4);n=min(D,oldD)
        B[1:n,1:n,:]=A[1:n,1:n,:]
        if D>oldD
            # Preserve the optimized old block. A contracting new diagonal block
            # plus weak off-diagonal couplings permits bond growth without
            # perturbing every entry of the already converged tensor.
            B[oldD+1:end,oldD+1:end,:]=0.05 .* randn(rng,ComplexF64,D-oldD,D-oldD,4)
            B[1:oldD,oldD+1:end,:]=1e-3.*randn(rng,ComplexF64,oldD,D-oldD,4)
            B[oldD+1:end,1:oldD,:]=1e-3.*randn(rng,ComplexF64,D-oldD,oldD,4)
        end
        canonicalization(B)
    end
    As=first.(pairs);ls=last.(pairs);mass=(2sqrt(x)*me-.25)/x;gC=1/x
    gates=[blockedXXGateTwisted(2pi*(m-1)/(2h+1)/N;mass=mass) for m in 1:2h+1]
    terms=vcat(kernelTermsMerged(N,gC),[(1.,liftingMPO(N÷2;qop=Zblk())...)])
    hm,h2,_=holonomyMats(h)
    println("N=$N D=$D x=$x me=$me theta=$theta h=$h starting from $cache; maxit=$maxit");flush(stdout)
    timed=@timed continueLBFGS(As,ls,gates,terms,gC,N,hm,h2;maxit=maxit,tol=1e-7)
    serialize(out,(physics=(N=N,x=x,me=me,theta=theta,hmax=h),D=D,source=cache,
                    result=timed.value,seconds=timed.time,tolerance=1e-7))
    println("saved $out; seconds=$(timed.time) eta=$(timed.value.eta) E=$(timed.value.E)")
end
abspath(PROGRAM_FILE)==(@__FILE__) && prepareBondGround(ARGS)
