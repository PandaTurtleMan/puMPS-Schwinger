# CACHE N x m/e theta hmax channels MOMENTA OUTPUT_DIR
include("holonomy_twoparticle.jl")
include("holonomy_pairkrylov.jl")
using Serialization,Printf
include("particle_channel_selection.jl")

function channelsFromLocalMatrices(As,mats;nchannels=2,nmodes=2,cutoff=1e-10,selection=:euclidean)
    selection in (:euclidean,:norm_balanced) || throw(ArgumentError("unknown channel selection"))
    M=length(As);locals=leftInsertionBasis.(As);newOffsets=cumsum([0;length.(locals)])
    V=spzeros(ComplexF64,last(mats.offsets),last(newOffsets))
    for m in 1:M
        V[mats.offsets[m]+1:mats.offsets[m+1],newOffsets[m]+1:newOffsets[m+1]]=hcat(vec.(locals[m])...)
    end
    groups=[collect(newOffsets[m]+1:newOffsets[m+1]) for m in 1:M]
    samples=[Vector{ComplexF64}[] for _ in 1:M];samplelevels=Vector{Float64}[]
    for (i,k) in enumerate(mats.momenta)
        H=Hermitian(Matrix(V'*(Matrix(mats.H[i])*V)));N=Hermitian(sparse(V'*(sparse(mats.N[i])*V)))
        whiten=twoParticleWhitener(N,cutoff,groups);W=whiten.W
        ground=zeros(ComplexF64,size(W,1),M)
        if k==0
            for m in 1:M;ground[newOffsets[m]+1,m]=1;end
        end
        overlap=W'*(sparse(N)*ground)
        n=size(W,2)
        if n>600
            f=svd(overlap);r=k==0 ? count(>(1e-10*maximum(f.S;init=0.)),f.S) : 0
            house=r>0 ? qr(f.U[:,1:r]) : nothing
            function expand(v)
                r==0 && return v
                z=zeros(ComplexF64,n);z[r+1:end]=v;lmul!(house.Q,z);z
            end
            function reduce(v)
                r==0 && return v
                lmul!(house.Q',v);v[r+1:end]
            end
            action=v->reduce(W'*(H*(W*expand(v))))
            vals,u,_=pairKrylovSolve(action,n-r,min(nmodes,n-r),1e-7)
            maximum(norm(action(u[:,j])-vals[j]*u[:,j]) for j in eachindex(vals))<=1e-7 || error("particle-channel residual")
            C=W*hcat([expand(u[:,j]) for j in axes(u,2)]...);push!(samplelevels,vals)
        else
            Q=k==0 ? nullspace(overlap') : Matrix{ComplexF64}(I,n,n)
            Z=W*Q; reduced=Z'*(Matrix(H)*Z)
            f=eigen(Hermitian((reduced+reduced')/2),1:min(nmodes,size(reduced,1)))
            C=Z*f.vectors;push!(samplelevels,f.values)
        end
        for m in 1:M,j in axes(C,2)
            v=zeros(ComplexF64,length(As[m]))
            for a in 2:length(locals[m]);v.+=C[newOffsets[m]+a,j].*vec(locals[m][a]);end
            norm(v)>1e-12 && push!(samples[m],v/norm(v))
        end
    end
    channels=Vector{Array{ComplexF64,3}}[];singular=Vector{Float64}[];diagnostics=NamedTuple[]
    for m in 1:M
        isempty(samples[m]) && error("empty particle samples")
        inds=mats.offsets[m]+1:mats.offsets[m+1]
        metric=selection==:euclidean ? nothing : sum(Matrix(n[inds,inds]) for n in mats.N)/length(mats.N)
        selected=selectParticleChannels(samples[m],size(As[m]),nchannels;metric=metric)
        push!(channels,selected.channels);push!(singular,selected.singular_values)
        push!(diagnostics,(metric_rank=selected.metric_rank,discarded_weight=selected.discarded_weight))
    end
    (channels=channels,singular_values=singular,sample_levels=samplelevels,selection=selection,diagnostics=diagnostics)
end

function solveMomentumMatrices(mats;nev=parse(Int,get(ENV,"PAIR_NEV","5")),cutoff=1e-10,residualtol=1e-7,
        previous_matrices=nothing,previous_solutions=nothing)
    isnothing(previous_matrices)==isnothing(previous_solutions) || throw(ArgumentError("both warm-start inputs are required"))
    starts=fill(nothing,length(mats.momenta))
    if !isnothing(previous_matrices)
        old=previous_matrices
        old.assembly_parameters==mats.assembly_parameters && old.momenta==mats.momenta &&
            length(old.bases)==length(mats.bases) || throw(ArgumentError("warm-start physics/basis mismatch"))
        length(previous_solutions)==length(starts) || throw(DimensionMismatch("warm-start momenta"))
        inds=vcat([mats.offsets[m].+pairBasisEmbedding(old.bases[m],mats.bases[m]) for m in eachindex(mats.bases)]...)
        starts=[begin
            size(s.coefficients,1)==last(old.offsets) || throw(DimensionMismatch("warm-start coefficients"))
            C=zeros(ComplexF64,last(mats.offsets),size(s.coefficients,2));C[inds,:]=s.coefficients;C
        end for s in previous_solutions]
    end
    groups=[collect(mats.offsets[m]+1:mats.offsets[m+1]) for m in 1:length(mats.bases)]
    [solveTwoParticleRitz(H,N;nev=nev,cutoff=cutoff,residualtol=residualtol,
        normblocks=groups,eigensolve=size(H,1)>parse(Int,get(ENV,"PAIR_DENSE_MAX","5000")) ? pairKrylovSolve : nothing,
        initial_coefficients=C) for (H,N,C) in zip(mats.H,mats.N,starts)]
end

function momentumChargeChecks(mats,solutions;localcharges=nothing,localbases=nothing)
    W,vl,vr=liftingMPO(first(mats.bases).Nb;qop=Zblk())
    chargeblocks=Vector{Matrix{ComplexF64}}[]
    values=[zeros(length(r.levels)) for r in solutions]
    for m in eachindex(mats.bases)
        b=mats.bases[m];r=mats.offsets[m]+1:mats.offsets[m+1]
        blocks=insertionMPOMomentumBlocks(b,b,fill(directMPOEdges(W),b.Nb),vl,vr;
            hermitian=true,translation_invariant=true,skip_local=!isnothing(localcharges))
        if !isnothing(localcharges)
            nl=length(localbases[m].patterns)
            for k in eachindex(blocks);blocks[k][1:nl,1:nl]=localcharges[m][k];end
        end
        push!(chargeblocks,blocks)
        for k in eachindex(blocks),j in eachindex(values[k])
            v=solutions[k].coefficients[r,j]
            values[k][j]+=real(dot(v,blocks[k]*v))
        end
    end
    values,chargeblocks
end

function runPairBondCheck(args)
    length(args)==9 || error("CACHE N x m/e theta hmax CHANNELS MOMENTA OUTPUT_DIR")
    cache=args[1];N=parse(Int,args[2]);x=parse(Float64,args[3]);me=parse(Float64,args[4]);theta=parse(Float64,args[5])
    h=parse(Int,args[6]);q=parse(Int,args[7]);ks=parse.(Int,split(args[8],','));dir=args[9]
    mkpath(dir);BLAS.set_num_threads(1);E0_THETA[]=theta
    source=deserialize(cache);result=hasproperty(source,:result) ? source.result : nothing
    E,As,ls=isnothing(result) ? source : (result.E,result.As,result.lams)
    D=size(first(As),1);gC=1/x;mass=(2sqrt(x)*me-.25)/x;M=2h+1
    physics=(N=N,D=D,x=x,me=me,theta=theta,hmax=h,lam=1.)
    final=joinpath(dir,"branches.jls");ispath(final) && error("refusing to overwrite $final")
    hm,h2,_=holonomyMats(h)
    gates=[blockedXXGateTwisted(2pi*(m-1)/M/N;mass=mass) for m in 1:M]
    terms=vcat(kernelTermsMerged(N,gC),[(1.,liftingMPO(N÷2;qop=Zblk())...)])
    HH,NN=buildHNFixed(As,ls,gates,terms,gC,N,hm,h2);check,_=solveC(HH,Matrix(NN))
    isapprox(E,check;atol=1e-8,rtol=1e-10) || error("cache energy disagrees with physics")
    println("physical bond check $physics, momenta=$ks q=$q, ground=$E, eta=$(isnothing(result) ? NaN : result.eta)");flush(stdout)
    basefile=joinpath(dir,"local_matrices.jls")
    if isfile(basefile)
        old=deserialize(basefile);old.physics==physics && old.momenta==ks || error("baseline cache physics mismatch")
        all(old.matrices.bases[m].tensors[1]==As[m] for m in 1:M) || error("baseline cache tensor mismatch")
        base=old.matrices;basetime=old.seconds
        println("reusing local assembly checkpoint");flush(stdout)
    else
        timed=@timed twoParticleMomentumMatrices(As,[Array{ComplexF64,3}[] for _ in As],N,gC,1.,h;
            mass=mass,theta=theta,momenta=ks,separations=Int[],verbose=true)
        base=timed.value;basetime=timed.time
        serialize(basefile,(physics=physics,momenta=ks,matrices=base,seconds=basetime))
    end
    baseSolve=@timed solveMomentumMatrices(base)
    println("baseline assembly=$basetime s solve=$(baseSolve.time) s");flush(stdout)
    seedfile=joinpath(dir,"channels_q$q.jls")
    if isfile(seedfile)
        old=deserialize(seedfile);old.physics==physics && old.momenta==ks && old.As==As || error("channel checkpoint mismatch")
        seeded=old.seeded;channeltime=old.seconds
    else
        channelStat=@timed channelsFromLocalMatrices(As,base;nchannels=q,nmodes=2)
        seeded=channelStat.value;channeltime=channelStat.time
        serialize(seedfile,(physics=physics,momenta=ks,As=As,seeded=seeded,seconds=channeltime))
    end
    println("channel selection $channeltime s; channels=$(length.(seeded.channels))");flush(stdout)
    pairfile=joinpath(dir,"pair_matrices_q$q.jls")
    if isfile(pairfile)
        old=deserialize(pairfile);old.physics==physics && old.momenta==ks && old.channels==seeded.channels || error("pair checkpoint mismatch")
        pairs=old.matrices;pairtime=old.seconds
        println("reusing pair assembly checkpoint");flush(stdout)
    else
        timed=@timed twoParticleMomentumMatrices(As,seeded.channels,N,gC,1.,h;
            mass=mass,theta=theta,momenta=ks,local_matrices=base,verbose=true)
        pairs=timed.value;pairtime=timed.time
        serialize(pairfile,(physics=physics,momenta=ks,channels=seeded.channels,matrices=pairs,seconds=pairtime))
    end
    pairSolve=@timed solveMomentumMatrices(pairs)
    for (a,b) in zip(baseSolve.value,pairSolve.value)
        all(b.levels.<=a.levels.+1e-7) || error("pair enlargement violated variational ordering")
    end
    basecharge,chargeblocks=momentumChargeChecks(base,baseSolve.value)
    paircharge,_=momentumChargeChecks(pairs,pairSolve.value;localcharges=chargeblocks,localbases=base.bases)
    timings=(baseline_assembly=basetime,baseline_solve=baseSolve.time,channels=channeltime,
             pair_extra_assembly=pairtime,pair_solve=pairSolve.time)
    serialize(final,(physics=physics,momenta=ks,channels=seeded,ground_energy=E,
        ground_eta=isnothing(result) ? NaN : result.eta,source=cache,timings=timings,
        one_insertion=baseSolve.value,with_pairs=pairSolve.value,one_charge2=basecharge,pair_charge2=paircharge,
        peak_rss=Sys.maxrss(),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        restricted_matter_momentum=true))
    open(joinpath(dir,"branches.csv"),"w") do io
        println(io,"N,D,x,me,theta,hmax,k,ansatz,level,E,residual,charge2")
        for (name,solutions,charges) in (("one",baseSolve.value,basecharge),("pair",pairSolve.value,paircharge)),(i,k) in enumerate(ks),j in eachindex(solutions[i].levels)
            r=solutions[i];println(io,"$N,$D,$x,$me,$theta,$h,$k,$name,$(j-1),$(r.levels[j]),$(r.residuals[j]),$(charges[i][j])")
        end
    end
    println("saved $final; $timings; maximum charge2=$(maximum(maximum.(paircharge))) peak RSS=$(Sys.maxrss()/2^30) GiB");flush(stdout)
end
abspath(PROGRAM_FILE)==(@__FILE__) && runPairBondCheck(ARGS)
