# Finite-ring adaptation of Vanderstraeten et al., arXiv:1506.01008, Eqs. (5)-(10).
# |chi(K,r,a,b)> = Nb^(-1/2) sum_x exp(i K x) |A ... B_a(x) ... B_b(x+r) ... A>.
# Independent coefficients for each separation and channel pair, plus the full
# one-insertion local term. This is a finite-volume Ritz calculation, not the
# infinite-chain scattering-boundary/S-matrix calculation in the paper.
include("holonomy_dispersion12.jl")
include("holonomy_direct.jl")
include("pair_work_control.jl")

# Coordinate tensors are materialized only when a generic contraction asks for
# one. The analytic local block never needs the dD² dense unit tensors.
struct CoordinateInsertionBasis <: AbstractVector{Array{ComplexF64,3}}
    dims::NTuple{3,Int}
end
Base.size(b::CoordinateInsertionBasis)=(prod(b.dims),)
Base.IndexStyle(::Type{CoordinateInsertionBasis})=IndexLinear()
function Base.getindex(b::CoordinateInsertionBasis,i::Int)
    @boundscheck checkbounds(b,i)
    B=zeros(ComplexF64,b.dims);B[i]=1;B
end
Base.copy(b::CoordinateInsertionBasis)=b
fullInsertionBasis(A)=CoordinateInsertionBasis(size(A))
ownLocalBasis(b::CoordinateInsertionBasis)=b
ownLocalBasis(b)=copy.(b)

struct InsertionTensorList{L<:AbstractVector{Array{ComplexF64,3}}} <: AbstractVector{Array{ComplexF64,3}}
    vacuum::Array{ComplexF64,3}
    locals::L
    channels::Vector{Array{ComplexF64,3}}
end
Base.size(b::InsertionTensorList)=(1+length(b.locals)+length(b.channels),)
Base.IndexStyle(::Type{<:InsertionTensorList})=IndexLinear()
function Base.getindex(b::InsertionTensorList,i::Int)
    @boundscheck checkbounds(b,i)
    i==1 ? b.vacuum : i<=1+length(b.locals) ? b.locals[i-1] : b.channels[i-1-length(b.locals)]
end
# Keep generic/custom tensor vectors supported, including old serialized bases.
function sameTensorPrefix(a,b,n)
    length(a)>=n && length(b)>=n || return false
    if a isa InsertionTensorList && b isa InsertionTensorList &&
            a.locals isa CoordinateInsertionBasis && b.locals isa CoordinateInsertionBasis &&
            a.locals.dims==b.locals.dims
        a.vacuum==b.vacuum || return false
        return all(a[i]==b[i] for i in max(2,2+length(a.locals)):n)
    end
    all(a[i]==b[i] for i in 1:n)
end

struct TranslatedInsertionPattern <: AbstractMatrix{Int}
    L::Int
    first::Int
    second::Int  # zero for a local insertion
    separation::Int
end
Base.size(p::TranslatedInsertionPattern)=(p.L,p.L)
function Base.getindex(p::TranslatedInsertionPattern,i::Int,x::Int)
    @boundscheck checkbounds(p,i,x)
    i==x ? p.first : p.second!=0 && i==mod1(x+p.separation,p.L) ? p.second : 1
end

# Common left gauge for constructing particle channels at different momenta.
# Include A separately in the local space to retain the finite-ring ground state.
function leftInsertionBasis(A)
    D,_,d=size(A)
    stacked=reshape(permutedims(A,(1,3,2)),D*d,D)
    V=nullspace(stacked')
    out=[copy(A)]
    for b in 1:D, j in axes(V,2)
        B=zeros(ComplexF64,D*d,D); B[:,b]=V[:,j]
        push!(out,permutedims(reshape(B,D,d,D),(1,3,2)))
    end
    out
end

struct RingInsertionBasis
    # Concrete alternatives preserve inference in the inner contraction loops
    # while accepting both historical dense storage and new implicit storage.
    tensors::Union{Vector{Array{ComplexF64,3}},InsertionTensorList{CoordinateInsertionBasis},InsertionTensorList{Vector{Array{ComplexF64,3}}}}
    patterns::Union{Vector{Matrix{Int}},Vector{TranslatedInsertionPattern}}
    labels::Vector{NamedTuple}
    momenta::Vector{Int}
    phases::Matrix{ComplexF64}          # origin × momentum, exp(+iKx)/sqrt(Nb)
    Nb::Int
end
Base.length(b::RingInsertionBasis)=length(b.patterns)*length(b.momenta)

# Check the ordered coordinate basis without concatenating dD² dense columns.
# Exact comparison is intentional: rotated or perturbed bases need projection.
function isUnitLocalBasis(b::RingInsertionBasis,motifs)
    n=length(first(b.tensors));length(motifs)==n || return false
    if b.tensors isa InsertionTensorList && b.tensors.locals isa CoordinateInsertionBasis
        return all(b.patterns[j][1,1]==column+1 for (column,j) in enumerate(motifs))
    end
    for (column,j) in enumerate(motifs)
        tensor=b.tensors[b.patterns[j][1,1]]
        for i in eachindex(tensor)
            tensor[i]==(i==column) || return false
        end
    end
    true
end

function ringInsertionBasis(A,channels,Nb;momenta=0:Nb-1,separations=1:Nb÷2,
                            localbasis=fullInsertionBasis(A))
    Nb>=2 || throw(ArgumentError("at least two blocks required"))
    size(A,1)==size(A,2) || throw(DimensionMismatch("square MPS bonds required"))
    ks=collect(Int,momenta); rs=collect(Int,separations)
    !isempty(ks) && all(k->0<=k<Nb,ks) && allunique(ks) || throw(ArgumentError("invalid momenta"))
    all(r->1<=r<Nb,rs) && allunique(rs) || throw(ArgumentError("invalid separations"))
    localbasis isa CoordinateInsertionBasis && localbasis.dims!=size(A) && throw(DimensionMismatch("insertion tensor shape"))
    explicit=localbasis isa CoordinateInsertionBasis ? channels : Iterators.flatten((localbasis,channels))
    all(B->size(B)==size(A),explicit) || throw(DimensionMismatch("insertion tensor shape"))
    all(isfinite,A) && all(B->all(isfinite,B),explicit) || throw(ArgumentError("nonfinite tensor"))
    storedlocals=localbasis isa CoordinateInsertionBasis ? localbasis : Array{ComplexF64,3}[B for B in localbasis]
    tensors=InsertionTensorList(ComplexF64.(A),storedlocals,Array{ComplexF64,3}[B for B in channels])
    patterns=TranslatedInsertionPattern[]; labels=NamedTuple[]
    for j in eachindex(localbasis)
        push!(patterns,TranslatedInsertionPattern(Nb,1+j,0,0));push!(labels,(kind=:local,r=0,a=j,b=0))
    end
    shift=1+length(localbasis)
    for r in rs, a in eachindex(channels), b in eachindex(channels)
        push!(patterns,TranslatedInsertionPattern(Nb,shift+a,shift+b,r));push!(labels,(kind=:pair,r=r,a=a,b=b))
    end
    isempty(patterns) && throw(ArgumentError("empty insertion basis"))
    phases=ComplexF64[cis(2pi*k*x/Nb)/sqrt(Nb) for x in 0:Nb-1,k in ks]
    RingInsertionBasis(tensors,patterns,labels,ks,phases,Nb)
end

# Contract two arbitrary nonuniform periodic MPS against a sparse, possibly
# position-dependent open MPO. Only D² × D² environments are used. Local mixed
# transfers are cached with a bounded count, independent of the Hilbert dimension.
function insertionMPOProjectReference(bra::RingInsertionBasis,ket::RingInsertionBasis,sites,vL,vR;
                             hermitian=false,cachebytes=128*2^20,translation_invariant=false)
    Nb=bra.Nb; Nb==ket.Nb==length(sites) || throw(DimensionMismatch("ring length"))
    D=size(first(bra.tensors),1); d=size(first(bra.tensors),3); D2=D^2
    size(first(ket.tensors))==size(first(bra.tensors)) || throw(DimensionMismatch("bra/ket dimensions"))
    chi=length(vL); length(vR)==chi || throw(DimensionMismatch("MPO boundaries"))
    hermitian && bra!==ket && throw(ArgumentError("Hermitian projection requires the same basis"))
    ops=Matrix{ComplexF64}[]; opids=Dict{Matrix{ComplexF64},Int}()
    edges=[Tuple{Int,Int,Int}[] for _ in 1:Nb]
    for i in 1:Nb, (l,r,op) in sites[i]
        1<=l<=chi && 1<=r<=chi && size(op)==(d,d) || throw(DimensionMismatch("MPO edge"))
        id=get!(opids,op) do; push!(ops,op);length(ops);end
        push!(edges[i],(l,r,id))
    end
    cache=Dict{NTuple{3,Int},Matrix{ComplexF64}}()
    maxcache=max(1,cachebytes÷(16D2^2))
    function transfer(b,k,o)
        key=(b,k,o)
        haskey(cache,key) && return cache[key]
        length(cache)>=maxcache && empty!(cache)
        transformed=reshape(reshape(ket.tensors[k],D2,d)*ops[o],D,D,d)
        cache[key]=fastMixedTransfer(bra.tensors[b],transformed)
    end
    current=[zeros(ComplexF64,D2,D2) for _ in 1:chi]; scratch=deepcopy(current)
    active=falses(chi); nextactive=copy(active)
    function element(pb,pk,x,y)
        foreach(E->fill!(E,0),current); active.=.!iszero.(vL)
        for l in 1:chi, j in 1:D2; current[l][j,j]=vL[l]; end
        for i in 1:Nb
            foreach(E->fill!(E,0),scratch);fill!(nextactive,false)
            for (l,r,o) in edges[i]
                active[l] || continue
                mul!(scratch[r],transfer(pb[i,x],pk[i,y],o),current[l],1.,1.)
                nextactive[r]=true
            end
            current,scratch=scratch,current;active,nextactive=nextactive,active
        end
        sum(vR[r]*tr(current[r]) for r in 1:chi)
    end
    H=zeros(ComplexF64,length(bra),length(ket)); G=zeros(ComplexF64,Nb,Nb)
    nkb=length(bra.momenta); nkk=length(ket.momenta)
    for j in eachindex(ket.patterns), i in 1:(hermitian ? j : length(bra.patterns))
        for y in 1:Nb, x in 1:Nb
            G[x,y]=element(bra.patterns[i],ket.patterns[j],x,y)
        end
        rr=(i-1)*nkb+1:i*nkb; cc=(j-1)*nkk+1:j*nkk
        H[rr,cc]=bra.phases'*G*ket.phases
        hermitian && i!=j && (H[cc,rr]=H[rr,cc]')
    end
    hermitian ? (H+H')/2 : H
end

include("holonomy_transferbatch.jl")
include("holonomy_pairsweep.jl")
include("holonomy_pairnorm.jl")
include("holonomy_pairmpo.jl")
include("holonomy_pairassemble.jl")

function insertionNorm(b;projector=insertionMPOProject,kwargs...)
    d=size(b.tensors[1],3)
    sites=fill([(1,1,Matrix{ComplexF64}(I,d,d))],b.Nb)
    projector(b,b,sites,[1.],[1.];hermitian=true,translation_invariant=true,kwargs...)
end

# Exact projection of the full finite-holonomy Hamiltonian, including all chosen
# matter-momentum couplings. No fixed-ket translation shortcut for open MPOs.
function twoParticleMatrices(As,channels,Nsites,gC,lam,hmax;
        mass=0.,theta=E0_THETA[],momenta=0:Nsites÷2-1,separations=1:Nsites÷4,
        localbases=fullInsertionBasis.(As),cachebytes=128*2^20,verbose=false,projector=insertionMPOProjectLargeD,compact=true)
    iseven(Nsites) && Nsites>=6 || throw(ArgumentError("even Nsites >= 6 required"))
    M=2hmax+1; Nb=Nsites÷2
    length(As)==length(channels)==length(localbases)==M || throw(DimensionMismatch("sector count"))
    isfinite(theta) && isfinite(mass) && isfinite(gC) && isfinite(lam) || throw(ArgumentError("nonfinite physics"))
    bs=[ringInsertionBasis(As[m],channels[m],Nb;momenta=momenta,separations=separations,
                          localbasis=localbases[m]) for m in 1:M]
    offsets=cumsum([0;length.(bs)]); H=zeros(ComplexF64,last(offsets),last(offsets)); S=spzeros(ComplexF64,last(offsets),last(offsets))
    hm,h2,_=holonomyMats(hmax)
    # Explicit theta argument: do not mutate the process-wide E0_THETA reference.
    weights=[ComplexF64[-(2i*Qbvals[s]+q1vals[s])+theta/pi for s in 1:4] for i in 0:Nb-1]
    terms=vcat(kernelTermsMerged(Nsites,gC),[(lam,liftingMPO(Nb;qop=Zblk())...)])
    for m in 1:M
        verbose && (println("[two-particle] diagonal sector $m/$M, basis=$(length(bs[m]))");flush(stdout))
        r=offsets[m]+1:offsets[m+1]; b=bs[m]
        S[r,r]=insertionNorm(b;projector=projector,cachebytes=cachebytes)
        dropzeros!(S) # Dense block assignment otherwise stores off-momentum zeros.
        sites,vl,vr=compact ? pairHoppingMPO(2pi*(m-1)/M/Nsites,Nb;mass=mass) :
                            gateMPOSites(blockedXXGateTwisted(2pi*(m-1)/M/Nsites;mass=mass),4,Nb)
        hh=projector(b,b,sites,vl,vr;hermitian=true,translation_invariant=true,cachebytes=cachebytes)
        if compact
            sites,vl,vr=pairDiagonalMPO(Nsites,gC,lam,real(h2[m,m]);theta=theta)
            hh.+=projector(b,b,sites,vl,vr;hermitian=true,cachebytes=cachebytes)
        else
            hh.+=projector(b,b,momentMPOSites(fill(ONES4,Nb),weights,2),[1.,0.,0.],
                               [gC*Nsites*real(h2[m,m]),0.,gC/Nsites];hermitian=true,cachebytes=cachebytes)
            for (coef,W,vl,vr) in terms
                iszero(coef) && continue
                hh.+=coef.*projector(b,b,fill(directMPOEdges(W),Nb),vl,vr;
                                          hermitian=true,cachebytes=cachebytes)
            end
        end
        H[r,r]=hh
    end
    for m in 1:M, mp in m+1:M
        verbose && (println("[two-particle] cross sectors $m,$mp");flush(stdout))
        phase=[dressPhase(2pi*(mp-m)/M,Nsites)(i) for i in 0:Nb-1]
        block=projector(bs[m],bs[mp],momentMPOSites(phase,weights,1),[1.,0.],
                                 [gC*Nsites*h2[m,mp],2gC*hm[m,mp]];cachebytes=cachebytes)
        r=offsets[m]+1:offsets[m+1]; s=offsets[mp]+1:offsets[mp+1]
        H[r,s]=block;H[s,r]=block'
    end
    (H=Hermitian(H),N=Hermitian(S),bases=bs,offsets=offsets)
end

# Normalize basis columns before removing null/gauge/finite-ring exchange modes.
# Return coefficients in the ORIGINAL basis so states can be reconstructed.
function solveTwoParticleRitz(H,N;nev=8,cutoff=1e-10,residualtol=1e-7,
                              normblocks=nothing,eigensolve=nothing,initial_coefficients=nothing)
    0<cutoff<1 && nev>=0 && residualtol>0 || throw(ArgumentError("invalid solver settings"))
    size(H)==size(N) && size(H,1)==size(H,2) || throw(DimensionMismatch("projected matrices"))
    all(isfinite,H) && all(isfinite,N) || throw(ArgumentError("nonfinite projected matrices"))
    isnothing(initial_coefficients) || (size(initial_coefficients,1)==size(H,1) && all(isfinite,initial_coefficients)) ||
        throw(ArgumentError("invalid initial Ritz coefficients"))
    whitening=twoParticleWhitener(N,cutoff,normblocks)
    W=whitening.W; rank=whitening.rank;count=min(nev,rank)
    if isnothing(eigensolve)
        reduced=W'*(Matrix(H)*W);hermerr=norm(reduced-reduced')/max(norm(reduced),1.)
        hermerr<=residualtol || error("projected Hamiltonian failed Hermiticity check: $hermerr")
        if count>0
            f=eigen(Hermitian((reduced+reduced')/2),1:count);vals=f.values;u=f.vectors
        else
            vals=Float64[];u=zeros(ComplexF64,rank,0)
        end
        action=v->reduced*v;solverinfo=(method=:dense,matvecs=0)
    else
        # This branch retains H in the original coordinates and sparse block W.
        # It does not form W' H W or compute the unwanted high-energy eigenvectors.
        ishermitian(H) || throw(ArgumentError("iterative solve requires a Hermitian Hamiltonian"))
        action=v->W'*(H*(W*v))
        if isnothing(initial_coefficients)
            vals,u,solverinfo=eigensolve(action,rank,count,residualtol)
        else
            # Coefficients live in the original nonorthogonal basis. Project
            # with the NEW metric, including any changed norm-null directions.
            initial=W'*(N*initial_coefficients)
            vals,u,solverinfo=eigensolve(action,rank,count,residualtol;initial_vectors=initial)
        end
    end
    coeffs=W*u
    residuals=[norm(action(u[:,j])-vals[j]*u[:,j]) for j in 1:count]
    maximum(residuals;init=0.)<=residualtol || error("Ritz residual exceeds tolerance")
    (levels=vals,coefficients=coeffs,rank=rank,residuals=residuals,
     norm_eigenvalues=whitening.values,norm_threshold=whitening.threshold,
     norm_diagonal_floor=whitening.columnfloor,norm_error=whitening.error,whitener=W,solver_info=solverinfo)
end

function twoParticleLevels(As,channels,Nsites,gC,lam,hmax;nev=8,cutoff=1e-10,
                           residualtol=1e-7,eigensolve=nothing,kwargs...)
    matrices=twoParticleMatrices(As,channels,Nsites,gC,lam,hmax;kwargs...)
    result=solveTwoParticleRitz(matrices.H,matrices.N;nev=nev,cutoff=cutoff,residualtol=residualtol,
                               normblocks=twoParticleNormBlocks(matrices.bases),eigensolve=eigensolve)
    (result=result,matrices=matrices)
end

# Build a common reduced B basis from one-insertion Ritz states, sampled at
# several matter momenta. Channels are SVD-compressed separately per holonomy
# sector. Their count is a variational truncation that must be converged.
function seedTwoParticleChannels(As,Nsites,gC,lam,hmax;momenta=0:Nsites÷2-1,
        nmodes=2,nchannels=2,svdcutoff=1e-10,cutoff=1e-10,mass=0.,theta=E0_THETA[],
        cachebytes=128*2^20,verbose=false)
    nmodes>0 && nchannels>0 && 0<svdcutoff<1 || throw(ArgumentError("invalid channel settings"))
    ks=collect(Int,momenta)
    !isempty(ks) && allunique(ks) && all(k->0<=k<Nsites÷2,ks) || throw(ArgumentError("invalid sample momenta"))
    M=length(As); locals=leftInsertionBasis.(As)
    samples=[Vector{ComplexF64}[] for _ in 1:M]; samplelevels=Vector{Float64}[]
    for k in ks
        r=twoParticleLevels(As,[Array{ComplexF64,3}[] for _ in 1:M],Nsites,gC,lam,hmax;
            momenta=[k],separations=Int[],localbases=locals,nev=nmodes+1,cutoff=cutoff,
            mass=mass,theta=theta,cachebytes=cachebytes,verbose=verbose)
        push!(samplelevels,r.result.levels)
        # At k=0, deflate the vacuum subspace before choosing particle samples.
        # Individual holonomy-sector ground vectors belong to this local basis.
        W=r.result.whitener; ground=zeros(ComplexF64,size(W,1),M)
        if k==0
            for m in 1:M; ground[r.matrices.offsets[m]+1,m]=1; end
        end
        overlap=W'*r.matrices.N*ground
        Q=k==0 ? nullspace(overlap') : Matrix{ComplexF64}(I,size(W,2),size(W,2))
        size(Q,2)>0 || continue
        f=eigen(Hermitian(Q'*W'*r.matrices.H*W*Q))
        C=W*Q*f.vectors[:,1:min(nmodes,length(f.values))]
        for m in 1:M, j in axes(C,2)
            v=zeros(ComplexF64,length(As[m]))
            for a in 2:length(locals[m])
                v.+=C[r.matrices.offsets[m]+a,j].*vec(locals[m][a])
            end
            norm(v)>1e-12 && push!(samples[m],v/norm(v))
        end
    end
    channels=Vector{Array{ComplexF64,3}}[]; spectra=Vector{Float64}[]
    for m in 1:M
        isempty(samples[m]) && error("no nonzero particle samples in sector $m")
        f=svd(reduce(hcat,samples[m])); keep=min(nchannels,count(>(svdcutoff*first(f.S)),f.S))
        push!(channels,[reshape(f.U[:,j],size(As[m])) for j in 1:keep]);push!(spectra,f.S)
    end
    (channels=channels,singular_values=spectra,sample_levels=samplelevels)
end
