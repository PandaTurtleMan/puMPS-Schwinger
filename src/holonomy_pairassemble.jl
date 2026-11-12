# Assemble the local/local block by opening BOTH physical tensors analytically.
# This removes the loop over dD² separate ket basis tensors, and keeps temporary
# arrays at D^4 instead of dD^6. Pair columns still use shared insertion sweeps.
function openPairWorkspace(D,d)
    (L=zeros(ComplexF64,D,D,D,D),temp=zeros(ComplexF64,D,D,D,D),
     product=zeros(ComplexF64,D,D,D,D),ordered=zeros(ComplexF64,D,D,D,D),
     X=[zeros(ComplexF64,D,D,D,D) for _ in 1:d],Y=[zeros(ComplexF64,D,D,D,D) for _ in 1:d])
end
function addMixedOpenPair!(target,Ab,Ak,L,Mid,w)
    D=size(Ab,1);d=size(Ab,3);D2=D^2
    permutedims!(w.L,reshape(L,D,D,D,D),(1,2,4,3))
    lm=reshape(w.L,D^3,D);mm=reshape(Mid,D^3,D);tmp=reshape(w.temp,D^3,D)
    for s in 1:d
        mul!(tmp,lm,transpose(@view(Ak[:,:,s])))
        permutedims!(w.X[s],w.temp,(1,3,2,4))
        mul!(tmp,mm,adjoint(@view(Ab[:,:,s])))
        permutedims!(w.Y[s],w.temp,(4,1,2,3))
    end
    block=reshape(target,D,D,d,D,D,d)
    for sb in 1:d,st in 1:d
        mul!(reshape(w.product,D2,D2),reshape(w.X[sb],D2,D2),reshape(w.Y[st],D2,D2))
        permutedims!(w.ordered,w.product,(3,2,1,4))
        @views block[:,:,sb,:,:,st].+=w.ordered
    end
    target
end
function mixedOpenPairBlock(Ab,Ak,L,Mid)
    D=size(Ab,1);d=size(Ab,3);D2=D^2
    l4=reshape(L,D,D,D,D);m4=reshape(Mid,D,D,D,D)
    lmat=reshape(permutedims(l4,(1,2,4,3)),D^3,D);mmat=reshape(m4,D^3,D)
    Xs=[reshape(permutedims(reshape(lmat*transpose(@view(Ak[:,:,s])),D,D,D,D),(1,3,2,4)),D2,D2) for s in 1:d]
    Ys=[reshape(permutedims(reshape(mmat*adjoint(@view(Ab[:,:,s])),D,D,D,D),(4,1,2,3)),D2,D2) for s in 1:d]
    H=zeros(ComplexF64,D,D,d,D,D,d)
    for sb in 1:d,st in 1:d
        R=reshape(Xs[sb]*Ys[st],D,D,D,D)
        H[:,:,sb,:,:,st]=permutedims(R,(3,2,1,4))
    end
    reshape(H,d*D2,d*D2)
end

# Progress bar and ETA for the local/local assembly (logging only: never changes any matrix element).
# twoParticleMomentumMatrices counts the work of every local sweep up front with a structure-only dry run
# (localMPOWorkUnits), installs a tracker in ASSEMBLY_PROGRESS, and localMPOProject adds its work as it goes.
# Work units: one addMixedOpenPair! call = 16 (it does d^2 = 16 D^4 x D^4 products), one arc P*S or one
# middle-transfer product = 1 (a single D^2 x D^2 product). Env: PUMPS_PROGRESS=0 disables it,
# PUMPS_PROGRESS_SECONDS sets the reporting interval (default 300 s).
const MIXED_PAIR_UNITS=16
mutable struct AssemblyProgress
    total::Int
    done::Threads.Atomic{Int}
    t0::Float64
    every::Float64
    last::Float64
    lock::ReentrantLock
end
const ASSEMBLY_PROGRESS=Ref{Union{Nothing,AssemblyProgress}}(nothing)
function fmtDuration(s)
    s=round(Int,max(s,0.));h,r=divrem(s,3600);m,sec=divrem(r,60)
    h>0 ? "$(h)h$(lpad(m,2,'0'))m" : "$(m)m$(lpad(sec,2,'0'))s"
end
function printAssemblyProgress(p,done)
    f=p.total>0 ? min(done/p.total,1.) : 1.;el=time()-p.t0;nb=30;nf=round(Int,nb*f)
    bar="["*"#"^nf*"."^(nb-nf)*"]"
    eta=f>0 ? el*(1-f)/f : NaN
    tail=f>=1 ? "done in $(fmtDuration(el))" :
        f>0 ? "elapsed $(fmtDuration(el))  ETA $(fmtDuration(eta))  (finish ~$(Libc.strftime("%Y-%m-%d %H:%M",time()+eta)))" :
        "elapsed $(fmtDuration(el))  ETA --"
    println("[assembly] $bar $(lpad(string(round(100f;digits=1)),5))%  $tail");flush(stdout)
end
function tickAssembly!(units)
    p=ASSEMBLY_PROGRESS[];(isnothing(p)||units==0) && return
    done=Threads.atomic_add!(p.done,units)+units
    now=time()
    if now-p.last>=p.every && trylock(p.lock)
        try
            now-p.last>=p.every && (p.last=now;printAssemblyProgress(p,done))
        finally
            unlock(p.lock)
        end
    end
    nothing
end
# Structure-only replica of localMPOProject's sweep: the same reachability flags (pa/sa), the same
# middle/outer key sets, no tensors. Returns the work units that localMPOProject will report.
function localMPOWorkUnits(sites,vL,vR;translation_invariant=false,hermitian=false)
    L=length(sites);chi=length(vL)
    pa=falses(chi,L+1);sa=falses(chi,L+1);pa[:,1].=.!iszero.(vL);sa[:,end].=.!iszero.(vR)
    for i in 1:L,(l,r,_) in sites[i];pa[l,i] || continue;pa[r,i+1]=true;end
    for i in L:-1:1,(l,r,_) in sites[i];sa[r,i+1] || continue;sa[l,i]=true;end
    mixed=translation_invariant||hermitian ? MIXED_PAIR_UNITS : 2MIXED_PAIR_UNITS
    units=0
    for x in (translation_invariant ? (1:1) : (1:L))
        middle=Set{Tuple{Int,Int}}((r,r) for r in 1:chi)
        for y in x+1:L
            outer=Set{Tuple{Int,Int}}()
            for (lx,rx,_) in sites[x],(ly,ry,_) in sites[y]
                pa[lx,x] && sa[ry,y+1] || continue
                (rx,ly) in middle || continue
                (lx,ry) in outer || (push!(outer,(lx,ry));units+=1)
                units+=mixed
            end
            if y<L
                next=Set{Tuple{Int,Int}}()
                for (r,u) in middle,(ly,ry,_) in sites[y]
                    ly==u || continue
                    push!(next,(r,ry));units+=1
                end
                middle=next
            end
        end
    end
    units
end

function localMPOProject(bra,ket,sites,vL,vR;translation_invariant=false,diagonal_momenta=false,
                         hermitian=false,destination=nothing,work_control=nothing)
    checkPairWork(work_control)
    hermitian && bra!==ket && throw(ArgumentError("same basis required for Hermitian local projection"))
    L=bra.Nb;D=size(first(bra.tensors),1);d=size(first(bra.tensors),3);dt=d*D^2
    ib=findall(l->l.kind==:local,bra.labels);ik=findall(l->l.kind==:local,ket.labels)
    nb=length(bra.momenta);nk=length(ket.momenta)
    diagonal_momenta && bra.momenta!=ket.momenta && throw(ArgumentError("matching momenta required"))
    !isnothing(destination) && !diagonal_momenta && throw(ArgumentError("destination requires matching momentum blocks"))
    H=diagonal_momenta ? (isnothing(destination) ? [zeros(ComplexF64,length(ib),length(ik)) for _ in 1:nk] : destination) : zeros(ComplexF64,length(ib)*nb,length(ik)*nk)
    diagonal_momenta && (length(H)!=nk || any(size(h)!=(length(ib),length(ik)) for h in H)) && throw(DimensionMismatch("local destination"))
    (isempty(ib)||isempty(ik)) && return H
    Ab=first(bra.tensors);Ak=first(ket.tensors)
    unitb=isUnitLocalBasis(bra,ib);unitk=isUnitLocalBasis(ket,ik)
    Vb=unitb ? nothing : hcat([vec(bra.tensors[bra.patterns[j][1,1]]) for j in ib]...)
    Vk=unitk ? nothing : hcat([vec(ket.tensors[ket.patterns[j][1,1]]) for j in ik]...)
    c=prepareDirectMPO(Ab,Ak,ones(D),ones(D),sites,vL,vR)
    workspace=openPairWorkspace(D,d)
    chi=c.chi;id=Matrix{ComplexF64}(I,D^2,D^2)
    # Contract physical op into the remaining vacuum tensor on each open site.
    transforms=Dict{Matrix{ComplexF64},Tuple{Array{ComplexF64,3},Array{ComplexF64,3}}}()
    for site in sites,(_,_,op) in site
        get!(transforms,op) do
            (reshape(reshape(Ab,D^2,d)*op',D,D,d),reshape(reshape(Ak,D^2,d)*op,D,D,d))
        end
    end
    function add!(raw,x,y)
        small=unitb ? raw : Vb'*raw
        unitk || (small=small*Vk)
        for kb in 1:nb,kk in 1:nk
            diagonal_momenta && kb!=kk && continue
            translation_invariant && bra.momenta[kb]!=ket.momenta[kk] && continue
            w=conj(bra.phases[y,kb])*ket.phases[x,kk]
            translation_invariant && (w*=L)
            if diagonal_momenta
                H[kk].+=w.*small
            else
                @views H[kb:nb:end,kk:nk:end].+=w.*small
            end
        end
    end
    for x in (translation_invariant ? (1:1) : (1:L))
        checkPairWork(work_control)
        same=zeros(ComplexF64,D,D,d,D,D,d)
        for e in c.edges[x]
            c.pa[e.left,x] && c.sa[e.right,x+1] || continue
            env=reshape(c.P[x][e.left]*c.S[x+1][e.right],D,D,D,D)
            for sb in 1:d,st in 1:d
                iszero(e.op[st,sb]) && continue
                same[:,:,sb,:,:,st].+=e.op[st,sb].*permutedims(env,(2,4,1,3))
            end
        end
        add!(reshape(same,dt,dt),x,x)
        # middle[r,u] propagates from after site x up to before site y.
        middle=Dict((r,r)=>copy(id) for r in 1:chi)
        for y in x+1:L
            checkPairWork(work_control)
            forward=zeros(ComplexF64,dt,dt)
            backward=translation_invariant || hermitian ? nothing : zeros(ComplexF64,dt,dt)
            outer=Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            units=0
            for ex in c.edges[x],ey in c.edges[y]
                c.pa[ex.left,x] && c.sa[ey.right,y+1] || continue
                mid=get(middle,(ex.right,ey.left),nothing);isnothing(mid) && continue
                haskey(outer,(ex.left,ey.right)) || (units+=1)
                units+=isnothing(backward) ? MIXED_PAIR_UNITS : 2MIXED_PAIR_UNITS
                arc=get!(outer,(ex.left,ey.right)) do;c.P[x][ex.left]*c.S[y+1][ey.right];end
                bx,kx=transforms[ex.op];by,ky=transforms[ey.op]
                addMixedOpenPair!(forward,bx,ky,arc,mid,workspace)
                isnothing(backward) || addMixedOpenPair!(backward,by,kx,mid,arc,workspace)
            end
            add!(forward,x,y)
            # For a Hermitian operator and identical bases, reversing the two
            # open sites gives the adjoint. This also conjugates the Bloch weight.
            translation_invariant || add!(hermitian ? forward' : backward,y,x)
            if y<L
                next=Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                for ((r,u),mid) in middle,ey in c.edges[y]
                    ey.left==u || continue
                    target=get!(next,(r,ey.right)) do;zeros(ComplexF64,D^2,D^2);end
                    mul!(target,ey.transfer,mid,1.,1.);units+=1
                end
                middle=next
            end
            tickAssembly!(units)
        end
    end
    H
end

function insertionMPOProjectLargeD(bra,ket,sites,vL,vR;hermitian=false,
                                 translation_invariant=false,cachebytes=128*2^20)
    hermitian && bra!==ket && throw(ArgumentError("same basis required for Hermitian projection"))
    nb=length(bra.momenta);nk=length(ket.momenta)
    bl=findall(l->l.kind==:local,bra.labels);kl=findall(l->l.kind==:local,ket.labels)
    bp=findall(l->l.kind==:pair,bra.labels);kp=findall(l->l.kind==:pair,ket.labels)
    inds(motifs,n)=vcat([collect((j-1)*n+1:j*n) for j in motifs]...)
    H=zeros(ComplexF64,length(bra),length(ket))
    if !isempty(bl)&&!isempty(kl)
        H[inds(bl,nb),inds(kl,nk)]=localMPOProject(bra,ket,sites,vL,vR;
            translation_invariant=translation_invariant,hermitian=hermitian)
    end
    sub(b,js)=RingInsertionBasis(b.tensors,b.patterns[js],b.labels[js],b.momenta,b.phases,b.Nb)
    if !isempty(kp)
        H[:,inds(kp,nk)]=insertionMPOProject(bra,sub(ket,kp),sites,vL,vR;
            translation_invariant=translation_invariant,cachebytes=cachebytes)
    end
    if !isempty(bp)&&!isempty(kl)
        if hermitian
            H[inds(bp,nb),inds(kl,nk)]=H[inds(bl,nb),inds(kp,nk)]'
        else
            adj=[[(l,r,Matrix(op')) for (l,r,op) in site] for site in sites]
            H[inds(bp,nb),inds(kl,nk)]=insertionMPOProject(sub(ket,kl),sub(bra,bp),adj,conj.(vL),conj.(vR);
                translation_invariant=translation_invariant,cachebytes=cachebytes)'
        end
    end
    hermitian ? (H+H')/2 : H
end

# As above, but store only matching matter-momentum blocks. This is the
# explicitly restricted branch ansatz used by the old dispersion benchmark;
# callers must not confuse it with the full all-momentum Hamiltonian.
pairBasisSubset(b,js)=RingInsertionBasis(b.tensors,b.patterns[js],b.labels[js],b.momenta,b.phases,b.Nb)

function hermitizePairBlocks!(blocks)
    for H in blocks
        for j in axes(H,2)
            H[j,j]=real(H[j,j])
            for i in 1:j-1
                z=(H[i,j]+conj(H[j,i]))/2;H[i,j]=z;H[j,i]=conj(z)
            end
        end
    end
    blocks
end

function allocatePairMomentumStorage(dim,nk,old,inds;consume_previous=false)
    H=Matrix{ComplexF64}[];N=SparseMatrixCSC{ComplexF64,Int}[]
    for k in 1:nk
        push!(H,zeros(ComplexF64,dim,dim));push!(N,spzeros(ComplexF64,dim,dim))
        if !isnothing(old)
            H[k][inds,inds]=old.H[k];N[k][inds,inds]=old.N[k]
            if consume_previous
                oldbytes=16length(old.H[k])
                old.H[k]=Hermitian(zeros(ComplexF64,0,0));old.N[k]=Hermitian(spzeros(ComplexF64,0,0))
                oldbytes>=64*2^20 && GC.gc()
            end
        end
    end
    H,N
end

# Exact embedding for a fixed background and a frozen prefix of channel tensors.
# Labels, not positions, identify old motifs: q growth interleaves new pair labels.
function pairBasisEmbedding(old,new)
    old.Nb==new.Nb && old.momenta==new.momenta && old.phases==new.phases ||
        throw(ArgumentError("cached basis ring/momentum mismatch"))
    count(l->l.kind==:local,old.labels)==count(l->l.kind==:local,new.labels) ||
        throw(ArgumentError("enrichment must preserve the local basis"))
    length(old.tensors)<=length(new.tensors) && sameTensorPrefix(old.tensors,new.tensors,length(old.tensors)) ||
        throw(ArgumentError("enrichment requires identical A/local tensors and a frozen channel prefix"))
    positions=Dict(label=>i for (i,label) in enumerate(new.labels))
    allunique(old.labels) && allunique(new.labels) || throw(ArgumentError("duplicate basis labels"))
    all(haskey(positions,l) for l in old.labels) || throw(ArgumentError("cached motifs are not a subset"))
    inds=[positions[l] for l in old.labels]
    all(old.patterns[j]==new.patterns[inds[j]] for j in eachindex(inds)) ||
        throw(ArgumentError("cached insertion patterns differ"))
    inds
end

# Compute only the complement of the old-bra × old-ket rectangle. The caller
# inserts the saved rectangle before adding the new parallel blocks.
function insertionMPOEnrichmentBlocks(bra,ket,sites,vL,vR,oldbra,oldket;
        hermitian=false,translation_invariant=false,cachebytes=128*2^20,transfer_mode=:auto,destination=nothing,sweep_options=(;))
    bra.momenta==ket.momenta || throw(ArgumentError("matching momenta required"))
    for (b,js) in ((bra,oldbra),(ket,oldket))
        allunique(js) && all(j->1<=j<=length(b.patterns),js) || throw(ArgumentError("invalid retained motifs"))
    end
    nb=setdiff(collect(eachindex(bra.patterns)),oldbra)
    nk=setdiff(collect(eachindex(ket.patterns)),oldket)
    all(j->bra.labels[j].kind==:pair,nb) && all(j->ket.labels[j].kind==:pair,nk) ||
        throw(ArgumentError("only pair motifs may be added during enrichment"))
    hermitian && (bra!==ket || oldbra!=oldket) && throw(ArgumentError("Hermitian enrichment requires the same basis/embedding"))
    H=isnothing(destination) ? [zeros(ComplexF64,length(bra.patterns),length(ket.patterns)) for _ in bra.momenta] : destination
    length(H)==length(bra.momenta) && all(size(h)==(length(bra.patterns),length(ket.patterns)) for h in H) || throw(DimensionMismatch("block destination"))
    if !isempty(nk)
        insertionMPOProject(bra,pairBasisSubset(ket,nk),sites,vL,vR;
            translation_invariant=translation_invariant,cachebytes=cachebytes,diagonal_momenta=true,transfer_mode=transfer_mode,
            destination=[view(h,:,nk) for h in H],sweep_options...)
    end
    if !isempty(nb)&&!isempty(oldket)
        if hermitian
            for k in eachindex(H);@views H[k][nb,oldket].=H[k][oldbra,nk]';end
        else
            adj=[[(l,r,Matrix(op')) for (l,r,op) in site] for site in sites]
            insertionMPOProject(pairBasisSubset(ket,oldket),pairBasisSubset(bra,nb),adj,conj.(vL),conj.(vR);
                translation_invariant=translation_invariant,cachebytes=cachebytes,diagonal_momenta=true,transfer_mode=transfer_mode,
                destination=[adjoint(view(h,nb,oldket)) for h in H],sweep_options...)
        end
    end
    hermitian ? hermitizePairBlocks!(H) : H
end

function insertionMPOMomentumBlocks(bra,ket,sites,vL,vR;hermitian=false,
                                    translation_invariant=false,cachebytes=128*2^20,skip_local=false,
                                    retained_bra=nothing,retained_ket=nothing,transfer_mode=:auto,destination=nothing,sweep_options=(;))
    isnothing(retained_bra)==isnothing(retained_ket) || throw(ArgumentError("both retained motif lists are required"))
    if !isnothing(retained_bra)
        skip_local && throw(ArgumentError("cannot combine retained motifs and skip_local"))
        return insertionMPOEnrichmentBlocks(bra,ket,sites,vL,vR,retained_bra,retained_ket;
            hermitian=hermitian,translation_invariant=translation_invariant,cachebytes=cachebytes,transfer_mode=transfer_mode,destination=destination,sweep_options=sweep_options)
    end
    bra.momenta==ket.momenta || throw(ArgumentError("matching momenta required"))
    bl=findall(l->l.kind==:local,bra.labels);kl=findall(l->l.kind==:local,ket.labels)
    bp=findall(l->l.kind==:pair,bra.labels);kp=findall(l->l.kind==:pair,ket.labels)
    H=isnothing(destination) ? [zeros(ComplexF64,length(bra.patterns),length(ket.patterns)) for _ in bra.momenta] : destination
    length(H)==length(bra.momenta) && all(size(h)==(length(bra.patterns),length(ket.patterns)) for h in H) || throw(DimensionMismatch("block destination"))
    sub(b,js)=RingInsertionBasis(b.tensors,b.patterns[js],b.labels[js],b.momenta,b.phases,b.Nb)
    if !skip_local&&!isempty(bl)&&!isempty(kl)
        localMPOProject(bra,ket,sites,vL,vR;
            translation_invariant=translation_invariant,diagonal_momenta=true,hermitian=hermitian,
            destination=[view(h,bl,kl) for h in H],work_control=get(sweep_options,:work_control,nothing))
    end
    if !isempty(kp)
        insertionMPOProject(bra,sub(ket,kp),sites,vL,vR;
            translation_invariant=translation_invariant,cachebytes=cachebytes,diagonal_momenta=true,transfer_mode=transfer_mode,
            destination=[view(h,:,kp) for h in H],sweep_options...)
    end
    if !isempty(bp)&&!isempty(kl)
        if hermitian
            for k in eachindex(H);@views H[k][bp,kl].=H[k][bl,kp]';end
        else
            adj=[[(l,r,Matrix(op')) for (l,r,op) in site] for site in sites]
            insertionMPOProject(sub(ket,kl),sub(bra,bp),adj,conj.(vL),conj.(vR);
                translation_invariant=translation_invariant,cachebytes=cachebytes,diagonal_momenta=true,transfer_mode=transfer_mode,
                destination=[adjoint(view(h,bp,kl)) for h in H],sweep_options...)
        end
    end
    hermitian ? hermitizePairBlocks!(H) : H
end

function twoParticleMomentumMatrices(As,channels,Nsites,gC,lam,hmax;
        mass=0.,theta=E0_THETA[],momenta=0:Nsites÷2-1,separations=1:Nsites÷4,
        localbases=fullInsertionBasis.(As),cachebytes=128*2^20,verbose=false,local_matrices=nothing,
        parallel=Threads.nthreads()>1,previous_matrices=nothing,consume_previous=false,transfer_mode=:auto,sweep_options=(;))
    iseven(Nsites) && Nsites>=6 && hmax>=0 || throw(ArgumentError("invalid N or hmax"))
    M=2hmax+1;L=Nsites÷2;ks=collect(momenta);nk=length(ks)
    length(As)==length(channels)==length(localbases)==M || throw(DimensionMismatch("sector count"))
    all(isfinite,(gC,lam,mass,theta)) || throw(ArgumentError("nonfinite physics"))
    parameters=(Nsites=Nsites,gC=gC,lam=lam,hmax=hmax,mass=mass,theta=theta)
    !isnothing(local_matrices)&&!isnothing(previous_matrices) && throw(ArgumentError("choose local_matrices or previous_matrices"))
    consume_previous && isnothing(previous_matrices) && throw(ArgumentError("consume_previous requires previous_matrices"))
    transfer_mode in (:auto,:dense,:factorized) || throw(ArgumentError("invalid transfer_mode"))
    # Own the cached basis tensors: later edits to an input channel pool must not
    # silently change the basis against which retained H/N blocks were computed.
    bases=[ringInsertionBasis(As[m],copy.(channels[m]),L;momenta=ks,separations=separations,
        localbasis=ownLocalBasis(localbases[m])) for m in 1:M]
    skip_local=!isnothing(local_matrices)
    retained=fill(nothing,M)
    if !isnothing(previous_matrices)
        old=previous_matrices
        hasproperty(old,:assembly_parameters) && old.assembly_parameters==parameters ||
            throw(ArgumentError("previous matrices lack matching assembly_parameters"))
        old.momenta==ks && length(old.bases)==M || throw(ArgumentError("previous momentum/sector mismatch"))
        old.offsets==cumsum([0;length.(getproperty.(old.bases,:patterns))]) || throw(ArgumentError("invalid previous offsets"))
        length(old.H)==length(old.N)==nk && all(size(h)==size(n)==(last(old.offsets),last(old.offsets)) for (h,n) in zip(old.H,old.N)) ||
            throw(DimensionMismatch("previous matrix dimensions"))
        retained=[pairBasisEmbedding(old.bases[m],bases[m]) for m in 1:M]
    end
    if skip_local
        hasproperty(local_matrices,:assembly_parameters) && local_matrices.assembly_parameters!=parameters &&
            throw(ArgumentError("cached local physics mismatch"))
        local_matrices.momenta==ks || throw(ArgumentError("cached local momentum mismatch"))
        length(local_matrices.bases)==M || throw(DimensionMismatch("cached local sectors"))
        for m in 1:M
            old=local_matrices.bases[m];new=bases[m];nl=length(localbases[m])
            all(l->l.kind==:local,old.labels) && length(old.labels)==nl &&
                sameTensorPrefix(old.tensors,new.tensors,nl+1) || throw(ArgumentError("cached local basis mismatch"))
        end
    end
    offsets=cumsum([0;length.(getproperty.(bases,:patterns))]);dim=last(offsets)
    inds=isnothing(previous_matrices) ? Int[] : vcat([offsets[m].+retained[m] for m in 1:M]...)
    # Copy and optionally release one old momentum at a time. Consuming is an
    # explicit ownership transfer: the old H/N cache is invalidated on return
    # (and may be partially consumed if an allocation/contraction throws).
    H,N=allocatePairMomentumStorage(dim,nk,previous_matrices,inds;consume_previous=consume_previous)
    hm,h2,_=holonomyMats(hmax);weights=[ComplexF64[-(2i*Qbvals[s]+q1vals[s])+theta/pi for s in 1:4] for i in 0:L-1]
    # Dense H blocks have disjoint destinations. Sparse CSC structure cannot be
    # mutated concurrently, so collect sector norms and insert them serially.
    diagonal_norms=Vector{Vector{Matrix{ComplexF64}}}(undef,M)
    function diagonal!(m)
        local r,b,norms,sites,vl,vr
        verbose && (println("[pair momentum] diagonal $m/$M, local+pair=$(length(bases[m].patterns)), momenta=$ks");flush(stdout))
        r=offsets[m]+1:offsets[m+1];b=bases[m]
        norms=insertionMPOMomentumBlocks(b,b,fill([(1,1,I4b)],L),[1.],[1.];
            hermitian=true,translation_invariant=true,cachebytes=cachebytes,skip_local=skip_local,
            retained_bra=retained[m],retained_ket=retained[m],transfer_mode=transfer_mode,sweep_options=sweep_options)
        sites,vl,vr=pairHoppingMPO(2pi*(m-1)/M/Nsites,L;mass=mass)
        insertionMPOMomentumBlocks(b,b,sites,vl,vr;hermitian=true,translation_invariant=true,cachebytes=cachebytes,skip_local=skip_local,
            retained_bra=retained[m],retained_ket=retained[m],transfer_mode=transfer_mode,destination=[view(h,r,r) for h in H],sweep_options=sweep_options)
        sites,vl,vr=pairDiagonalMPO(Nsites,gC,lam,real(h2[m,m]);theta=theta)
        insertionMPOMomentumBlocks(b,b,sites,vl,vr;hermitian=true,cachebytes=cachebytes,skip_local=skip_local,
            retained_bra=retained[m],retained_ket=retained[m],transfer_mode=transfer_mode,destination=[view(h,r,r) for h in H],sweep_options=sweep_options)
        diagonal_norms[m]=norms
    end
    function cross!(sectors)
        local m,mp,phase,r,s
        m,mp=sectors
        verbose && (println("[pair momentum] cross $m,$mp");flush(stdout))
        phase=[dressPhase(2pi*(mp-m)/M,Nsites)(i) for i in 0:L-1]
        r=offsets[m]+1:offsets[m+1];s=offsets[mp]+1:offsets[mp+1]
        insertionMPOMomentumBlocks(bases[m],bases[mp],momentMPOSites(phase,weights,1),[1.,0.],
                    [gC*Nsites*h2[m,mp],2gC*hm[m,mp]];cachebytes=cachebytes,skip_local=skip_local,
                    retained_bra=retained[m],retained_ket=retained[mp],transfer_mode=transfer_mode,destination=[view(h,r,s) for h in H],sweep_options=sweep_options)
        for k in 1:nk;@views H[k][s,r].=H[k][r,s]';end
    end
    crosses=[(m,mp) for m in 1:M for mp in m+1:M]
    # Progress/ETA over the local/local sweeps (they are the whole cost of a one-insertion spectrum; pair
    # columns, when present, are not counted and the line says so).
    progress_on=get(ENV,"PUMPS_PROGRESS","1")!="0" && !skip_local && all(isnothing,retained) &&
        all(m->any(l->l.kind==:local,bases[m].labels),1:M)
    if progress_on
        tdry=time()
        total=sum(1:M) do m
            localMPOWorkUnits(fill([(1,1,I4b)],L),[1.],[1.];translation_invariant=true,hermitian=true)+
            localMPOWorkUnits(pairHoppingMPO(2pi*(m-1)/M/Nsites,L;mass=mass)...;translation_invariant=true,hermitian=true)+
            localMPOWorkUnits(pairDiagonalMPO(Nsites,gC,lam,real(h2[m,m]);theta=theta)...;hermitian=true)
        end
        total+=sum(crosses;init=0) do (m,mp)
            phase=[dressPhase(2pi*(mp-m)/M,Nsites)(i) for i in 0:L-1]
            localMPOWorkUnits(momentMPOSites(phase,weights,1),[1.,0.],[gC*Nsites*h2[m,mp],2gC*hm[m,mp]])
        end
        every=parse(Float64,get(ENV,"PUMPS_PROGRESS_SECONDS","300"))
        haspairs=any(m->any(l->l.kind==:pair,bases[m].labels),1:M)
        println("[assembly] $(M+length(crosses)) sector jobs on $(parallel ? min(Threads.nthreads(),M+length(crosses)) : 1) threads, ",
                "$total work units (counted in $(round(time()-tdry;digits=1)) s); report every $(round(Int,every)) s",
                haspairs ? "; local/local block only (pair columns not counted)" : "");flush(stdout)
        ASSEMBLY_PROGRESS[]=AssemblyProgress(total,Threads.Atomic{Int}(0),time(),every,time(),ReentrantLock())
    end
    try
    if parallel
        # One queue lets cross blocks fill the tail of the diagonal work.
        # A bounded worker pool takes one whole block at a time. Use an explicit
        # queue to retain Julia 1.10 support (:greedy was introduced in 1.11).
        jobs=vcat([(m,m) for m in 1:M],crosses)
        work=Channel{Tuple{Int,Int}}(length(jobs))
        foreach(job->put!(work,job),jobs);close(work)
        @sync for _ in 1:min(Threads.nthreads(),length(jobs))
            Threads.@spawn for (m,mp) in work
                m==mp ? diagonal!(m) : cross!((m,mp))
            end
        end
    else
        foreach(diagonal!,1:M)
        foreach(cross!,crosses)
    end
    finally
        p=ASSEMBLY_PROGRESS[]
        if !isnothing(p)
            ASSEMBLY_PROGRESS[]=nothing
            p.done[]==p.total || println("[assembly] note: counted $(p.done[]) of $(p.total) predicted work units")
            printAssemblyProgress(p,p.total)
        end
    end
    # All sector tasks have joined. Mutate sparse CSC structure only here.
    for m in 1:M,k in 1:nk
        r=offsets[m]+1:offsets[m+1];N[k][r,r]+=sparse(diagonal_norms[m][k])
        # Release each private dense norm block as its sparse destination is filled.
        diagonal_norms[m][k]=Matrix{ComplexF64}(undef,0,0)
    end
    if skip_local
        inds=vcat([collect(offsets[m]+1:offsets[m]+length(localbases[m])) for m in 1:M]...)
        for k in 1:nk;H[k][inds,inds]=local_matrices.H[k];N[k][inds,inds]=local_matrices.N[k];end
    end
    (H=Hermitian.(H),N=Hermitian.(N),bases=bases,offsets=offsets,momenta=ks,assembly_parameters=parameters)
end
