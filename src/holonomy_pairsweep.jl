# Shared insertion sweeps for RingInsertionBasis. Included by holonomy_twoparticle.
# Fix a ket configuration, then evaluate every local bra and every two-insertion
# bra from the same prefix/suffix environments. No physical Hilbert-space vectors.

# Add G[j,l,s] = sum(i,a,b,k) P[i,j,a,b] K[i,k,s] S[a,b,k,l].
# Batch the physical slices of P*K in one GEMM, then close S per slice.
# These are ordinary transposes: bra conjugation is already in the environments.
# Cost O(d D^5), scratch d D^4, instead of forming the O(D^6) product P*S.
function addOpenBraKetFirst!(opened,P,S,K,scratch)
    D,_,d=size(opened);D3=D^3
    mul!(reshape(scratch,D3,D*d),transpose(reshape(P,D,D3)),reshape(K,D,D*d))
    slices=reshape(scratch,D,D3,d);right=reshape(S,D3,D)
    for s in 1:d
        mul!(@view(opened[:,:,s]),@view(slices[:,:,s]),right,1.,1.)
    end
    opened
end

# All first-bra channels share S and K. At large D contract S*K once,
# then close all channel prefixes with a single GEMM. Unlike copying the old
# ket-first workspace B times, the dD^4 intermediates are shared across B.
function openBraChannelWorkspace(D,d,B,ketfirst)
    if ketfirst
        packed=zeros(ComplexF64,D,B,D,D,D);ordered=zeros(ComplexF64,D,D,D,D)
        first=zeros(ComplexF64,D,D,D,D,d);second=similar(first)
        result=zeros(ComplexF64,D,B,D,d);ket=zeros(ComplexF64,D,D,d)
        (packed=packed,packedmat=reshape(packed,D*B,D^3),ordered=ordered,orderedmat=reshape(ordered,D^3,D),
         first=first,firstmat=reshape(first,D^3,D*d),second=second,secondmat=reshape(second,D^3,D*d),
         result=result,resultmat=reshape(result,D*B,D*d),ket=ket,ketmat=reshape(ket,D,D*d))
    else
        packed=zeros(ComplexF64,D^2,B,D^2);ordered=zeros(ComplexF64,D,D,D,D,B)
        first=zeros(ComplexF64,D,D,B,D,D);result=zeros(ComplexF64,d,D,D,B)
        (packed=packed,packedmat=reshape(packed,D^2*B,D^2),ordered=ordered,orderedmat=reshape(ordered,D^2,D^2*B),
         first=first,firstmat=reshape(first,D^2*B,D^2),result=result,resultmat=reshape(result,d,D^2*B))
    end
end
function addOpenBraChannels!(opened,P,S,K,w,ketfirst;suffix_factor=nothing)
    D,_,d,B=size(opened);D2=D^2
    if ketfirst
        permutedims!(w.ordered,reshape(S,D,D,D,D),(1,2,4,3))
        permutedims!(w.ket,reshape(K,D,D,d),(2,1,3))
        mul!(w.firstmat,w.orderedmat,w.ketmat)
        # first[a,b,l,i,s] -> second[i,a,b,l,s]
        permutedims!(w.second,w.first,(4,1,2,3,5))
        permutedims!(w.packed,reshape(P,D,D,D,D,B),(2,5,1,3,4))
        mul!(w.resultmat,w.packedmat,w.secondmat)
        raw=w.result
        @inbounds for c in 1:B,s in 1:d,l in 1:D,j in 1:D;opened[j,l,s,c]+=raw[j,c,l,s];end
    else
        permutedims!(w.packed,reshape(P,D2,D2,B),(1,3,2))
        if isnothing(suffix_factor)
            mul!(w.firstmat,w.packedmat,S)
        else
            mul!(w.firstmat,w.packedmat*suffix_factor.left,suffix_factor.right)
        end
        permutedims!(w.ordered,w.first,(1,4,2,5,3))
        mul!(w.resultmat,transpose(reshape(K,D2,d)),w.orderedmat)
        raw=w.result
        @inbounds for c in 1:B,s in 1:d,l in 1:D,j in 1:D;opened[j,l,s,c]+=raw[s,j,l,c];end
    end
    opened
end

function insertionMPOProject(bra::RingInsertionBasis,ket::RingInsertionBasis,sites,vL,vR;
        hermitian=false,cachebytes=128*2^20,translation_invariant=false,diagonal_momenta=false,
        openbra_order=:auto,transfer_mode=:auto,batch_origins=true,fourier_batch_size=8,channel_batch_size=4,destination=nothing,
        batch_openbra=true,batch_factorized=false,batch_workspace_bytes=256*2^20,work_control=nothing,
        suffix_rtol=0.,suffix_min_sites=8,suffix_max_rank_fraction=.5)
    checkPairWork(work_control)
    Nb=bra.Nb; Nb==ket.Nb==length(sites) || throw(DimensionMismatch("ring length"))
    D=size(bra.tensors[1],1);d=size(bra.tensors[1],3);D2=D^2
    openbra_order in (:auto,:environment_first,:ket_first) || throw(ArgumentError("invalid openbra_order"))
    # Measured crossover for the blocked Schwinger physical dimension, BLAS=1.
    # Other physical dimensions keep the established path unless explicitly selected.
    ketfirst=openbra_order==:ket_first || (openbra_order==:auto && d==4 && D>=12)
    0<=suffix_rtol<1 && suffix_min_sites>=1 && 0<suffix_max_rank_fraction<=1 ||
        throw(ArgumentError("invalid suffix compression controls"))
    # Experimental: exact rolling environments, low-rank closure only. Keep
    # short suffixes exact, and fall back to dense if the accepted rank is large.
    suffix_rtol>0 && ketfirst && throw(ArgumentError("suffix compression currently needs environment_first closure"))
    transfer_mode in (:auto,:dense,:factorized) || throw(ArgumentError("invalid transfer_mode"))
    # The automatic crossover is selected by the matched Feynman benchmark.
    factorized=transfer_mode==:factorized || (transfer_mode==:auto && d==4 && D>=24)
    size(ket.tensors[1])==size(bra.tensors[1]) || throw(DimensionMismatch("bra/ket dimensions"))
    chi=length(vL);length(vR)==chi || throw(DimensionMismatch("MPO boundaries"))
    hermitian && bra!==ket && throw(ArgumentError("Hermitian projection requires the same basis"))
    ops=Matrix{ComplexF64}[];opids=Dict{Matrix{ComplexF64},Int}()
    edges=[Tuple{Int,Int,Int}[] for _ in 1:Nb]
    for i in 1:Nb,(l,r,op) in sites[i]
        1<=l<=chi && 1<=r<=chi && size(op)==(d,d) || throw(DimensionMismatch("MPO edge"))
        id=get!(opids,op) do;push!(ops,op);length(ops);end
        push!(edges[i],(l,r,id))
    end
    # Active channels depend only on the MPO graph and boundary vectors.
    pa=falses(chi,Nb+1);sa=copy(pa);pa[:,1].=.!iszero.(vL);sa[:,end].=.!iszero.(vR)
    for i in 1:Nb,(l,r,_) in edges[i];pa[l,i] && (pa[r,i+1]=true);end
    for i in Nb:-1:1,(l,r,_) in edges[i];sa[r,i+1] && (sa[l,i]=true);end
    # Omit edges that cannot belong to a complete boundary-to-boundary path.
    live=[[(l,r,o) for (l,r,o) in edges[i] if pa[l,i] && sa[r,i+1]] for i in 1:Nb]
    maxcache=max(1,cachebytes÷(16*(D2^2+D2*d)))
    transfers=Dict{NTuple{3,Int},Matrix{ComplexF64}}()
    ketops=Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    function ketop(k,o)
        get!(ketops,(k,o)) do
            length(ketops)>=maxcache && empty!(ketops)
            reshape(ket.tensors[k],D2,d)*ops[o]
        end
    end
    function transfer(b,k,o)
        get!(transfers,(b,k,o)) do
            length(transfers)>=maxcache && empty!(transfers)
            fastMixedTransfer(bra.tensors[b],reshape(ketop(k,o),D,D,d))
        end
    end
    work=factorized ? BatchedMixedWorkspace(D,d) : nothing
    brafactors=Dict{Int,Any}();ketfactors=Dict{Tuple{Int,Int},Any}()
    braFactor(b)=get!(() -> mixedBraFactors(bra.tensors[b]),brafactors,b)
    function ketFactor(k,o)
        get!(ketfactors,(k,o)) do
            length(ketfactors)>=maxcache && empty!(ketfactors)
            mixedKetFactors(reshape(ketop(k,o),D,D,d))
        end
    end
    function left!(out,b,k,o,X)
        factorized ? batchedMixedLeftApply!(out,braFactor(b),ketFactor(k,o),X,work;beta=1.) :
            mul!(out,transfer(b,k,o),X,1.,1.)
    end
    function right!(out,X,b,k,o)
        factorized ? batchedMixedRightApply!(out,X,braFactor(b),ketFactor(k,o),work;beta=1.) :
            mul!(out,X,transfer(b,k,o),1.,1.)
    end
    bank()=[zeros(ComplexF64,D2,D2) for _ in 1:chi]
    # Consume prefixes in site order instead of retaining the whole ring.
    # S[x+1] is still needed at every possible second bra insertion.
    P=bank();Pnext=bank();S=[i==1 ? Matrix{ComplexF64}[] : bank() for i in 1:Nb+1]
    suffix_factors=Matrix{Union{Nothing,NamedTuple{(:left,:right),Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}}}(nothing,chi,Nb+1)
    for c in 1:chi,j in 1:D2;S[end][c][j,j]=vR[c];end
    env=zeros(ComplexF64,D2,ketfirst ? d*D2 : D2)
    flat=ketfirst ? zeros(ComplexF64,0,0) : similar(env)
    F=zeros(ComplexF64,d,D2);opened=zeros(ComplexF64,D,D,d)
    env4=ketfirst ? nothing : reshape(env,D,D,D,D)
    flat4=ketfirst ? nothing : reshape(flat,D,D,D,D)
    F3=reshape(F,d,D,D);openedvector=vec(opened)
    function openbra!(prefix,suffix,k,i)
        if ketfirst
            fill!(opened,0)
            for (l,r,o) in live[i]
                addOpenBraKetFirst!(opened,prefix[l],suffix[r],ketop(k,o),env)
            end
        else
            fill!(F,0)
            for (l,r,o) in live[i]
                f=suffix_factors[r,i+1]
                isnothing(f) ? mul!(env,prefix[l],suffix[r]) : mul!(env,prefix[l]*f.left,f.right)
                permutedims!(flat4,env4,(1,3,2,4))
                mul!(F,transpose(ketop(k,o)),flat,1.,1.)
            end
            permutedims!(opened,F3,(2,3,1))
        end
        openedvector
    end
    localmotifs=findall(l->l.kind==:local,bra.labels)
    # Exact test: custom, reordered or rotated local bases still need projection.
    unitlocal=isUnitLocalBasis(bra,localmotifs)
    localvectors=unitlocal ? nothing : zeros(ComplexF64,D2*d,length(localmotifs))
    if !unitlocal
        for (a,j) in enumerate(localmotifs);localvectors[:,a]=vec(bra.tensors[bra.patterns[j][1,1]]);end
    end
    # labels is Vector{NamedTuple}; constrain the reduction result so channel
    # ranges, workspace views and the inner sweep remain type-stable.
    q::Int=maximum((max(l.a,l.b) for l in bra.labels if l.kind==:pair);init=0)
    channel_batch_size isa Integer && channel_batch_size>=1 || throw(ArgumentError("positive channel batch size required"))
    batch_workspace_bytes>=0 || throw(ArgumentError("nonnegative batch workspace budget required"))
    # Budget extra persistent scratch per contraction worker, above scalar work.
    # H/N and the pre-existing environment/cache banks are outside this budget.
    requested_batch=max(1,min(q,channel_batch_size))
    function extraBytes(B)
        B==1 && return 0
        openbytes=batch_openbra ? 16*(ketfirst ? (B+1+2d)*D^4+2d*D2*B : 3B*D^4+2d*D2*B) : 0
        factorbytes=factorized && batch_factorized ? 16*(2d+1)*B*D^4 : 0
        openbytes+factorbytes+16*2chi*(B-1)*D^4
    end
    # Bind abatch once: mutating a variable captured by slices() boxes it and
    # makes the inner channel loops allocate scalar objects on every iteration.
    abatch::Int=let candidate=requested_batch
        while candidate>1 && extraBytes(candidate)>batch_workspace_bytes;candidate-=1;end
        candidate
    end
    channelwork=factorized && batch_factorized && abatch>1 ? BatchedMixedWorkspace(D,d,abatch) : work
    openwork=batch_openbra && abatch>1 ? openBraChannelWorkspace(D,d,abatch,ketfirst) : nothing
    channelopened=isnothing(openwork) ? nothing : zeros(ComplexF64,D,D,d,abatch)
    channelvalues=zeros(ComplexF64,q,abatch)
    Q=[zeros(ComplexF64,D2,D2*abatch) for _ in 1:chi];Qnext=deepcopy(Q)
    slices(bank)=[[@view(bank[c][:,(a-1)*D2+1:a*D2]) for c in 1:chi] for a in 1:abatch]
    Qs=slices(Q);Qnexts=slices(Qnext)
    openshape=ketfirst ? (D,D,D,D,abatch) : (D2,D2,abatch)
    Qopen=[reshape(E,openshape) for E in Q];Qnextopen=[reshape(E,openshape) for E in Qnext]
    Sopen=ketfirst ? [[reshape(E,D,D,D,D) for E in bank] for bank in S] : S
    channelids=zeros(Int,q);targets=zeros(Int,Nb,q,q)
    for (j,l) in enumerate(bra.labels)
        l.kind==:pair || continue
        targets[l.r,l.a,l.b]=j
        channelids[l.a]=bra.patterns[j][1,1]
        channelids[l.b]=bra.patterns[j][mod1(1+l.r,Nb),1]
    end
    channelvectors=zeros(ComplexF64,D2*d,q)
    for a in 1:q;channelvectors[:,a]=vec(bra.tensors[channelids[a]]);end
    gaps=[any(!iszero,targets[g,:,:]) || any(!iszero,targets[Nb-g,:,:]) for g in 1:Nb-1]
    lastsites=[begin z=findlast(g->g<=Nb-x && gaps[g],1:Nb-1);isnothing(z) ? x : x+z end for x in 1:Nb-1]
    localvalues=zeros(ComplexF64,length(localmotifs));pairvalues=zeros(ComplexF64,q)
    nkb=length(bra.momenta);nkk=length(ket.momenta)
    diagonal_momenta && bra.momenta!=ket.momenta && throw(ArgumentError("matching momenta required"))
    fourier_batch_size isa Integer && fourier_batch_size>=1 || throw(ArgumentError("positive Fourier batch size required"))
    # In matching momentum blocks the two phases depend only on x-y. Sum
    # origins in relative coordinates BEFORE Fourier transformation. Custom
    # phase matrices keep the general contraction, without an approximate test.
    standardphases(b)=all(b.phases[x,k]==cis(2pi*b.momenta[k]*(x-1)/Nb)/sqrt(Nb)
        for k in eachindex(b.momenta),x in 1:Nb)
    batched=diagonal_momenta && batch_origins && standardphases(bra) && standardphases(ket)
    batchsize=batched ? min(isnothing(work_control) ? fourier_batch_size : work_control.columns,max(1,length(ket.patterns))) : 1
    accumulated=batched ? zeros(ComplexF64,Nb,length(bra.patterns),batchsize) : nothing
    transformed=batched ? zeros(ComplexF64,nkb,length(bra.patterns)*batchsize) : nothing
    column=zeros(ComplexF64,length(bra))
    !isnothing(destination) && !diagonal_momenta && throw(ArgumentError("destination requires matching momentum blocks"))
    H=diagonal_momenta ? (isnothing(destination) ? [zeros(ComplexF64,length(bra.patterns),length(ket.patterns)) for _ in 1:nkb] : destination) : zeros(ComplexF64,length(bra),length(ket))
    diagonal_momenta && (length(H)!=nkb || any(size(h)!=(length(bra.patterns),length(ket.patterns)) for h in H)) && throw(DimensionMismatch("projection destination"))
    amplitudes=zeros(ComplexF64,Nb,length(bra.patterns))
    fouriercolumn=reshape(column,nkb,length(bra.patterns))
    function add!(j,x,y,slot,value)
        iszero(j) && return
        if batched
            r=x-y+1;r<1 && (r+=Nb)
            accumulated[r,j,slot]+=value
        else
            amplitudes[x,j]+=value
        end
    end
    # Only for an EXACTLY translation-invariant operator, one ket origin is
    # sufficient; different momenta vanish. Never apply this to dressed electric
    # terms or to individual open kernel MPOs merely because the MPS is uniform.
    origins=translation_invariant ? (1:1) : (1:Nb)
    checkpointkey=diagonal_momenta ? pairCheckpointKey(work_control,:columns,bra,ket,sites,vL,vR,
        hermitian,translation_invariant,openbra_order,transfer_mode,batch_origins,batchsize,
        channel_batch_size,batch_openbra,batch_factorized,batch_workspace_bytes,
        suffix_rtol,suffix_min_sites,suffix_max_rank_fraction) : nothing
    chunkbefore=nothing;chunkrange=1:0;skipchunk=false
    for (j,pattern) in enumerate(ket.patterns)
      slot=mod1(j,batchsize)
      if slot==1
          checkPairWork(work_control)
          chunkrange=j:min(j+batchsize-1,length(ket.patterns))
          saved=pairLoadChunk(checkpointkey,chunkrange);skipchunk=!isnothing(saved)
          if skipchunk
              for k in eachindex(H);@views H[k][:,chunkrange].+=saved[k];end
          elseif !isnothing(checkpointkey)
              chunkbefore=[copy(h[:,chunkrange]) for h in H]
          end
      end
      skipchunk && continue
      batched && slot==1 && fill!(accumulated,0)
      for y in origins
        checkPairWork(work_control)
        for i in Nb:-1:2
            foreach(E->fill!(E,0),S[i]);k=pattern[i,y]
            for (l,r,o) in live[i];right!(S[i][l],S[i+1][r],1,k,o);end
        end
        if suffix_rtol>0
            fill!(suffix_factors,nothing)
            for i in 2:Nb+1-suffix_min_sites,c in 1:chi
                sa[c,i] || continue
                checkPairWork(work_control)
                f=svd(S[i][c]);scale=maximum(f.S;init=0.)
                rank=count(>(suffix_rtol*scale),f.S)
                if 0<rank<=floor(Int,suffix_max_rank_fraction*D2)
                    suffix_factors[c,i]=(left=f.U[:,1:rank]*Diagonal(f.S[1:rank]),right=f.Vt[1:rank,:])
                end
            end
        end
        foreach(E->fill!(E,0),P)
        for c in 1:chi,t in 1:D2;P[c][t,t]=vL[c];end
        batched || fill!(amplitudes,0)
        for x in 1:Nb
            if !isempty(localmotifs)
                openbra!(P,S[x+1],pattern[x,y],x)
                if unitlocal
                    copyto!(localvalues,openedvector)
                else
                    mul!(localvalues,localvectors',openedvector)
                end
                for (a,i) in enumerate(localmotifs);add!(i,x,y,slot,localvalues[a]);end
            end
            for firsta in 1:abatch:(x<Nb ? q : 0)
                lastz=lastsites[x];lastz==x && continue
                active=min(abatch,q-firsta+1)
                foreach(E->fill!(E,0),Q)
                for ai in 1:active,(l,r,o) in live[x]
                    left!(Qs[ai][r],channelids[firsta+ai-1],pattern[x,y],o,P[l])
                end
                for z in x+1:lastz
                    g=z-x
                    if gaps[g]
                        if !isnothing(openwork)
                            fill!(channelopened,0)
                            for (l,r,o) in live[z]
                                addOpenBraChannels!(channelopened,Qopen[l],Sopen[z+1][r],ketop(pattern[z,y],o),openwork,ketfirst;
                                    suffix_factor=suffix_factors[r,z+1])
                            end
                            mul!(channelvalues,channelvectors',reshape(channelopened,D2*d,abatch))
                        end
                        for ai in 1:active
                            a=firsta+ai-1
                            if isnothing(openwork)
                                mul!(pairvalues,channelvectors',openbra!(Qs[ai],S[z+1],pattern[z,y],z))
                            end
                            for b in 1:q
                                value=isnothing(openwork) ? pairvalues[b] : channelvalues[b,ai]
                                add!(targets[g,a,b],x,y,slot,value)
                                add!(targets[Nb-g,b,a],z,y,slot,value)
                            end
                        end
                    end
                    if z<lastz
                        foreach(E->fill!(E,0),Qnext)
                        for (l,r,o) in live[z]
                            if factorized && batch_factorized
                                batchedMixedLeftApply!(Qnext[r],braFactor(1),ketFactor(pattern[z,y],o),Q[l],channelwork;beta=1.)
                            elseif factorized
                                for ai in 1:active
                                    left!(Qnexts[ai][r],1,pattern[z,y],o,Qs[ai][l])
                                end
                            else
                                @views mul!(Qnext[r][:,1:active*D2],transfer(1,pattern[z,y],o),Q[l][:,1:active*D2],1.,1.)
                            end
                        end
                        Q,Qnext=Qnext,Q
                        Qs,Qnexts=Qnexts,Qs
                        Qopen,Qnextopen=Qnextopen,Qopen
                    end
                end
            end
            if x<Nb
                foreach(E->fill!(E,0),Pnext)
                for (l,r,o) in live[x];left!(Pnext[r],1,pattern[x,y],o,P[l]);end
                P,Pnext=Pnext,P
            end
        end
        # Batch the Fourier transform rather than looping over momenta at
        # every insertion pair. This matters when all Nb momenta are retained.
        batched && continue
        mul!(fouriercolumn,bra.phases',amplitudes)
        for kk in 1:nkk
            col=(j-1)*nkk+kk
            if diagonal_momenta
                for i in eachindex(bra.patterns)
                    H[kk][i,j]+=(translation_invariant ? Nb : 1)*column[(i-1)*nkb+kk]*ket.phases[y,kk]
                end
                continue
            end
            if translation_invariant
                kb=findfirst(==(ket.momenta[kk]),bra.momenta)
                if !isnothing(kb)
                    for i in eachindex(bra.patterns)
                        row=(i-1)*nkb+kb
                        H[row,col]+=Nb*column[row]*ket.phases[y,kk]
                    end
                end
            else
                for row in eachindex(column);H[row,col]+=column[row]*ket.phases[y,kk];end
            end
        end
      end
      if batched && (slot==batchsize || j==length(ket.patterns))
          mul!(transformed,bra.phases',reshape(accumulated,Nb,:))
          factor=(translation_invariant ? Nb : 1)/sqrt(Nb)
          for s in 1:slot,k in 1:nkb,i in eachindex(bra.patterns)
              H[k][i,j-slot+s]+=factor*transformed[k,i+(s-1)*length(bra.patterns)]
          end
      end
      if !isnothing(checkpointkey) && (slot==batchsize || j==length(ket.patterns))
          pairSaveChunk(work_control,checkpointkey,chunkrange,[h[:,chunkrange]-b for (h,b) in zip(H,chunkbefore)])
      end
    end
    if diagonal_momenta
        return hermitian ? hermitizePairBlocks!(H) : H
    end
    hermitian ? (H+H')/2 : H
end
