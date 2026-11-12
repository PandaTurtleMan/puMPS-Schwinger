# Direct tangent contractions. Included after holonomy_fixed_solve.jl.
# A supplied ket impurity replaces the entire bank of dD² basis impurities.
# Every environment is D² × D²; no dD² × D⁴ tangent intermediate is formed.
struct DirectMPOEdge
    left::Int
    right::Int
    op::Matrix{ComplexF64} # physical [ket,bra]
    transfer::Matrix{ComplexF64}
    ketop::Matrix{ComplexF64}
end

function directMPOEdges(W)
    [(l,r,Matrix{ComplexF64}(W[l,r,:,:])) for l in axes(W,1), r in axes(W,2)
        if !all(iszero,@view(W[l,r,:,:]))] |> vec
end

function prepareDirectMPO(Ab, Ak, liB, liK, sites, vL, vR)
    D=size(Ak,1); d=size(Ak,3); chi=length(vL); Nb=length(sites); D2=D^2
    cache=Dict{Matrix{ComplexF64},Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
    edges=Vector{Vector{DirectMPOEdge}}(undef,Nb)
    for i in 1:Nb
        edges[i]=DirectMPOEdge[]
        for (l,r,op) in sites[i]
            T,U=get!(cache,op) do
                U=reshape(Ak,D2,d)*op
                fastMixedTransfer(Ab,reshape(U,D,D,d)),U
            end
            push!(edges[i],DirectMPOEdge(l,r,op,T,U))
        end
    end
    P=[[zeros(ComplexF64,D2,D2) for _ in 1:chi] for _ in 1:Nb+1]
    S=[[zeros(ComplexF64,D2,D2) for _ in 1:chi] for _ in 1:Nb+1]
    for c in 1:chi
        P[1][c].=vL[c].*Matrix{ComplexF64}(I,D2,D2)
        S[Nb+1][c].=vR[c].*Matrix{ComplexF64}(I,D2,D2)
    end
    pa=falses(chi,Nb+1); sa=copy(pa); pa[:,1].=.!iszero.(vL); sa[:,end].=.!iszero.(vR)
    for i in 1:Nb, e in edges[i]
        pa[e.left,i] || continue
        mul!(P[i+1][e.right],e.transfer,P[i][e.left],1.0,1.0)
        pa[e.right,i+1]=true
    end
    for i in Nb:-1:1, e in edges[i]
        sa[e.right,i+1] || continue
        mul!(S[i][e.left],S[i+1][e.right],e.transfer,1.0,1.0)
        sa[e.left,i]=true
    end
    return (bra=Ab,ket=Ak,liB=liB,liK=liK,edges=edges,P=P,S=S,pa=pa,sa=sa,
            sites=sites,vL=vL,vR=vR,D=D,d=d,chi=chi,Nb=Nb)
end

function adjointDirectMPO(c)
    sites=[[(l,r,Matrix(op')) for (l,r,op) in site] for site in c.sites]
    prepareDirectMPO(c.ket,c.bra,c.liK,c.liB,sites,conj.(c.vL),conj.(c.vR))
end

# Sum_y braweights[y] <bra impurity at y| O |sum_x ketweights[x] B at x>.
# Weights include the Bloch phases but not the Schmidt inverses (handled here).
function directMPOAction(c, v, ketweights, braweights)
    D=c.D; d=c.d; D2=D^2; chi=c.chi; Nb=c.Nb
    B=reshape(v,D,D,d).*reshape(c.liK,1,D,1)
    Bm=reshape(B,D2,d)
    cache=Dict{Matrix{ComplexF64},Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
    for site in c.edges, e in site
        get!(cache,e.op) do
            U=Bm*e.op
            fastMixedTransfer(c.bra,reshape(U,D,D,d)),U
        end
    end
    Q=[[zeros(ComplexF64,D2,D2) for _ in 1:chi] for _ in 1:Nb+1]
    qa=falses(chi,Nb+1)
    for i in 1:Nb, e in c.edges[i]
        if qa[e.left,i]
            mul!(Q[i+1][e.right],e.transfer,Q[i][e.left],1.0,1.0)
            qa[e.right,i+1]=true
        end
        if !iszero(ketweights[i]) && c.pa[e.left,i]
            K,_=cache[e.op]
            mul!(Q[i+1][e.right],K,c.P[i][e.left],ketweights[i],1.0)
            qa[e.right,i+1]=true
        end
    end
    R=[zeros(ComplexF64,D2,D2) for _ in 1:chi]; Rnext=deepcopy(R)
    ra=falses(chi); ranext=copy(ra)
    F=zeros(ComplexF64,d,D2); env=zeros(ComplexF64,D2,D2)
    flatbuffer=similar(env)
    function flat(E)
        permutedims!(reshape(flatbuffer,D,D,D,D),reshape(E,D,D,D,D),(1,3,2,4))
        return flatbuffer
    end
    for i in Nb:-1:1
        for e in c.edges[i]
            iszero(braweights[i]) && continue
            fill!(env,0); live=false
            if qa[e.left,i] && c.sa[e.right,i+1]
                mul!(env,Q[i][e.left],c.S[i+1][e.right]); live=true
            end
            if c.pa[e.left,i] && ra[e.right]
                mul!(env,c.P[i][e.left],R[e.right],1.0,1.0); live=true
            end
            live && mul!(F,transpose(e.ketop),flat(env),braweights[i],1.0)
            if !iszero(ketweights[i]) && c.pa[e.left,i] && c.sa[e.right,i+1]
                mul!(env,c.P[i][e.left],c.S[i+1][e.right])
                _,U=cache[e.op]
                mul!(F,transpose(U),flat(env),braweights[i]*ketweights[i],1.0)
            end
        end
        foreach(M->fill!(M,0),Rnext); fill!(ranext,false)
        for e in c.edges[i]
            if ra[e.right]
                mul!(Rnext[e.left],R[e.right],e.transfer,1.0,1.0); ranext[e.left]=true
            end
            if !iszero(ketweights[i]) && c.sa[e.right,i+1]
                K,_=cache[e.op]
                mul!(Rnext[e.left],c.S[i+1][e.right],K,ketweights[i],1.0); ranext[e.left]=true
            end
        end
        R,Rnext=Rnext,R; ra,ranext=ranext,ra
    end
    return vec(permutedims(reshape(F,d,D,D),(2,3,1)).*reshape(c.liB,1,D,1))
end

function momentMPOSites(phases, weights, order)
    # Upper triangular moment automaton: entry (1,3) is (sum_i weights_i)^2.
    [[(l,r,Matrix(Diagonal(phases[i].*weights[i].^(r-l).*binomial(r-1,l-1))))
        for l in 1:order+1 for r in l:order+1] for i in eachindex(phases)]
end

# Exact decomposition of a two-site gate into local matrix units. No SVD cutoff.
# Add an open-chain nearest-neighbor automaton and explicit wrap-bond channels.
function gateMPOSites(hmat, d, Nb)
    terms=Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}[]
    for st in 1:d, sb in 1:d
        R=Matrix(transpose(hmat[(sb-1)*d+1:sb*d,(st-1)*d+1:st*d]))
        all(iszero,R) && continue
        L=zeros(ComplexF64,d,d); L[st,sb]=1
        push!(terms,(L,R))
    end
    nr=length(terms); chi=2nr+2; last=chi; id=Matrix{ComplexF64}(I,d,d)
    sites=Vector{Vector{Tuple{Int,Int,Matrix{ComplexF64}}}}(undef,Nb)
    for i in 1:Nb
        edges=[(1,1,id),(last,last,id)]
        for (j,(L,R)) in enumerate(terms)
            push!(edges,(1,1+j,L),(1+j,last,R))
            wrap=1+nr+j
            i==1 && push!(edges,(1,wrap,R))
            1<i<Nb && push!(edges,(wrap,wrap,id))
            i==Nb && push!(edges,(wrap,last,L))
        end
        sites[i]=edges
    end
    vL=zeros(ComplexF64,chi); vR=copy(vL); vL[1]=1; vR[end]=1
    return sites,vL,vR
end

# Norm separation blocks via two D² GEMMs per physical pair, replacing the
# dense dD² × D⁴ impurity arrays. This prepares the unchanged norm cutoff.
function directPairBlock(A, li, L, Mid)
    D=size(A,1); d=size(A,3); D2=D^2
    l4=reshape(L,D,D,D,D); m4=reshape(Mid,D,D,D,D)
    H=zeros(ComplexF64,D,D,d,D,D,d) # bra a,c,s; ket a,c,s
    for sb in 1:d, st in 1:d
        # X[va,ap,uc,a] = sum_b L[va,ap,b,uc] A[a,b,sb]
        X=reshape(reshape(permutedims(l4,(1,2,4,3)),D^3,D)*transpose(A[:,:,sb]),D,D,D,D)
        # Y[a,ua,vc,ap] = sum_bp Mid[a,ua,vc,bp] conj(A[ap,bp,st])
        Y=reshape(reshape(m4,D^3,D)*adjoint(A[:,:,st]),D,D,D,D)
        R=reshape(reshape(permutedims(X,(1,3,2,4)),D2,D2)*
                  reshape(permutedims(Y,(4,1,2,3)),D2,D2),D,D,D,D) # va,uc,ua,vc
        H[:,:,sb,:,:,st].=permutedims(R,(3,2,1,4)).*reshape(li,1,D,1,1).*reshape(li,1,1,1,D)
    end
    return reshape(H,d*D2,d*D2)
end

function directNormBlocks(A, li, kset, Nb)
    D=size(A,1); d=size(A,3); D2=D^2; dt=d*D2
    powers=transferPowers(fastTransferMatrix(A),Nb-1)
    blocks=[zeros(ComplexF64,dt,dt) for _ in kset]
    same=zeros(ComplexF64,dt,dt); env=reshape(powers[Nb],D,D,D,D)
    for s in 1:d, ua in 1:D, uc in 1:D, va in 1:D, vc in 1:D
        same[ua+(uc-1)*D+(s-1)*D2,va+(vc-1)*D+(s-1)*D2]=li[uc]*li[vc]*env[va,ua,vc,uc]
    end
    for b in blocks; b .+= Nb.*same; end
    for sep in 1:Nb-1
        b=directPairBlock(A,li,powers[Nb-sep],powers[sep])
        for (i,k) in enumerate(kset); blocks[i] .+= (Nb*cis(-2pi*k*sep/Nb)).*b; end
    end
    return blocks
end
