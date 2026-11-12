# Gauge-covariant OBC-to-uniform seed conversion, included by ground_obc.jl.
# A physical OBC even-odd cell matches the puMPS staggered charge convention.

function obcLeftBlocks(psi, D; first_site=1)
    work=deepcopy(psi);N=length(work)
    iseven(N)||throw(ArgumentError("even chain length required"))
    maximum(linkdims(work))>D && truncate!(work;maxdim=D,cutoff=0.)
    orthogonalize!(work,N)
    first_site in (1,2)||throw(ArgumentError("first_site must be 1 or 2"))
    blocks=Array{ComplexF64,3}[]
    windows=[(i,i+1) for i in first_site:2:N-1]
    first_site==2 && push!(windows,(N,N))
    for (i,j) in windows
        ix=Index[]
        dl=i>1 ? dim(linkind(work,i-1)) : 1
        dr=j<N ? dim(linkind(work,j)) : 1
        i>1 && push!(ix,linkind(work,i-1))
        append!(ix,[siteind(work,k) for k in i:j])
        j<N && push!(ix,linkind(work,j))
        T=i==j ? work[i] : work[i]*work[j]
        raw=reshape(Array(T,ix...),dl,2^(j-i+1),dr)
        push!(blocks,permutedims(ComplexF64.(raw),(1,3,2)))
    end
    blocks,work
end

"""Identify neighboring virtual spaces using a translated-prefix overlap.

For left-canonical blocks B_i, propagate X <- sum_s B_i[s]' X B_{i+1}[s].
At the selected block b, X maps its right virtual space to its left. The unitary
polar factor U supplies the missing identification: A[s] = B_b[s] U'. Under
independent unitary bond changes G_i, this A transforms by a single similarity
G_{b-1}' A G_{b-1}, leaving every periodic state invariant. Simply repeating B_b
does not have that property. Physical translation invariance is approximate;
gauge covariance does not itself guarantee optimal seed quality.
"""
function alignedOBCBlock(blocks,b; boundary_component=nothing,rank_tol=1e-12)
    2<=b<length(blocks)||throw(ArgumentError("select an interior block"))
    A=blocks[b];size(A,1)==size(A,2)||throw(ArgumentError("full square bulk block required"))
    firstblock=first(blocks)
    # Use PHYSICAL first-cell populations. Isometry row norms can all be one
    # while representing very different populations; choosing an unoccupied or
    # charged component can project the shifted overlap into the wrong sector.
    R=ones(ComplexF64,1,1)
    for i in length(blocks):-1:2
        B=blocks[i];R=sum(B[:,:,s]*R*B[:,:,s]' for s in axes(B,3))
        R/=norm(R)
    end
    # The first even-origin cell follows one left-canonical physical site, whose
    # traced left environment is identity; its left bond need not have size one.
    populations=[real(tr(firstblock[:,:,s]*R*firstblock[:,:,s]')) for s in axes(firstblock,3)]
    # Resolve numerical ties by physical index, not floating-point order.
    p=boundary_component===nothing ? findfirst(x->x>=maximum(populations)*(1-1e-12),populations) : Int(boundary_component)
    X=copy(firstblock[:,:,p]);norm(X)>0||error("zero boundary component")
    X/=norm(X)
    for i in 1:b-1
        L=blocks[i];R=blocks[i+1]
        Y=zeros(ComplexF64,size(L,2),size(R,2))
        for s in axes(L,3);Y .+= L[:,:,s]'*X*R[:,:,s];end
        norm(Y)>0||error("zero translated-prefix overlap")
        X=Y/norm(Y)
    end
    F=svd(X);ratio=minimum(F.S)/maximum(F.S)
    ratio>rank_tol||error("rank-deficient virtual-space identification: ratio=$ratio")
    U=F.U*F.Vt
    aligned=similar(A)
    for s in axes(A,3);aligned[:,:,s]=A[:,:,s]*U';end
    aligned,(block=b,boundary_component=p,boundary_populations=populations/sum(populations),overlap_singular_values=F.S,
        singular_ratio=ratio,unitarity_error=norm(U'*U-I),bridge=U)
end

function extractAlignedBulkSeed(psi,N,D;block=nothing,boundary_component=nothing,first_site=2)
    N==length(psi)||throw(ArgumentError("N mismatch"))
    blocks,_=obcLeftBlocks(psi,D;first_site=first_site);center=clamp((N÷2)÷2,2,length(blocks)-1)
    candidates=sort(collect(2:length(blocks)-1);by=b->abs(b-center))
    b=block===nothing ? findfirst(b->size(blocks[b],1)==D && size(blocks[b],2)==D,candidates) : nothing
    block===nothing && b===nothing && error("no full-rank central block at target D")
    chosen=block===nothing ? candidates[b] : Int(block)
    A,meta=alignedOBCBlock(blocks,chosen;boundary_component=boundary_component)
    A,merge(meta,(first_site=first_site,physical_sites=(first_site+2(chosen-1),first_site+2(chosen-1)+1)))
end

function obcCentralDensity(psi,firstsite,count=2)
    work=deepcopy(psi);lastsite=firstsite+count-1;N=length(work)
    1<firstsite<=lastsite<N||throw(ArgumentError("interior density window required"))
    orthogonalize!(work,firstsite)
    ix=Index[linkind(work,firstsite-1)]
    append!(ix,[siteind(work,j) for j in firstsite:lastsite]);push!(ix,linkind(work,lastsite))
    T=work[firstsite]
    for j in firstsite+1:lastsite;T*=work[j];end
    raw=reshape(Array(T,ix...),dim(first(ix)),2^count,dim(last(ix)))
    F=reshape(permutedims(raw,(2,1,3)),2^count,:)
    rho=F*F';rho/tr(rho)
end

function uniformRingFidelity(A,B,ncells)
    function overlap(C,D)
        E=sum(kron(conj(C[:,:,s]),D[:,:,s]) for s in axes(C,3))
        tr(E^ncells)
    end
    abs2(overlap(A,B))/real(overlap(A,A)*overlap(B,B))
end

function finiteBlockOverlap(A,B)
    X=ones(ComplexF64,1,1)
    for (a,b) in zip(A,B)
        X=sum(a[:,:,s]'*X*b[:,:,s] for s in axes(a,3))
    end
    only(X)
end
