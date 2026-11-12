# Sector/momentum blocks of the norm. Hamiltonian momentum couplings are retained.
using SparseArrays
function twoParticleNormBlocks(bases;localonly=false)
    blocks=Vector{Int}[];offset=0
    for b in bases
        nk=length(b.momenta)
        motifs=localonly ? count(l->l.kind==:local,b.labels) : length(b.patterns)
        n=nk*motifs
        for k in 1:nk;push!(blocks,collect(offset+k:nk:offset+n));end
        offset+=n
    end
    blocks
end

function twoParticleWhitener(N,cutoff,blocks)
    n=size(N,1);dn=real.(Vector(diag(N)));mx=maximum(dn;init=0.)
    mx>0 || throw(ArgumentError("zero-norm variational space"))
    # A symmetry-null Bloch column can have a small positive diagonal from
    # cancellation. Normalizing it promotes roundoff to a unit-norm fake state.
    # Remove diagonals at the accumulation-error scale BEFORE rescaling.
    columnfloor=100eps(Float64)*n*mx
    structured=!isnothing(blocks)
    groups=structured ? blocks : [collect(1:n)]
    sort(vcat(groups...))==collect(1:n) || throw(ArgumentError("norm blocks must partition all coordinates"))
    if structured
        membership=zeros(Int,n)
        for (j,g) in enumerate(groups);membership[g].=j;end
        # Check the structural promise without making another dense matrix.
        limit=100eps(Float64)*mx
        if parent(N) isa SparseMatrixCSC
            rows,cols,values=findnz(parent(N))
            any(t->membership[rows[t]]!=membership[cols[t]] && abs(values[t])>limit,eachindex(values)) &&
                throw(ArgumentError("norm couples the supplied blocks"))
        else
            for j in 1:n,i in 1:n
                membership[i]!=membership[j] && abs(N[i,j])>limit &&
                    throw(ArgumentError("norm couples the supplied blocks"))
            end
        end
    end
    pieces=NamedTuple[];allvalues=Float64[]
    for group in groups
        inds=[i for i in group if dn[i]>columnfloor]
        isempty(inds) && continue
        scales=1 ./ sqrt.(dn[inds])
        Sn=Matrix(N[inds,inds]).*(scales*scales')
        f=eigen(Hermitian((Sn+Sn')/2))
        append!(allvalues,f.values);push!(pieces,(inds=inds,scales=scales,f=f))
    end
    largest=maximum(allvalues);threshold=cutoff*largest
    minimum(allvalues)>=-max(100eps(Float64)*length(allvalues),cutoff)*largest ||
        error("projected norm is not positive semidefinite")
    rank=sum(count(>(threshold),p.f.values) for p in pieces)
    if structured
        rows=Int[];cols=Int[];vals=ComplexF64[];offset=0
        for p in pieces
            keep=findall(>(threshold),p.f.values)
            B=p.scales.*p.f.vectors[:,keep].*reshape(1 ./ sqrt.(p.f.values[keep]),1,:)
            for j in axes(B,2),i in axes(B,1)
                push!(rows,p.inds[i]);push!(cols,offset+j);push!(vals,B[i,j])
            end
            offset+=length(keep)
        end
        W=sparse(rows,cols,vals,n,rank)
    else
        p=only(pieces);keep=findall(>(threshold),p.f.values);W=zeros(ComplexF64,n,rank)
        W[p.inds,:]=p.scales.*p.f.vectors[:,keep].*reshape(1 ./ sqrt.(p.f.values[keep]),1,:)
    end
    # Hermitian{SparseMatrixCSC} does not dispatch to sparse-sparse matmul in
    # Julia 1.12. Materialize the sparse wrapper, not a dense global norm.
    Nwork=structured ? sparse(N) : Matrix(N)
    normerror=norm(W'*(Nwork*W)-I,Inf)
    normerror<=1e-6 || error("norm whitening lost accuracy ($normerror); increase the norm cutoff")
    (W=W,values=sort!(allvalues),threshold=threshold,columnfloor=columnfloor,error=normerror,rank=rank)
end
