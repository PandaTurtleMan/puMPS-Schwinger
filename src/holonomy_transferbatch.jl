# Exact mixed-transfer application, batching physical slices into two GEMMs.
# Factors are prepared once per tensor/operator; no D²-square transfer is formed.
struct BatchedMixedWorkspace
    first::Vector{ComplexF64}
    second::Vector{ComplexF64}
    result::Vector{ComplexF64}
    D::Int
    d::Int
end
BatchedMixedWorkspace(D,d,batch=1)=BatchedMixedWorkspace(zeros(ComplexF64,batch*d*D^4),
    zeros(ComplexF64,batch*d*D^4),zeros(ComplexF64,batch*D^4),D,d)
function mixedBraFactors(B)
    D,_,d=size(B)
    (left=adjoint(reshape(permutedims(B,(1,3,2)),D*d,D)),
     right=reshape(permutedims(conj.(B),(2,1,3)),D,D*d))
end
function mixedKetFactors(K)
    D,_,d=size(K)
    (left=transpose(reshape(K,D,D*d)),right=reshape(permutedims(K,(2,3,1)),D*d,D))
end
function batchedMixedLeftApply!(out,B,K,X,w::BatchedMixedWorkspace;alpha=1.,beta=0.)
    D=w.D;d=w.d;n=size(X,2)
    size(X,1)==D^2 && size(out)==size(X) || throw(DimensionMismatch("mixed left batch"))
    needed=D^2*n
    length(w.result)>=needed || throw(DimensionMismatch("mixed left workspace"))
    # Preserve ordinary Array workspaces on the common full-width path. Views
    # are needed only when a caller uses a larger workspace for a partial batch.
    if length(w.result)==needed
        return mixedLeftBatchCore!(out,B,K,X,w.first,w.second,w.result,D,d,n,alpha,beta)
    end
    mixedLeftBatchCore!(out,B,K,X,view(w.first,1:d*needed),view(w.second,1:d*needed),
        view(w.result,1:needed),D,d,n,alpha,beta)
end
function mixedLeftBatchCore!(out,B,K,X,first,second,raw,D,d,n,alpha,beta)
    mul!(reshape(first,D*d,D*n),K.left,reshape(X,D,D*n))
    permutedims!(reshape(second,D,d,D,n),reshape(first,D,d,D,n),(3,2,1,4))
    mul!(reshape(raw,D,D*n),B.left,reshape(second,D*d,D*n))
    result=reshape(raw,D,D,n);dest=reshape(out,D,D,n)
    if iszero(beta)
        @inbounds for t in 1:n,j in 1:D,i in 1:D;dest[i,j,t]=alpha*result[j,i,t];end
    else
        @inbounds for t in 1:n,j in 1:D,i in 1:D;dest[i,j,t]=beta*dest[i,j,t]+alpha*result[j,i,t];end
    end
    out
end
function batchedMixedRightApply!(out,X,B,K,w::BatchedMixedWorkspace;alpha=1.,beta=0.)
    D=w.D;d=w.d
    mul!(reshape(w.first,D^3,D*d),reshape(X,D^3,D),B.right)
    permutedims!(reshape(w.second,D^2,D,D,d),reshape(w.first,D^2,D,D,d),(1,3,2,4))
    mul!(reshape(w.result,D^3,D),reshape(w.second,D^3,D*d),K.right)
    result=reshape(w.result,D^2,D,D);dest=reshape(out,D^2,D,D)
    if iszero(beta)
        @inbounds for j in 1:D,i in 1:D,t in 1:D^2;dest[t,i,j]=alpha*result[t,j,i];end
    else
        @inbounds for j in 1:D,i in 1:D,t in 1:D^2;dest[t,i,j]=beta*dest[t,i,j]+alpha*result[t,j,i];end
    end
    out
end
