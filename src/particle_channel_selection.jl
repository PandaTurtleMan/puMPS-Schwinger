# Shared channel compression, with optional finite-ring norm weighting.
# One full pool is selected once; channel ladders use its nested prefixes.
function selectParticleChannels(samples,shape,nchannels;metric=nothing,cutoff=1e-10)
    nchannels>0 && 0<cutoff<1 || throw(ArgumentError("invalid channel truncation"))
    isempty(samples)&&throw(ArgumentError("empty particle samples"))
    X=hcat(samples...);all(isfinite,X)||throw(ArgumentError("nonfinite particle samples"))
    if isnothing(metric)
        f=svd(X);n=min(nchannels,count(>(cutoff*first(f.S)),f.S))
        return (channels=[reshape(f.U[:,j],shape) for j in 1:n],singular_values=f.S,
            metric_rank=size(X,1),discarded_weight=sum(abs2,f.S[n+1:end])/sum(abs2,f.S))
    end
    size(metric)==(size(X,1),size(X,1)) || throw(DimensionMismatch("channel metric"))
    e=eigen(Hermitian((metric+metric')/2));scale=maximum(e.values;init=0.)
    scale>0 || throw(ArgumentError("empty channel norm metric"))
    minimum(e.values)>=-100cutoff*scale || throw(ArgumentError("indefinite channel norm metric"))
    keep=findall(>(cutoff*scale),e.values);U=e.vectors[:,keep];s=sqrt.(e.values[keep])
    Y=Diagonal(s)*(U'*X)
    norms=vec(sqrt.(sum(abs2,Y;dims=1)));valid=findall(>(cutoff*maximum(norms)),norms)
    isempty(valid)&&throw(ArgumentError("samples are null in the channel metric"))
    Y=Y[:,valid]./reshape(norms[valid],1,:)
    f=svd(Y);n=min(nchannels,count(>(cutoff*first(f.S)),f.S))
    B=U*(Diagonal(1 ./s)*f.U[:,1:n])
    (channels=[reshape(B[:,j],shape) for j in 1:n],singular_values=f.S,
        metric_rank=length(keep),discarded_weight=sum(abs2,f.S[n+1:end])/sum(abs2,f.S))
end
