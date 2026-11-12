# Optional low-memory excitation backend. Run with --project=matrixfree_env.
# Same one-block Bloch ansatz, norm cutoff and Hermitian Hamiltonian as the dense
# solver. Only the contraction order and eigenproblem solution method change.
include("holonomy_dispersion12.jl")
include("holonomy_direct.jl")
using KrylovKit

function prepareBlochDirect(As,lambdas,Nsites,gC,lam,hmax;mass=0.0)
    iseven(Nsites) && Nsites>=6 || throw(ArgumentError("matrix-free backend requires even Nsites >= 6"))
    M=2hmax+1; Nb=Nsites÷2; D=size(As[1],1); dt=4D^2
    length(As)==length(lambdas)==M || throw(DimensionMismatch("sector count"))
    lis=[1 ./ max.(l,1e-12) for l in lambdas]
    hm,h2,_=holonomyMats(hmax)
    terms=vcat(kernelTermsMerged(Nsites,gC),[(lam,liftingMPO(Nb;qop=Zblk())...)])
    diag=NamedTuple[]; cross=NamedTuple[]
    weights=[eBlk(i) for i in 0:Nb-1]
    for m in 1:M
        A=As[m]; li=lis[m]
        hmat=blockedXXGateTwisted(2pi*(m-1)/M/Nsites;mass=mass)
        sites,vl,vr=gateMPOSites(hmat,4,Nb)
        gate=prepareDirectMPO(A,A,li,li,sites,vl,vr)
        se=gC*Nsites*real(h2[m,m])
        electric=prepareDirectMPO(A,A,li,li,momentMPOSites(fill(ONES4,Nb),weights,2),[1.,0.,0.],[se,0.,gC/Nsites])
        mpos=NamedTuple[]
        for (coef,W,vL,vR) in terms
            iszero(coef) && continue
            c=prepareDirectMPO(A,A,li,li,fill(directMPOEdges(W),Nb),vL,vR)
            hermitian_local=all(e.op==e.op' for e in c.edges[1]) && vL==conj.(vL) && vR==conj.(vR)
            push!(mpos,(coef=coef,forward=c,backward=hermitian_local ? c : adjointDirectMPO(c)))
        end
        push!(diag,(gate=gate,electric=electric,mpos=mpos))
    end
    for m in 1:M, mp in m+1:M
        phase=[dressPhase(2pi*(mp-m)/M,Nsites)(i) for i in 0:Nb-1]
        c=prepareDirectMPO(As[m],As[mp],lis[m],lis[mp],momentMPOSites(phase,weights,1),
                           [1.,0.],[gC*Nsites*h2[m,mp],2gC*hm[m,mp]])
        push!(cross,(m=m,mp=mp,forward=c,backward=adjointDirectMPO(c)))
    end
    return (As=As,lis=lis,Nb=Nb,M=M,D=D,dt=dt,diagonal=diag,cross=cross)
end

function blochDirectAction(c,k,v)
    length(v)==c.M*c.dt || throw(DimensionMismatch("tangent vector length"))
    z=cis.(2pi*k/c.Nb.*(0:c.Nb-1)); zb=conj.(z)
    atfirst=zeros(ComplexF64,c.Nb); atfirst[1]=1
    out=zeros(ComplexF64,length(v))
    for m in 1:c.M
        r=(m-1)*c.dt+1:m*c.dt; vm=@view v[r]; d=c.diagonal[m]
        out[r].=directMPOAction(d.gate,vm,z,zb)+directMPOAction(d.electric,vm,z,zb)
        for t in d.mpos
            # Preserve the reference's fixed-ket Fourier convention, including
            # its final Hermitian symmetrization. An individual open MPO need
            # not be translation invariant, so a full Bloch projection of that
            # term alone is NOT an interchangeable replacement.
            out[r].+=(c.Nb/2).*(t.coef.*directMPOAction(t.forward,vm,atfirst,zb)+
                               conj(t.coef).*directMPOAction(t.backward,vm,z,atfirst))
        end
    end
    for t in c.cross
        rm=(t.m-1)*c.dt+1:t.m*c.dt; rp=(t.mp-1)*c.dt+1:t.mp*c.dt
        out[rm].+=directMPOAction(t.forward,@view(v[rp]),z,zb)
        out[rp].+=directMPOAction(t.backward,@view(v[rm]),z,zb)
    end
    return out
end

function directNormWhitening(c,k;cutoff=1e-10)
    cutoff>=0 || throw(ArgumentError("cutoff must be nonnegative"))
    fs=[eigen(Hermitian(begin
        b=only(directNormBlocks(c.As[m],c.lis[m],[k],c.Nb)); (b+b')/2
    end)) for m in 1:c.M]
    threshold=cutoff*maximum(maximum(f.values) for f in fs)
    ks=[findall(>(threshold),f.values) for f in fs]
    ws=[f.vectors[:,keep].*reshape(1 ./ sqrt.(f.values[keep]),1,:) for (f,keep) in zip(fs,ks)]
    offsets=cumsum([0;length.(ks)])
    return ws,offsets
end

function whitenedDirectAction(c,k,ws,offsets,v)
    full=zeros(ComplexF64,c.M*c.dt)
    for m in 1:c.M
        r=(m-1)*c.dt+1:m*c.dt; s=offsets[m]+1:offsets[m+1]
        mul!(@view(full[r]),ws[m],@view(v[s]))
    end
    h=blochDirectAction(c,k,full); out=zeros(ComplexF64,length(v))
    for m in 1:c.M
        r=(m-1)*c.dt+1:m*c.dt; s=offsets[m]+1:offsets[m+1]
        mul!(@view(out[s]),ws[m]',@view(h[r]))
    end
    return out
end

function directLowestLevels(action,n,nev;tol=1e-8,krylovdim=max(160,4nev),maxiter=50,seed=821,verbose=false)
    nev>=0 || throw(ArgumentError("nev must be nonnegative"))
    tol>0 && maxiter>0 || throw(ArgumentError("positive tol and maxiter required"))
    count=min(nev,n); calls=Ref(0)
    function apply(v)
        calls[]+=1
        verbose && calls[]%25==0 && (println("[direct] matrix-vector product $(calls[])");flush(stdout))
        return action(v)
    end
    count==0 && return Float64[],(residuals=Float64[],matvecs=0,converged=0)
    rng=Xoshiro(seed)
    start=KrylovKit.Block([randn(rng,ComplexF64,n) for _ in 1:count])
    alg=KrylovKit.BlockLanczos(;krylovdim=min(n,max(krylovdim,2count+1)),maxiter=maxiter,tol=tol,eager=true)
    vals,vecs,info=KrylovKit.eigsolve(apply,start,count,:SR,alg)
    info.converged>=count || error("matrix-free eigensolve did not converge: $(info.converged)/$count eigenpairs, $(calls[]) matrix-vector products")
    vals=real.(vals[1:count]); vecs=vecs[1:count]
    residuals=[norm(apply(v)-val*v)/norm(v) for (val,v) in zip(vals,vecs)]
    maximum(residuals)<=tol || error("matrix-free eigenpair residual $(maximum(residuals)) exceeds tolerance $tol")
    return vals,(residuals=residuals,matvecs=calls[],converged=count)
end

function blochDispersionMatrixFree(As,lambdas,Nsites,gC,lam,hmax;
        nev=8,cutoff=1e-10,mass=0.0,momenta=0:(Nsites÷2-1),
        eigtol=1e-8,krylovdim=max(160,4nev),maxiter=50,seed=821,verbose=false,returninfo=false)
    ks=collect(Int,momenta); Nb=Nsites÷2
    nev>=0 && cutoff>=0 && eigtol>0 || throw(ArgumentError("invalid nev, cutoff or eigtol"))
    all(k->0<=k<Nb,ks) && length(unique(ks))==length(ks) || throw(ArgumentError("invalid or duplicate momenta"))
    levels=Vector{Float64}[]; ranks=Int[]; diagnostics=NamedTuple[]
    isempty(ks) && return (returninfo ? (levels,ranks,diagnostics) : (levels,ranks))
    verbose && (println("[direct] preparing transfer environments");flush(stdout))
    c=prepareBlochDirect(As,lambdas,Nsites,gC,lam,hmax;mass=mass)
    for k in ks
        ws,offsets=directNormWhitening(c,k;cutoff=cutoff); rank=last(offsets)
        verbose && (println("[direct] k=$k rank=$rank; block Krylov solve");flush(stdout))
        vals,info=directLowestLevels(v->whitenedDirectAction(c,k,ws,offsets,v),rank,nev;
                                    tol=eigtol,krylovdim=krylovdim,maxiter=maxiter,seed=seed+k,verbose=verbose)
        push!(levels,vals);push!(ranks,rank);push!(diagnostics,info)
        verbose && (println("[direct] k=$k matvecs=$(info.matvecs) residuals=$(info.residuals)");flush(stdout))
    end
    return returninfo ? (levels,ranks,diagnostics) : (levels,ranks)
end
