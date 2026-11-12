# Optional: include AFTER holonomy_twoparticle.jl with --project=matrixfree_env.
# Existing pinned KrylovKit dependency; no new package environment is required.
using KrylovKit
function pairKrylovStart(n,count,rng,initial)
    isnothing(initial) && return [randn(rng,ComplexF64,n) for _ in 1:count],0
    size(initial,1)==n && all(isfinite,initial) || throw(ArgumentError("invalid initial Krylov vectors"))
    f=svd(Matrix{ComplexF64}(initial[:,1:min(count,size(initial,2))]))
    keep=findall(>(1e-10*maximum(f.S;init=0.)),f.S)
    warm=[copy(f.U[:,j]) for j in keep];nwarm=length(warm)
    starts=Vector{ComplexF64}[]
    # Give every start an independent random component. This retains overlap
    # with new lower states and degenerate eigenspaces without doubling the
    # Krylov block (which can cost more than the warm start saves).
    for j in 1:count
        v=randn(rng,ComplexF64,n)
        v./=norm(v)
        j<=nwarm && (v .= warm[j] .+ 0.1 .* v)
        for pass in 1:2,u in starts;v.-=dot(u,v).*u;end
        norm(v)>1e-12 || error("random Krylov completion lost rank")
        push!(starts,v/norm(v))
    end
    starts,nwarm
end

function pairKrylovSolve(action,n,nev,tol;krylovdim=max(160,4nev),maxiter=50,seed=821,verbose=false,initial_vectors=nothing)
    count=min(nev,n);calls=Ref(0)
    count==0 && return Float64[],zeros(ComplexF64,n,0),(method=:block_krylov,matvecs=0)
    function apply(v)
        calls[]+=1
        verbose && calls[]%50==0 && (println("[pair Krylov] matrix-vector product $(calls[])");flush(stdout))
        action(v)
    end
    for attempt in 0:2
        rng=Xoshiro(seed+attempt)
        vectors,nwarm=pairKrylovStart(n,count,rng,attempt==0 ? initial_vectors : nothing)
        start=KrylovKit.Block(vectors)
        alg=KrylovKit.BlockLanczos(;krylovdim=min(n,max(krylovdim,2length(vectors)+1)),maxiter=maxiter,tol=tol,eager=attempt==0)
        result=try
            KrylovKit.eigsolve(apply,start,count,:SR,alg)
        catch err
            # KrylovKit 0.10.4 estimates ||r*u|| via sqrt(u'*(r'*r)*u).
            # Near an exact degeneracy this quadratic form can round slightly
            # negative. Retry with a fresh block and less frequent Ritz checks;
            # never clamp a residual or accept unconverged eigenpairs.
            upstream=any(f->occursin("blocklanczos.jl",String(f.file)),stacktrace(catch_backtrace()))
            if attempt<2 && upstream && err isa DomainError && err.val isa Real && -1e-12<err.val<0
                verbose && (println("[pair Krylov] retrying after residual-norm roundoff");flush(stdout))
                continue
            end
            rethrow()
        end
        vals,vecs,info=result
        info.converged>=count || error("pair Krylov solve did not converge: $(info.converged)/$count eigenpairs in $(calls[]) matvecs")
        return real.(vals[1:count]),hcat(vecs[1:count]...),(method=:block_krylov,matvecs=calls[],roundoff_retries=attempt,
            warm_vectors=nwarm,starting_block=length(vectors))
    end
end
