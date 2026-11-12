# Neutral-sector bit basis and magnetic-translation orbits in the concentrated
# gauge. No 2^N dense Hamiltonian. A finite cyclic flux window is the SAME window
# as holonomyMats; it is not assumed to be exactly translation invariant.
using LinearAlgebra,Random,Serialization

function neutralBits(N)
    iseven(N) && 2<=N<=30 || throw(ArgumentError("even N in 2:30 required"))
    bits=Vector{UInt32}(undef,binomial(N,N÷2));x=(UInt32(1)<<(N÷2))-1
    for i in eachindex(bits)
        bits[i]=x;c=x & -x;r=x+c;x=(((r ⊻ x)>>2)÷c)|r
    end
    bits
end

function fluxDiagonal(bits,N,h,gC,mass,theta)
    charge=0.;electric=0.;staggered=0.
    for j in 0:N-1
        n=1-Int((bits>>j)&1);charge+=n-(j%2)
        electric+=(h+charge+theta/(2pi))^2
        staggered+=(iseven(j) ? 1 : -1)*n
    end
    gC*electric+mass*staggered
end

function prepareFluxOrbits(N,hmax;gC=.25,mass=-.0625,theta=0.,verbose=false)
    bits=neutralBits(N);n=length(bits);M=2hmax+1;L=N÷2
    rank=zeros(Int32,1<<N)
    for (a,b) in enumerate(bits);rank[Int(b)+1]=a;end
    orbit=zeros(Int32,n,M);shift=zeros(UInt8,n,M)
    reps=Tuple{Int32,Int16}[];lengths=UInt8[];diagonal=Float64[];variance=Float64[]
    mask=(UInt32(1)<<N)-1
    rotate(b)=((b<<2)&mask)|(b>>(N-2))
    for hi in 1:M,a in 1:n
        orbit[a,hi]!=0 && continue
        push!(reps,(a,hi));id=length(reps);b=bits[a];hh=hi;t=0;energies=Float64[]
        while true
            aa=rank[Int(b)+1];orbit[aa,hh]=id;shift[aa,hh]=t
            push!(energies,fluxDiagonal(b,N,hh-hmax-1,gC,mass,theta))
            kap=Int((b>>(N-2))&1)+Int((b>>(N-1))&1)-1
            b=rotate(b);hh=mod1(hh+kap,M);t+=1
            b==bits[a] && hh==hi && break
        end
        push!(lengths,t);mu=sum(energies)/t;push!(diagonal,mu)
        push!(variance,sum(abs2,e-mu for e in energies)/t)
    end
    verbose && (println("flux orbits: N=$N h=$hmax neutral=$n, full=$(n*M), orbits=$(length(reps))");flush(stdout))
    # A CSR gather action; each edge carries only an orbit id and translation.
    offsets=Vector{Int64}(undef,length(reps)+1);offsets[1]=1
    neighbors=Int32[];phases=UInt8[]
    sizehint!(neighbors,length(reps)*(N÷2+1));sizehint!(phases,length(reps)*(N÷2+1))
    for (id,(a,hi)) in enumerate(reps)
        b=bits[a]
        for j in 0:N-1
            jp=mod(j+1,N);bj=(b>>j)&1;bp=(b>>jp)&1;bj==bp && continue
            bb=b ⊻ (UInt32(1)<<j) ⊻ (UInt32(1)<<jp)
            # H is real symmetric. On the cut, occupied site 1 hopping to N
            # raises the cut flux (DFT of exp(+i theta_m)).
            hh=j==N-1 ? mod1(hi+(bp==0 ? 1 : -1),M) : hi
            aa=rank[Int(bb)+1];push!(neighbors,orbit[aa,hh]);push!(phases,shift[aa,hh])
        end
        offsets[id+1]=length(neighbors)+1
    end
    (N=N,hmax=hmax,gC=gC,mass=mass,theta=theta,bits=bits,orbit=orbit,shift=shift,
     reps=reps,lengths=lengths,diagonal=diagonal,variance=variance,
     offsets=offsets,neighbors=neighbors,phases=phases)
end

function fluxMomentumBasis(c,k)
    L=c.N÷2;0<=k<L || throw(ArgumentError("momentum"))
    keep=findall(l->mod(k*Int(l),L)==0,c.lengths)
    inverse=zeros(Int32,length(c.reps));inverse[keep]=1:length(keep)
    # The self-conjugate momenta have exactly real phases. Keeping these real
    # halves the memory needed by the large ED Krylov basis.
    phase=k==0 ? ones(L) : 2k==L ? [isodd(j) ? -1. : 1. for j in 0:L-1] : cis.(-2pi*k/L.*(0:L-1))
    root=sqrt.(Float64.(c.lengths))
    (keep=keep,inverse=inverse,phase=phase,root=root,k=k)
end

function fluxMomentumAction(c,b,v)
    length(v)==length(b.keep) || throw(DimensionMismatch("ED vector"))
    out=similar(v,promote_type(eltype(v),eltype(b.phase)))
    Threads.@threads :static for ii in eachindex(b.keep)
        i=b.keep[ii];z=c.diagonal[i]*v[ii]
        @inbounds for e in c.offsets[i]:c.offsets[i+1]-1
            j=c.neighbors[e];jj=b.inverse[j]
            jj==0 && continue
            z+=b.root[i]/b.root[j]*b.phase[Int(c.phases[e])+1]*v[jj]
        end
        out[ii]=z
    end
    out
end

# Embed only for SMALL independent tests, never for production N=28.
function fluxMomentumEmbedding(c,b)
    n=length(c.bits);U=zeros(ComplexF64,n*(2c.hmax+1),length(b.keep))
    for hi in axes(c.orbit,2),a in 1:n
        id=c.orbit[a,hi];j=b.inverse[id];j==0 && continue
        U[(hi-1)*n+a,j]=b.phase[Int(c.shift[a,hi])+1]/b.root[id]
    end
    U
end
