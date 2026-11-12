# Exact model-specific MPOs. Physical matrices use [ket,bra] orientation.
# The hopping gate needs two interaction channels, two wrap channels and an
# onsite transition, rather than a separate channel for every matrix unit.
function pairHoppingMPO(phi,L;mass=0.)
    z=cis(phi)
    hon=blkop(sm2,sp2).*z+blkop(sp2,sm2).*conj(z)+mass.*Matrix(Diagonal(massBlkVals))
    terms=[(Matrix(transpose(blkop(sp2,I2b).*z)),Matrix(transpose(blkop(I2b,sm2)))),
           (Matrix(transpose(blkop(sm2,I2b).*conj(z))),Matrix(transpose(blkop(I2b,sp2))))]
    sites=[Tuple{Int,Int,Matrix{ComplexF64}}[] for _ in 1:L]
    for i in 1:L
        append!(sites[i],[(1,1,I4b),(6,6,I4b),(1,6,Matrix(transpose(hon)))])
        for (j,(left,right)) in enumerate(terms)
            push!(sites[i],(1,1+j,left),(1+j,6,right))
            i==1 && push!(sites[i],(1,3+j,right))
            1<i<L && push!(sites[i],(3+j,3+j,I4b))
            i==L && push!(sites[i],(3+j,6,left))
        end
    end
    sites,[1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.]
end

# Combining e²/N with the Coulomb kernel cancels the j*k coefficient.
# For j<k, the q_j q_k coefficient is f(j)+g(k), with the lifting included.
# This identity holds on charged as well as neutral configurations.
function pairDiagonalMPO(N,gC,lam,h2diag;theta=0.)
    L=N÷2;sites=[Tuple{Int,Int,Matrix{ComplexF64}}[] for _ in 1:L]
    f(j)=gC*(j^2/N+j)+4lam
    g(j)=gC*(j^2/N-j)+4lam
    a(j)=gC*j^2/N+4lam
    q0=ComplexF64.(q0vals);q1=ComplexF64.(q1vals)
    for i in 1:L
        j=2(i-1);k=j+1
        onsite=a(j).*q0.^2+a(k).*q1.^2+(f(j)+g(k)).*q0.*q1-
               (gC*theta/pi).*(j.*q0+k.*q1)
        for c in 1:4;push!(sites[i],(c,c,I4b));end
        for (l,r,v) in ((1,2,f(j).*q0+f(k).*q1),(1,3,q0+q1),
                        (2,4,q0+q1),(3,4,g(j).*q0+g(k).*q1),(1,4,onsite))
            push!(sites[i],(l,r,Matrix(Diagonal(v))))
        end
    end
    constant=gC*N*(h2diag+(theta/(2pi))^2)
    sites,[1.,0.,0.,0.],[constant,0.,0.,1.]
end
