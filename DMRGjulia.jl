using LinearAlgebra
using ITensors
using Plots
using LaTeXStrings
using ITensorMPS
using Combinatorics
using ProgressBars

function hamiltonian(sites::Vector{<:Index}, t1,t2;U=10^10,tJ=false)
    #sites = siteinds("Electron", N;conserve_qns=true)
    L = length(sites)
    ampo = OpSum()
    for j=1:L-1 #1:N-2
        ampo -= t1, "Cdagup", j, "Cup", j+1
        ampo -= t1, "Cdagup", j+1, "Cup", j
        ampo -= t1, "Cdagdn", j, "Cdn", j+1
        ampo -= t1, "Cdagdn", j+1, "Cdn", j
    end
    K = div(L-1,2)
    for j = 1:K
        ampo -= t2, "Cdagup", 2*j-1, "Cup", 2*j+1
        ampo -= t2, "Cdagup", 2*j+1, "Cup", 2*j-1
        ampo -= t2, "Cdagdn", 2*j-1, "Cdn", 2*j+1
        ampo -= t2, "Cdagdn", 2*j+1, "Cdn", 2*j-1
    end
    if !tJ
        for j = 1:L
            ampo += U, "Nup", j, "Ndn", j
        end
    end
    #@show ampo
    H = MPO(ampo, sites)
    return H
end

function RVB_state(L)
    @assert isodd(L) "L must be odd"
    Npair = (L - 1) ÷ 2
    sites = collect(0:L-1)
    rvb_basis = []
    rvb_coeffs = []
    for hole in sites
        singlet_pairs = []
        a = 0
        while a < L
            if a == hole
                a += 1
                continue
            end
            b = a + 1
            if b == hole
                b += 1
            end
            if b < L
                push!(singlet_pairs, (a, b))
                a = b + 1
            else
                break
            end
        end
        # Generate all possible up_sites for the singlet pairs
        up_sites_list = [pair for pair in singlet_pairs]
        for up_sites in Iterators.product((pair for pair in singlet_pairs)...)
            sign = 1
            for (idx, pair) in enumerate(singlet_pairs)
                if up_sites[idx] == pair[1]
                    sign *= -1
                end
            end
            push!(rvb_basis, (hole, Tuple(up_sites)))
            push!(rvb_coeffs, sign / (2.0 ^ (length(singlet_pairs) / 2)))
        end
    end
    return rvb_basis, rvb_coeffs
end

function exact_ground_state(L, t1, t2, rvb_basis, rvb_coeffs)
    # Tight-binding Hamiltonian in the hole basis
    Htb = zeros(L, L)
    for j in 1:L-1
        Htb[j, j+1] = -t1
        Htb[j+1, j] = -t1
    end
    for j in 1:2:L-2
        Htb[j, j+2] = -t2
        Htb[j+2, j] = -t2
    end
    e_vals, e_vecs = eigen(Htb)
    e_coeff = e_vecs[:, argmin(e_vals)]  # ground state coefficients in the hole basis

    # Build the RVB vector in the many-body basis
    basis_dict = Dict{Any, Int}()
    for (idx, state) in enumerate(rvb_basis)
        basis_dict[state] = idx
    end
    rvb_vec = zeros(length(rvb_basis))
    for (state, coeff) in zip(rvb_basis, rvb_coeffs)
        idx = basis_dict[state]
        rvb_vec[idx] += coeff * e_coeff[state[1]+1] # Julia is 1-based
    end
    rvb_vec ./= norm(rvb_vec)
    return rvb_vec
end

function basis_tuple_to_statevec(basis_tuple, L)
    hole, up_sites = basis_tuple
    statevec = fill("Dn", L)
    statevec[hole+1] = "Emp"  # Julia is 1-based, basis is 0-based
    for up in up_sites
        statevec[up+1] = "Up"
    end
    return statevec
end

N = 101#50 + 1
#sites = siteinds("Electron", N; conserve_qns=true)
sites = siteinds("tJ", N; conserve_qns=true)
t1 = 1.0
t2 = 0.5 #0.5 #-1.0
U = 10^10 # need to be very large to both prohibit double occupancy and improve convergence
tJ = true

HMPO = hamiltonian(sites, t1, t2; U=U, tJ=tJ);

state0 = [isodd(i) ? "Up" : "Dn" for i in 1:length(sites)]
#state0[6] = "Emp" # add one hole
#state0[5] = "Emp" 
#state0 = ["Up" for i in 1:length(sites)] # empty lattice
state0[1] = "Emp"
#psi0 = randomMPS(sites,state0; linkdims=20)
psi0 = productMPS(sites, state0) # better for large U

nsweeps = 200 # number of sweeps is 5
maxdim = [1024]#[16,16,16,16,16,32,32,32,32,32,
        #64,64,64,64,64,128,128,128,128,128,
        #256,256,256,256,256,512,512,512,512,512,
        #1024,1024,1024,1024,1024,2048,2048] # gradually increase states kept
cutoff = [1E-10] # desired truncation error
energy, psi = dmrg(HMPO,psi0; nsweeps, cutoff, maxdim);

energy, psi = dmrg(HMPO,psi; nsweeps, cutoff, maxdim);


# use analytical state as initial state
rvb_basis, rvb_coeffs = RVB_state(N)
gs = exact_ground_state(N, t1, t2, rvb_basis, rvb_coeffs)

product_states = [basis_tuple_to_statevec(b, N) for b in rvb_basis]
gs_exact = gs[1]*MPS(sites,product_states[1])
for k = ProgressBar(2:length(product_states))
    gs_exact += gs[k] * MPS(sites, product_states[k])
    #println(k)
end
norm(gs_exact)
@show maxlinkdim(gs_exact)


energy, psi = dmrg(HMPO, gs_exact; nsweeps, cutoff, maxdim);

inner(gs_exact', HMPO, gs_exact)

inner(psi, gs_exact)