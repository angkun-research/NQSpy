using LinearAlgebra
using ITensors
using Plots
using LaTeXStrings
using ITensorMPS
using Combinatorics
using ProgressBars

function hamiltonian(sites::Vector{<:Index}, t1,t2;
    U=10^10,tJ=true,J1=1.0, J2=1.0)
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
    else
        for j = 1:L-1
            ampo += J1/2, "Splus", j, "Sminus", j+1
            ampo += J1/2, "Sminus", j, "Splus", j+1
            ampo += J1, "Sz", j, "Sz", j+1
        end
        for j = 1:K
            ampo += J2/2, "Splus", 2*j-1, "Sminus", 2*j+1
            ampo += J2/2, "Sminus", 2*j-1, "Splus", 2*j+1 
            ampo += J2, "Sz", 2*j-1, "Sz", 2*j+1
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

function EntanglementEntropy(psi::MPS; cut=div(length(siteinds(psi)), 2))
    #sites = siteinds(psi)
    #cut = div(length(sites), 2) # cut in the middle
    orthogonalize!(psi, cut)
    U, S, V = svd(psi[cut]*psi[cut+1], inds(psi[cut]))
    svals = diag(S)
    p = svals .^ 2
    #p ./= sum(p)
    S_ent = -sum(p .* log.(p))
    return S_ent
    #return p
end

N = 21 #31 #50 + 1
#sites = siteinds("Electron", N; conserve_qns=true)
sites = siteinds("tJ", N; conserve_qns=true)
t1 = 1.0 #1.0
t2 = 0.5 # 0.5 #0.5 #-1.0
U = 10^10 # need to be very large to both prohibit double occupancy and improve convergence
tJ = true
J1 = 1.0 #1.0
J2 = 0.9 #0.5

HMPO = hamiltonian(sites, t1, t2; U=U, tJ=tJ, J1=J1, J2=J2);

state0 = [isodd(i) ? "Up" : "Dn" for i in 1:length(sites)]
state0[5] = "Emp" # add one hole
#state0[6] = "Emp" 
#state0 = ["Up" for i in 1:length(sites)] # empty lattice
#state0[1] = "Emp"
#state0[10] = "Emp"
# for j in 11:length(sites)
#     if isodd(j)
#         state0[j] = "Dn"
#     else
#         state0[j] = "Up"
#     end
# end
#state0[2] = "Emp"
psi0 = randomMPS(sites,state0; linkdims=50)
#psi0 = productMPS(sites, state0) # better for large U

nsweeps = 100 # number of sweeps is 5
maxdim = [1024]#[1024]#[16,16,16,16,16,32,32,32,32,32,
        #64,64,64,64,64,128,128,128,128,128,
        #256,256,256,256,256,512,512,512,512,512,
        #1024,1024,1024,1024,1024,2048,2048] # gradually increase states kept
cutoff = [1E-12] # desired truncation error
energy, psi = dmrg(HMPO,psi0; nsweeps, cutoff, maxdim);
# EE = EntanglementEntropy(psi, cut=div(N,2))
# EE = sort(EE; rev=true)
# scatter(EE[1:80]; xlabel=L"k", ylabel=L"\lambda_k", yaxis=:log)
#energy, psi = dmrg(HMPO,psi; nsweeps, cutoff, maxdim);
EEs = zeros(N-1)
for cut = 1:N-1
    EEs[cut] = EntanglementEntropy(psi, cut=cut)
    if isnan(EEs[cut]) # NaN
        EEs[cut] = 0.0
    end
    #println("Cut: $cut, S_ent: $(EEs[cut])")
end
plot(1:N-1, EEs;marker=:circle, xlabel="Cut", ylabel="S", legend=false)
# savefig("~/Desktop/Fig3.pdf")
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





"""
Return dense parameter count = sum over sites of prod(dim.(inds(psi[j]))).

If `complex=true`, returns the number of real scalar parameters assuming each
complex entry has 2 real parameters.
"""
function mps_dense_parameter_count(psi::MPS; complex::Bool=false)
    n = 0
    for j in 1:length(psi)
        n += prod(dim.(inds(psi[j])))
    end
    return complex ? 2n : n
end

"""
Return stored parameter count for QN/block-sparse tensors: sum(nnz(psi[j])).

This is often the best "how many numbers are actually stored" measure for
QN-conserving MPS.
If `complex=true`, returns real-scalar count for complex storage (2*nnz).
"""
function mps_stored_parameter_count(psi::MPS; complex::Bool=false)
    n = 0
    for j in 1:length(psi)
        n += nnz(psi[j])
    end
    return complex ? 2n : n
end

@show mps_dense_parameter_count(psi)
@show mps_stored_parameter_count(psi)