using LinearAlgebra
using ITensors
using Plots
using LaTeXStrings
using ITensorMPS

function hamiltonian(sites::Vector{<:Index}, t1,t2;U=1000.0)
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
    for j = 1:K-1
        ampo -= t2, "Cdagup", 2*j-1, "Cup", 2*j+1
        ampo -= t2, "Cdagup", 2*j+1, "Cup", 2*j-1
        ampo -= t2, "Cdagdn", 2*j-1, "Cdn", 2*j+1
        ampo -= t2, "Cdagdn", 2*j+1, "Cdn", 2*j-1
    end
    for j = 1:L
        ampo += U, "Nup", j, "Ndn", j
    end
    H = MPO(ampo, sites)
    return H
end

N = 10 + 1
sites = siteinds("Electron", N; conserve_qns=true)
t1 = -1.0
t2 = -0.5 #-1.0
U = 1000.0

HMPO = hamiltonian(sites, t1, t2; U=U)

state0 = [isodd(i) ? "Up" : "Dn" for i in 1:length(sites)]
#state0[6] = "Emp" # add one hole
#state0[5] = "Emp" 
#state0 = ["Up" for i in 1:length(sites)] # empty lattice
state0[1] = "Emp"
#psi0 = randomMPS(sites,state0; linkdims=20)
psi0 = productMPS(sites, state0) # better for large U

nsweeps = 1000 # number of sweeps is 5
maxdim = [1024]#[16,16,16,16,16,32,32,32,32,32,
        #64,64,64,64,64,128,128,128,128,128,
        #256,256,256,256,256,512,512,512,512,512,
        #1024,1024,1024,1024,1024,2048,2048] # gradually increase states kept
cutoff = [1E-10] # desired truncation error
energy, psi = dmrg(HMPO,psi0; nsweeps, cutoff, maxdim);


