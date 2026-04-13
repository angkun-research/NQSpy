import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from vmc_utils import state_to_onehot, local_energy,GlobalSampler, local_energy_from_adjlist
from vmc_utils import local_energy_on_the_fly, propose_move
from vmc_utils import BalancedSampler
from vmc_utils import TightBinding_coeff, find_state_coeff
from vmc_utils import generate_initial_state
from vmc_utils import VBSNN, sr_update, Obtain_Sampling_batch, sr_update_optimizer
import random
from tqdm import trange 
import numpy.linalg as la

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

L = 31 #11
t1 = 1.0
t2 = 0.5
J1 = 0.1 #1.0
J2 = 0.09 #0.9 #0.81/100
# basis = build_MB_basis(L)
# basis_dict = {state: idx for idx, state in enumerate(basis)}
# #H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
# #Hsparse = csr_matrix(H)
# H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
# Hsparse = adjlist_to_csr(H_ind, H_val)
# eigvals, eigvecs = eigsh(Hsparse, k=3, which='SA')
# # eigvals, eigvecs = la.eigh(H)
# print(f"Exact ground state energy for L {L}: {eigvals[0:3]}")
# exact_gs = eigvecs[:, 0]
# print("Dimension of Hilbert space:", len(basis))
# exit()

E_exact = -3.66565952 # eigvals[0] # Ls = [11,21,31] E_exact = [-6.156705, -10.009270,-13.79471558]
# [-2.80219475,-3.26788669,-3.66565952]
print(f"Exact ground state energy for L={L}: {E_exact}")

# h 128, k 5 for L=7;
hidden_dim = 32 #64 #16 
kernel_size = 5 
print("hidden_dim:", hidden_dim, "kernel_size:", kernel_size)
psi = VBSNN(L, Conv_dim=hidden_dim,kernel_size=kernel_size)
psi.to(device)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# start VMC 
# Initialize state
#initial_state = generate_initial_state(L) #random.choice(basis)
n_walkers = 128 #32 #64 
initial_states = [generate_initial_state(L) for _ in range(n_walkers)]  

#optimizer = optim.Adam(psi.parameters(), lr=1e-2) # 1e-2 3e-3
optimizer = optim.SGD(psi.parameters(), lr=1e-2)

n_samples = 256 #128 #64  
print("n_walkers:", n_walkers, "n_samples:", n_samples)
epochs = 500 
N_eff = n_samples * n_walkers / 10  # effective sample size, adjusted for autocorrelation (heuristic factor of 10)
burn_in = 64 #1000
Sampler = BalancedSampler if J2 == J1 else GlobalSampler
lr = 5e-1 # learning rate for SR updates
sr_tau = 1e0 # regularization shift for SR (tau)

#_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True) # burn-in
_, _, states, _ = Obtain_Sampling_batch(psi, L, initial_states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers

# record energy for every step after burn-in
energy_record = [] 
error_record = []
for step in trange(epochs, desc="VMC Sampling"):
    if step % 100 == 0:
        states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]  # reinitialize walkers every 100 steps to reduce autocorrelation
        _, _, states, _ = Obtain_Sampling_batch(psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers
        print(f"Reinitialized walkers at step {step} to reduce autocorrelation.")
    psis, energies, states, sampled_states = Obtain_Sampling_batch(psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device, print_rate=False, Sampler=Sampler)
    # Optimization step every batch
    if torch.is_tensor(energies):
        E_tensor = energies.squeeze()
    else:
        E_tensor = torch.stack(energies).squeeze()
    if torch.is_tensor(psis):
        psis_tensor = psis.squeeze()
    else:
        psis_tensor = torch.stack(psis).squeeze()
    E_mean = E_tensor.mean()
    E_std = E_tensor.var().sqrt()
    E_se = E_std / np.sqrt(len(E_tensor))  # standard error of the mean, adjusted for autocorrelation
    # loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    #sr_update(psi, sampled_states, E_tensor, L, lr, sr_tau, device)
    
    print(f"Step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")
    energy_record.append(E_mean.item())
    error_record.append(E_se.item())
    if E_mean.item() > E_exact*0.75:
        lr = 5e-1 
        sr_tau = 1e0 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #sr_update(psi, sampled_states, E_tensor, L, lr, sr_tau, device)
        sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device, adaptive_lr=True, lr=lr)
    else:
        lr = 5e-2 #5e-2 #2e-2 
        sr_tau = 1e-1 #1e-3 #1e-4 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device)
    # if E_mean.item() < E_exact+0.01:
    #     break 

print(f"Step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")

print("VMC finished.")

states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]  # reinitialize walkers every 100 steps to reduce autocorrelation
_, _, states, _ = Obtain_Sampling_batch(psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers
for k in range(10):
    _, energies, states, _ = Obtain_Sampling_batch(psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device, Sampler=Sampler)
    if torch.is_tensor(energies):
        E_tensor = energies.squeeze()
    else:
        E_tensor = torch.stack(energies).squeeze()
    E_mean = E_tensor.mean()
    E_std = E_tensor.var().sqrt()
    E_se = E_std / np.sqrt(len(E_tensor))
    print(f"Final evaluation {k}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")
    energy_record.append(E_mean.item())
    error_record.append(E_se.item())

# output energy_record to csv
import pandas as pd
totalsam = n_walkers* n_samples
totalsam = int(totalsam)
df = pd.DataFrame({'Energy': energy_record, 'Error': error_record})
csv_path = f"data/energy_error_nonexact_L{L}_t2{t2}_J1{J1}_J2{J2}_hidden{hidden_dim}_kernel{kernel_size}_sam{totalsam}.csv"
df.to_csv(csv_path, index=False)
print(f"Saved energy and error record to {csv_path}")