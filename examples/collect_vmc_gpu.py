import torch
import torch.optim as optim
import numpy as np
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vmc_utils import generate_initial_state
from vmc_utils import VBSNN, sr_update_optimizer
from vmc_utils import cleanup_memory
from vmc_utils_gpu import BalancedSampler_gpu, GlobalSampler_gpu
from vmc_utils_gpu import Obtain_Sampling_batch_gpu
from tqdm import trange 
from utils import pick_best_device, set_seed

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = pick_best_device()
print(f"Using device: {device}")
set_seed(42)  # for reproducibility

L = 11 #11
t1 = 1.0
t2 = 0.5
J1 = 0.0 #0.0
J2 = 0.0 #0.0 #0.81/100
use_sr = True #False

E_exact = -2.42988202 #eigvals[0] # Ls = [11,21,31,41,51] E_exact = [-2.42988202, -2.519684,-2.541281,-2.5496381,-2.553721]
print(f"Exact ground state energy for L={L}: {E_exact}")

# h 128, k 5 for L=7;
hidden_dim = 16 #16 #64 #32 #16 
kernel_size = 2 
print("hidden_dim:", hidden_dim, "kernel_size:", kernel_size)
psi = VBSNN(L, Conv_dim=hidden_dim,kernel_size=kernel_size)
psi.to(device)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# start VMC 
# Initialize state
#initial_state = generate_initial_state(L) #random.choice(basis)
n_walkers = 1000 #64 
initial_states = [generate_initial_state(L) for _ in range(n_walkers)]  

if use_sr:
    optimizer = optim.SGD(psi.parameters(), lr=1e-1) # 1e-2
else:
    optimizer = optim.Adam(psi.parameters(), lr=1e-1) # no sr need adam

n_samples = 100 #64 #128  
print("n_walkers:", n_walkers, "n_samples:", n_samples)
epochs = 100 #1000 
N_eff = n_samples * n_walkers / 10  # effective sample size, adjusted for autocorrelation (heuristic factor of 10)
burn_in = 64 #1000
Sampler = BalancedSampler_gpu if J2 == J1 else GlobalSampler_gpu
lr = 5e-1 # learning rate for SR updates
sr_tau = 1e0 # regularization shift for SR (tau)

_, _, states, _ = Obtain_Sampling_batch_gpu(psi, L, initial_states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers

# record energy for every step after burn-in
energy_record = [] 
error_record = []
#for step in trange(epochs, desc="VMC Sampling"):
for step in range(epochs):
    if step % 100 == 0:
        states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]  # reinitialize walkers every 100 steps to reduce autocorrelation
        _, _, states, _ = Obtain_Sampling_batch_gpu(psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers
        print(f"Reinitialized walkers at step {step} to reduce autocorrelation.")
    psis, energies, states, sampled_states = Obtain_Sampling_batch_gpu(psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device, print_rate=False, Sampler=Sampler)
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
    E_se = E_std / math.sqrt(E_tensor.numel())  # standard error of the mean, adjusted for autocorrelation

    print(f"Step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")
    energy_record.append(E_mean.item())
    error_record.append(E_se.item())

    if use_sr:
        if E_mean.item() > E_exact*0.75: #0.85:
            #lr = np.abs(E_exact - E_mean.item()) #1.0 #5e-1 
            lr = 5e-1 
            sr_tau = 1e0 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device, adaptive_lr=True, lr=lr)
        else:
            lr = 5e-2 #5e-2 #2e-2 
            sr_tau = 1e-1 #1e-3 #1e-4 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device)
    else:
        loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # free Python containers / references you no longer need
    sampled_states = None           # list of many tuples
    psis, psis_tensor = None, None                      # if you stored another copy
    energies, E_tensor = None, None
    cleanup_memory(free_vars=None, optimizer=optimizer)

print(f"Step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")

print("VMC finished.")

states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]  # reinitialize walkers every 100 steps to reduce autocorrelation
_, _, states, _ = Obtain_Sampling_batch_gpu(psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers
for k in range(10):
    _, energies, states, _ = Obtain_Sampling_batch_gpu(psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device, Sampler=Sampler)
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
    if E_mean.item() > -2.0:
        print(f"Final energy {E_mean.item():.6f} is unreasonably high. Exiting.")
        exit() # Exit the program if the energy is unreasonably high
    # free Python containers / references you no longer need
    energies, E_tensor = None, None

# output energy_record to csv
# import pandas as pd
# df = pd.DataFrame({'Energy': energy_record, 'Error': error_record})
# totalsam = n_walkers* n_samples
# totalsam = int(totalsam)
# folder = "data/" #"/nfs/home/awu14/data/NQSdata/"
# filename = f"energy_error_exact_L{L}_t2{t2}_hidden{hidden_dim}_kernel{kernel_size}_sam{totalsam}.csv"
# save_path = folder + filename
# df.to_csv(save_path, index=False)
# print(f"Saved energy and error record to {save_path}")