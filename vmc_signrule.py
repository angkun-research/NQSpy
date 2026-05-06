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
from vmc_utils import cleanup_memory
from NeuralNetworks import ConvSkipHole,HoleOnLocalRule,LocalRule
import random
from tqdm import trange 
import numpy.linalg as la
from ExactGS import exact_ground_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Obtain_Sampling(initial_state, n_samples, L, pretrain=False, burnin = False,tb_coeff=None,print_rate=False,
                    Sampler=GlobalSampler):
    Psis = []
    Elocs = []
    state = initial_state
    new_state = state
    accept_rate = 0.0
    for _ in range(n_samples):
        #new_state = GlobalSampler(state, L)
        #new_state = propose_move(state, L)
        new_state = Sampler(state, L)
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        new_s_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_val = psi(s_onehot)
        psi_new_val = psi(new_s_onehot)
        #accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2+1e-12).item()))
        if psi_val.abs() < 1e-16:
            accept_prob = 1.0
        else:
            accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2).item()))
        if random.random() < accept_prob:
            state = new_state
            Psis.append(psi_new_val)
            accept_rate += 1.0
        else:
            Psis.append(psi_val)

        if burnin:
            continue
        #if  not pretrain:
        if not pretrain:
            #E_loc = local_energy_from_adjlist(state, psi, basis, basis_dict, H_ind, H_val, device, L)
            E_loc = local_energy_on_the_fly(state, psi, L, t1, t2, J1=J1, J2=J2, device=device)
            Elocs.append(E_loc)
        else:
            Elocs.append(torch.tensor(find_state_coeff(L, state, tb_coeff), device=device))
    if print_rate:
        print(f"Acceptance rate: {accept_rate / n_samples:.4f}")
    return Psis, Elocs, state


L = 51 #31
t1 = 1.0
t2 = 0.5
J1 = 0.0 #0.0
J2 = 0.0 #0.0 #0.81/100
# basis = build_MB_basis(L)
# basis_dict = {state: idx for idx, state in enumerate(basis)}
# #H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
# #Hsparse = csr_matrix(H)
# H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
# Hsparse = adjlist_to_csr(H_ind, H_val)
# eigvals, eigvecs = eigsh(Hsparse, k=3, which='SA')
# # eigvals, eigvecs = la.eigh(H)
# print(f"Exact ground state energy: {eigvals[0:3]}")
# exact_gs = eigvecs[:, 0]
# print("Dimension of Hilbert space:", len(basis))

E_exact = -2.5537213198833166 # L=51
print(f"Exact ground state energy for L={L}: {E_exact}")

LocalRule = LocalRule(in_channels=4, hidden_dim=16, kernel_size=2, stride=2)
#ConvSkipHole(in_channels=4, hidden_dim=32, kernel_size=2, stride=2, activation='tanh')
optimizer = optim.Adam(LocalRule.parameters(), lr=0.01)#, weight_decay=1e-3) #lr=0.01

folder = "data/"
filename = f"SignRule_t2{t2}_hidden16.pth"
save_path = folder + filename
checkpoint = torch.load(save_path, map_location=device)
LocalRule.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
LocalRule.to(device)
print("Loaded SignRule.")

# LocalRule.eval()
# test_state = (15,(2,3,5,7,9,11,13))#generate_initial_state(L)
# print("Test state:", test_state)
# test_onehot = torch.tensor(state_to_onehot(test_state, L), dtype=torch.float32, device=device).unsqueeze(0)
# with torch.no_grad():
#     sign_pred = LocalRule(test_onehot)
# print("Predicted sign for a test state:", sign_pred.item())
# exit()

hidden_dim = L #128 #32
psi = HoleOnLocalRule(LocalRule, L,hidden_dim=hidden_dim)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")


# start VMC 
# Initialize state
initial_state = generate_initial_state(L) #random.choice(basis)

optimizer = optim.Adam(psi.parameters(), lr=1e-1) #1e-2

n_samples = 10000
epochs = 100
burn_in = 1000

_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True) # burn-in
for step in trange(epochs, desc="VMC Sampling"):
    psis, energies, state = Obtain_Sampling(state, n_samples, L,print_rate=False, Sampler=BalancedSampler)
    # Optimization step every batch
    E_tensor = torch.stack(energies).squeeze()
    psis_tensor = torch.stack(psis).squeeze()
    E_mean = E_tensor.mean()
    loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Use stochastic reconfiguration update (encapsulated)
    #stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=1e-1, reg=1e-3, device=device, noise_sigma=noise_sigma)
    # if step % (epochs//20) == 0:
    #     print(f"Step {step}: <E> = {E_mean.item():.6f}")
    print(f"Step {step}: <E> = {E_mean.item():.6f}")

state = generate_initial_state(L) 
n_samples = 10000
for k in range(5):
    _, energies, state = Obtain_Sampling(state, n_samples, L)
    E_tensor = torch.stack(energies).squeeze()
    E_mean = E_tensor.mean()
    print(f"Next sample {k}: <E> = {E_mean.item():.6f}")

# n_walkers = 64 #32 #64 
# initial_states = [generate_initial_state(L) for _ in range(n_walkers)]  

# n_samples = 128 #64  
# print("n_walkers:", n_walkers, "n_samples:", n_samples)
# epochs = 200
# burn_in = 64 #1000
# Sampler = BalancedSampler
# lr = 5e-1 # learning rate for SR updates
# sr_tau = 1e0 # regularization shift for SR (tau)

# optimizer = optim.SGD(psi.parameters(), lr=1e-2)

# _, _, states, _ = Obtain_Sampling_batch(psi, L, initial_states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers

# # record energy for every step after burn-in
# energy_record = [] 
# error_record = []
# for step in trange(epochs, desc="VMC Sampling"):
#     if step % 100 == 0:
#         states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]  # reinitialize walkers every 100 steps to reduce autocorrelation
#         _, _, states, _ = Obtain_Sampling_batch(psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers
#         print(f"Reinitialized walkers at step {step} to reduce autocorrelation.")
#     psis, energies, states, sampled_states = Obtain_Sampling_batch(psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device, print_rate=False, Sampler=Sampler)
#     # Optimization step every batch
#     if torch.is_tensor(energies):
#         E_tensor = energies.squeeze()
#     else:
#         E_tensor = torch.stack(energies).squeeze()
#     if torch.is_tensor(psis):
#         psis_tensor = psis.squeeze()
#     else:
#         psis_tensor = torch.stack(psis).squeeze()
#     E_mean = E_tensor.mean()
#     E_std = E_tensor.var().sqrt()
#     E_se = E_std / np.sqrt(len(E_tensor))  # standard error of the mean, adjusted for autocorrelation

#     print(f"Step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")
#     energy_record.append(E_mean.item())
#     error_record.append(E_se.item())
#     if E_mean.item() > E_exact*0.75:
#         lr = 5e-1 
#         sr_tau = 1e0 
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         #sr_update(psi, sampled_states, E_tensor, L, lr, sr_tau, device)
#         sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device, adaptive_lr=True, lr=lr)
#     else:
#         lr = 5e-2 #5e-2 #2e-2 
#         sr_tau = 1e-1 #1e-3 #1e-4 
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device)
#     # if E_mean.item() < E_exact+0.01:
#     #     break 
#     # free Python containers / references you no longer need
#     sampled_states.clear()           # list of many tuples
#     psis, psis_tensor = None, None                      # if you stored another copy
#     energies, E_tensor = None, None
#     cleanup_memory(free_vars=None, optimizer=optimizer)

# print(f"Step {step}: <E> = {E_mean.item():.6f}")

# states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]  # reinitialize walkers every 100 steps to reduce autocorrelation
# _, _, states, _ = Obtain_Sampling_batch(psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers
# for k in range(10):
#     _, energies, states, _ = Obtain_Sampling_batch(psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device, Sampler=Sampler)
#     if torch.is_tensor(energies):
#         E_tensor = energies.squeeze()
#     else:
#         E_tensor = torch.stack(energies).squeeze()
#     E_mean = E_tensor.mean()
#     E_std = E_tensor.var().sqrt()
#     E_se = E_std / np.sqrt(len(E_tensor))
#     print(f"Final evaluation {k}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")
#     energy_record.append(E_mean.item())
#     error_record.append(E_se.item())
#     # free Python containers / references you no longer need
#     energies, E_tensor = None, None

print("VMC finished.")

# output energy_record to csv
# import pandas as pd
# totalsam = n_walkers* n_samples
# totalsam = int(totalsam)
# df = pd.DataFrame({'Energy': energy_record, 'Error': error_record})
# csv_path = f"data/energy_error_signrule_L{L}_t2{t2}_hidden{hidden_dim}_kernel{kernel_size}_sam{totalsam}_epoch{epochs}.csv"
# df.to_csv(csv_path, index=False)
# print(f"Saved energy and error record to {csv_path}")

# # check final fidelity
# with torch.no_grad():
#     predicted_coeffs = psi(X_tensor).squeeze()
#     print("norm of predicted coeffs:", torch.norm(predicted_coeffs).item())
#     # compute fidelity: ensure both normalized
#     pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
#     exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
#     print("Final fidelity:", torch.sum(pred_norm * exact_gs_tensor).item())



# Save checkpoint
# save_path = f"data/psi_L{L}_t2{t2}_hidden{hidden_dim}.pth" #"data/psi_checkpoint.pth"
# torch.save({
#     'model_state_dict': psi.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': step,
# }, save_path)
# print(f"Saved checkpoint to {save_path}")


# checkpoint = torch.load(save_path, map_location=device)
# psi.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
# psi.to(device)
# print("Loaded checkpoint.")
