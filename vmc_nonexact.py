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
from vmc_utils import NNConvStrides
from vmc_utils import stochastic_reconfiguration_matrix_free_real
from NeuralNetworks import LdepConvPlusFC
import random
from tqdm import trange 
import numpy.linalg as la

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

def Obtain_Sampling_batch(initial_states, n_steps, L, pretrain=False, burnin=False, tb_coeff=None, print_rate=False,
                          Sampler=GlobalSampler):
    """Run multiple independent walkers in parallel (batched NN evaluation).

    initial_state: single state (will be copied to initialize all walkers)
    n_steps: number of MCMC steps per walker
    n_walkers: number of parallel walkers
    Returns flattened lists of psis and elocs and the list of final walker states.
    """
    # initialize walkers
    n_walkers = len(initial_states)
    states = initial_states.copy()
    Psis = []
    Elocs = []
    accept_count = 0

    for _ in range(n_steps):
        # propose moves for all walkers (kept simple: use Sampler per-walker)
        proposed = [Sampler(s, L) for s in states]

        # build batched one-hot inputs efficiently via numpy.stack -> torch.from_numpy
        s_np = np.stack([state_to_onehot(s, L) for s in states])
        new_np = np.stack([state_to_onehot(s, L) for s in proposed])
        s_batch = torch.from_numpy(s_np).float().to(device)
        new_batch = torch.from_numpy(new_np).float().to(device)

        # evaluate psi for all current and proposed states in one forward pass
        psi_vals = psi(s_batch).squeeze()
        psi_new_vals = psi(new_batch).squeeze()

        # compute acceptance probabilities (vectorized)
        denom = psi_vals.abs()**2
        ratio = (psi_new_vals.abs()**2) / (denom + 1e-12)
        accept_prob = torch.clamp(ratio, max=1.0)

        rand = torch.rand(n_walkers, device=device)
        accept = (rand < accept_prob).cpu().numpy()

        # update walkers and collect psi values
        for i in range(n_walkers):
            if accept[i]:
                states[i] = proposed[i]
                Psis.append(psi_new_vals[i])
                accept_count += 1
            else:
                Psis.append(psi_vals[i])

        if burnin:
            continue

        # compute local energies per walker (still per-walker calls)
        for s in states:
            if not pretrain:
                E_loc = local_energy_on_the_fly(s, psi, L, t1, t2, J1=J1, J2=J2, device=device)
                Elocs.append(E_loc)
            else:
                Elocs.append(torch.tensor(find_state_coeff(L, s, tb_coeff), device=device))

    if print_rate:
        print(f"Acceptance rate (batched): {accept_count / (n_steps * n_walkers):.4f}")

    # flatten Psis/Elocs are lists of tensors -> stack
    if len(Psis) > 0:
        Psis_t = torch.stack(Psis).squeeze()
    else:
        Psis_t = torch.tensor([], device=device)
    if len(Elocs) > 0:
        Elocs_t = torch.stack(Elocs).squeeze()
    else:
        Elocs_t = torch.tensor([], device=device)

    # return psis, elocs, and final states (return first state for compatibility)
    return Psis_t, Elocs_t, states


L = 11 #31
t1 = 1.0
t2 = 0.5
J1 = 1.0 #0.0
J2 = 0.9 #0.0 #0.81/100
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
#H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
#Hsparse = csr_matrix(H)
H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
Hsparse = adjlist_to_csr(H_ind, H_val)
eigvals, eigvecs = eigsh(Hsparse, k=3, which='SA')
# eigvals, eigvecs = la.eigh(H)
print(f"Exact ground state energy for L {L}: {eigvals[0:3]}")
exact_gs = eigvecs[:, 0]
print("Dimension of Hilbert space:", len(basis))

# L =21 J2=1.2, [-10.15876638 -10.04104634  -9.98258285]
# L=21, J2=0.9, [-10.00927021  -9.88692634  -9.6925758 ]


#psi = NNConvStrides(L, nhole=1, in_channels=4, hidden_dim=32, strides=(1,2,3,4)).to(device)
# h 128, k 5 for L=7;
hidden_dim = 32 #512 #128 #16 #32 
kernel_size = 5 #16
print("hidden_dim:", hidden_dim, "kernel_size:", kernel_size)
psi = LdepConvPlusFC(L, Conv_dim=hidden_dim,kernel_size=kernel_size)
psi.to(device)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# start VMC 
# Initialize state
#initial_state = generate_initial_state(L) #random.choice(basis)
n_walkers = 32 # 4
initial_states = [generate_initial_state(L) for _ in range(n_walkers)]  # 4 parallel walkers

optimizer = optim.Adam(psi.parameters(), lr=3e-3) # 1e-2 3e-3

n_samples = 64 #250 #1000
epochs = 100
burn_in = 64 #1000
Sampler = BalancedSampler if J2 == J1 else GlobalSampler

#_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True) # burn-in
_, _, states = Obtain_Sampling_batch(initial_states, burn_in, L, burnin=True)  # burn-in for all walkers

# record energy for every step after burn-in
energy_record = [] 
for step in trange(epochs, desc="VMC Sampling"):
    #psis, energies, state = Obtain_Sampling(state, n_samples, L,print_rate=False, Sampler=BalancedSampler)
    psis, energies, states = Obtain_Sampling_batch(states, n_samples, L, print_rate=False, Sampler=Sampler)
    # Optimization step every batch
    # E_tensor = torch.stack(energies).squeeze()
    # psis_tensor = torch.stack(psis).squeeze()
    if torch.is_tensor(energies):
        E_tensor = energies.squeeze()
    else:
        E_tensor = torch.stack(energies).squeeze()
    if torch.is_tensor(psis):
        psis_tensor = psis.squeeze()
    else:
        psis_tensor = torch.stack(psis).squeeze()
    E_mean = E_tensor.mean()
    # loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # Use stochastic reconfiguration update (encapsulated)
    #logpsis = torch.log(psis_tensor + 1e-12)  # add small constant for numerical stability
    # #stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=3e-3, reg=1e-7, device=device)
    # stochastic_reconfiguration_matrix_free(psi, energies, logpsis, optimizer, lr=1e-2, reg=1e-7, device=device)
    # need more testing
    #stochastic_reconfiguration_matrix_free_real(psi, energies, logpsis, optimizer, lr=3e-3, reg=1e-7, device=device)
    # if step % (epochs//20) == 0:
    #     print(f"Step {step}: <E> = {E_mean.item():.6f}")
    print(f"Step {step}: <E> = {E_mean.item():.6f}")
    energy_record.append(E_mean.item())

print(f"Step {step}: <E> = {E_mean.item():.6f}")

print("VMC finished.")

for k in range(5):
    #_, energies, state = Obtain_Sampling(state, n_samples, L)
    _, energies, states = Obtain_Sampling_batch(states, n_samples, L)
    if torch.is_tensor(energies):
        E_tensor = energies.squeeze()
    else:
        E_tensor = torch.stack(energies).squeeze()
    E_mean = E_tensor.mean()
    print(f"Next sample {k}: <E> = {E_mean.item():.6f}")

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

# L=21, J1=1.0, J2=0.5, E=-10.22192679
# L=11, E=-6.155918052962847; no J: -2.4298820

# output energy_record to csv
# import pandas as pd
# df = pd.DataFrame({'Energy': energy_record})
# csv_path = f"data/energy_record_L{L}_t2{t2}_hidden{hidden_dim}.csv"
# df.to_csv(csv_path, index=False)
# print(f"Saved energy record to {csv_path}")