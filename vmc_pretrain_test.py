import torch
import torch.optim as optim
import numpy as np

from ExactGS import exact_ground_state

from vmc_utils import FCNet,PhysicalNN
from vmc_utils import build_MB_basis
from vmc_utils import state_to_onehot,GlobalSampler,local_energy
from vmc_utils import TightBinding_coeff, find_state_coeff
import random
from tqdm import trange 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L = 11
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
t1 = 1.0
t2 = 0.5
exact_gs = exact_ground_state(L, t1, t2,basis, TBcoeff=True)
exact_gs = exact_gs / np.linalg.norm(exact_gs)  # normalize
print("Dimension of Hilbert space:", len(basis))
# # H = |gs><gs| for fidelity calculation
# # Build density matrix (explicit projector) -- full matrix (may be large)
# psi0 = np.array(exact_gs, dtype=np.float32)
# H = np.outer(psi0, psi0.conj())  # numpy matrix: H_{ij} = psi0[i] * psi0[j].conj()

#psi = FCNet(L,hidden_dim=32).to(device)
psi = PhysicalNN(L, hidden_dim=32, kernel_size=2, holewave=True).to(device)
#strides = (1,2,3,4,5)#(2,)
#psi = LdepConvStrides(L, nhole=1, hidden_dim=32,kernel_size=3, strides=strides)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# Initialize state
state = random.choice(basis)

optimizer = optim.Adam(psi.parameters(), lr=1e-3) #1e-3)

batch_size = 100
n_steps = 10000*batch_size
burn_in = batch_size
samples = []
#energies = []
psis = []
logpsis = []
psi0 = []
logpsi0 = []
noise_sigma = 0.0 #0.1 #0.05

tb_coeff = TightBinding_coeff(L, t1, t2)

for step in trange(n_steps, desc="VMC Sampling"):
    # Propose move
    new_state = GlobalSampler(state, L)
    s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
    new_s_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
    psi_val = psi(s_onehot)
    psi_new_val = psi(new_s_onehot)
    accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2).item()))
    # old_prob = (psi_val.abs()**2).item() + exact_gs[basis_dict[state]]**2
    # new_prob = (psi_new_val.abs()**2).item() + exact_gs[basis_dict[new_state]]**2
    # accept_prob = min(1, float(new_prob / (old_prob)))
    if random.random() < accept_prob:
        state = new_state

    if step > burn_in:
        samples.append(state)
        # Compute local energy
        #E_loc = local_energy(state, psi, basis, basis_dict, H, device, L)
        #E_loc = local_fidelity(state, psi, basis, basis_dict, exact_gs, device, L)
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_val = psi(s_onehot)
        #energies.append(E_loc)
        logpsis.append(torch.log(psi_val.abs() + 1e-12))  # log|psi| for gradient estimator
        psis.append(psi_val)
        #psi0.append(torch.tensor(exact_gs[basis_dict[state]], device=device))
        psi0.append(torch.tensor(find_state_coeff(L, state, tb_coeff), device=device))
        #logpsi0.append(torch.log(torch.tensor(exact_gs[basis_dict[state]],device=device).abs() + 1e-12))
    # Optimization step every batch
    if step % batch_size == 0 and len(psis) > 0:
        #E_tensor = torch.stack(energies).squeeze()
        psis_tensor = torch.stack(psis).squeeze()
        logpsi_tensor = torch.stack(logpsis).squeeze()
        #logpsi_mean = logpsi_tensor.mean()
        #E_mean = E_tensor.mean()
        psi0_tensor = torch.stack(psi0).squeeze()
        #logpsi0_tensor = torch.stack(logpsi0).squeeze()
        #loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
        #loss = torch.mean((logpsi_tensor - logpsi0_tensor) ** 2)
        loss = torch.mean((psis_tensor - psi0_tensor.detach()) ** 2)
        #loss = torch.mean((logpsi_tensor - logpsi0_tensor.detach()) ** 2)
        optimizer.zero_grad() 
        loss.backward()
        if noise_sigma > 0.0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        # max of grad
                        max_grad = p.grad.abs().max().item()
                        p.grad.add_(torch.randn_like(p.grad) * noise_sigma* np.sqrt(group['lr']* max_grad))
        optimizer.step()
        # Use stochastic reconfiguration update (encapsulated)
        #stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=1e-1, reg=1e-3, device=device, noise_sigma=noise_sigma)
        if step % (n_steps//20) == 0:
            print(f"loss: {loss.item()}")
            # E_tensor = torch.stack(energies).squeeze()
            # E_mean = E_tensor.mean()
            # print(f"Step {step}: <E> = {E_mean.item():.6f}")
        #energies = []
        #logpsis = []
        samples = []
        psis = []
        psi0 = []
        logpsi0 = []
        logpsis = []



print("VMC finished.")

# E_tensor = torch.stack(energies).squeeze()
# E_mean = E_tensor.mean()
# print(f"Step {step}: <E> = {E_mean.item():.6f}")

X = np.stack([state_to_onehot(state, L) for state in basis])
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
# check final fidelity
with torch.no_grad():
    predicted_coeffs = psi(X_tensor).squeeze()
    print("Predicted norm:", torch.norm(predicted_coeffs).item())
    print("expected norm:", (2 ** ((L-1) / 4)))
    # compute fidelity: ensure both normalized
    pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
    print("Predicted norm after normalization:", torch.norm(pred_norm).item())
    exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
    print("Exact GS norm:", torch.norm(exact_gs_tensor).item())
    print("Final fidelity:", torch.sum(pred_norm * exact_gs_tensor).item())