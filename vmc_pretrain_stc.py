import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from vmc_utils import FCNet,PhysicalNN  # or ConvNet, etc.
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from vmc_utils import state_to_onehot, local_energy,GlobalSampler, local_energy_from_adjlist
from vmc_utils import local_energy_on_the_fly
from vmc_utils import TightBinding_coeff, find_state_coeff
from NeuralNetworks import PhysicsLocalLayer
import random
from tqdm import trange 
import numpy.linalg as la
from ExactGS import exact_ground_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Obtain_Sampling(initial_state, n_samples, L, pretrain=False, burnin = False,tb_coeff=None):
    Psis = []
    Elocs = []
    state = initial_state
    new_state = state
    for _ in range(n_samples):
        new_state = GlobalSampler(state, L)
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

    return Psis, Elocs, state


L = 19
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
t1 = 1.0
t2 = 0.5
J1 = 0.0
J2 = 0.0 #0.81/100
#H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
#Hsparse = csr_matrix(H)
H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
Hsparse = adjlist_to_csr(H_ind, H_val)
eigvals, eigvecs = eigsh(Hsparse, k=3, which='SA')
# eigvals, eigvecs = la.eigh(H)
print(f"Exact ground state energy: {eigvals[0:3]}")
exact_gs = eigvecs[:, 0]
print("Dimension of Hilbert space:", len(basis))

hidden_dim = 128 #32
#psi = FCNet(L,hidden_dim=32).to(device)
psi = PhysicalNN(L, hidden_dim=hidden_dim, kernel_size=2, holewave=True).to(device)
#strides = (1,2,3,4,5)#(2,)
#psi = LdepConvStrides(L, nhole=1, hidden_dim=32,kernel_size=3, strides=strides)
#psi = PhysicsLocalLayer(n_freqs=16)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# Initialize state
initial_state = random.choice(basis)

optimizer = optim.Adam(psi.parameters(), lr=1e-3) #1e-3)

tb_coeff = TightBinding_coeff(L, t1, t2)

n_samples = 100
epochs = 10000
burn_in = 1000

_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True)

for step in trange(epochs, desc="VMC Sampling"):
    psis, psi0s, state = Obtain_Sampling(state, n_samples, L, pretrain=True, tb_coeff=tb_coeff)
    # Optimization step every batch
    psis_tensor = torch.stack(psis).squeeze()
    psi0s_tensor = torch.stack(psi0s).squeeze()
    loss = torch.mean((psis_tensor - psi0s_tensor.detach()) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Use stochastic reconfiguration update (encapsulated)
    #stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=1e-1, reg=1e-3, device=device, noise_sigma=noise_sigma)
    if step % (epochs//20) == 0:
        print(f"loss: {loss.item()}")



# print("Pretrain finished.")

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


# start VMC fine-tuning
# Initialize state
initial_state = random.choice(basis)

optimizer = optim.Adam(psi.parameters(), lr=1e-3) #1e-3, 1e-2)

n_samples = 10000
epochs = 100
burn_in = 1000

_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True) # burn-in

for step in trange(epochs, desc="VMC Sampling"):
    psis, energies, state = Obtain_Sampling(state, n_samples, L)
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
    if step % (epochs//20) == 0:
        #print(f"loss: {loss.item()}")
        #E_tensor = torch.stack(energies).squeeze()
        #E_mean = E_tensor.mean()
        print(f"Step {step}: <E> = {E_mean.item():.6f}")
    # if E_mean.item() < -2.5:
    #     print(step)
    #     break  # early stop

print("VMC finished.")

for k in range(5):
    _, energies, state = Obtain_Sampling(state, n_samples, L)
    E_tensor = torch.stack(energies).squeeze()
    E_mean = E_tensor.mean()
    print(f"Next sample {k}: <E> = {E_mean.item():.6f}")

# check final fidelity
with torch.no_grad():
    predicted_coeffs = psi(X_tensor).squeeze()
    print("norm of predicted coeffs:", torch.norm(predicted_coeffs).item())
    # compute fidelity: ensure both normalized
    pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
    exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
    print("Final fidelity:", torch.sum(pred_norm * exact_gs_tensor).item())