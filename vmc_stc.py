import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from vmc_utils import FCNet,PhysicalNN  # or ConvNet, etc.
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from vmc_utils import state_to_onehot, propose_move, local_energy,GlobalSampler, local_energy_from_adjlist
from NeuralNetworks import LdepConvStrides
import random
from tqdm import trange 
import numpy.linalg as la

from utils import total_squared_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=1e-1, reg=1e-3, device=device,
                                    noise_sigma=0.0):
    """
    Perform one stochastic reconfiguration (natural gradient) update.
    - psi: model
    - energies: list of local energies (torch tensors)
    - logpsis: list of log|psi| values (torch tensors) corresponding to samples
    - optimizer: used only to read default lr if lr is None
    - reg: diagonal regularization for S
    """
    E_tensor = torch.stack(energies).squeeze()
    logpsi_tensor = torch.stack(logpsis).squeeze()
    E_mean = E_tensor.mean()
    n_samples = len(logpsi_tensor)

    params = [p for p in psi.parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    n_params = sum(sizes)

    # Build Jacobian J (n_samples, n_params) of ∂_θ lnψ(s)
    J = torch.zeros((n_samples, n_params), device=device, dtype=logpsi_tensor.dtype)
    for i in range(n_samples):
        psi.zero_grad()
        grads = torch.autograd.grad(logpsi_tensor[i], params, retain_graph=True, allow_unused=True)
        row_parts = []
        for g, s in zip(grads, sizes):
            if g is None:
                row_parts.append(torch.zeros(s, device=device, dtype=logpsi_tensor.dtype))
            else:
                row_parts.append(g.contiguous().view(-1))
        J[i] = torch.cat(row_parts)

    O_mean = J.mean(dim=0, keepdim=True)               # (1, n_params)
    Oc = J - O_mean                                    # (n_samples, n_params) 

    S = (Oc.t() @ Oc) / n_samples                      # Fisher / S matrix = ⟨∂_θ lnψ ∂_θ lnψ⟩_c (n_params, n_params)
    E_det = E_tensor.detach().unsqueeze(1)
    EO_mean = (E_det * J).mean(dim=0, keepdim=True)    # (1, n_params)
    g = 2.0 * (EO_mean.squeeze() - E_mean * O_mean.squeeze())  # force vector

    S += reg * torch.eye(n_params, device=device, dtype=S.dtype)
    delta = torch.linalg.solve(S, -g)                  # solve S δ = -g

    if lr is None:
        lr = optimizer.param_groups[0]['lr'] # default lr from optimizer is typicall 1e-3, which is too small for SR

    # Apply update: θ <- θ + lr * δ
    with torch.no_grad():
        idx = 0
        for p, shape, size in zip(params, shapes, sizes):
            seg = delta[idx: idx + size].view(shape)
            if noise_sigma > 0.0:
                # noise scaled by sqrt(lr_eff) like SGLD: std = noise_sigma * sqrt(lr_eff)
                noise = torch.randn_like(seg, device=device) * (noise_sigma * np.sqrt(lr))
            else:
                noise = torch.zeros_like(seg, device=device)
            max_seg = seg.abs().max().item()
            p.add_(lr * seg + noise* max_seg)
            idx += size

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

#psi = FCNet(L,hidden_dim=32).to(device)
psi = PhysicalNN(L, hidden_dim=32, kernel_size=2, holewave=True).to(device)
#strides = (1,2,3,4,5)#(2,)
#psi = LdepConvStrides(L, nhole=1, hidden_dim=32,kernel_size=3, strides=strides)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# Supervised pre training based on the exact ground state
X = np.stack([state_to_onehot(state, L) for state in basis])
# epsilon = 1e-12  # Small value to avoid log(0)
# y = np.array([np.log(np.abs(exact_gs[basis_dict[state]]) + epsilon) for state in basis])
# y = np.array([exact_gs[basis_dict[state]] for state in basis])

X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
# y_tensor = torch.tensor(y, dtype=torch.float32, device=device)# shape [N, 1]

# criterion = total_squared_loss

# #psi = FCNet(L,hidden_dim=32).to(device)
# pretrain_optimizer = optim.Adam(psi.parameters(), lr=1e-2)
# for epoch in range(5000):
#     pretrain_optimizer.zero_grad()
#     outputs = psi(X_tensor)
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     pretrain_optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Pretrain Epoch {epoch}, Loss: {loss.item()}")

# # Example: predict coefficients for all configurations
# with torch.no_grad():
#     predicted_coeffs = psi(X_tensor)
#     print("total loss:", torch.sum((predicted_coeffs - y_tensor) ** 2).item())
#     print("fidelity:", torch.sum(predicted_coeffs * y_tensor).item())
#     #print("fidelity:", torch.sum(torch.exp(predicted_coeffs) * exact_gs).item())


# Initialize state
state = random.choice(basis)

optimizer = optim.Adam(psi.parameters(), lr=1e-3) #1e-3, 1e-2)

batch_size = 1000
n_steps = 1000*batch_size
burn_in = 1000
samples = []
energies = []
psis = []
logpsis = []
noise_sigma = 0.0 #0.1 #0.05

for step in trange(n_steps, desc="VMC Sampling"):
    # Propose move
    #new_state = propose_move(state, L)
    new_state = GlobalSampler(state, L)
    s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
    new_s_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
    psi_val = psi(s_onehot)
    psi_new_val = psi(new_s_onehot)
    accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2).item()))
    if random.random() < accept_prob:
        state = new_state

    if step > burn_in:
        samples.append(state)
        # Compute local energy
        #E_loc = local_energy(state, psi, basis, basis_dict, H, device, L)
        E_loc = local_energy_from_adjlist(state, psi, basis, basis_dict, H_ind, H_val, device, L)
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_val = psi(s_onehot)
        energies.append(E_loc)
        logpsis.append(torch.log(psi_val.abs() + 1e-12))  # log|psi| for gradient estimator
        psis.append(psi_val)

    # Optimization step every batch
    if step % batch_size == 0 and len(energies) > 0:
        E_tensor = torch.stack(energies).squeeze()
        psis_tensor = torch.stack(psis).squeeze()
        logpsi_tensor = torch.stack(logpsis).squeeze()
        logpsi_mean = logpsi_tensor.mean()
        E_mean = E_tensor.mean()
        #loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * (logpsi_tensor - logpsi_mean))
        #loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * logpsi_tensor)
        loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
        #loss = torch.mean((E_tensor - E_mean) ** 2)
        #print(f"E_mean: {E_mean.item()}, E_var: {E_tensor.var().item()}")
        #print(f"logpsi_tensor min/max: {logpsi_tensor.min().item()}, {logpsi_tensor.max().item()}")
        #print(f"loss: {loss.item()}")
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
            #print(f"loss: {loss.item()}")
            E_tensor = torch.stack(energies).squeeze()
            E_mean = E_tensor.mean()
            print(f"Step {step}: <E> = {E_mean.item():.6f}")
            if E_mean.item() < -2.5:
                print(step)
                break  # early stop
        energies = []
        logpsis = []
        samples = []
        psis = []



print("VMC finished.")

E_tensor = torch.stack(energies).squeeze()
E_mean = E_tensor.mean()
print(f"Step {step}: <E> = {E_mean.item():.6f}")

# check final fidelity
with torch.no_grad():
    predicted_coeffs = psi(X_tensor).squeeze()
    print("norm of predicted coeffs:", torch.norm(predicted_coeffs).item())
    # compute fidelity: ensure both normalized
    pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
    exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
    print("Final fidelity:", torch.sum(pred_norm * exact_gs_tensor).item())