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
from vmc_utils import VBSNN, sr_update, Obtain_Sampling_batch
import random
from tqdm import trange 
import numpy.linalg as la

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

L = 21 #7 #11
t1 = 1.0
t2 = 0.5
J1 = 1.0 #0.0
J2 = 0.9 #0.0 #0.81/100
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

E_exact = -10.00927 #eigvals[0] # L = 31, E=-13.795; L=21, E=-10.00927

# h 128, k 5 for L=7;
hidden_dim = 64 #64 
kernel_size = 10 #5
print("hidden_dim:", hidden_dim, "kernel_size:", kernel_size)
psi = VBSNN(L, Conv_dim=hidden_dim,kernel_size=kernel_size)
psi.to(device)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

# folder = "data/"
# filename = f"psi_L{L}_sr_test.pth"
# save_path = folder + filename
# checkpoint = torch.load(save_path, map_location=device)
# psi.load_state_dict(checkpoint['model_state_dict'])
# psi.to(device)
# print("Loaded Pre-trained model.")

# start VMC 
# Initialize state
#initial_state = generate_initial_state(L) #random.choice(basis)
n_walkers = 64 #64 # 32
initial_states = [generate_initial_state(L) for _ in range(n_walkers)]  # 4 parallel walkers

optimizer = optim.Adam(psi.parameters(), lr=1e-2) # 1e-2 3e-3

n_samples = 128 #64 #64 
epochs = 200 # 200
N_eff = n_samples * n_walkers / 10  # effective sample size, adjusted for autocorrelation (heuristic factor of 10)
burn_in = 64 #1000
Sampler = BalancedSampler if J2 == J1 else GlobalSampler
lr = 5e-1 # learning rate for SR updates
sr_tau = 1e0 # regularization shift for SR (tau)

#_,_, state = Obtain_Sampling(initial_state, burn_in, L, burnin=True) # burn-in
_, _, states, _ = Obtain_Sampling_batch(psi, L, initial_states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True)  # burn-in for all walkers

# record energy for every step after burn-in
energy_record = [] 
for step in trange(epochs, desc="VMC Sampling"):
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

    sr_update(psi, sampled_states, E_tensor, L, lr, sr_tau, device)
    
    # if step % (epochs//20) == 0:
    #     print(f"Step {step}: <E> = {E_mean.item():.6f}")
    print(f"Step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")
    #energy_record.append(E_mean.item())
    if E_mean.item() > E_exact*0.85:#-3: #-6: #-2.0:
        lr = 5e-1 #2e-1
        sr_tau = 1e0 #1e-2
        optimizer = optim.Adam(psi.parameters(), lr=lr)
    else:
        lr = 2e-2 #5e-2 2e-2
        sr_tau = 1e-4 #1e-3 #1e-5 
        optimizer = optim.Adam(psi.parameters(), lr=lr)
    if E_mean.item() < E_exact+0.01:#-4.5: #-2.518: #-2.48: #-2.286:
        break

print(f"Step {step}: <E> = {E_mean.item():.6f}")

print("VMC finished.")


#check final fidelity
# X = np.stack([state_to_onehot(state, L) for state in basis])
# X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
# with torch.no_grad():
#     predicted_coeffs = psi(X_tensor).squeeze()
#     print("norm of predicted coeffs:", torch.norm(predicted_coeffs).item())
#     # compute fidelity: ensure both normalized
#     pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
#     exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
#     print("Final fidelity:", torch.sum(pred_norm * exact_gs_tensor).item())


# Save checkpoint
# save_path = f"data/psi_L{L}_sr_test.pth"
# torch.save({
#     'model_state_dict': psi.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': step,
# }, save_path)
# print(f"Saved checkpoint to {save_path}")


# def compute_log_psi_jacobian(psi, samples_onehot, device):
#     """Compute the Jacobian of log|psi| w.r.t. all model parameters.

#     Args:
#         psi: the neural network wavefunction model.
#         samples_onehot: np.ndarray of shape (Ns, L, 4), one-hot encoded samples.
#         device: torch device.

#     Returns:
#         jacobian: torch.Tensor of shape (Ns, Np), where Np is the total
#                   number of trainable parameters.
#     """
#     Ns = samples_onehot.shape[0]
#     x = torch.from_numpy(samples_onehot).float().to(device)

#     # Forward pass for all samples
#     psi_vals = psi(x)  # (Ns,)

#     # log|psi| for each sample
#     log_psi = torch.log(torch.abs(psi_vals) + 1e-12)  # (Ns,)

#     # Collect parameter count
#     params = [p for p in psi.parameters() if p.requires_grad]
#     Np = sum(p.numel() for p in params)

#     jacobian = torch.zeros(Ns, Np, device=device)

#     # Compute gradient of log|psi(sigma_i)| for each sample
#     for i in range(Ns):
#         psi.zero_grad()
#         if log_psi.grad_fn is not None:
#             log_psi[i].backward(retain_graph=(i < Ns - 1))
#         grads = []
#         for p in params:
#             if p.grad is not None:
#                 grads.append(p.grad.detach().reshape(-1))
#                 p.grad = None  # reset for next sample
#             else:
#                 grads.append(torch.zeros(p.numel(), device=device))
#         jacobian[i] = torch.cat(grads)
#         print(f"Computed Jacobian for sample {i+1}/{Ns}", end='\r')

#     return jacobian


# s_np = np.stack([state_to_onehot(s, L) for s in basis])
# J = compute_log_psi_jacobian(psi, s_np, device)  # (Ns, Np)
# print("Computed Jacobian shape:", J.shape)

# J_vmap = compute_log_psi_jacobian_vmap(psi, s_np, device)
# print("Computed Jacobian via vmap shape:", J_vmap.shape)

# # compare J and J_vmap
# diff = torch.norm(J - J_vmap).item()
# print(f"Difference between manual Jacobian and vmap Jacobian: {diff:.6e}")


# from scipy.sparse.linalg import cg

# def sr_gradient_cg(J, E_loc, tau=1e-3, maxiter=100):
#     """CG-based SR for large parameter counts (avoids forming S)."""
#     J64 = J.detach().double()        # promote to float64 for CG accuracy
#     E64 = E_loc.detach().double()
#     def matvec(v):
#         v_t = torch.from_numpy(v).to(J64)
#         return (J64.T @ (J64 @ v_t) + tau * v_t).cpu().numpy()

#     from scipy.sparse.linalg import LinearOperator
#     Np = J.shape[1]
#     A = LinearOperator((Np, Np), matvec=matvec, dtype=np.float64)
#     rhs = (J64.T @ E64).cpu().numpy()
#     sol, info = cg(A, rhs, maxiter=maxiter)
#     if info > 0:
#         print(f"Warning: CG did not converge (info={info})")
#     return torch.from_numpy(sol).float().to(J.device)

# Elocs = []
# for s in basis:
#     E_loc = local_energy_on_the_fly(s, psi, L, t1, t2, J1=J1, J2=J2, device=device)
#     Elocs.append(E_loc)
# E_tensor = torch.stack(Elocs).squeeze()
# print(f"Mean local energy over basis: {E_tensor.mean().item():.6f}")

# tau = 1e-2 #1e-3
# delta_theta = sr_gradient(J, E_tensor, tau=tau)

# #print(J.mean(dim=0, keepdim=True).shape)
# J_centered = J - J.mean(dim=0, keepdim=True)
# E_centered = E_tensor - E_tensor.mean()
# delta_theta_center = sr_gradient(J_centered, E_centered, tau=tau)

# diff_cg = torch.norm(delta_theta - delta_theta_center).item()
# print(f"Difference between direct SR and centered SR updates: {diff_cg:.6e}")

# delta_theta_cg = sr_gradient_cg(J, E_tensor, tau=tau, maxiter=10000) # only works for larger tau
# diff_cg = torch.norm(delta_theta - delta_theta_cg).item()
# print(f"Difference between direct SR and CG SR updates: {diff_cg:.6e}")