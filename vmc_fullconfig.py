import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from vmc_utils import build_MB_basis
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from vmc_utils import state_to_onehot
from vmc_utils import VBSNN
from vmc_utils import compute_log_psi_jacobian_vmap, sr_gradient
from tqdm import trange 
import numpy.linalg as la

def sr_update_optimizer_fullconfig(psi, psi_vals, Hpsi, probs,E_mean, optimizer, tau=1e-3, device='cpu'):
    # avoid unstable division near psi=0
    mask = probs > 1e-14
    probs_m = probs[mask]
    probs_m = probs_m / probs_m.sum()

    psi_m = psi_vals[mask]
    Hpsi_m = Hpsi[mask]
    E_loc = Hpsi_m / psi_m

    Joc = compute_log_psi_jacobian_vmap(psi, s_np[mask.cpu().numpy()], device)
    E_centered = E_loc.detach() - E_mean.detach()

    sqrt_p = probs_m.sqrt().unsqueeze(1)
    J_weighted = sqrt_p * Joc
    E_weighted = probs_m.sqrt() * E_centered

    delta_theta = sr_gradient(J_weighted, E_weighted, tau=tau)

    optimizer.zero_grad()
    offset = 0
    for p in psi.parameters():
        if p.requires_grad:
            numel = p.numel()
            # Reshape the 1D chunk of delta_theta back to the parameter's shape.
            # .clone() ensures we don't have overlapping memory issues from the 1D tensor.
            p.grad = delta_theta[offset:offset + numel].reshape(p.shape).clone()
            offset += numel

    optimizer.step()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

L = 19 #11
t1 = 1.0
t2 = 0.5
J1 = 1.0 #1.0
J2 = 0.9 #0.9 #0.81/100
# basis = build_MB_basis(L)
# # Build basis tensor once
# s_np = np.stack([state_to_onehot(s, L) for s in basis])
# s_tensor = torch.from_numpy(s_np).float().to(device)

# # Build Hamiltonian once
# H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
# Hsparse = adjlist_to_csr(H_ind, H_val).tocoo()

# indices = torch.tensor(
#     np.vstack([Hsparse.row, Hsparse.col]),
#     dtype=torch.long,
#     device=device,
# )
# values = torch.tensor(Hsparse.data, dtype=torch.float32, device=device)
# H_torch = torch.sparse_coo_tensor(
#     indices,
#     values,
#     size=Hsparse.shape,
#     device=device,
# ).coalesce() # shape (num_states, num_states)


# # compared to exact diagonalization
# eigvals, eigvecs = eigsh(Hsparse, k=3, which='SA')
# # eigvals, eigvecs = la.eigh(H)
# print(f"Exact ground state energy for L {L}: {eigvals[0:3]}")
# exact_gs = eigvecs[:, 0]
# print("Dimension of Hilbert space:", len(basis))
# E_exact = eigvals[0] 
# print(f"Exact ground state energy for L={L}: {E_exact}")

# h 128, k 5 for L=7;
hidden_dim = 10 #64 #16 
kernel_size = 8
print("hidden_dim:", hidden_dim, "kernel_size:", kernel_size)
psi = VBSNN(L, Conv_dim=hidden_dim,kernel_size=kernel_size)
psi.to(device)
# print number of parameters
n_params = sum(p.numel() for p in psi.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")
exit()
#optimizer = optim.Adam(psi.parameters(), lr=1e-2) # 1e-2 3e-3
optimizer = optim.SGD(psi.parameters(), lr=1e-0)

sr_tau = 1e-1
lr = 1e-0
epochs = 5000 #2000

# record energy for every step after burn-in
energy_record = [] 
error_record = []
for step in trange(epochs, desc="full Sampling"):
    psi_vals = psi(s_tensor).reshape(-1)            
    Hpsi = torch.sparse.mm(H_torch, psi_vals.unsqueeze(1)).squeeze(1) # shape (num_states,)

    norm = torch.dot(psi_vals, psi_vals).clamp_min(1e-12)
    probs = psi_vals.abs().square() / norm
    E_mean = torch.dot(psi_vals, Hpsi) / norm

    # optimizer.zero_grad()
    # E_mean.backward()
    # optimizer.step()

    sr_update_optimizer_fullconfig(psi, psi_vals, Hpsi, probs, E_mean, optimizer, tau=sr_tau, device=device)
    
    if step % 100 == 0: #if step % (epochs//20) == 0:
        print(f"Step {step}: <E> = {E_mean.item():.6f}")

    #print(f"Step {step}: <E> = {E_mean.item():.6f}")
    # free Python containers / references you no longer need
    weights, energies = None, None
    weights_tensor, energies_tensor = None, None

print(f"Step {step}: <E> = {E_mean.item():.6f}")

print("Full states optimization finished.")
print(f"Relative error in energy: {(E_mean.item() - E_exact) / abs(E_exact):.6e}")


# check final fidelity against exact ground state
with torch.no_grad():
    predicted_coeffs = psi(s_tensor).squeeze()
    print("norm of predicted coeffs:", torch.norm(predicted_coeffs).item())
    # compute fidelity: ensure both normalized
    pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
    exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
    fidelity = torch.sum(pred_norm * exact_gs_tensor).item()
    print(f"Final fidelity: {fidelity:.8f}")
