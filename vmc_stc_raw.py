import torch
import torch.optim as optim
import numpy as np
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from vmc_utils import state_to_onehot, propose_move, local_energy,GlobalSampler
from vmc_utils import CoeffAnsatz, local_energy_coeff
import random
from tqdm import trange 
import numpy.linalg as la
from scipy.sparse.linalg import eigsh

from utils import total_squared_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def explicit_vmc_update(psi, energies, logpsis, optimizer):
    """
    Explicitly compute gradients for VMC update:
    2⟨(E_loc - ⟨E_loc⟩) ∂θ_k ln|Ψ_θ(s)|⟩
    and manually update parameters.
    """
    E_tensor = torch.stack(energies).squeeze()
    logpsi_tensor = torch.stack(logpsis).squeeze()
    E_mean = E_tensor.mean()
    weights = (E_tensor - E_mean).detach()

    # Compute gradients of logpsi_tensor w.r.t. parameters for all samples
    params = list(psi.parameters())
    #print("pass 1\n")
    # grads is a tuple of (param_grad for all samples summed)
    # We need per-sample grads, so we need to call grad for each sample
    param_grads = [torch.zeros_like(p) for p in params]
    for i in range(len(logpsi_tensor)):
        psi.zero_grad()
        grad_i = torch.autograd.grad(
            outputs=logpsi_tensor[i],
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )
        for j, g in enumerate(grad_i):
            if g is not None:
                param_grads[j] += 2 * weights[i] * g / len(logpsi_tensor)
    #print("pass 2\n")
    with torch.no_grad():
        lr = optimizer.param_groups[0]['lr']
        for param, grad in zip(params, param_grads):
            param -= lr * grad

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

L = 7
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
t1 = 1.0
t2 = 0.5
H = build_Hamiltonian(L, t1, t2, basis)
eigvals, eigvecs = la.eigh(H)
# H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis)
# H_csr = adjlist_to_csr(H_ind, H_val)
# eigvals, eigvecs = eigsh(H_csr, k=3, which='SA')
print(f"Exact ground state energy: {eigvals[0:3]}")
exact_gs = eigvecs[:, 0]
print("Dimension of Hilbert space:", len(basis))


# --- Pinning / trapping potential (diagonal in computational basis) ---
# define site potential profile v_i (example: harmonic trap centered at L/2)
v_profile = np.array([((i - (L - 1) / 2.0) ** 2) for i in range(L)], dtype=float)
# build diagonal H_pin such that H_pin[state_index] = sum_i v_i * n_i(state)
H_pin_diag = np.zeros(len(basis), dtype=float)
for idx, state in enumerate(basis):
    onehot = state_to_onehot(state, L)
    occ = onehot[:, 1] + onehot[:, 2]
    H_pin_diag[idx] = float(np.dot(v_profile, occ))

psi = CoeffAnsatz(len(basis), complex=False, init_scale=0.01, device=device).to(device)

# # Supervised pre training based on the exact ground state
# X = np.stack([state_to_onehot(state, L) for state in basis])
# # epsilon = 1e-12  # Small value to avoid log(0)
# # y = np.array([np.log(np.abs(exact_gs[basis_dict[state]]) + epsilon) for state in basis])
# #y = np.array([exact_gs[basis_dict[state]] for state in basis])
# #y = np.random.rand(len(basis)).astype(np.float32)# shape [N,]
# y = np.ones(len(basis), dtype=np.float32)
# y /= np.linalg.norm(y)
# print(y.shape)
# X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
# y_tensor = torch.tensor(y, dtype=torch.float32, device=device)# shape [N, 1]

# criterion = total_squared_loss

# pretrain_optimizer = optim.Adam([p for p in psi.parameters()], lr=1e-2)

# for epoch in range(2000):
#     pretrain_optimizer.zero_grad()
#     # get full predicted vector from coefficient ansatz
#     outputs = psi.get_all()
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     pretrain_optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Pretrain Epoch {epoch}, Loss: {loss.item()}")

# # Example: predict coefficients for all configurations
# with torch.no_grad():
#     predicted_coeffs = psi.get_all()
#     print("total loss:", torch.sum((predicted_coeffs - y_tensor) ** 2).item())
#     # compute fidelity: ensure both normalized
#     pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
#     exact_t = torch.tensor(exact_gs, dtype=torch.float32, device=device)
#     exact_norm = exact_t / torch.norm(exact_t)
#     print("fidelity:", torch.sum(pred_norm * exact_norm).item())
# # make grad zero
# psi.zero_grad()

# Initialize state
state = random.choice(basis)
#optimizer = optim.Adam([p for p in psi.parameters()], lr=1e-2)
optimizer = optim.Adam(psi.parameters(), lr=1e-2)
#optimizer = optim.SGD(psi.parameters(), lr=1e-3)

batch_size = 140
n_steps = 1000*batch_size
burn_in = 1000
noise_sigma = 0.05
samples = []
energies = []
logpsis = []
psis = []

# annealing schedule parameters for pinning field
h0 = 0.0 * t1             # initial pinning strength (tunable)
anneal_frac = 0.5         # fraction of total steps to anneal over
T_anneal = max(1, int(n_steps * anneal_frac))
def h_schedule(step):
    # linear anneal to zero over T_anneal steps, then zero
    if step >= T_anneal:
        return 0.0
    return h0 * (1.0 - (step / float(T_anneal)))

for step in trange(n_steps, desc="VMC Sampling"):
    # Propose move
    #new_state = propose_move(state, L)
    new_state = GlobalSampler(state, L)

    # get amplitudes directly from coefficient ansatz
    psi_val = psi.get_by_index(basis_dict[state])
    psi_new_val = psi.get_by_index(basis_dict[new_state])
    accept_prob = min(1, float((psi_new_val.abs()**2).item() / (psi_val.abs()**2).item()))
    if random.random() < accept_prob:
        state = new_state

    if step > burn_in:
        samples.append(state)
        # compute current pinning strength and form H_current (diagonal addition)
        h = h_schedule(step)
        if h != 0.0:
            H_current = H + h * np.diag(H_pin_diag)
        else:
            H_current = H
        # Compute local energy (local_energy now handles CoeffAnsatz)
        E_loc = local_energy_coeff(state, psi, basis, basis_dict, H_current, device, L)
        #E_loc = local_energy_coeff(state, psi, basis, basis_dict, H, device, L)
        energies.append(E_loc)
        logpsis.append(torch.log(psi.get_by_index(basis_dict[state]).abs() + 1e-14))
        psis.append(psi.get_by_index(basis_dict[state]))

    # Optimization step every batch
    if step % batch_size == 0 and len(energies) > 0:
        E_tensor = torch.stack(energies).squeeze()
        psis_tensor = torch.stack(psis).squeeze()
        logpsi_tensor = torch.stack(logpsis).squeeze()
        logpsi_mean = logpsi_tensor.mean()
        E_mean = E_tensor.mean()
        #loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor/psis_tensor.detach())
        loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * logpsi_tensor) #* (logpsi_tensor - logpsi_mean))
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
        # #explicit_vmc_update(psi, energies, logpsis, optimizer)
        # #print(f"Step {step}: <E> = {E_mean.item():.6f}")
        # Use stochastic reconfiguration update (encapsulated)
        #stochastic_reconfiguration_step(psi, energies, logpsis, optimizer, lr=1e-1, reg=1e-3, device=device, noise_sigma=noise_sigma)
        if step % (n_steps//20) == 0:
            #print(f"loss: {loss.item()}")
            E_tensor = torch.stack(energies).squeeze()
            E_mean = E_tensor.mean()
            print(f"Step {step}: <E> = {E_mean.item():.6f}")
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
    predicted_coeffs = psi.get_all()
    # compute fidelity: ensure both normalized
    pred_norm = predicted_coeffs / torch.norm(predicted_coeffs)
    exact_gs_tensor = torch.tensor(exact_gs, dtype=torch.float32, device=device)
    print("Final fidelity:", torch.sum(pred_norm * exact_gs_tensor).item())


# test = np.zeros(len(basis))
# for step in trange(10000, desc="VMC Sampling"):
#     # Propose move
#     new_state = GlobalSampler(state, L)
#     test[basis_dict[new_state]] +=1
#     state = new_state

