import torch
import torch.optim as optim
import numpy as np
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import state_to_onehot, propose_move, local_energy,GlobalSampler
from vmc_utils import CoeffAnsatz, local_energy_coeff
import random
from tqdm import trange 
import numpy.linalg as la

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

L = 7
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
t1 = 1.0
t2 = 0.5
H = build_Hamiltonian(L, t1, t2, basis)
eigvals, eigvecs = la.eigh(H)
print(f"Exact ground state energy: {eigvals[0:3]}")
exact_gs = eigvecs[:, 0]

psi = CoeffAnsatz(len(basis), complex=False, init_scale=0.01, device=device).to(device)

# # Supervised pre training based on the exact ground state
# X = np.stack([state_to_onehot(state, L) for state in basis])
# # epsilon = 1e-12  # Small value to avoid log(0)
# # y = np.array([np.log(np.abs(exact_gs[basis_dict[state]]) + epsilon) for state in basis])
# y = np.array([exact_gs[basis_dict[state]] for state in basis])

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
#optimizer = optim.Adam(psi.parameters(), lr=1e-1)
optimizer = optim.SGD(psi.parameters(), lr=1e-3)

batch_size = 10000
n_steps = 100*batch_size
burn_in = 10000
samples = []
energies = []
logpsis = []
psis = []

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
        # Compute local energy (local_energy now handles CoeffAnsatz)
        E_loc = local_energy_coeff(state, psi, basis, basis_dict, H, device, L)
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
        optimizer.step()
        #explicit_vmc_update(psi, energies, logpsis, optimizer)
        #print(f"Step {step}: <E> = {E_mean.item():.6f}")
        energies = []
        logpsis = []
        samples = []
        psis = []
        if step % (n_steps//20) == 0:
            print(f"loss: {loss.item()}")
            print(f"Step {step}: <E> = {E_mean.item():.6f}")


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

