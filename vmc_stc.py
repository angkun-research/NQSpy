import torch
import torch.optim as optim
import numpy as np
from vmc_utils import FCNet  # or ConvNet, etc.
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import state_to_onehot, propose_move, local_energy
import random
from tqdm import trange 
import numpy.linalg as la

from utils import total_squared_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L = 7
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
t1 = 1.0
t2 = 0.5
H = build_Hamiltonian(L, t1, t2, basis)
eigvals, eigvecs = la.eigh(H)
print(f"Exact ground state energy: {eigvals[0:3]}")
exact_gs = eigvecs[:, 0]

psi = FCNet(L,hidden_dim=32).to(device)

# Supervised pre training based on the exact ground state
X = np.stack([state_to_onehot(state, L) for state in basis])
# epsilon = 1e-12  # Small value to avoid log(0)
# y = np.array([np.log(np.abs(exact_gs[basis_dict[state]]) + epsilon) for state in basis])
y = np.array([exact_gs[basis_dict[state]] for state in basis])

X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.float32, device=device)# shape [N, 1]

criterion = total_squared_loss

#psi = FCNet(L,hidden_dim=32).to(device)
pretrain_optimizer = optim.Adam(psi.parameters(), lr=1e-2)
for epoch in range(2000):
    pretrain_optimizer.zero_grad()
    outputs = psi(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    pretrain_optimizer.step()
    if epoch % 100 == 0:
        print(f"Pretrain Epoch {epoch}, Loss: {loss.item()}")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = psi(X_tensor)
    print("total loss:", torch.sum((predicted_coeffs - y_tensor) ** 2).item())
    print("fidelity:", torch.sum(predicted_coeffs * y_tensor).item())
    #print("fidelity:", torch.sum(torch.exp(predicted_coeffs) * exact_gs).item())


# Initialize state
state = random.choice(basis)

optimizer = optim.Adam(psi.parameters(), lr=1e-5) #1e-2)

n_steps = 100000
burn_in = 1000
samples = []
energies = []
logpsis = []

for step in trange(n_steps, desc="VMC Sampling"):
    # Propose move
    new_state = propose_move(state, L)
    s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
    new_s_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
    psi_val = psi(s_onehot)
    psi_new_val = psi(new_s_onehot)
    accept_prob = min(1, float((psi_new_val.abs()**2) / (psi_val.abs()**2)))
    if random.random() < accept_prob:
        state = new_state

    if step > burn_in:
        samples.append(state)
        # Compute local energy
        E_loc = local_energy(state, psi, basis, basis_dict, H, device, L)
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_val = psi(s_onehot)
        energies.append(E_loc)
        logpsis.append(torch.log(psi_val.abs() + 1e-12))  # log|psi| for gradient estimator

    # Optimization step every batch
    if step % 10000 == 0 and len(energies) > 0:
        E_tensor = torch.stack(energies)
        logpsi_tensor = torch.stack(logpsis).squeeze()
        logpsi_mean = logpsi_tensor.mean()
        E_mean = E_tensor.mean()
        #VMC loss: log-derivative trick
        loss = 2*torch.mean((E_tensor.detach() - E_mean.detach()) * (logpsi_tensor - logpsi_mean))
        # Variance loss
        #loss = torch.mean((E_tensor - E_mean) ** 2)
        print(f"logpsi_tensor min/max: {logpsi_tensor.min().item()}, {logpsi_tensor.max().item()}")
        print(f"loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}: <E> = {E_mean.item():.6f}")
        energies = []
        logpsis = []


print("VMC finished.")