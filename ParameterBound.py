import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from ExactGS import build_MB_basis_holes,build_Hamiltonian_holes
from ExactGS import basis_to_spinconfig_holes,basis_to_onehot
from NeuralNetworks import LdepConvHoles,FCNet, LdepConvStrides,LdepConvPlusFC
from utils import total_squared_loss
from vmc_utils import build_MB_basis,build_Hamiltonian_adjlist,adjlist_to_csr

L = 13
t1 = 1.0 #0.0 #1.0
t2 = 0.5 #0.0 #0.5
nhole = 1 # 2
J1 = 1.0
J2 = 0.0 #0.9
print(f"J2: {J2}")

basis = build_MB_basis_holes(L,nholes=nhole)
basis_old = build_MB_basis(L)
print(f"Number of basis states with {nhole} holes in L={L}: {len(basis)}")

H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis_old, J1=J1, J2=J2)
Hsparse = adjlist_to_csr(H_ind, H_val)
e_vals, e_vecs = eigsh(Hsparse, k=3, which='SA')
print(f"Ground state energy with  {nhole} holes in L={L}: {e_vals[:10]}")
print(f"Gap to first excited state: {e_vals[1] - e_vals[0]}")
# evaluate the ground state
# e0 = e_vecs[:,0].conj() @ Hsparse.dot(e_vecs[:,0])
# print(f"Ground state energy (from expectation value): {e0}")
# exit(0)

configs = basis_to_spinconfig_holes(basis, L)
onehot_configs = basis_to_onehot(configs, L)
onehot_configs = onehot_configs.reshape(-1, L, 4)  # (num_states, L, 4)
X = torch.tensor(onehot_configs, dtype=torch.float32)

# Ground state coefficients (normalize once; eigsh output is typically normalized, but keep it explicit)
y = torch.tensor(e_vecs[:, 0], dtype=torch.float32)
y = y / torch.norm(y)

hidden_dim = 16 #8 #580 #1024
kernel_size = 2 #8
print("hidden_dim:", hidden_dim, "kernel_size:", kernel_size)

criterion = total_squared_loss

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep behavior stable across runs (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_once(seed: int, epochs: int = 2000, lr: float = 1e-3, log_every: int = 100):
    set_seed(seed)

    #model = FCNet(L, hidden_dim=hidden_dim)
    model = LdepConvPlusFC(L, Conv_dim=hidden_dim,kernel_size=kernel_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_avg = loss.item() / len(X)
        if log_every is not None and epoch % log_every == 0:
            print(f"[seed={seed}] Epoch {epoch}, Loss: {loss.item()}, Loss Avg: {loss_avg}")
        if loss_avg < 1e-8:  # Early stopping if loss is very low; 1e-7 for FF, 1e-6 for Conv
            print(f"Early stopping at epoch {epoch} with loss {loss_avg}")
            break

    with torch.no_grad():
        predicted_coeffs = model(X)
        predicted_coeffs = predicted_coeffs.reshape(-1)
        predicted_coeffs = predicted_coeffs / torch.norm(predicted_coeffs)

        # Fidelity = |<psi_true | psi_pred>|
        fidelity = torch.abs(torch.dot(predicted_coeffs, y)).item()
        # energy = <psi_pred | H | psi_pred>
        pred = predicted_coeffs.detach().cpu().numpy()
        energy = float(pred.conj() @ Hsparse.dot(pred))
        print(f"Predicted ground state energy: {energy}")

        # (optional) squared error after normalization (for reference)
        normed_mse = torch.sum((predicted_coeffs - y) ** 2).item()

    return fidelity, energy.item(), normed_mse

# ===== 10 independent runs =====
num_runs = 1 #10 # 10
base_seed = 1234 # random seed #1234 

fidelities = []
energies = []
mses = []

# Parameter count (same each run for same architecture)
#tmp_model = FCNet(L, hidden_dim=hidden_dim)
tmp_model = LdepConvPlusFC(L, Conv_dim=hidden_dim,kernel_size=kernel_size)

total_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

for i in range(num_runs):
    seed = base_seed + i
    fidelity, energy, normed_mse = train_once(seed=seed, epochs=2000, lr=1e-3, log_every=100)
    fidelities.append(fidelity)
    energies.append(energy)
    mses.append(normed_mse)
    print(f"[run {i+1:02d}/{num_runs}] seed={seed} fidelity={fidelity:.8f} energy={energy:.8f} normed_mse={normed_mse:.8e}")

fidelities = np.array(fidelities, dtype=np.float64)
energies = np.array(energies, dtype=np.float64)
mses = np.array(mses, dtype=np.float64)

print("\n=== Summary over 10 runs ===")
print(f"Fidelity mean = {fidelities.mean():.8f}")
#print(f"Fidelity std  = {fidelities.std(ddof=1):.8f}")
print(f"Energy mean   = {energies.mean():.8f}")
#print(f"Energy std    = {energies.std(ddof=1):.8f}")
print(f"MSE mean      = {mses.mean():.8e}")
#print(f"MSE std       = {mses.std(ddof=1):.8e}")