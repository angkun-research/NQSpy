import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from vmc_utils import FCNet,PhysicalNN, state_to_onehot  # or ConvNet, etc.
from vmc_utils import build_MB_basis, build_Hamiltonian
from NeuralNetworks import LdepConvStrides
from tqdm import trange 

from utils import total_squared_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def obtain_fidelity(L, J1, J2, t1=1.0, t2=0.5, device=device):
    basis = build_MB_basis(L)
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
    Hsparse = csr_matrix(H)
    eigvals, eigvecs = eigsh(Hsparse, k=3, which='SA')
    #print(f"Exact ground state energy: {eigvals[0:3]}")
    exact_gs = eigvecs[:, 0]
    #print("Dimension of Hilbert space:", len(basis))

    #psi = FCNet(L,hidden_dim=32).to(device)
    #psi = PhysicalNN(L, hidden_dim=32, kernel_size=2, holewave=True).to(device)
    strides = (1,2,3)#(2,)
    psi = LdepConvStrides(L, nhole=1, hidden_dim=32,kernel_size=3, strides=strides)

    # Supervised pre training based on the exact ground state
    X = np.stack([state_to_onehot(state, L) for state in basis])
    y = np.array([exact_gs[basis_dict[state]] for state in basis])
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)# shape [N, 1]

    criterion = total_squared_loss

    pretrain_optimizer = optim.Adam(psi.parameters(), lr=1e-2)
    for epoch in range(5000):
        pretrain_optimizer.zero_grad()
        outputs = psi(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        pretrain_optimizer.step()

    # Example: predict coefficients for all configurations
    with torch.no_grad():
        predicted_coeffs = psi(X_tensor)
        lossFinal = torch.sum((predicted_coeffs - y_tensor) ** 2).item()
        print("total loss:", lossFinal)
        fidelity = torch.sum(predicted_coeffs * y_tensor).item()
        print("fidelity:", fidelity)

    return lossFinal,fidelity


# L = 7
# t1 = 1.0
# t2 = 0.5
# Js = [0.01,0.1,0.2,0.5,0.8,1.0,2.0,5.0,10.0]

# losses = np.zeros((len(Js),len(Js)))
# fidelities = np.zeros((len(Js),len(Js)))

# for i,J1 in enumerate(Js):
#     for j,J2 in enumerate(Js):
#         print(f"J1={J1}, J2={J2}")
#         lossFinal,fidelity = obtain_fidelity(L, J1, J2, t1=t1, t2=t2, device=device)
#         losses[i,j] = lossFinal
#         fidelities[i,j] = fidelity

# print("Losses:")
# print(losses)
# print("Fidelities:")
# print(fidelities)

# # save to file
# folder = "./data/"
# np.savez_compressed(folder + "fidelity_J1J2_L7_strides.npz", Js=Js, losses=losses, fidelities=fidelities)


# read from file
folder = "./data/"
data = np.load(folder + "fidelity_J1J2_L7_strides.npz")
Js = data['Js']
losses = data['losses']
fidelities = data['fidelities']

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.imshow(np.log(np.abs(1.0 - fidelities)), cmap="viridis")
plt.colorbar()
plt.xticks(ticks=np.arange(len(Js)), labels=Js)
plt.yticks(ticks=np.arange(len(Js)), labels=Js)
plt.xlabel("J2")
plt.ylabel("J1")
plt.title(f"Fidelity")
#plt.savefig(f"fidelity_heatmap_L{L}_t1{t1}_t2{t2}.png")
plt.show()


# plt.figure(figsize=(8,6))
# plt.imshow(losses, cmap="magma")
# plt.colorbar()
# plt.xticks(ticks=np.arange(len(Js)), labels=Js)
# plt.yticks(ticks=np.arange(len(Js)), labels=Js)
# plt.xlabel("J2")
# plt.ylabel("J1")
# plt.title(f"Loss")
# #plt.savefig(f"loss_heatmap_L{L}_t1{t1}_t2{t2}.png")
# plt.show()