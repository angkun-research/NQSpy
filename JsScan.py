import torch
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from vmc_utils import FCNet,PhysicalNN, state_to_onehot  # or ConvNet, etc.
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from NeuralNetworks import LdepConvStrides,LdepConvPlusFC
from tqdm import trange 

from utils import total_squared_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def obtain_fidelity(L, J1, J2, t1=1.0, t2=0.5, device=device):
    basis = build_MB_basis(L)
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
    H_csr = adjlist_to_csr(H_ind, H_val)
    eigval_csr, eigvecs_csr = eigsh(H_csr, k=3, which='SA')
    print(f"Exact ground state energy: {eigval_csr[0:3]}")
    exact_gs = eigvecs_csr[:, 0]
    #print("Dimension of Hilbert space:", len(basis))

    #psi = FCNet(L,hidden_dim=32).to(device)
    #psi = PhysicalNN(L, hidden_dim=32, kernel_size=2, holewave=True).to(device)
    # strides = (1,2,3)#(2,)
    # psi = LdepConvStrides(L, nhole=1, hidden_dim=32,kernel_size=3, strides=strides)
    hidden_dim = 64 #16 #32
    kernel_size = 5 #2 #16
    psi = LdepConvPlusFC(L, Conv_dim=hidden_dim,kernel_size=kernel_size)

    # Supervised pre training based on the exact ground state
    X = np.stack([state_to_onehot(state, L) for state in basis])
    y = np.array([exact_gs[basis_dict[state]] for state in basis])
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)# shape [N, 1]

    criterion = total_squared_loss

    pretrain_optimizer = optim.Adam(psi.parameters(), lr=1e-2)
    for epoch in range(2000):
        pretrain_optimizer.zero_grad()
        outputs = psi(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        pretrain_optimizer.step()
        if epoch % 500 == 0:
            print(f"Pretrain Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # Example: predict coefficients for all configurations
    with torch.no_grad():
        predicted_coeffs = psi(X_tensor)
        lossFinal = torch.sum((predicted_coeffs - y_tensor) ** 2).item()
        print("total loss:", lossFinal)
        predicted_coeffs = predicted_coeffs / torch.norm(predicted_coeffs)
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
# folder = "./data/"
# data = np.load(folder + "fidelity_J1J2_L7_strides.npz")
# Js = data['Js']
# losses = data['losses']
# fidelities = data['fidelities']

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,6))
# plt.imshow(np.log(np.abs(1.0 - fidelities)), cmap="viridis")
# plt.colorbar()
# plt.xticks(ticks=np.arange(len(Js)), labels=Js)
# plt.yticks(ticks=np.arange(len(Js)), labels=Js)
# plt.xlabel("J2")
# plt.ylabel("J1")
# plt.title(f"Fidelity")
# #plt.savefig(f"fidelity_heatmap_L{L}_t1{t1}_t2{t2}.png")
# plt.show()


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

print("kernel size is 5, hidden dim is 64")
L = 13
J2s = np.arange(0.0, 2.1, 0.1)

losses = np.zeros(len(J2s))
fidelities = np.zeros(len(J2s))

for i,J2 in enumerate(J2s):
    lossFinal,fidelity = obtain_fidelity(L, J1=1.0, J2=J2, device=device)
    losses[i] = lossFinal
    fidelities[i] = fidelity
    print(f"J2={J2:.1f}, Loss={lossFinal:.4f}, Fidelity={fidelity:.8f}")
