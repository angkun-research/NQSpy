import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from ExactGS import build_MB_basis_holes,build_Hamiltonian_holes,basis_to_spinconfig_holes,basis_to_onehot
from ExactGS import build_Hamiltonian_holes_adjlist
import NeuralNetworks as nnets
from utils import total_squared_loss
from vmc_utils import adjlist_to_csr
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


class Class3States(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=2, stride=2):
        super(Class3States, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride,)
                                #padding_mode='circular')
        self.pool = nnets.GeometricPool1d()
        self.finallayer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, L, 4) -> (batch, 4, L)
        x = torch.tanh(self.layer1(x).pow(5)) # .pow(3)
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #x = torch.tanh(x.pow(3))
        return x.squeeze(-1)


def obtain_train_data(L, J1=1.0,J2=1.0,pbc=False):
    t1, t2 = 0.0, 0.0
    basis = build_MB_basis_holes(L,nholes=0,Nup=L//2) # only works for even L
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    #print("Dimension of Hilbert space:", len(basis))
    # H_dense = build_Hamiltonian_holes(L, t1, t2, basis, J1=J1, J2=J2)
    # eigvals, eigvecs = eigh(H_dense)
    H_ind, H_val = build_Hamiltonian_holes_adjlist(L, t1, t2, basis, J1=J1, J2=J2, pbc=pbc)
    Hsparse = adjlist_to_csr(H_ind, H_val)
    eigvals, eigvecs = eigsh(Hsparse, k=1, which='SA')
    #print(f"Exact ground state energy for L {L}: {eigvals[0:3]}")

    spin_configs = basis_to_spinconfig_holes(basis, L)
    #coeffs = np.array(eigvecs[:, 0]*2**(L/4), dtype=np.float32)  # Ground state coefficients
    coeffs = np.array(eigvecs[:, 0], dtype=np.float64)
    even_sites = [i for i in range(L) if i % 2 == 0]
    candidate_state = ((),tuple(even_sites))
    print(coeffs[basis_dict[candidate_state]], np.max(np.abs(coeffs)))
    sign = np.sign(coeffs[basis_dict[candidate_state]])
    coeffs = coeffs * sign / np.max(np.abs(coeffs)) # gauge fix
    #coeffs = coeffs / coeffs[basis_dict[candidate_state]] # gauge fix

    onehot_configs = basis_to_onehot(spin_configs, L)
    onehot_configs = onehot_configs.reshape(-1, L, 4)  # (num_states, L, 4)
    #onehot_configs = onehot_configs.transpose(0, 2, 1)  # (num_states, 4, L)

    X = torch.tensor(onehot_configs, dtype=torch.float32)
    y = torch.tensor(coeffs, dtype=torch.float32)
    return X, y

J1, J2 = 1.0,1.5 #1.0 #0.9
print(f"J1={J1}, J2={J2}")
pbc = False
all_X = []
all_y = []
for L in [12]: #[2,4,6,8,10]:
    X, y = obtain_train_data(L, J1=J1, J2=J2, pbc=pbc)
    print(X.shape)
    print(f"L={L}, number of configurations: {len(y)}")
    # filter X, y to keep only |y| > 0
    mask = torch.abs(y) > -1 #1e-6
    print(f"After filtering, number of configurations: {torch.sum(mask).item()}")
    all_X.append(X[mask])
    all_y.append(y[mask]) # take absolute value of y to ignore the

print("max of y", torch.max(torch.cat(all_y)))
print("min of y", torch.min(torch.cat(all_y)))
#print(all_y)

in_channels=4
hidden_dim= 64 #16 #64 
kernel_size= 5 #2 #5  
stride=2
model = Class3States(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size, stride=stride)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the neural network: {n_params}")

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-3) #lr=0.01

# Training loop (example: 500 epochs)
for epoch in range(2000): #4000
    optimizer.zero_grad()
    loss = 0
    for X, y in zip(all_X, all_y):
        outputs = model(X)
        loss += criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = model(X)
    #print("True coeffs:", y.numpy())
    #print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
    print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2)/len(y))
    predicted_coeffs = predicted_coeffs / torch.norm(predicted_coeffs)
    y = y / torch.norm(y)
    fidelity = torch.abs(torch.dot(predicted_coeffs, y)).item()
    print(f"fidelity: {fidelity:.8f}")
    #print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))
    #print("size", len(y))




# L = 14
# Xtest, ytest = obtain_train_data(L, J1=J1, J2=J2)
# mask = torch.abs(ytest) > 1e-6
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("True coeffs:", ytemp.numpy())
#     print("Predicted coeffs:", predicted_coeffs.numpy())
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
#     print("size", len(ytemp))

# mask = torch.abs(ytest) < 0.1 
# Xtemp = Xtest[mask]
# ytemp = ytest[mask] # take absolute value of y to ignore the
# with torch.no_grad():
#     predicted_coeffs = model(Xtemp)
#     print("True coeffs:", ytemp.numpy())
#     print("Predicted coeffs:", predicted_coeffs.numpy())
#     print("Mean Squared loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2)/len(ytemp))
#     print("total loss:", np.sum((predicted_coeffs.numpy() - ytemp.numpy())**2))
#     #print("fidelity:", np.sum((predicted_coeffs.numpy() * ytemp.numpy())))
#     print("size", len(ytemp))

J2 = 1.5 #1.0 #0.9
Ls_test = [2,4,6,8,10]#[12,14,16,18,20]
for L in Ls_test:
    Xtest, ytest = obtain_train_data(L, J1=J1, J2=J2, pbc=pbc)
    predicted_coeffs = model(Xtest)
    predicted_coeffs = predicted_coeffs / torch.norm(predicted_coeffs)
    ytest = ytest / torch.norm(ytest)
    fidelity = torch.abs(torch.dot(predicted_coeffs, ytest)).item()
    print(f"L={L}, Fidelity: {fidelity}")