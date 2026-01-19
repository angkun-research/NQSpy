import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from ExactGS import build_MB_basis_holes,build_Hamiltonian_holes
from ExactGS import basis_to_spinconfig_holes,basis_to_onehot
from NeuralNetworks import LdepConvHoles,FCNet, LdepConvStrides
from utils import total_squared_loss

L = 9
t1 = 0.0
t2 = 0.0
nhole = 3
J1 = 1.0
J2 = 1.0

basis = build_MB_basis_holes(L,nholes=nhole)
print(f"Number of basis states with {nhole} holes in L={L}: {len(basis)}")
Hmb = build_Hamiltonian_holes(L, t1, t2, basis, J1=J1, J2=J2)
e_vals, e_vecs = np.linalg.eigh(Hmb)
# correct energy should be -4.41381638; -4.413816383873234
# solve only for the ground state
# Hsparse = csr_matrix(Hmb)
# e_vals, e_vecs = eigsh(Hsparse, k=1, which='SA')
print(f"Ground state energy with 2 holes in L={L}: {e_vals[:10]}")

# compare with J1=J2=0.0 case
# Hmb = build_Hamiltonian_holes(L, t1, t2, basis)
# Hsparse = csr_matrix(Hmb)
# e_val0, e_vec0 = eigsh(Hsparse, k=1, which='SA')
# print(f"Ground state energy with 2 holes in L={L}: {e_val0[0]}")
# print(f"Ground state fidelity: {np.abs(np.dot(e_vecs[:,0], e_vec0[:,0]))}")


# configs = basis_to_spinconfig_holes(basis, L)
# onehot_configs = basis_to_onehot(configs, L)
# onehot_configs = onehot_configs.reshape(-1, L, 4)  # for reshaping into (num_states, L, 4)
# X = torch.tensor(onehot_configs, dtype=torch.float32)
# y = torch.tensor(e_vecs[:,0], dtype=torch.float32)  # ground state

# hidden_dim = 32 #32
# kernel_size = 3 #16
# stride = 2
# strides = (1,2,3)#(2,)
# #model = LdepConvHoles(L, nhole=nhole, hidden_dim=hidden_dim,kernel_size=kernel_size, stride=stride)
# #model = FCNet(L, hidden_dim=hidden_dim)
# model = LdepConvStrides(L, nhole=nhole, hidden_dim=hidden_dim,kernel_size=kernel_size, strides=strides)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable parameters: {total_params}")

# criterion = total_squared_loss
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Training loop (example: 500 epochs)
# for epoch in range(2000):
#     optimizer.zero_grad()
#     outputs = model(X)
#     loss = criterion(outputs, y)
#     loss.backward()
#     optimizer.step()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")


# with torch.no_grad():
#     predicted_coeffs = model(X)
#     #print("True coeffs:", y.numpy())
#     #print("Predicted coeffs:", predicted_coeffs.numpy())
#     print("total loss:", np.sum((predicted_coeffs.numpy() - y.numpy())**2))
#     print("fidelity:", np.sum((predicted_coeffs.numpy() * y.numpy())))
