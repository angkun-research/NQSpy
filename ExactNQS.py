import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import NeuralNetworks as nnets

def build_MB_basis(L):
    assert L % 2 == 1, "L must be odd"
    Nup = (L - 1) // 2
    sites = list(range(L))
    basis = []
    for hole in sites:
        remaining_sites = [s for s in sites if s != hole]
        for up_sites in itertools.combinations(remaining_sites, Nup):
            # up_sites is a tuple of sites occupied by up spins
            basis.append((hole, up_sites))
    return basis

def RVB_state(L):
    assert L % 2 == 1, "L must be odd"
    Npair = (L - 1) // 2
    sites = list(range(L))
    rvb_basis = []
    rvb_coeffs = []
    for hole in sites:
        singlet_pairs = []
        a = 0
        while a < L:
            if a == hole:
                a += 1
                continue
            b = a + 1
            if b == hole:
                b += 1
            if b < L:
                singlet_pairs.append((a,b))
                a = b + 1
            else:
                break
        for up_sites in itertools.product(*singlet_pairs):
            sign = 1
            for idx, pair in enumerate(singlet_pairs):
                if up_sites[idx] == pair[0]:
                    sign *= 1
                elif up_sites[idx] == pair[1]:
                    sign *= -1
            rvb_basis.append((hole, up_sites))
            #rvb_coeffs.append(sign / (2 ** (len(singlet_pairs) / 2)))
            rvb_coeffs.append(sign)
    return rvb_basis, rvb_coeffs

def exact_ground_state(L, t1, t2,basis, TBcoeff=True):
    Htb = np.zeros((L,L), dtype=float) # tight-binding Hamiltonian in the hole basis
    for j in range(L-1):
        Htb[j,j+1] = -t1
        Htb[j+1,j] = -t1
    for j in range(0,L-2,2):
        Htb[j,j+2] = -t2
        Htb[j+2,j] = -t2
    e_vals, e_vecs = la.eigh(Htb)
    e_coeff = e_vecs[:,0]  # ground state coefficients in the hole basis

    rvb_basis, rvb_coeffs = RVB_state(L)
    rvb_vec = np.zeros(len(basis), dtype=float)
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    if TBcoeff:
        for state, coeff in zip(rvb_basis, rvb_coeffs):
            rvb_vec[basis_dict[state]] += coeff*e_coeff[state[0]]
    else:
        for state, coeff in zip(rvb_basis, rvb_coeffs):
            rvb_vec[basis_dict[state]] += coeff
    #rvb_vec /= la.norm(rvb_vec)  # Normalize # do not normalize for training
    return rvb_vec

# Convert basis to binary spin configurations
def basis_to_spinconfig(basis, L):
    configs = []
    for state in basis:
        config = np.ones(L)
        config[state[0]] = 0  # 0 for hole
        for idx in state[1]:
            config[idx] = 2  # 2 for up spin, 1 for down spin
        configs.append(config)
    return np.array(configs)

# one-hot encoding for empty, up, down, doubly occupied states
def basis_to_onehot(configs, L):
    onehot_configs = []
    for config in configs:
        onehot = np.zeros(4 * L)
        for i, state in enumerate(config):
            idx = 4 * i + int(state)  # Ensure integer index
            onehot[idx] = 1
        onehot_configs.append(onehot)
    return np.array(onehot_configs)

def total_squared_loss(output, target):
    return torch.sum((output - target) ** 2)

L = 7
#rvb_basis, rvb_coeffs = RVB_state(L)
basis = build_MB_basis(L)
rvb_vec = exact_ground_state(L, t1=1.0, t2=1.0, basis=basis, TBcoeff=False)

spin_configs = basis_to_spinconfig(basis, L)
coeffs = np.array(rvb_vec)

onehot_configs = basis_to_onehot(spin_configs, L)

# Convert to torch tensors
X = torch.tensor(onehot_configs, dtype=torch.float32)
y = torch.tensor(coeffs, dtype=torch.float32)
model = nnets.FCNet(L)

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-10)

# Training loop (example: 500 epochs)
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = model(X)
    print("True coeffs:", coeffs)
    print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - coeffs)**2))

# Initialize network, loss, optimizer
#model = nnets.FCNet(L)
in_channels=4
hidden_dim=32 #32
kernel_size=16
model = nnets.ConvNet(in_channels=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size)
onehot_temp = onehot_configs.reshape(-1, L, 4) 

X = torch.tensor(onehot_temp, dtype=torch.float32)
y = torch.tensor(coeffs, dtype=torch.float32)

criterion = total_squared_loss
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-10)

# Training loop (example: 500 epochs)
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example: predict coefficients for all configurations
with torch.no_grad():
    predicted_coeffs = model(X)
    print("True coeffs:", coeffs)
    print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - coeffs)**2))


L = 5
basis = build_MB_basis(L)
rvb_vec = exact_ground_state(L, t1=1.0, t2=1.0, basis=basis, TBcoeff=False)
spin_configs = basis_to_spinconfig(basis, L)
coeffs_test = np.array(rvb_vec)
onehot_configs_test = basis_to_onehot(spin_configs, L)
onehot_temp = onehot_configs_test.reshape(-1, L, 4)
Xtest = torch.tensor(onehot_temp, dtype=torch.float32)
ytest = torch.tensor(coeffs_test, dtype=torch.float32)

with torch.no_grad():
    predicted_coeffs = model(Xtest)
    print("True coeffs:", coeffs_test)
    print("Predicted coeffs:", predicted_coeffs.numpy())
    print("total loss:", np.sum((predicted_coeffs.numpy() - coeffs_test)**2))



model = nnets.PhysicsLocalLayer(L)

