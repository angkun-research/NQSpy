import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

def RVB_state(L):
    assert L % 2 == 0, "L must be even"
    Npair = L // 2
    sites = list(range(L))
    rvb_basis = []
    rvb_coeffs = []
    singlet_pairs = []
    a = 0
    while a < L:
        b = a + 1
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
        rvb_basis.append(up_sites)
        rvb_coeffs.append(sign / (2 ** (len(singlet_pairs) / 2)))
    return rvb_basis, rvb_coeffs

# Neural network definition
class RVBNet(nn.Module):
    def __init__(self, L, hidden_dim=32):
        super(RVBNet, self).__init__()
        self.fc1 = nn.Linear(4*L, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)

L = 10
rvb_basis, rvb_coeffs = RVB_state(L)

# Convert basis to binary spin configurations
def basis_to_spinconfig(basis, L):
    configs = []
    for up_sites in basis:
        config = np.zeros(L)
        for idx in up_sites:
            config[idx] = 1  # 1 for up spin
        configs.append(config)
    return np.array(configs)

spin_configs = basis_to_spinconfig(rvb_basis, L)
coeffs = np.array(rvb_coeffs)

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

onehot_configs = basis_to_onehot(spin_configs, L)

# Convert to torch tensors
X = torch.tensor(onehot_configs, dtype=torch.float32)
y = torch.tensor(coeffs, dtype=torch.float32)

# Initialize network, loss, optimizer
model = RVBNet(L)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (example: 500 epochs)
for epoch in range(500):
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