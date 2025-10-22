import numpy as np
import random
import itertools
import torch
import torch.nn as nn

# Neural network definition
# fully connected
class FCNet(nn.Module):
    def __init__(self, L, hidden_dim=32):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(4*L, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        #self.act = nn.Tanh()

    def forward(self, x):
        # x shape: (batch, L, 4)
        x = x.view(x.shape[0], -1)  # flatten to (batch, 4*L)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)

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

def build_Hamiltonian(L, t1, t2, basis):
    """
    Build the many-body Hamiltonian for the infinite-U Hubbard model.
    Args:
        L (int): Number of sites.
        t1 (float): NN hopping amplitude.
        t2 (float): NNN hopping amplitude (odd sites only).
        basis (list): Many-body basis from build_MB_basis(L).
    Returns:
        H (np.ndarray): Hamiltonian matrix.
    """
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)), dtype=float)

    for idx, (hole, up_sites) in enumerate(basis):
        # NN hopping: hole exchanges with a neighbor
        for delta in [-1, 1]:
            neighbor = hole + delta
            # open boundary conditions
            if neighbor < 0 or neighbor >= L:
                continue
            if neighbor not in up_sites:
                # Move hole to neighbor, spin configuration unchanged
                new_state = (neighbor, up_sites)
                H[idx, basis_dict[new_state]] -= t1
            else:
                # Move hole to neighbor, spin at neighbor moves to hole
                new_up_sites = tuple(sorted([s if s != neighbor else hole for s in up_sites]))
                new_state = (neighbor, new_up_sites)
                H[idx, basis_dict[new_state]] -= t1

        # NNN hopping: only between odd sites (even in python)
        if hole % 2 == 0:
            for delta in [-2, 2]:
                neighbor = hole + delta
                if neighbor < 0 or neighbor >= L:
                    continue
                if neighbor not in up_sites:
                    new_up_sites = up_sites
                    new_state = (neighbor, up_sites)
                    if new_state == (0,(1,2)):
                        print(idx, "Found it!", basis_dict[new_state])
                    H[idx, basis_dict[new_state]] -= t2*(-1) # sign factor for NNN hopping for many-body
                else:
                    new_up_sites = tuple(sorted([s if s != neighbor else hole for s in up_sites]))
                    new_state = (neighbor, new_up_sites)
                    H[idx, basis_dict[new_state]] -= t2*(-1) # sign factor for NNN hopping for many-body

    return H

def state_to_onehot(state, L):
    """
    Convert a basis state (hole, up_sites) to a (L, 4) one-hot array.
    Channels: [hole, up, down, empty]
    """
    hole, up_sites = state
    arr = np.zeros((L, 4), dtype=np.float32)
    for i in range(L):
        if i == hole:
            arr[i, 0] = 1.0  # hole
        elif i in up_sites:
            arr[i, 1] = 1.0  # up
        else:
            arr[i, 2] = 1.0  # down
    # Optional: mark empty (should be zero for this model)
    arr[:, 3] = (arr.sum(axis=1) == 0).astype(np.float32)
    return arr

# Metropolis sampling
def propose_move(state, L):
    """
    Propose a new state by moving the hole to a random NN or NNN site,
    or swapping an up spin (to ensure ergodicity).
    """
    hole, up_sites = state
    up_sites = list(up_sites)
    # Move hole to NN or NNN
    moves = []
    for delta in [-2, -1, 1, 2]:
        neighbor = hole + delta
        if 0 <= neighbor < L and neighbor != hole:
            moves.append(neighbor)
    if moves:
        new_hole = random.choice(moves)
        if new_hole in up_sites:
            # swap hole and up spin
            idx = up_sites.index(new_hole)
            up_sites[idx] = hole
            up_sites = tuple(sorted(up_sites))
        else:
            up_sites = tuple(sorted(up_sites))
        return (new_hole, up_sites)
    else:
        return state  # no move possible

def local_energy(state, psi, basis, basis_dict, H, device, L):
    """
    Compute local energy for a given state.
    psi: neural network, returns log-amplitude
    """
    idx = basis_dict[state]
    # Find all connected states (nonzero H[idx, :])
    connected = np.nonzero(H[idx])[0]
    E_loc = torch.zeros(1, device=device)
    s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
    psi_s = psi(s_onehot)
    for j in connected:
        coeff = H[idx, j]
        state_p = basis[j]
        s_p_onehot = torch.tensor(state_to_onehot(state_p, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_sp = psi(s_p_onehot)
        #ratio = (psi_sp.abs()) / (psi_s.abs() + 1e-12)
        ratio = psi_sp / (psi_s)
        E_loc = E_loc + coeff * ratio
    return E_loc