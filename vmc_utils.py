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
        x = torch.tanh(self.fc1(x)) #torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
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
    moves = []
    # Move hole to NN or NNN
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
    
def GlobalSampler(state, L):
    """
    Propose a new state by moving the hole to a random site,
    or keep hole position but swap a random two spins (to ensure ergodicity).
    """
    hole, up_sites = state
    up_sites = list(up_sites)
    # randomly choose two sites
    while True:
        site1, site2 = sorted(random.sample(range(L), 2))
        if site1 == hole:
            break
        elif site2 == hole:
            break
        elif (site1 in up_sites) != (site2 in up_sites):
            break

    if site1 == hole:
        new_hole = site2
        up_sites_copy = up_sites.copy()
        if site2 in up_sites:
            # swap hole and up spin
            idx = up_sites.index(site2)
            up_sites_copy[idx] = hole
        up_sites_copy = tuple(sorted(up_sites_copy))
        return (new_hole, up_sites_copy)
    elif site2 == hole:
        new_hole = site1
        up_sites_copy = up_sites.copy()
        if site1 in up_sites:
            idx = up_sites.index(site1)
            up_sites_copy[idx] = hole
        up_sites_copy = tuple(sorted(up_sites_copy))
        return (new_hole, up_sites_copy)
    else:
        # swap spins at site1 and site2
        up_sites_copy = up_sites.copy()
        if site1 in up_sites:
            idx = up_sites.index(site1)
            up_sites_copy[idx] = site2
        else:
            idx = up_sites.index(site2)
            up_sites_copy[idx] = site1
        up_sites_copy = tuple(sorted(up_sites_copy))
        return (hole, up_sites_copy)


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




# New: simple coefficient ansatz (real coefficients)
class CoeffAnsatz(nn.Module):
    def __init__(self, nbasis, complex=False, init_scale=0.1, device='cpu'):
        super().__init__()
        self.complex = complex
        self.device = device
        if self.complex:
            self.real = nn.Parameter(init_scale * torch.randn(nbasis, device=device))
            self.imag = nn.Parameter(init_scale * torch.randn(nbasis, device=device))
        else:
            self.coeffs = nn.Parameter(init_scale * torch.randn(nbasis, device=device))

    # return scalar amplitude for an integer index (torch scalar)
    def get_by_index(self, idx):
        if self.complex:
            return torch.complex(self.real[idx], self.imag[idx])
        else:
            return self.coeffs[idx]

    # return full vector of amplitudes (torch tensor)
    def get_all(self):
        if self.complex:
            return torch.complex(self.real, self.imag)
        else:
            return self.coeffs

    # don't implement generic forward to avoid accidental one-hot calls
    def forward(self, x):
        raise RuntimeError("CoeffAnsatz.forward is not implemented. Use get_by_index or get_all().")

# ...existing code...

def local_energy_coeff(state, psi, basis, basis_dict, H, device, L):
    """
    Compute local energy for a given state.
    psi: neural network or CoeffAnsatz
    """
    idx = basis_dict[state]
    # Find all connected states (nonzero H[idx, :])
    connected = np.nonzero(H[idx])[0]
    E_loc = torch.zeros(1, device=device)
    # psi_s using coefficient ansatz if available
    if hasattr(psi, "get_by_index"):
        psi_s = psi.get_by_index(idx)
        # ensure tensor dtype/shape
        if not torch.is_tensor(psi_s):
            psi_s = torch.tensor(psi_s, device=device)
    else:
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_s = psi(s_onehot)

    for j in connected:
        coeff = H[idx, j]
        state_p = basis[j]
        if hasattr(psi, "get_by_index"):
            psi_sp = psi.get_by_index(j)
            if not torch.is_tensor(psi_sp):
                psi_sp = torch.tensor(psi_sp, device=device)
        else:
            s_p_onehot = torch.tensor(state_to_onehot(state_p, L), dtype=torch.float32, device=device).unsqueeze(0)
            psi_sp = psi(s_p_onehot)
        ratio = psi_sp / psi_s #(psi_s + 1e-12)
        E_loc = E_loc + coeff * ratio
    return E_loc



class GeometricPool1d(nn.Module):
    """Wrapper to signed geometric-mean (product) pooling.

    Modes:
      - 'geom': signed geometric mean (exp(mean(log(|x|)))) * sign(prod(x))
                numerically stable via clamp(eps). Zero entries produce zero output.
    Input expected shape: (B, C, L) -> output shape: (B, C)
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):
        # x: (B, C, L)
        # geom: signed geometric mean across last dim
        # compute sign of product
        sign = torch.prod(torch.sign(x), dim=-1)   # (B, C)
        # compute mean log of absolute values (stable)
        log_abs = torch.log(torch.clamp(torch.abs(x), min=self.eps))
        mean_log = torch.mean(log_abs, dim=-1)    # (B, C)
        geom = torch.exp(mean_log)                # (B, C)
        return sign * geom
    
class PhysicalNN(nn.Module):
    def __init__(self, L, in_channels=4, hidden_dim=32, kernel_size=2, stride=2, 
                 holewave=False):
        super(PhysicalNN, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        self.pool = GeometricPool1d()
        #self.fc1 = nn.Linear(hidden_dim*(L//2)+L, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim+L, hidden_dim)
        self.finallayer = nn.Linear(hidden_dim, 1)

        self.holewave=holewave

    def forward(self, x):
        # remove hole site before convolution
        batch_size, L, C = x.shape
        device = x.device
        # assume exactly one hole per sample encoded as channel 0 == 1
        # get hole indices (shape: (B,))
        hole_idx = torch.argmax(x[:, :, 0], dim=1)
        # build mask: True for positions that are NOT the hole
        idx = torch.arange(L, device=device)
        mask = idx.unsqueeze(0) != hole_idx.unsqueeze(1)  # (B, L)
        # select non-hole positions and reshape -> (B, L-1, C)
        x_nohole = x[mask].view(batch_size, L - 1, C)
        # x shape: (batch, L, 4) -> (batch, 4, L)
        x = x_nohole.permute(0, 2, 1)
        x = torch.tanh(self.layer1(x)) # shape: (batch, hidden_dim, L//2)
        #x = x.view(batch_size, -1)  # flatten
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        # add hole position encoding to flattened vector
        # make hole position one-hot vector
        hole_vec = torch.zeros(batch_size, L, device=device)
        if self.holewave:
            hole_vec[torch.arange(batch_size), hole_idx] = 1.0
        x = torch.cat([x, hole_vec], dim=1)
        x = torch.tanh(self.fc1(x))  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #x = torch.tanh(x)
        
        return x.squeeze(-1)





