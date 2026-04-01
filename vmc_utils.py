import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy.linalg as la
import math

from torch.func import vmap, jacrev, functional_call

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

def build_Hamiltonian(L, t1, t2, basis,J1=0.0, J2=0.0):
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
        # NN spin exchange interaction
        if J1 != 0.0:
            for site in range(L-1):
                if site != hole and site+1 != hole:
                    spin_i = 1 if site in up_sites else -1
                    spin_j = 1 if site+1 in up_sites else -1
                    H[idx, idx] += (J1 / 4.0) * spin_i * spin_j # S_i^z S_j^z term
                    if spin_i != spin_j:
                        if spin_i == 1:
                            flipped_up_sites = tuple(sorted([s if s != site else site+1 for s in up_sites]))
                        else:
                            flipped_up_sites = tuple(sorted([s if s != site+1 else site for s in up_sites]))
                        flipped_state = (hole, flipped_up_sites)
                        H[idx, basis_dict[flipped_state]] += (J1 / 2.0) # S_i^+ S_j^- + S_i^- S_j^+ term

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
        # NNN spin exchange interaction
        if J2 != 0.0:
            for site in range(0, L-2, 2): # even sites only
                if site != hole and site+2 != hole:
                    spin_i = 1 if site in up_sites else -1
                    spin_j = 1 if site+2 in up_sites else -1
                    H[idx, idx] += (J2 / 4.0) * spin_i * spin_j # S_i^z S_j^z term
                    if spin_i != spin_j:
                        if spin_i == 1:
                            flipped_up_sites = tuple(sorted([s if s != site else site+2 for s in up_sites]))
                        else:
                            flipped_up_sites = tuple(sorted([s if s != site+2 else site for s in up_sites]))
                        flipped_state = (hole, flipped_up_sites)
                        H[idx, basis_dict[flipped_state]] += (J2 / 2.0) # S_i^+ S_j^- + S_i^- S_j^+ term
               
    return H

def build_Hamiltonian_adjlist(L, t1, t2, basis, J1=0.0, J2=0.0):
    """
    Build Hamiltonian as adjacency lists: for each basis index i return arrays
    of connected indices and matrix elements H[i, j].
    Returns:
      neighbors_idx: list of 1D np.int32 arrays (columns for each row)
      neighbors_val: list of 1D np.float64 arrays (values for each row)
    """
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    nbasis = len(basis)
    neighbors = [[] for _ in range(nbasis)]

    for idx, (hole, up_sites) in enumerate(basis):
        # NN hopping
        for delta in [-1, 1]:
            neighbor = hole + delta
            if neighbor < 0 or neighbor >= L:
                continue
            if neighbor not in up_sites:
                new_state = (neighbor, up_sites)
                j = basis_dict[new_state]
                neighbors[idx].append((j, -t1))
            else:
                new_up_sites = tuple(sorted([s if s != neighbor else hole for s in up_sites]))
                new_state = (neighbor, new_up_sites)
                j = basis_dict[new_state]
                neighbors[idx].append((j, -t1))
        # NN spin exchange interaction
        if J1 != 0.0:
            for site in range(L-1):
                if site != hole and site+1 != hole:
                    spin_i = 1 if site in up_sites else -1
                    spin_j = 1 if site+1 in up_sites else -1
                    diag_coeff = (J1 / 4.0) * spin_i * spin_j
                    neighbors[idx].append((idx, diag_coeff))  # diagonal term
                    if spin_i != spin_j:
                        if spin_i == 1:
                            flipped_up_sites = tuple(sorted([s if s != site else site+1 for s in up_sites]))
                        else:
                            flipped_up_sites = tuple(sorted([s if s != site+1 else site for s in up_sites]))
                        flipped_state = (hole, flipped_up_sites)
                        j = basis_dict[flipped_state]
                        neighbors[idx].append((j, J1 / 2.0))  # off-diagonal term

        # NNN hopping (same rule as original)
        if hole % 2 == 0:
            for delta in [-2, 2]:
                neighbor = hole + delta
                if neighbor < 0 or neighbor >= L:
                    continue
                if neighbor not in up_sites:
                    j = basis_dict[(neighbor, up_sites)]
                    neighbors[idx].append((j, -t2 * (-1)))
                else:
                    new_up_sites = tuple(sorted([s if s != neighbor else hole for s in up_sites]))
                    j = basis_dict[(neighbor, new_up_sites)]
                    neighbors[idx].append((j, -t2 * (-1)))
        # NNN spin exchange interaction
        if J2 != 0.0:
            for site in range(0, L-2, 2): # even sites only
                if site != hole and site+2 != hole:
                    spin_i = 1 if site in up_sites else -1
                    spin_j = 1 if site+2 in up_sites else -1
                    diag_coeff = (J2 / 4.0) * spin_i * spin_j
                    neighbors[idx].append((idx, diag_coeff))  # diagonal term
                    if spin_i != spin_j:
                        if spin_i == 1:
                            flipped_up_sites = tuple(sorted([s if s != site else site+2 for s in up_sites]))
                        else:
                            flipped_up_sites = tuple(sorted([s if s != site+2 else site for s in up_sites]))
                        flipped_state = (hole, flipped_up_sites)
                        j = basis_dict[flipped_state]
                        neighbors[idx].append((j, J2 / 2.0))  # off-diagonal term

    neighbors_idx = [np.array([p[0] for p in row], dtype=np.int32) for row in neighbors]
    neighbors_val = [np.array([p[1] for p in row], dtype=np.float64) for row in neighbors]
    return neighbors_idx, neighbors_val

def adjlist_to_csr(neighbors_idx, neighbors_val):
    """
    Convert adjacency lists to a scipy.sparse.csr_matrix.
    neighbors_idx / neighbors_val: lists of 1D arrays per row.
    Returns: scipy.sparse.csr_matrix
    """
    rows = []
    cols = []
    data = []
    for i, (cols_i, vals_i) in enumerate(zip(neighbors_idx, neighbors_val)):
        if cols_i.size == 0:
            continue
        rows.extend([i] * int(cols_i.size))
        cols.extend(cols_i.tolist())
        data.extend(vals_i.tolist())

    if len(data) == 0:
        nbasis = len(neighbors_idx)
        return sp.csr_matrix((nbasis, nbasis), dtype=np.float64)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float64)
    nbasis = len(neighbors_idx)
    return sp.csr_matrix((data, (rows, cols)), shape=(nbasis, nbasis))


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
        #x = torch.tanh(self.layer1(x).pow(5))
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
        #x = torch.tanh(x) #does not matter??

        normscale = 1 #(2 ** ((L-1) // 4))*np.sqrt(L)  # scaling factor for normalization
        x = x / normscale
        return x.squeeze(-1)



def local_energy_from_adjlist(state, psi, basis, basis_dict, neighbors_idx, neighbors_val, device, L):
    """
    Compute local energy using adjacency lists neighbors_idx / neighbors_val.
    neighbors_idx[i], neighbors_val[i] are 1D numpy arrays for row i.
    """
    idx = basis_dict[state]
    nbrs = neighbors_idx[idx]
    vals = neighbors_val[idx]

    # get psi_s
    if hasattr(psi, "get_by_index"):
        psi_s = psi.get_by_index(idx)
        if not torch.is_tensor(psi_s):
            psi_s = torch.tensor(psi_s, device=device)
        else:
            psi_s = psi_s.to(device)
    else:
        s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
        psi_s = psi(s_onehot)

    E_loc = torch.zeros(1, device=device, dtype=psi_s.dtype)

    # iterate neighbors
    for j, val in zip(nbrs.tolist(), vals.tolist()):
        coeff = torch.tensor(val, device=device, dtype=psi_s.dtype)
        # get psi_sp
        if hasattr(psi, "get_by_index"):
            psi_sp = psi.get_by_index(int(j))
            if not torch.is_tensor(psi_sp):
                psi_sp = torch.tensor(psi_sp, device=device, dtype=psi_s.dtype)
            else:
                psi_sp = psi_sp.to(device, dtype=psi_s.dtype)
        else:
            state_p = basis[int(j)]
            s_p_onehot = torch.tensor(state_to_onehot(state_p, L), dtype=torch.float32, device=device).unsqueeze(0)
            psi_sp = psi(s_p_onehot)

        ratio = psi_sp / psi_s
        E_loc = E_loc + coeff * ratio

    return E_loc


def TightBinding_coeff(L,t1,t2):
    Htb = np.zeros((L,L), dtype=float) # tight-binding Hamiltonian in the hole basis
    for j in range(L-1):
        Htb[j,j+1] = -t1
        Htb[j+1,j] = -t1
    for j in range(0,L-2,2):
        Htb[j,j+2] = -t2
        Htb[j+2,j] = -t2
    e_vals, e_vecs = la.eigh(Htb)
    e_coeff = e_vecs[:,0]  # ground state coefficients in the hole basis
    return e_coeff

def find_state_coeff(L, state, tb_coeff):
    hole, up_sites = state
    sign = 1
    first_spin = 0
    while first_spin < L:
        if first_spin == hole:
            first_spin += 1
            continue
        second_spin = first_spin + 1
        if second_spin == hole:
            second_spin += 1
        if first_spin in up_sites:
            if second_spin in up_sites:
                sign = 0  # zero coefficient for invalid state
                break
            else:
                sign *= 1
        else:
            if second_spin in up_sites:
                sign *= -1
            else:
                sign = 0  # zero coefficient for invalid state
                break
        first_spin = second_spin + 1

    coeff = sign*tb_coeff[hole]
    return coeff



def local_energy_on_the_fly(state, psi, L, t1, t2, J1=0.0, J2=0.0, device='cpu'):
    """
    Compute local energy for `state` by enumerating connected states on-the-fly
    (no global Hamiltonian / basis required). Returns a torch scalar on `device`.
    """
    hole, up_sites = state
    up_sites = tuple(up_sites)
    # psi_s
    s_onehot = torch.tensor(state_to_onehot(state, L), dtype=torch.float32, device=device).unsqueeze(0)
    psi_s = psi(s_onehot)
    dtype = psi_s.dtype
    E_loc = torch.zeros(1, device=device, dtype=dtype)
    if psi_s.abs() < 1e-12:
        psi_s = torch.tensor(1e-12, device=device, dtype=dtype)

    # NN hopping: deltas -1, +1
    for delta in (-1, 1):
        nbr = hole + delta
        if 0 <= nbr < L:
            if nbr not in up_sites:
                new_state = (nbr, up_sites)
            else:
                new_up = tuple(sorted([s if s != nbr else hole for s in up_sites]))
                new_state = (nbr, new_up)
            s_p_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
            psi_sp = psi(s_p_onehot)
            coeff = -t1
            E_loc = E_loc + coeff * (psi_sp / psi_s)

    # NNN hopping (same rule as build_Hamiltonian / adjlist): only when hole is even (hole % 2 == 0)
    if hole % 2 == 0:
        for delta in (-2, 2):
            nbr = hole + delta
            if 0 <= nbr < L:
                if nbr not in up_sites:
                    new_state = (nbr, up_sites)
                else:
                    new_up = tuple(sorted([s if s != nbr else hole for s in up_sites]))
                    new_state = (nbr, new_up)
                s_p_onehot = torch.tensor(state_to_onehot(new_state, L), dtype=torch.float32, device=device).unsqueeze(0)
                psi_sp = psi(s_p_onehot)
                coeff = -t2 * (-1)  # consistent with build_Hamiltonian / adjlist
                E_loc = E_loc + coeff * (psi_sp / psi_s)

    # NN spin exchange (diagonal + possible off-diagonal flips)
    if J1 != 0.0:
        for site in range(L - 1):
            if site == hole or (site + 1) == hole:
                continue
            spin_i = 1 if site in up_sites else -1
            spin_j = 1 if (site + 1) in up_sites else -1
            diag_coeff = (J1 / 4.0) * spin_i * spin_j
            E_loc = E_loc + diag_coeff  # diagonal term (ratio = 1)
            if spin_i != spin_j:
                # construct flipped configuration
                if spin_i == 1:
                    flipped_up = tuple(sorted([s if s != site else site + 1 for s in up_sites]))
                else:
                    flipped_up = tuple(sorted([s if s != (site + 1) else site for s in up_sites]))
                flipped_state = (hole, flipped_up)
                s_p_onehot = torch.tensor(state_to_onehot(flipped_state, L), dtype=torch.float32, device=device).unsqueeze(0)
                psi_sp = psi(s_p_onehot)
                E_loc = E_loc + (J1 / 2.0) * (psi_sp / psi_s)

    # NNN spin exchange (even sites only)
    if J2 != 0.0:
        for site in range(0, L - 2, 2):
            if site == hole or (site + 2) == hole:
                continue
            spin_i = 1 if site in up_sites else -1
            spin_j = 1 if (site + 2) in up_sites else -1
            diag_coeff = (J2 / 4.0) * spin_i * spin_j
            E_loc = E_loc + diag_coeff
            if spin_i != spin_j:
                if spin_i == 1:
                    flipped_up = tuple(sorted([s if s != site else site + 2 for s in up_sites]))
                else:
                    flipped_up = tuple(sorted([s if s != (site + 2) else site for s in up_sites]))
                flipped_state = (hole, flipped_up)
                s_p_onehot = torch.tensor(state_to_onehot(flipped_state, L), dtype=torch.float32, device=device).unsqueeze(0)
                psi_sp = psi(s_p_onehot)
                E_loc = E_loc + (J2 / 2.0) * (psi_sp / psi_s)

    return E_loc


def generate_initial_state(L):
    hole = random.randint(0, L-1) #L // 2 # randomly choose a hole position
    L_list = list(range(L))
    L_remove_hole_list = [x for x in L_list if x != hole] # remove the hole site from the list
    upsites = L_remove_hole_list[::2] # choose half of the remaining sites
    initial_state = (hole, tuple(upsites))
    return initial_state


def BalancedSampler(state, L):
    hole, up_sites = state
    up_sites = list(up_sites)
    if random.random() < 0.5: # move hole
        moves = []
        # Move hole to NN
        for delta in [-1, 1]:
            neighbor = hole + delta
            if 0 <= neighbor < L and neighbor != hole:
                moves.append(neighbor)
        new_hole = random.choice(moves)
        if new_hole in up_sites:
            # swap hole and up spin
            idx = up_sites.index(new_hole)
            up_sites[idx] = hole
            up_sites = tuple(sorted(up_sites))
        else:
            up_sites = tuple(sorted(up_sites))
        return (new_hole, up_sites)
    else: # swap two nearest neighbor spins
        spins = np.zeros(L, dtype=int)
        for site in range(L):
            if site == hole:
                spins[site] = 0  # hole
            elif site in up_sites:
                spins[site] = 1   # up spin
            else:
                spins[site] = -1   # down spin
        pairs = []
        for k in range(L - 1):
            if spins[k] * spins[k + 1] == -1:
                pairs.append((k, k + 1))
            elif spins[k+1] == 0 and k+2 < L and spins[k]*spins[k+2]==-1:
                pairs.append((k, k + 2))
        # randomly select a pair with opposite spins to swap
        site1, site2 = random.choice(pairs)
        if site1 in up_sites and site2 not in up_sites:
            idx_up = up_sites.index(site1)
            up_sites[idx_up] = site2
            up_sites = tuple(sorted(up_sites))
        elif site2 in up_sites and site1 not in up_sites:
            idx_up = up_sites.index(site2)
            up_sites[idx_up] = site1
            up_sites = tuple(sorted(up_sites))
        return (hole, up_sites)
        



class NNConvStrides(nn.Module):
    def __init__(self, L, nhole=2, in_channels=4, hidden_dim=32, strides=(1,2,3,4),kernel_size=3):
        super(NNConvStrides, self).__init__()
        self.L = L
        self.nhole = nhole
        self.kernel_size = kernel_size
        self.pad = (kernel_size-1)//2
        # create parallel conv branches with different strides
        self.strides = tuple(strides)
        self.convs = nn.ModuleList([
            #nn.Conv1d(in_channels, hidden_dim, s, padding=(s-1)//2, stride=s)
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=s)
            for s in self.strides
        ])
        # one pooling module per branch (keeps feature dim = hidden_dim)
        self.pools = nn.ModuleList([GeometricPool1d() for _ in self.strides])
        #self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(5) for _ in self.strides])

        # fc input: concatenated pooled features from each branch + L-sized hole one-hot
        self.fc1 = nn.Linear(hidden_dim * len(self.strides) + L, hidden_dim)
        #self.fc1 = nn.Linear(hidden_dim * len(self.strides)*5 + L, hidden_dim)
        self.finallayer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # obtain holes per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1)   # (B, L) boolean mask
        idxs = torch.nonzero(hole_mask, as_tuple=False)    # (B*nhole, 2)
        hole_pos = idxs[:, 1].view(batch_size, self.nhole).to(device)  # (B, nhole)

        # build mask for non-hole positions
        hole_positions_mask = torch.zeros(batch_size, L, dtype=torch.bool, device=device)
        rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.nhole)  # (B, nhole)
        hole_positions_mask[rows, hole_pos] = True
        mask = ~hole_positions_mask  # True for non-hole sites

        # select non-hole positions -> (B, L-nhole, C)
        x_nohole = x[mask].view(batch_size, L - self.nhole, C)
        x_perm = x_nohole.permute(0, 2, 1)  # (B, C, L-nhole)

        # apply each conv branch and pool (each yields (B, hidden_dim))
        feats = []
        for conv, pool in zip(self.convs, self.pools):
            y = torch.tanh(conv(x_perm).pow(1))
            #y = torch.relu(conv(x_perm))
            y = pool(y).squeeze(-1)  # (B, hidden_dim,5)
            #y = y.flatten(1)
            feats.append(y)
        feats_cat = torch.cat(feats, dim=1)  # (B, hidden_dim * n_branches)
        #print(feats_cat.shape)
        # hole one-hot vector of length L
        hole_vec = torch.zeros(batch_size, L, device=device, dtype=dtype)
        hole_vec.scatter_(1, hole_pos, 1.0)

        # combine branch features + hole encoding
        x_comb = torch.cat([feats_cat, hole_vec], dim=1)  # (B, hidden_dim*n_branches + L)
        x_out = torch.tanh(self.fc1(x_comb))
        #x_out = torch.relu(self.fc1(x_comb))
        x_out = self.finallayer(x_out)

        return x_out.squeeze(-1)
    

class VBSNN(nn.Module):
    """
    Fast jacobian ready stable version of VBS neural network. 
    """
    def __init__(self, L, in_channels=4, Conv_dim=32, kernel_size=3, stride=2):
        super(VBSNN, self).__init__()
        self.L = L
        self.nhole = 1
        self.pad = (kernel_size-1)//2
        # create parallel conv branches with different strides
        self.conv = nn.Conv1d(in_channels, Conv_dim, kernel_size=kernel_size, padding=self.pad, stride=stride)
        
        # one pooling module per branch (keeps feature dim = hidden_dim)
        self.pool = GeometricPool1d()

        self.fc_2 = nn.Linear(Conv_dim+L, Conv_dim)
        self.finallayer = nn.Linear(Conv_dim, 1)

    def forward(self, x):
        batch_size, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # obtain holes per sample encoded as channel 0 == 1
        hole_mask = (x[:, :, 0] == 1).float()  # (B, L)
        hole_pos = torch.argmax(hole_mask, dim=1)  # (B,) — fixed shape, vmap-safe 

        # Build indices [0,1,...,hole-1, hole+1,...,L-1] per sample
        idx = torch.arange(L - self.nhole, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L-nhole)
        idx = idx + (idx >= hole_pos.unsqueeze(1)).long()  # shift indices past the hole
        x_nohole = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # (B, L-nhole, C)
        x_perm = x_nohole.permute(0, 2, 1)  # (B, C, L-nhole)

        # apply each conv branch and pool (each yields (B, hidden_dim))
        feats = []
        y1 = torch.tanh(self.conv(x_perm).pow(5))
        y1 = self.pool(y1).squeeze(-1)  # (B, hidden_dim,5)
        feats.append(y1)
        hole_vec = hole_mask  # already (B, L) with 1.0 at hole position
        feats.append(hole_vec)

        feats_cat = torch.cat(feats, dim=1)  # (B, hidden_dim * n_branches)

        x_out = torch.tanh(self.fc_2(feats_cat).pow(3)) # best pow(3)
        x_out = self.finallayer(x_out)

        return x_out.squeeze(-1)


def compute_log_psi_jacobian_vmap(psi, samples_onehot, device):
    """Vectorized Jacobian via torch.func (PyTorch >= 2.0)."""
    x = torch.from_numpy(samples_onehot).float().to(device)

    params_dict = {k: v for k, v in psi.named_parameters() if v.requires_grad}
    param_names = list(params_dict.keys())

    def log_psi_single(params_dict, x_single):
        out = functional_call(psi, params_dict, (x_single.unsqueeze(0),))
        return torch.log(torch.abs(out.squeeze()) + 1e-12)

    # jacrev over params_dict for a single sample, then vmap over batch
    jac_fn = vmap(jacrev(log_psi_single), in_dims=(None, 0))
    jac_dict = jac_fn(params_dict, x)  # dict of {name: (Ns, *param_shape)}

    # Flatten and concatenate
    rows = []
    for name in param_names:
        rows.append(jac_dict[name].reshape(x.shape[0], -1))
    return torch.cat(rows, dim=1)  # (Ns, Np)

def sr_gradient(J, E_loc, tau=1e-3):
    """Stochastic reconfiguration: solve (J^T J + tau*I) delta_theta = J^T E_loc.

    Args:
        J: Jacobian of log|psi|, shape (Ns, Np).
        E_loc: local energies, shape (Ns,).
        tau: diagonal regularization shift.

    Returns:
        delta_theta: natural gradient update, shape (Np,).
    """
    J64 = J.detach().double()
    E64 = E_loc.detach().double()
    Ns = J64.shape[0]
    S = J64.T @ J64 # /Ns
    S.diagonal().add_(tau)
    f = J64.T @ E64 # /Ns
    delta_theta = torch.linalg.solve(S, f.unsqueeze(-1)).squeeze(-1)
    #delta_theta = torch.linalg.lstsq(S, f.unsqueeze(-1)).solution.squeeze(-1)
    return delta_theta.float()


def sr_update(psi, sampled_states, E_tensor, L, lr, tau, device):
    """Perform one stochastic reconfiguration parameter update in-place.
    
    Args:
        psi: neural network wavefunction model.
        sampled_states: list of sampled states from MCMC.
        E_tensor: tensor of local energies, shape (Ns,).
        L: system size.
        lr: learning rate for SR update.
        tau: diagonal regularization shift.
        device: torch device.
    """
    s_np = np.stack([state_to_onehot(s, L) for s in sampled_states])
    # Compute Jacobian of log|psi| over sampled states
    J = compute_log_psi_jacobian_vmap(psi, s_np, device)       # (Ns, Np)
    E_loc = E_tensor.detach() - E_tensor.mean().detach()        # center local energies
    #J_centered = J - J.mean(dim=0, keepdim=True)
    delta_theta = sr_gradient(J, E_loc, tau=tau)
    with torch.no_grad():
        offset = 0
        for p in psi.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.data -= lr * delta_theta[offset:offset + numel].reshape(p.shape)
                offset += numel


def Obtain_Sampling_batch(psi, L, initial_states, n_steps, pretrain=False, burnin=False, tb_coeff=None, print_rate=False,
                          Sampler=GlobalSampler, t1=1.0,t2=0.5, J1=0.0, J2=0.0, device='cpu'):
    """Run multiple independent walkers in parallel (batched NN evaluation).

    initial_state: single state (will be copied to initialize all walkers)
    n_steps: number of MCMC steps per walker
    n_walkers: number of parallel walkers
    Returns flattened lists of psis and elocs and the list of final walker states.
    """
    # initialize walkers
    n_walkers = len(initial_states)
    states = initial_states.copy()
    Psis = []
    Elocs = []
    Sampled_states = []
    accept_count = 0

    for _ in range(n_steps):
        # propose moves for all walkers (kept simple: use Sampler per-walker)
        proposed = [Sampler(s, L) for s in states]

        # build batched one-hot inputs efficiently via numpy.stack -> torch.from_numpy
        s_np = np.stack([state_to_onehot(s, L) for s in states])
        new_np = np.stack([state_to_onehot(s, L) for s in proposed])
        s_batch = torch.from_numpy(s_np).float().to(device)
        new_batch = torch.from_numpy(new_np).float().to(device)

        # evaluate psi for all current and proposed states in one forward pass
        psi_vals = psi(s_batch).squeeze()
        psi_new_vals = psi(new_batch).squeeze()

        # compute acceptance probabilities (vectorized)
        denom = psi_vals.abs()**2
        ratio = (psi_new_vals.abs()**2) / (denom + 1e-12)
        accept_prob = torch.clamp(ratio, max=1.0)

        rand = torch.rand(n_walkers, device=device)
        accept = (rand < accept_prob).cpu().numpy()

        # update walkers and collect psi values
        for i in range(n_walkers):
            if accept[i]:
                states[i] = proposed[i]
                Psis.append(psi_new_vals[i])
                accept_count += 1
            else:
                Psis.append(psi_vals[i])
            Sampled_states.append(tuple(states[i]))

        if burnin:
            continue

        # compute local energies per walker (still per-walker calls)
        for s in states:
            if not pretrain:
                E_loc = local_energy_on_the_fly(s, psi, L, t1, t2, J1=J1, J2=J2, device=device)
                Elocs.append(E_loc)
            else:
                Elocs.append(torch.tensor(find_state_coeff(L, s, tb_coeff), device=device))

    if print_rate:
        print(f"Acceptance rate (batched): {accept_count / (n_steps * n_walkers):.4f}")

    # flatten Psis/Elocs are lists of tensors -> stack
    if len(Psis) > 0:
        Psis_t = torch.stack(Psis).squeeze()
    else:
        Psis_t = torch.tensor([], device=device)
    if len(Elocs) > 0:
        Elocs_t = torch.stack(Elocs).squeeze()
    else:
        Elocs_t = torch.tensor([], device=device)

    # return psis, elocs, and final states (return first state for compatibility)
    return Psis_t, Elocs_t, states, Sampled_states