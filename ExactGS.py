import numpy as np
import numpy.linalg as la
import itertools

import torch

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


def obtain_train_data(L, t1=1.0, t2=1.0, TBcoeff=False, Reshape=False,Normalize=False):
    basis = build_MB_basis(L)
    rvb_vec = exact_ground_state(L, t1=t1, t2=t2, basis=basis, TBcoeff=TBcoeff)

    if Normalize:
        rvb_vec = rvb_vec / la.norm(rvb_vec)

    spin_configs = basis_to_spinconfig(basis, L)
    coeffs = np.array(rvb_vec)

    onehot_configs = basis_to_onehot(spin_configs, L)
    if Reshape:
        onehot_configs = onehot_configs.reshape(-1, L, 4) 

    X = torch.tensor(onehot_configs, dtype=torch.float32)
    y = torch.tensor(coeffs, dtype=torch.float32)
    return X, y