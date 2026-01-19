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

def build_MB_basis_holes(L, nholes=1, Nup=None):
    """
    Build many-body basis with `nholes` holes and the rest singly-occupied sites
    carrying spin up/down. Each basis state is represented as
      (holes_tuple, up_sites_tuple)
    where holes_tuple are the hole positions and up_sites_tuple are the positions
    with spin-up electrons. Spin-down positions are the occupied sites not in up_sites.

    Args:
        L (int): number of lattice sites
        nholes (int): number of holes (0 <= nholes < L)
        Nup (int or None): number of up spins. If None, set to floor((L-nholes)/2).

    Returns:
        list of tuples: basis states as (holes_tuple, up_sites_tuple)
    """
    assert isinstance(L, int) and L > 0
    assert isinstance(nholes, int) and 0 <= nholes < L

    sites = list(range(L))
    Nev = L - nholes  # number of electrons
    if Nup is None:
        Nup = Nev // 2  # "around half" up spins by default
    assert 0 <= Nup <= Nev, "Nup must be between 0 and number of electrons"

    basis = []
    for holes in itertools.combinations(sites, nholes):
        remaining_sites = [s for s in sites if s not in holes]
        for up_sites in itertools.combinations(remaining_sites, Nup):
            basis.append((tuple(sorted(holes)), tuple(sorted(up_sites))))
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


def build_Hamiltonian_holes(L, t1, t2, basis,J1=0.0, J2=0.0):
    """
    Build the many-body Hamiltonian for the infinite-U Hubbard model
    for a basis with an arbitrary number of holes.

    Each basis state is (holes_tuple, up_sites_tuple). Holes are empty sites.
    Hopping: an electron from a neighbor site moves into a hole -> the hole
    moves to the neighbor position. If the electron was up, update up_sites.

    Spin interactions: add S_i^z S_j^z diagonal term (J/4 * s_i * s_j) and
    S_i^+ S_j^- + S_i^- S_j^+ off-diagonal flip terms (J/2) when both sites
    are occupied (not holes).
    """
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)), dtype=float)

    for idx, state in enumerate(basis):
        holes, up_sites = state

        # normalize types: allow older code where hole was an int
        if not isinstance(holes, tuple):
            holes = (holes,)
        if not isinstance(up_sites, tuple):
            up_sites = (up_sites,)

        # NN hopping: hole exchanges with a neighbor (delta = ±1)
        for hole in holes:
            for delta in (-1, 1):
                neighbor = hole + delta
                if neighbor < 0 or neighbor >= L:
                    continue
                # If neighbor is also a hole, there is no electron to hop
                if neighbor in holes:
                    continue

                # compute new holes set: replace this hole by neighbor
                new_holes = tuple(sorted([h for h in holes if h != hole] + [neighbor]))

                # update up_sites if the electron at neighbor was up
                if neighbor in up_sites:
                    new_up_sites = tuple(sorted([s if s != neighbor else hole for s in up_sites]))
                else:
                    new_up_sites = up_sites

                new_state = (new_holes, new_up_sites)
                try:
                    H[idx, basis_dict[new_state]] -= t1
                except KeyError:
                    # new_state not in basis (shouldn't happen if basis built consistently)
                    continue

        # NN spin exchange interaction (J1)
        if J1 != 0.0:
            for site in range(L-1):
                if site in holes or (site+1) in holes:
                    continue
                spin_i = 1 if site in up_sites else -1
                spin_j = 1 if site+1 in up_sites else -1
                H[idx, idx] += (J1 / 4.0) * spin_i * spin_j  # S_i^z S_j^z term
                if spin_i != spin_j:
                    # construct flipped up_sites: move up from i to j or j to i
                    if spin_i == 1:
                        flipped_up_sites = tuple(sorted([s if s != site else site+1 for s in up_sites]))
                    else:
                        flipped_up_sites = tuple(sorted([s if s != site+1 else site for s in up_sites]))
                    flipped_state = (holes, flipped_up_sites)
                    try:
                        H[idx, basis_dict[flipped_state]] += (J1 / 2.0)  # S_i^+ S_j^- + S_i^- S_j^+ term
                    except KeyError:
                        pass

        # NNN hopping: only for holes on even python indices (same rule as before)
        for hole in holes:
            if hole % 2 == 0:
                other_holes = [h for h in holes if h != hole]
                for delta in (-2, 2):
                    neighbor = hole + delta
                    if neighbor < 0 or neighbor >= L:
                        continue
                    if neighbor in holes:
                        continue

                    new_holes = tuple(sorted([h for h in holes if h != hole] + [neighbor]))

                    if neighbor in up_sites:
                        new_up_sites = tuple(sorted([s if s != neighbor else hole for s in up_sites]))
                    else:
                        new_up_sites = up_sites
                    middle_site  = (hole + neighbor) // 2
                    if middle_site in other_holes:
                        sign_factor = 1
                    else:
                        sign_factor = -1
                    new_state = (new_holes, new_up_sites)
                    try:
                        H[idx, basis_dict[new_state]] -= t2 * sign_factor # preserve original sign convention
                    except KeyError:
                        continue

        # NNN spin exchange interaction (J2) on even sites only (site, site+2)
        if J2 != 0.0:
            for site in range(0, L-2, 2):
                if site in holes or (site+2) in holes:
                    continue
                spin_i = 1 if site in up_sites else -1
                spin_j = 1 if site+2 in up_sites else -1
                H[idx, idx] += (J2 / 4.0) * spin_i * spin_j  # S_i^z S_j^z term
                if spin_i != spin_j:
                    if spin_i == 1:
                        flipped_up_sites = tuple(sorted([s if s != site else site+2 for s in up_sites]))
                    else:
                        flipped_up_sites = tuple(sorted([s if s != site+2 else site for s in up_sites]))
                    flipped_state = (holes, flipped_up_sites)
                    try:
                        H[idx, basis_dict[flipped_state]] += (J2 / 2.0)  # flip term
                    except KeyError:
                        pass

    return H

def basis_to_spinconfig_holes(basis, L):
    configs = []
    for state in basis:
        config = np.ones(L)
        for hole in state[0]:
            config[hole] = 0  # 0 for hole
        for idx in state[1]:
            config[idx] = 2  # 2 for up spin, 1 for down spin
        configs.append(config)
    return np.array(configs)
