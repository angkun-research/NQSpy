import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools


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
            rvb_coeffs.append(sign / (2 ** (len(singlet_pairs) / 2)))
    return rvb_basis, rvb_coeffs

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

def exact_ground_state(L, t1, t2,basis):
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
    for state, coeff in zip(rvb_basis, rvb_coeffs):
        rvb_vec[basis_dict[state]] += coeff*e_coeff[state[0]]
    rvb_vec /= la.norm(rvb_vec)  # Normalize
    return rvb_vec


L = 7
basis = build_MB_basis(L)
t1 = 1.0
t2 = 0.5
H = build_Hamiltonian(L, t1, t2, basis)
# check H is Hermitian
assert np.allclose(H, H.conj().T), "Hamiltonian is not Hermitian"
rvb_gs = exact_ground_state(L, t1, t2, basis)

eigvals, eigvecs = la.eigh(H)
ground_state = eigvecs[:, 0]  # Ground state vector
print(f"Exact ground state energy: {eigvals[0]:.6f}")

# Step 3: Compute overlap
overlap = np.abs(np.dot(rvb_gs, ground_state))
print(f"Overlap between RVB superposition and ground state: {overlap:.6f}")

# compute all overlaps
all_overlaps = np.abs(np.dot(rvb_gs, eigvecs))
for idx, val in enumerate(all_overlaps):
    if val > 0.5:
        print(f"Overlap with eigenstate {idx}: {val:.6f} (E={eigvals[idx]:.3f})")

# find the position of the largest overlap
max_idx = np.argmax(all_overlaps)
