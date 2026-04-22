import numpy as np
from vmc_utils import build_MB_basis, build_Hamiltonian
from vmc_utils import build_Hamiltonian_adjlist,adjlist_to_csr
from tqdm import trange 
import numpy.linalg as la
from scipy.sparse.linalg import eigsh


# L = 7
# basis = build_MB_basis(L)
# basis_dict = {state: idx for idx, state in enumerate(basis)}
# lam = 0.0001
# t1 = lam
# t2 = t1
# J1 = 1.0 - lam
# J2 = J1
# H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
# eigvals, eigvecs = la.eigh(H)
# print(f"Exact ground state energy: {eigvals[0:3]}")
# exact_gs = eigvecs[:, 0]
# print("Dimension of Hilbert space:", len(basis))


# H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
# H_csr = adjlist_to_csr(H_ind, H_val)
# eigval_csr, eigvecs_csr = eigsh(H_csr, k=3, which='SA')
# print(f"(CSR) Exact ground state energy: {eigval_csr[0:3]}")
# exact_gs_csr = eigvecs_csr[:, 0]
# print(f"Fidelity between dense and csr eigvecs: {np.abs(np.dot(exact_gs, exact_gs_csr))}")


L = 13
basis = build_MB_basis(L)
basis_dict = {state: idx for idx, state in enumerate(basis)}
t1 = 1.0
t2 = 0.5
J1 = 0.0
J2 = 0.0
H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
H_csr = adjlist_to_csr(H_ind, H_val)
# H = build_Hamiltonian(L, t1, t2, basis, J1=J1, J2=J2)
# # compare the memory of H and H_csr
# import sys
# print(f"Memory of dense H: {sys.getsizeof(H)} bytes")
# print(f"Memory of sparse H_csr: {sys.getsizeof(H_csr.data) + sys.getsizeof(H_csr.indptr) + sys.getsizeof(H_csr.indices)} bytes")
eigval_csr, eigvecs_csr = eigsh(H_csr, k=3, which='SA')
print(f"(CSR) Exact ground state energy: {eigval_csr[0:3]}")
exact_gs0 = eigvecs_csr[:, 0]

# J1 = 10.0
# J2 = 10.0
# H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
# H_csr = adjlist_to_csr(H_ind, H_val)
# eigval_csr, eigvecs_csr = eigsh(H_csr, k=3, which='SA')
# print(f"(CSR) Exact ground state energy: {eigval_csr[0:3]}")
# exact_gs1 = eigvecs_csr[:, 0]
# print(f"Fidelity between dense and csr eigvecs: {np.abs(np.dot( exact_gs0, exact_gs1))}")

J1 = 1.0
J2s = np.arange(0.0, 2.1, 0.1)
fidelitys = np.zeros(len(J2s))
for i, J2 in enumerate(J2s):
    H_ind, H_val = build_Hamiltonian_adjlist(L, t1, t2, basis, J1=J1, J2=J2)
    H_csr = adjlist_to_csr(H_ind, H_val)
    eigval_csr, eigvecs_csr = eigsh(H_csr, k=3, which='SA')
    print(f"(CSR) Exact ground state energy for J2={J2:.1f}: {eigval_csr[0]:.8f}, {eigval_csr[1]:.4f}, {eigval_csr[2]:.4f}")
    fidelitys[i] = np.abs(np.dot(exact_gs0, eigvecs_csr[:, 0]))
    print(f"Fidelity with J2={J2:.1f} state: {fidelitys[i]:.8f}")
    energy = float(exact_gs0.conj() @ H_csr.dot(exact_gs0))
    print(f"Energy with J2={J2:.1f} state: {energy:.8f}")

