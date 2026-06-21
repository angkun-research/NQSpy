import netket as nk
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
#from scipy.sparse.linalg import eigsh
import numpy as np
from numpy import linalg

L = 7
graph = nk.graph.Chain(L, pbc=False)
N = graph.n_nodes

hi = nk.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions=L-1)

t1 = 1
t2 = 0.5
U = 10**10
H = 0.0

up, down = 1, -1
for i, j in zip(list(range(L-1)), list(range(1, L))):
    H -= t1 * (cdag(hi, i, sz=up) @ c(hi, j, sz=up) + cdag(hi, j, sz=up) @ c(hi, i, sz=up))
    H -= t1 * (cdag(hi, i, sz=down) @ c(hi, j, sz=down) + cdag(hi, j, sz=down) @ c(hi, i, sz=down))

for i, j in zip(list(range(0, L-2, 2)), list(range(2, L, 2))):
     H -= t2 * (cdag(hi, i, sz=up) @ c(hi, j, sz=up) + cdag(hi, j, sz=up) @ c(hi, i, sz=up))
     H -= t2 * (cdag(hi, i, sz=down) @ c(hi, j, sz=down) + cdag(hi, j, sz=down) @ c(hi, i, sz=down))
    
for i in range(L):
    H += U * nc(hi, i, 1) @ nc(hi, i, -1)
#print("Hamiltonian =", H.operator_string())
#print(H.operators)
# print(H.max_conn_size)
#from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd
#H_pnc = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(H)
# print(H_pnc.max_conn_size)

#print("ed energy was:", eigsh(H.to_sparse(), k=1, which="SA")[0][0])
# diagonalize a full matrix
print("ed energy was:", linalg.eigvalsh(H.to_dense())[:5])
# analytical solution
H_sp = np.zeros((L, L))
for i in range(L-1):
    H_sp[i, i+1] = -t1
    H_sp[i+1, i] = -t1
for i in range(0, L-2, 2):
    H_sp[i, i+2] = -t2
    H_sp[i+2, i] = -t2
print(H_sp)
print("single-particle eigenvalues:", np.linalg.eigvalsh(H_sp))