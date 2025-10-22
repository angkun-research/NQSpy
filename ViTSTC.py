import netket as nk
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import struct

class NoDoubleOccupancyConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    n_sites: int = struct.field(pytree_node=False)

    def __init__(self, n_sites):
        self.n_sites = n_sites

    def __call__(self, x):
        return jax.pure_callback(
            self._call_py,
            jax.ShapeDtypeStruct(x.shape[:-1], bool),
            x,
            vmap_method="expand_dims"
        )

    def _call_py(self, x):
        # x shape: (..., n_sites*2)
        n_sites = self.n_sites
        n_up = x[..., :n_sites] # first half are up spins
        n_down = x[..., n_sites:] # second half are down spins
        # Double occupancy if n_up + n_down > 1 at any site
        occ_sum = n_up + n_down
        return np.all(occ_sum <= 1, axis=-1)

    def __hash__(self):
        return hash(("NoDoubleOccupancyConstraint", self.n_sites))

    def __eq__(self, other):
        return isinstance(other, NoDoubleOccupancyConstraint) and self.n_sites == other.n_sites

# --- Model parameters ---
L = 7
t1 = 1
t2 = 0.5
#U = 1e10  # Large U for strong interaction

# --- Build graph and Hilbert space ---
graph = nk.graph.Chain(L, pbc=False)
N = graph.n_nodes
N_up = L // 2
N_down = L // 2
#hi = nk.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=(L//2, L//2))
nodoubleoccupancy_constraint = NoDoubleOccupancyConstraint(N)
hi = nk.hilbert.SpinOrbitalFermions(
    N, s=1/2, n_fermions_per_spin=(L//2, L//2),
    constraint=nodoubleoccupancy_constraint
    )

#basis_states = np.array([s for s in hi.states()]) # [n_0_up, n_1_up, n_2_up, n_0_down, n_1_down, n_2_down]

# --- Build Hamiltonian ---
H = 0.0
up, down = 1, -1
for i, j in zip(range(L-1), range(1, L)):
    H -= t1 * (cdag(hi, i, sz=up) @ c(hi, j, sz=up) + cdag(hi, j, sz=up) @ c(hi, i, sz=up))
    H -= t1 * (cdag(hi, i, sz=down) @ c(hi, j, sz=down) + cdag(hi, j, sz=down) @ c(hi, i, sz=down))
for i, j in zip(range(0, L-2, 2), range(2, L, 2)):
    H -= t2 * (cdag(hi, i, sz=up) @ c(hi, j, sz=up) + cdag(hi, j, sz=up) @ c(hi, i, sz=up))
    H -= t2 * (cdag(hi, i, sz=down) @ c(hi, j, sz=down) + cdag(hi, j, sz=down) @ c(hi, i, sz=down))
#for i in range(L):
#    H += U * nc(hi, i, 1) @ nc(hi, i, -1)

hamiltonian = H.to_jax_operator()

# --- Simple variational ansatz: RBM ---
model = nk.models.RBM(alpha=2)

# --- Sampler ---
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    graph=graph,
    d_max=2,
    n_chains=256,
    sweep_size=N,
)

# --- VMC State ---
vstate = nk.vqs.MCState(
    sampler,
    model,
    n_samples=2048,
    n_discard_per_chain=16,
    chunk_size=1024,
)
# --- Optimizer ---
optimizer = nk.optimizer.Adam(learning_rate=0.05)

# --- VMC Driver ---
vmc = nk.VMC(
    hamiltonian=H,
    optimizer=optimizer,
    variational_state=vstate,
)

# --- Run optimization ---
log = nk.logging.RuntimeLog()
vmc.run(n_iter=1000, out=log)


# --- Exact diagonalization for fidelity calculation ---

# 1. Get all basis states
basis_states = np.array([s for s in hi.states()])  # shape: (dim, N)

# 2. Get VMC amplitudes
log_psi_vmc = vstate.log_value(basis_states)
psi_vmc = np.exp(log_psi_vmc)
psi_vmc /= np.linalg.norm(psi_vmc)

# 3. Get ED ground state
H_dense = H.to_dense()
eigvals, eigvecs = np.linalg.eigh(H_dense)
gs_ed = eigvecs[:, 0]  # ground state

# 4. Fidelity
fidelity = np.abs(np.vdot(gs_ed, psi_vmc))**2
print("Fidelity between VMC and ED ground state:", fidelity)