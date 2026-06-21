import netket as nk
import jax.numpy as jnp
from netket.utils import struct
from scipy.sparse.linalg import eigsh

# --------------------------
# 1) Parameters
# --------------------------
L = 10
N_up = 3
N_dn = 3
t = 1.0
J = 0.4
pbc = False

graph = nk.graph.Chain(L, pbc=pbc)

# --------------------------
# 2) No-double-occupancy constraint
# --------------------------
class NoDoubleOcc(nk.hilbert.constraint.DiscreteHilbertConstraint):
    n_orb: int = struct.field(pytree_node=False)

    def __init__(self, n_orb):
        self.n_orb = n_orb

    def __call__(self, x):
        # NetKet spinful convention: two spin blocks of length L
        n_up = x[..., : self.n_orb]
        n_dn = x[..., self.n_orb : 2 * self.n_orb]
        return jnp.all((n_up + n_dn) <= 1, axis=-1)

    def __hash__(self):
        return hash(("NoDoubleOcc", self.n_orb))

    def __eq__(self, other):
        return isinstance(other, NoDoubleOcc) and (self.n_orb == other.n_orb)

hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=L,
    s=0.5,
    n_fermions_per_spin=(N_up, N_dn),
    constraint=NoDoubleOcc(L),
)

print("Hilbert size:", hi.n_states)

# --------------------------
# 3) Build t-J Hamiltonian
#    H = -t Σ_<ij>,σ (c†_{iσ} c_{jσ} + h.c.)
#        + J Σ_<ij> (S_i·S_j - 1/4 n_i n_j)
# --------------------------
def cdag(i, sz):
    return nk.operator.fermion.create(hi, i, sz=sz)

def c(i, sz):
    return nk.operator.fermion.destroy(hi, i, sz=sz)

def n(i, sz):
    return nk.operator.fermion.number(hi, i, sz=sz)

H = 0.0
for i, j in graph.edges():
    # projected hopping (projection enforced by constrained Hilbert)
    for sz in (+1, -1):  # spin-1/2: sz = ±1 in NetKet API
        H = H - t * (cdag(i, sz) * c(j, sz) + cdag(j, sz) * c(i, sz))

    # spin operators
    Splus_i  = cdag(i, +1) * c(i, -1)
    Sminus_i = cdag(i, -1) * c(i, +1)
    Sz_i     = 0.5 * (n(i, +1) - n(i, -1))
    n_i      = n(i, +1) + n(i, -1)

    Splus_j  = cdag(j, +1) * c(j, -1)
    Sminus_j = cdag(j, -1) * c(j, +1)
    Sz_j     = 0.5 * (n(j, +1) - n(j, -1))
    n_j      = n(j, +1) + n(j, -1)

    H_ex = 0.5 * (Splus_i * Sminus_j + Sminus_i * Splus_j) + Sz_i * Sz_j
    H = H + J * (H_ex - 0.25 * n_i * n_j)

# --------------------------
# 4) VMC setup
# --------------------------
model = nk.models.RBM(alpha=2, param_dtype=float)

# Prefer fermion-hop sampler if available in your NetKet version
sampler = nk.sampler.MetropolisHamiltonian(hilbert=hi, hamiltonian=H, n_chains=32)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=4096,
    n_discard_per_chain=64,
    seed=1234,
)

opt = nk.optimizer.Sgd(learning_rate=0.02)
sr = nk.optimizer.SR(diag_shift=1e-3)
vmc = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

log = nk.logging.RuntimeLog()
vmc.run(n_iter=400, out=log)
print("Final VMC energy:", vstate.expect(H))

# --------------------------
# 5) Optional ED check (small sizes)
# --------------------------
if hi.n_states < 50000:
    e0 = eigsh(H.to_sparse(), k=1, which="SA", return_eigenvectors=False)[0]
    print("ED ground energy:", e0)