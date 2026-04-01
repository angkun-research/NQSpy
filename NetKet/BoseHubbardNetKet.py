import netket as nk
from scipy.sparse.linalg import eigsh
from netket.experimental.driver import VMC_SR

# System
L = 10
Np = L // 2               # unit-filling for bosons
n_max = Np               # maximum number of bosons per site
t = 1.0
U_user = 2.0            # your convention: U * n(n-1)
U_nk = 2.0 * U_user     # convert to NetKet convention \frac{U}{2}\sum_i n_i(n_i-1)

g = nk.graph.Chain(L, pbc=False)
#g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
hi = nk.hilbert.Fock(n_max=n_max, N=L, n_particles=Np)
#nk.hilbert.Boson(n_max=n_max, N=Np, n_sites=g.n_nodes) # dose not exist
#print("Hilbert space size:", hi.n_states)

# H = -t sum_<ij> (b_i^† b_j + h.c.) + U_user sum_i n_i(n_i-1)
H = nk.operator.BoseHubbard(  # BoseHubbard = BoseHubbardJax
    hilbert=hi,
    graph=g,
    J=t,
    U=U_nk,
    V=0.0,
    mu=0.0,
)
# BoseHubbardJax: JAX-native operator, best for jit, accelerators, sharding, and modern NetKet workflows.
# BoseHubbardNumba: CPU/Numba backend, useful in some CPU-only or legacy contexts.

# Variational ansatz for multi-valued local states (0..n_max)
model = nk.models.RBMMultiVal(alpha=2, n_classes=n_max + 1, param_dtype=float) # hidden units are int(alpha * N_visible).
# RBM: takes your input vector directly and applies one dense layer + log_cosh.
# RBMMultiVal: first does a one-hot encoding of each site value into n_classes channels, flattens, then applies a standard RBM.
#model = nk.models.Jastrow()

# Particle-number-conserving sampler
sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=1, n_chains=32)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=2048,     # sample per chain 2048/32 = 64
    n_discard_per_chain=64, # number of samples to discard for burn-in
    seed=1234,
)
# print the number of variational parameters
print("Number of variational parameters:", vstate.n_parameters)


opt = nk.optimizer.Sgd(learning_rate=0.02)
#opt = nk.optimizer.Sgd(learning_rate=0.001) # for Jastrow
sr = nk.optimizer.SR(diag_shift=1e-3)
vmc = nk.driver.VMC(H, opt, variational_state=vstate , preconditioner=sr)

# from netket.experimental.driver import VMC_SR
# vmc = VMC_SR(
#     hamiltonian=H,
#     optimizer=opt,
#     diag_shift=1e-4,
#     variational_state=vstate,
# )

log = nk.logging.RuntimeLog()
vmc.run(n_iter=400, out=log)

print("Final VMC energy:", vstate.expect(H))
# -7.45807 ± 0.00096

# Optional exact check for small L
if hi.n_states < 50000:
    e0 = eigsh(H.to_sparse(), k=2, which="SA", return_eigenvectors=False)#[0]
    # hilbert space size
    print("Hilbert space size:", hi.n_states)
    print("ED ground energy:", e0)
    # -3.5347167150067964

# OBC
# L = 8, Np = 8: D=6435, ED [-4.93109728 -6.26371598], VMC: -6.2617 ± 0.0078 
# L = 10, Np = 5: D=2002, ED [-6.95802959 -7.45873858], VMC: -7.45763 ± 0.00070
# L = 10, Np = 5, n_max=4: D=1992, ED [-6.95802959 -7.45873858], VMC: -7.45807 ± 0.00096; Jastrow -6.78 ± 0.19
# PBC
# L = 8, Np = 8: D=6435, ED [-5.4130141  -7.45979063], VMC: -7.4524 ± 0.0077
# L = 10, Np = 5: D=2002, ED [-7.09392643 -8.10700553], VMC:-8.10702 ± 0.00090