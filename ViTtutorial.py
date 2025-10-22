import matplotlib.pyplot as plt

import netket as nk

import jax
import jax.numpy as jnp

print(jax.devices())

import flax
from flax import linen as nn

from einops import rearrange

from functools import partial
from ViTutils import ViT

seed = 0
key = jax.random.key(seed)

L = 4
n_dim = 2
J2 = 0.5

# initialize a batch of spin configurations, considering a system on a LxL square lattice
M = 200

key, subkey = jax.random.split(key)
spin_configs = jax.random.randint(subkey, shape=(M, L * L), minval=0, maxval=1) * 2 - 1

print(f"{spin_configs.shape = }")

lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule

# Intiialize the ViT variational wave function
vit_module = ViT(
    num_layers=2, d_model=10, n_heads=2, patch_size=2, transl_invariant=True
)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
N_samples = 4096
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

optimizer = nk.optimizer.Sgd(learning_rate=0.0075)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)

# Variational monte carlo driver
from netket.experimental.driver import VMC_SR

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="complex",
)

# Optimization
log = nk.logging.RuntimeLog()

N_opt = 800
vmc.run(n_iter=N_opt, out=log)