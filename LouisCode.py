import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
import netket as nk
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
import jax; import jax.numpy as jnp
from netket.utils import struct
from flax import nnx
from netket.nn.masked_linear import default_kernel_init
from typing import Any
from functools import partial
from scipy.sparse.linalg import eigsh

DType = Any

L = 7
hi = nk.hilbert.SpinOrbitalFermions(L, s=1/2, n_fermions_per_spin=(L//2, L//2))
graph = nk.graph.Chain(L)

t1 = 1
t2 = 0.5
U = 1e5 #1e10
H = 0.0

for i, j in zip(list(range(L-1)), list(range(1, L))):
    H -= t1 * (cdag(hi, i, -1) @ c(hi, j, -1) + cdag(hi, j, -1) @ c(hi, i, -1))
    H -= t1 * (cdag(hi, i, 1) @ c(hi, j, 1) + cdag(hi, j, 1) @ c(hi, i, 1))

for i, j in zip(list(range(0, L-2, 2)), list(range(2, L, 2))):
    H -= t2 * (cdag(hi, i, -1) @ c(hi, j, -1) + cdag(hi, j, -1) @ c(hi, i, -1))
    H -= t2 * (cdag(hi, i, 1) @ c(hi, j, 1) + cdag(hi, j, 1) @ c(hi, i, 1))

for i in range(L):
    H += U * nc(hi, i, 1) @ nc(hi, i, -1)
    
def _logdet_cmplx(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))

class LogSlaterDeterminant(nnx.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions

    def __init__(
        self,
        hilbert,
        kernel_init=default_kernel_init,
        param_dtype=float,
        *,
        rngs: nnx.Rngs,
    ):
        self.hilbert = hilbert
        key = rngs.params()
        self.M = nnx.Param(
            kernel_init(
                key,
                (
                    self.hilbert.n_orbitals * int(1 + self.hilbert.spin * 2),
                    self.hilbert.n_fermions,
                ),
                param_dtype,
            )
        )

    def __call__(self, n: jax.Array) -> jax.Array:
        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            A = self.M[R]
            return _logdet_cmplx(A)
        return log_sd(n)

model = LogSlaterDeterminant(hi, rngs=nnx.Rngs(0))
sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)
vstate = nk.vqs.MCState(sa, model, n_samples=64, n_discard_per_chain=16)
op = nk.optimizer.Sgd(learning_rate=0.001)
gs = nk.VMC(H, op, variational_state=vstate)

slater_log = nk.logging.RuntimeLog()
gs.run(n_iter=10000, out=slater_log)

print("nn energy was:", gs.energy)
print("ed energy was:", eigsh(H.to_sparse(), k=1, which="SA")[0][0])
