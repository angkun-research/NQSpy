import torch

# Lanczos process.
# Louis Primeau April 9th 2026
# A: function that performs A(x) = A @ x. For matrices wrap with lambda x: A @ x.
# b: starting vector, need not be normalized.
# krylov_dim: how many krylov vectors you want to keep.
# keep_V: whether to save the krylov vectors or just build Tm
# use keep_V to compute matrix functions as f(A) ≈ Vm^T f(Tm) Vm
# beware of ghosts!
def lanczos_process(A, b, krylov_dim, keep_V=True):

    alphas = []
    betas = []
    
    

    b = b / torch.norm(b)
    if keep_V: Vm = [b]
    qi = b
    qim1 = torch.zeros_like(b)
    beta = 0

    for i in range(1, krylov_dim):

        v = A(qi)

        alpha = qi.conj().dot(v)
        alphas.append(alpha)
        
        v = v - beta * qim1 - alpha * qi

        beta = torch.linalg.norm(v)
        betas.append(beta)
        
        v /= beta

        if keep_V: Vm.append(v)
        qim1, qi = qi, v

    alphas = torch.tensor(alphas)
    betas = torch.tensor(betas)
    Tm = torch.diag(alphas) + torch.diag(betas[:-1], diagonal=-1) + torch.diag(betas[:-1], diagonal=1)

    if not keep_V:
        return Tm
    else:
        return Tm, torch.cat([V.unsqueeze(1) for V in Vm], dim=1)

A = torch.randn(100, 100, dtype=torch.float64)# + 1j * torch.randn(100, 100, dtype=torch.float64)
A = A + A.conj().T
Tm, Vm = lanczos_process(lambda x: A @ x, torch.randn(100, dtype=torch.float64), 20, keep_V=True)

print(torch.linalg.eigvalsh(Tm))
print(torch.linalg.eigvalsh(A))
