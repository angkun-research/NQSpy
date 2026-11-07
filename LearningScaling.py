import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from ExactGS import build_MB_basis_holes,build_Hamiltonian_holes,basis_to_spinconfig_holes,basis_to_onehot
from ExactGS import build_Hamiltonian_holes_adjlist
import NeuralNetworks as nnets
from utils import total_squared_loss
from vmc_utils import adjlist_to_csr
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class Class3States(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=16, kernel_size=2, stride=2,
                 activation="tanh", power=5):
        super(Class3States, self).__init__()
        self.pad = (kernel_size-1)//2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride,)
                                #padding_mode='circular')
        self.pool = nnets.GeometricPool1d()
        self.finallayer = nn.Linear(hidden_dim, 1)
        self.activation = activation
        self.power = power

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, L, 4) -> (batch, 4, L)
        x = self.layer1(x)

        if self.activation == "tanh":
            x = torch.tanh(x.pow(self.power))
        elif self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        x = self.pool(x).squeeze(-1)  # shape: (batch, hidden_dim)
        x = self.finallayer(x)
        #x = torch.tanh(x.pow(3))
        return x.squeeze(-1)


def obtain_train_data(L, J1=1.0,J2=1.0,pbc=False):
    t1, t2 = 0.0, 0.0
    basis = build_MB_basis_holes(L,nholes=0,Nup=L//2) # only works for even L
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    #print("Dimension of Hilbert space:", len(basis))
    # H_dense = build_Hamiltonian_holes(L, t1, t2, basis, J1=J1, J2=J2)
    # eigvals, eigvecs = eigh(H_dense)
    H_ind, H_val = build_Hamiltonian_holes_adjlist(L, t1, t2, basis, J1=J1, J2=J2, pbc=pbc)
    Hsparse = adjlist_to_csr(H_ind, H_val)
    eigvals, eigvecs = eigsh(Hsparse, k=1, which='SA')
    #print(f"Exact ground state energy for L {L}: {eigvals[0:3]}")

    spin_configs = basis_to_spinconfig_holes(basis, L)
    #coeffs = np.array(eigvecs[:, 0]*2**(L/4), dtype=np.float32)  # Ground state coefficients
    coeffs = np.array(eigvecs[:, 0], dtype=np.float64)
    even_sites = [i for i in range(L) if i % 2 == 0]
    candidate_state = ((),tuple(even_sites))
    print(coeffs[basis_dict[candidate_state]], np.max(np.abs(coeffs)))
    sign = np.sign(coeffs[basis_dict[candidate_state]])
    coeffs = coeffs * sign / np.max(np.abs(coeffs)) # gauge fix
    #coeffs = coeffs / coeffs[basis_dict[candidate_state]] # gauge fix

    onehot_configs = basis_to_onehot(spin_configs, L)
    onehot_configs = onehot_configs.reshape(-1, L, 4)  # (num_states, L, 4)
    #onehot_configs = onehot_configs.transpose(0, 2, 1)  # (num_states, 4, L)

    X = torch.tensor(onehot_configs, dtype=torch.float32)
    y = torch.tensor(coeffs, dtype=torch.float32)
    E_inf = Hsparse.diagonal().sum().real / Hsparse.shape[0]
    return X, y, Hsparse, E_inf

r"""
The V-score measures variational-state accuracy using the energy
and its variance, without needing the exact ground state:

V = \frac{N \sigma_E^2}{(E - E_\infty)^2},
E = \langle H \rangle_\psi,
\sigma_E^2 = \langle H^2 \rangle_\psi - \langle H \rangle_\psi^2.

Here:
- N is the number of spins (L in this calculation).
- E_inf = Tr(H) / dim(H) is the infinite-temperature energy
  over the chosen Hilbert space or symmetry sector.
- Expectations use the normalized variational state.
"""
def compute_v_score(predictions, Hsparse, L, E_inf):
    psi = predictions.detach().cpu().numpy().astype(np.float64)
    norm = np.linalg.norm(psi)
    if norm == 0:
        return float("nan")
    psi = psi / norm

    Hpsi = Hsparse @ psi
    energy = np.vdot(psi, Hpsi).real
    residual = Hpsi - energy * psi
    variance = np.vdot(residual, residual).real

    denominator = (energy - E_inf)**2
    return L * variance / denominator if denominator > 0 else float("nan")

def train_activation(X, y, activation, power, seed, epochs, lr,
                     Hsparse, E_inf, scheduleLR=False):
    # Reset before constructing each model so all activations start
    # with identical weights for a given seed.
    torch.manual_seed(seed)

    model = Class3States(
        in_channels=4,
        hidden_dim=32,
        kernel_size=5,
        stride=2,
        activation=activation,
        power=power,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        if scheduleLR else None
    )

    # Entry 0 is before training; entry n is after n optimizer updates.
    mse_history = np.empty(epochs + 1)
    v_history = np.empty(epochs + 1)
    L = X.shape[1]
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(X)
        loss = total_squared_loss(predictions, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite loss: {activation=}, {power=}, "
                f"{seed=}, {epoch=}"
            )

        mse_history[epoch] = loss.item() / y.numel()
        v_history[epoch] = compute_v_score(predictions, Hsparse, L, E_inf)
        if epoch % 500 == 0:
            print(
                f"{activation}, power={power}, seed={seed}, epoch={epoch}: "
                f"MSE={mse_history[epoch]:.6e}, V={v_history[epoch]:.6e}"
            )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X)
        v_history[-1] = compute_v_score(predictions, Hsparse, L, E_inf)
        mse_history[-1] = (
            total_squared_loss(predictions, y).item() / y.numel()
        )

        norm_product = predictions.norm() * y.norm()
        overlap = (
            (torch.abs(torch.dot(predictions, y)) / norm_product).item()
            if norm_product.item() > 0 else float("nan")
        )

    return mse_history, overlap, v_history


ACTIVATIONS = [
    ("tanh", 1, r"$\tanh(x)$"),
    ("tanh", 3, r"$\tanh(x^3)$"),
    ("tanh", 5, r"$\tanh(x^5)$"),
    ("tanh", 7, r"$\tanh(x^7)$"),
    ("relu", 1, "ReLU"),
    ("sigmoid", 1, "Sigmoid"),
]


def collect_data(
    system_sizes=(12,), seeds=(0,), epochs=2000,
    lr=0.01, J1=1.0, J2=1.0, pbc=False, scheduleLR=False,
):
    results = {
        "settings": {
            "J1": J1,
            "J2": J2,
            "pbc": pbc,
            "epochs": epochs,
            "lr": lr,
            "seeds": tuple(seeds),
        },
        "systems": {},
    }

    for L in system_sizes:
        if L < 2 or L % 2:
            raise ValueError("System sizes must be even integers >= 2.")

        # Compute the ED target once per system size.
        X, y, Hsparse, E_inf = obtain_train_data(L, J1=J1, J2=J2, pbc=pbc)
        runs = {}
        results["systems"][L] = runs

        print(f"\nL={L}, configurations={len(y)}, J1={J1}, J2={J2}")

        for activation, power, label in ACTIVATIONS:
            histories = []
            overlaps = []
            v_histories = []

            for seed in results["settings"]["seeds"]:
                history, overlap, v_history = train_activation(
                    X, y, activation, power, seed, epochs, lr,
                    Hsparse, E_inf, scheduleLR=scheduleLR
                )
                histories.append(history)
                overlaps.append(overlap)
                v_histories.append(v_history)

                print(
                    f"{activation:7s} power={power}, seed={seed}: "
                    f"final MSE={history[-1]:.6e}, "
                    f"|overlap|={overlap:.8f}, "
                    f"V={v_history[-1]:.6e}, "
                )

            runs[(activation, power)] = {
                "label": label,
                "mse": np.stack(histories),
                "overlap": np.asarray(overlaps),
                "v_score": np.stack(v_histories),
            }

    return results


def plot_results(results, log_x=True, show_spread=True, metric="mse"):
    settings = results["settings"]
    systems = results["systems"]

    fig, axes = plt.subplots(
        1, len(systems),
        figsize=(7 * len(systems), 5),
        squeeze=False,
    )

    floor = np.finfo(float).tiny

    for ax, (L, runs) in zip(axes[0], systems.items()):
        for run in runs.values():
            histories = run[metric]
            steps = np.arange(histories.shape[1])

            # Epoch 0 can only appear on a linear x-axis.
            start = 1 if log_x else 0
            steps = steps[start:]
            plotted = histories[:, start:]
            final_fidelity = run["overlap"].mean()

            line, = ax.plot(
                steps,
                np.maximum(plotted.mean(axis=0), floor),
                label=f'{run["label"]}, F={final_fidelity:.6f}',
            )

            if show_spread and len(histories) > 1:
                ax.fill_between(
                    steps,
                    np.maximum(plotted.min(axis=0), floor),
                    np.maximum(plotted.max(axis=0), floor),
                    color=line.get_color(),
                    alpha=0.15,
                )

        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")

        ax.set(
            xlabel="Epoch (completed optimizer updates)",
            ylabel="Mean squared loss" if metric == "mse" else "V score",
            title=(
                f"L={L}, J1={settings['J1']}, "
                f"J2={settings['J2']}"
            ),
        )
        ax.grid(True, which="both", alpha=0.2)
        ax.legend()

    fig.tight_layout()
    return fig, axes


# %% Collect training data
if __name__ == "__main__":
    results = collect_data(
        system_sizes=[12],
        seeds=[0],
        epochs=2000,
        lr=0.01,
        J1=1.0,
        J2=1.5,
        scheduleLR= True #True #False
    )

    # fig, axes = plot_results(results, log_x=False)
    # plt.show()

    fig_loss_logx, _ = plot_results(results, log_x=True)
    fig_v_logx, _ = plot_results(results, log_x=True, metric="v_score")
    fig_loss, _ = plot_results(results, log_x=False)
    fig_v, _ = plot_results(results, log_x=False, metric="v_score")

    fig_loss_logx.savefig("/Users/angkunwu/Desktop/training_loss_logx.pdf", bbox_inches="tight")
    fig_v_logx.savefig("/Users/angkunwu/Desktop/v_score_logx.pdf", bbox_inches="tight")
    fig_loss.savefig("/Users/angkunwu/Desktop/training_loss.pdf", bbox_inches="tight")
    fig_v.savefig("/Users/angkunwu/Desktop/v_score.pdf", bbox_inches="tight")
    plt.show()