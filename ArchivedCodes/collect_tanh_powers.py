import itertools
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import NeuralNetworks as nnets
from ExactGS import build_MB_basis_holes, basis_to_spinconfig_holes, basis_to_onehot
from utils import set_seed, total_squared_loss

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
NUM_GPUS = 2  # one worker process pinned to each GPU

TRAIN_LS = [2, 4, 6, 8, 10]
TEST_L = 20
EPOCHS, LR = 5000, 1e-2
NONZERO_TOL = 1e-6   # |y| > tol -> sign = +-1 ; |y| <= tol -> sign = 0

MODEL_CFG = dict(in_channels=4, hidden_dim=16, kernel_size=2, stride=2, activation='tanh')
POWERS = [1, 3, 5]
MODELS = {  # p=p binds the loop value at definition time
    f"Class3States_p{p}": (lambda p=p: Class3States(**MODEL_CFG, power=p))
    for p in POWERS
}

RESULTS_FILE = (f"data/VBS_generalization_seeds_h{MODEL_CFG['hidden_dim']}"
                f"_{MODEL_CFG['activation']}_powers.json")
CRITERION = total_squared_loss


# ----------------------------------------------------------------------------- data
def Build_VBS(L):
    assert L % 2 == 0, "L must be even"
    singlet_pairs = [(2 * i, 2 * i + 1) for i in range(L // 2)]
    vbs_basis, vbs_coeffs = [], []
    for up_sites in itertools.product(*singlet_pairs):
        sign = 1
        for idx, pair in enumerate(singlet_pairs):
            if up_sites[idx] == pair[1]:
                sign *= -1
        vbs_basis.append(((), up_sites))
        vbs_coeffs.append(sign)
    return vbs_basis, vbs_coeffs


def obtain_train_data_VBS(L, Reshape=False, Normalize=False):
    basis = build_MB_basis_holes(L, nholes=0)  # no holes for VBS
    basis_dict = {state: idx for idx, state in enumerate(basis)}
    vbs_basis, vbs_coeffs = Build_VBS(L)
    vbs_vec = np.zeros(len(basis))
    for config, coeff in zip(vbs_basis, vbs_coeffs):
        vbs_vec[basis_dict[config]] = coeff
    if Normalize:
        vbs_vec = vbs_vec / np.sqrt(2) ** (L // 2)

    spin_configs = basis_to_spinconfig_holes(basis, L)
    onehot_configs = basis_to_onehot(spin_configs, L)
    if Reshape:
        onehot_configs = onehot_configs.reshape(-1, L, 4)  # (num_states, L, 4)

    X = torch.tensor(onehot_configs, dtype=torch.float32)
    y = torch.tensor(vbs_vec, dtype=torch.float32)
    return X, y


@lru_cache
def load_train_data():  # built once per worker, reused by all its tasks (CPU tensors)
    return [(L, *obtain_train_data_VBS(L, Reshape=True)) for L in TRAIN_LS]


@lru_cache
def load_test_data():  # L=20 split into sign=+-1 and sign=0 subsets (CPU tensors)
    X, y = obtain_train_data_VBS(TEST_L, Reshape=True)
    nz = torch.abs(y) > NONZERO_TOL
    return X[nz], y[nz], X[~nz], y[~nz]


# ----------------------------------------------------------------------------- model
class Class3States(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, kernel_size=2, stride=2, activation='tanh', power=5):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.layer1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=self.pad, stride=stride)
        self.pool = nnets.GeometricPool1d()
        self.power = power
        self.finallayer = nn.Linear(hidden_dim, 1)
        self.activation = activation

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, L, 4) -> (batch, 4, L)
        if self.activation == 'tanh':
            x = torch.tanh(self.layer1(x).pow(self.power))
        elif self.activation == 'relu':
            x = torch.relu(self.layer1(x))
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(self.layer1(x))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        return self.finallayer(x).squeeze(-1)


# ----------------------------------------------------------------------------- experiment
def init_worker(gpu_queue):
    global GPU_ID
    GPU_ID = gpu_queue.get()  # each worker claims one GPU for its whole lifetime


@torch.no_grad()
def squared_error_sum(model, X, y, device):
    """Sum of squared errors over the full set at once; returns (sse, n)."""
    X, y = X.to(device), y.to(device)
    sse = ((model(X) - y) ** 2).sum().double().item()
    return sse, len(y)


def run_single_experiment(name: str, seed: int) -> dict:
    device = torch.device(f"cuda:{GPU_ID % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    train = [(L, X.to(device), y.to(device)) for L, X, y in load_train_data()]
    set_seed(seed)  # after data loading, so the model init depends only on the seed
    model = MODELS[name]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    tag = f"[{name} seed={seed} gpu={GPU_ID}]"
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = sum(CRITERION(model(X), y) for _, X, y in train)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"{tag} epoch {epoch}: loss = {loss.item():.6e}", flush=True)

    model.eval()
    result = {
        "model": name, "power": model.power, "seed": seed, "gpu_id": GPU_ID,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_train_loss": loss.item(),
    }

    # (1) training MSE: pooled over all L=2..10 configurations, plus per-L
    tot_sse, tot_n = 0.0, 0
    for L, X, y in train:
        sse, n = squared_error_sum(model, X, y, device)
        result[f"train_mse_L{L}"] = sse / n
        tot_sse, tot_n = tot_sse + sse, tot_n + n
    result["train_mse"] = tot_sse / tot_n

    # (2) L=20 test, sign = +-1   (3) L=20 test, sign = 0
    X_nz, y_nz, X_z, y_z = load_test_data()
    sse, n = squared_error_sum(model, X_nz, y_nz, device)
    result["test_mse_sign_pm1"], result["n_test_sign_pm1"] = sse / n, n
    sse, n = squared_error_sum(model, X_z, y_z, device)
    result["test_mse_sign0"], result["n_test_sign0"] = sse / n, n
    return result


# ----------------------------------------------------------------------------- summary
REPORT = {  # column label -> metric key in the per-seed results
    "train L=2-10": "train_mse",
    "test L=20 (1,-1)": "test_mse_sign_pm1",
    "test L=20 (0)": "test_mse_sign0",
}


def summarize(results):
    df = pd.DataFrame(results.values())
    agg = {"power": ("power", "first"), "n_seeds": ("seed", "count")}
    for m in REPORT.values():
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")  # sample std (ddof=1) over seeds
    summary = df.groupby("model").agg(**agg).sort_values("power")
    summary.to_csv(RESULTS_FILE.replace(".json", "_summary.csv"))

    print("\nMSE over seeds: mean +- std")
    for model, r in summary.iterrows():
        print(f"\n{model}  (power={int(r['power'])}, n_seeds={int(r['n_seeds'])})")
        for label, m in REPORT.items():
            print(f"  {label:<18} {r[f'{m}_mean']:.2e} +- {r[f'{m}_std']:.2e}")

    missing = summary.index[summary["n_seeds"] < len(SEEDS)].tolist()
    if missing:
        print(f"\nWARNING: fewer than {len(SEEDS)} seeds finished for {missing}")
    return summary


# ----------------------------------------------------------------------------- main
def main(table_only=False):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    tasks = [(name, seed) for name in MODELS for seed in SEEDS if f"{name}_seed{seed}" not in results]
    if table_only:
        tasks = []
    print(f"Pending tasks: {len(tasks)}, using {NUM_GPUS} GPU workers")

    if tasks:
        ctx = mp.get_context("spawn")  # workers never inherit CUDA state from the parent
        gpu_queue = ctx.Queue()
        for gpu_id in range(NUM_GPUS):
            gpu_queue.put(gpu_id)

        with ProcessPoolExecutor(max_workers=NUM_GPUS, mp_context=ctx,
                                 initializer=init_worker, initargs=(gpu_queue,)) as executor:
            future_map = {executor.submit(run_single_experiment, name, seed): (name, seed) for name, seed in tasks}
            for future in as_completed(future_map):
                name, seed = future_map[future]
                try:
                    r = future.result()
                    results[f"{name}_seed{seed}"] = r
                    with open(RESULTS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"done {name} seed={seed} gpu={r['gpu_id']}: train_mse={r['train_mse']:.6e}, "
                          f"test_pm1={r['test_mse_sign_pm1']:.6e}, test_0={r['test_mse_sign0']:.6e}", flush=True)
                except Exception as exc:
                    print(f"failed {name} seed={seed}: {exc}", flush=True)

    if not results:
        print("No results to summarize.")
        return
    summarize(results)


if __name__ == "__main__":
    main(table_only="--table-only" in sys.argv)
    # full run:   nohup python collect_tanh_powers.py > data/collect_tanh_powers.log 2>&1 &
    # table only: python collect_tanh_powers.py --table-only   (reprints the summary from the existing JSON)