import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vmc_utils import generate_initial_state, VBSNN, sr_update_optimizer, cleanup_memory
from vmc_utils_gpu import BalancedSampler_gpu, GlobalSampler_gpu, Obtain_Sampling_batch_gpu
from utils import set_seed

SEEDS = [42, 123, 456, 789]
NUM_GPUS = 2  # one seed runs per GPU at a time

# parameter sets to sweep: each dict fully describes one experiment configuration
PARAM_SETS = [
    dict(L=23, t2=0.5, J1=0.0, J2=0.0, hidden_dim=64, kernel_size=2,
         n_walkers=100, n_samples=100, epochs=2000, burn_in=64, use_sr=False,
         E_exact= -2.526143),
]
# L = 21, 23,25,27,29,31
# -2.519684, -2.526143, -2.531216, -2.535273, -2.538568, -2.541281 

RESULTS_FILE = "data/vmc_seeds_summary.json"

def gpu_device_for(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id % torch.cuda.device_count()}")


def run_single_experiment(params: dict, seed: int, gpu_id: int) -> dict:
    set_seed(seed)
    device = gpu_device_for(gpu_id)

    L = params["L"]
    t2 = params["t2"]
    J1 = params["J1"]
    J2 = params["J2"]
    hidden_dim = params["hidden_dim"]
    kernel_size = params["kernel_size"]
    n_walkers = params["n_walkers"]
    n_samples = params["n_samples"]
    epochs = params["epochs"]
    burn_in = params["burn_in"]
    use_sr = params["use_sr"]

    psi = VBSNN(L, Conv_dim=hidden_dim, kernel_size=kernel_size).to(device)

    if use_sr:
        optimizer = optim.SGD(psi.parameters(), lr=1e-1)
    else:
        optimizer = optim.Adam(psi.parameters(), lr=1e-2)

    Sampler = GlobalSampler_gpu #BalancedSampler_gpu if J2 == J1 else GlobalSampler_gpu

    initial_states = [generate_initial_state(L) for _ in range(n_walkers)]
    _, _, states, _ = Obtain_Sampling_batch_gpu(
        psi, L, initial_states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True
    )

    energy_record, error_record = [], []

    for step in range(epochs):
        if step % 100 == 0:
            states = [generate_initial_state(L, singlet=True) for _ in range(n_walkers)]
            _, _, states, _ = Obtain_Sampling_batch_gpu(
                psi, L, states, burn_in, t2=t2, J1=J1, J2=J2, device=device, burnin=True
            )

        psis, energies, states, sampled_states = Obtain_Sampling_batch_gpu(
            psi, L, states, n_samples, t2=t2, J1=J1, J2=J2, device=device,
            print_rate=False, Sampler=Sampler
        )

        E_tensor = energies.squeeze() if torch.is_tensor(energies) else torch.stack(energies).squeeze()
        psis_tensor = psis.squeeze() if torch.is_tensor(psis) else torch.stack(psis).squeeze()

        E_mean = E_tensor.mean()
        E_se = E_tensor.var().sqrt() / math.sqrt(E_tensor.numel())

        energy_record.append(E_mean.item())
        error_record.append(E_se.item())

        if step % 20 == 0:
            print(f"[seed={seed} gpu={gpu_id}] step {step}: <E> = {E_mean.item():.6f} ± {E_se.item():.6f}")

        if use_sr:
            if E_mean.item() > params["E_exact"] * 0.75:
                lr, sr_tau = 5e-1, 1e0
            else:
                lr, sr_tau = 5e-2, 1e-1
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            sr_update_optimizer(psi, sampled_states, E_tensor, L, optimizer, sr_tau, device,
                                 adaptive_lr=use_sr)
        else:
            loss = 2 * torch.mean((E_tensor.detach() - E_mean.detach()) * psis_tensor / psis_tensor.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sampled_states = None
        psis, psis_tensor = None, None
        energies, E_tensor = None, None
        cleanup_memory(free_vars=None, optimizer=optimizer)

    # save per-epoch record for this (params, seed) to data/
    tag = f"L{L}_t2{t2}_hidden{hidden_dim}_kernel{kernel_size}_sam{n_walkers * n_samples}_seed{seed}"
    df = pd.DataFrame({"epoch": range(epochs), "Energy": energy_record, "Error": error_record})
    os.makedirs("data", exist_ok=True)
    epoch_csv = f"data/energy_error_{tag}.csv"
    df.to_csv(epoch_csv, index=False)

    best_idx = int(np.argmin(energy_record))
    return {
        "seed": seed,
        "gpu_id": gpu_id,
        "params": params,
        "epoch_csv": epoch_csv,
        "min_energy": energy_record[best_idx],
        "min_energy_error": error_record[best_idx],
        "min_energy_epoch": best_idx,
        "final_energy": energy_record[-1],
        "final_energy_error": error_record[-1],
        "final_epoch": epochs - 1,
    }


def task_key(params: dict, seed: int) -> str:
    return f"L{params['L']}_hidden{params['hidden_dim']}_kernel{params['kernel_size']}_seed{seed}"


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    tasks = []
    for params in PARAM_SETS:
        for i, seed in enumerate(SEEDS):
            key = task_key(params, seed)
            if key not in results:
                gpu_id = i % NUM_GPUS
                tasks.append((key, params, seed, gpu_id))

    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_GPUS} GPU workers")

    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        future_map = {
            executor.submit(run_single_experiment, params, seed, gpu_id): (key, params, seed, gpu_id)
            for key, params, seed, gpu_id in tasks
        }

        for future in as_completed(future_map):
            key, params, seed, gpu_id = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"done seed={seed:<4} gpu={gpu_id} "
                    f"min_E={result['min_energy']:.6f} ± {result['min_energy_error']:.6f} "
                    f"(epoch {result['min_energy_epoch']}) -> {result['epoch_csv']}"
                )
            except Exception as exc:
                print(f"failed seed={seed}: {exc}")

    print(f"Saved summary to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
    # nohup python collect_vmc_seeds.py > data/collect_vmc_seeds.log 2>&1 &