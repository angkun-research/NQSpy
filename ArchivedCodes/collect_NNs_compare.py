import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import NeuralNetworks as nnets
from ExactGS import obtain_train_data
from utils import set_seed, total_squared_loss

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
NUM_GPUS = 2  # one worker process pinned to each GPU

L, T1, T2 = 13, 1.0, 0.5
EPOCHS, LR = 2000, 1e-2

MODELS = {
    "FCNet": lambda: nnets.FCNet(L, hidden_dim=1024),
    "ConvNet": lambda: nnets.ConvNet(hidden_dim=64, kernel_size=3),
    "VisionTransformer1D": lambda: nnets.VisionTransformer1D(),
    "LdepConvPlusFC": lambda: nnets.LdepConvPlusFC(L, Conv_dim=16, kernel_size=2),
}

RESULTS_FILE = f"data/nns4_seeds_L{L}_t2{T2}.json"
CRITERION = F.mse_loss  # or total_squared_loss

def init_worker(gpu_queue):
    global GPU_ID
    GPU_ID = gpu_queue.get()  # each worker claims one GPU for its whole lifetime


@lru_cache
def load_data():  # exact GS computed once per worker, reused by all its tasks
    return obtain_train_data(L, t1=T1, t2=T2, TBcoeff=True, Reshape=True, Normalize=True)


def run_single_experiment(name: str, seed: int) -> dict:
    device = torch.device(f"cuda:{GPU_ID % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    X, y = (t.to(device) for t in load_data())
    set_seed(seed)  # after data loading, so the model init depends only on the seed
    model = MODELS[name]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = CRITERION(model(X), y)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"[{name} seed={seed} gpu={GPU_ID}] epoch {epoch}: loss = {loss.item():.6e}", flush=True)

    model.eval()
    with torch.no_grad():
        out = model(X)
        out_norm = out.norm()
        if out_norm > 0:
            fidelity = (out / out_norm * y).sum().item()
        else:
            fidelity = 0.0
        fidelity = abs(fidelity)
        return {
            "model": name, "seed": seed, "gpu_id": GPU_ID,
            "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "loss": CRITERION(out, y).item(),
            "fidelity": fidelity,
        }


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    tasks = [(name, seed) for name in MODELS for seed in SEEDS if f"{name}_seed{seed}" not in results]
    print(f"Pending tasks: {len(tasks)}, using {NUM_GPUS} GPU workers")

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
                print(f"done {name} seed={seed} gpu={r['gpu_id']}: "
                      f"loss={r['loss']:.6e}, fidelity={r['fidelity']:.6f}", flush=True)
            except Exception as exc:
                print(f"failed {name} seed={seed}: {exc}", flush=True)

    summary = pd.DataFrame(results.values()).groupby("model").agg(
        n_params=("n_params", "first"), n_seeds=("seed", "count"),
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
        fidelity_mean=("fidelity", "mean"), fidelity_std=("fidelity", "std"),
    )
    print(summary.to_string(float_format="%.6g"))
    summary.to_csv(RESULTS_FILE.replace(".json", "_summary.csv"))


if __name__ == "__main__":
    main()
    # nohup python collect_nn_seeds.py > data/collect_nn_seeds.log 2>&1 &