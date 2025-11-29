"""
run_week3_ckks_fedavg.py

Week-3 experiment:
- Full Flower federated learning pipeline using CKKS-based secure aggregation
  on the server side.
- Uses the same synthetic telemetry + MLP model as Week-1, but replaces
  plain FedAvg aggregation with CKKS homomorphic aggregation.

Outputs:
- results/results_ckks.csv     (per-round metrics)
- results/acc_vs_rounds.png   (accuracy vs rounds, overwritten each run)
- results/bytes_vs_rounds.png (bytes vs rounds, overwritten each run)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# 0) Make sure the project root (containing "src/") is on sys.path
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # one level up from "experiments/"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can safely import from src.*
import flwr as fl
import torch
import torch.utils.data as data

from src.data import generate_synthetic_telemetry
from src.model import MLP
from src.client import get_client_fn
from src.server import make_strategy
from src.utils import set_global_seeds, plot_metrics


def main() -> None:
    # -----------------------------
    # 1) Global config
    # -----------------------------
    NUM_CLIENTS = 8
    ROUNDS = 20
    DROPOUT_PROB = 0.3
    SEED = 42
    SECURE_MODE = "ckks"  # <--- key: enables CKKS aggregation on the server

    set_global_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Week-3] Using device: {device}")

    # -----------------------------
    # 2) Generate synthetic telemetry
    # -----------------------------
    # Each element: (X_tensor, y_tensor) for one client
    clients_data = generate_synthetic_telemetry(
        num_clients=NUM_CLIENTS,
        samples_per_client=2000,
        window_size=32,
        anomaly_prob=0.05,
        seed=SEED,
    )

    # -----------------------------
    # 3) Build a global validation loader (e.g., from client 0)
    # -----------------------------
    X0, y0 = clients_data[0]
    n0 = len(X0)
    split = int(0.8 * n0)

    val_ds = data.TensorDataset(X0[split:], y0[split:])
    val_loader = data.DataLoader(val_ds, batch_size=64, shuffle=False)

    # -----------------------------
    # 4) Global model for the server/strategy
    # -----------------------------
    input_dim = X0.shape[1]  # should be 32
    model = MLP(input_dim=input_dim, hidden_dim=32)

    # -----------------------------
    # 5) Results path + strategy
    # -----------------------------
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    results_csv = results_dir / "results_ckks.csv"
    if results_csv.exists():
        # Overwrite old results to avoid confusion
        results_csv.unlink()

    strategy = make_strategy(
        model=model,
        val_loader=val_loader,
        device=device,
        results_csv=str(results_csv),
        seed=SEED,
        secure_mode=SECURE_MODE,
        num_clients=NUM_CLIENTS,
        dropout_prob=DROPOUT_PROB,
    )

    # -----------------------------
    # 6) Flower client_fn (re-uses src.client helper)
    # -----------------------------
    client_fn = get_client_fn(
        clients_data=clients_data,
        device=device,
        model_cls=MLP,
        local_epochs=1,
        secure_mode=SECURE_MODE,
        noise_std=0.01,  # used only if mode="mask"; ignored for "ckks"
    )

    # -----------------------------
    # 7) Run Flower simulation
    # -----------------------------
    print("[Week-3] Starting Flower simulation with CKKS aggregation...")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    # -----------------------------
    # 8) Plot metrics from CSV
    # -----------------------------
    if results_csv.exists():
        print(f"[Week-3] Simulation finished. Results stored in: {results_csv}")
        plot_metrics(str(results_csv), save_dir=str(results_dir))
        print("[Week-3] Plots updated: acc_vs_rounds.png, bytes_vs_rounds.png")
    else:
        print(
            "[Week-3] WARNING: results_ckks.csv not found. "
            "Check that the strategy logging ran correctly."
        )


if __name__ == "__main__":
    main()
