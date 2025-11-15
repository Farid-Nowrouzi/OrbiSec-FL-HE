import os
import sys
import argparse

import flwr as fl
import torch
import torch.utils.data as data
import pandas as pd

# --- Make sure Python can find src/ (so src.data, src.model, ...) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import generate_synthetic_telemetry
from src.model import MLP
from src.utils import set_global_seeds, plot_metrics
from src.client import get_client_fn
from src.server import make_strategy


def main(args: argparse.Namespace) -> None:
    # --- Reproducibility & device selection ---
    set_global_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure results/ exists
    os.makedirs("results", exist_ok=True)

    # Decide whether this run is "baseline" (no security) or "secure"
    tag = "baseline" if args.secure == "none" else "secure"
    csv_path = os.path.join("results", f"results_{tag}.csv")

    # If a previous CSV with this tag exists, remove it to start fresh
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # --- Generate synthetic telemetry for all clients ---
    clients_data = generate_synthetic_telemetry(
        num_clients=args.clients,
        samples_per_client=2000,
        window_size=32,
        anomaly_prob=0.05,
        seed=args.seed,
    )

    # --- Global validation loader (use last 20% of client 0) ---
    X0, y0 = clients_data[0]
    n0 = len(X0)
    split0 = int(0.8 * n0)

    val_ds_global = data.TensorDataset(X0[split0:], y0[split0:])
    val_loader_global = data.DataLoader(
        val_ds_global,
        batch_size=64,
        shuffle=False,
    )

    # --- Dummy global model used by the server for logging & eval ---
    dummy_model = MLP(input_dim=X0.shape[1], hidden_dim=32)

    # --- Build Flower strategy with per-round CSV logging ---
    strategy = make_strategy(
        model=dummy_model,
        val_loader=val_loader_global,
        device=device,
        results_csv=csv_path,
        seed=args.seed,
        secure_mode=args.secure,
        num_clients=args.clients,
        dropout_prob=args.dropout_prob,
    )

    # --- Server config (Flower <= 1.23 style) ---
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)

    # --- Client function (still using classic `client_fn(cid: str)` signature) ---
    client_fn = get_client_fn(
        clients_data=clients_data,
        device=device,
        model_cls=MLP,
        local_epochs=1,
        secure_mode=args.secure,
        noise_std=0.01,
    )

    # --- Run simulation ---
    print(
        f"Starting simulation:\n"
        f"  rounds        = {args.rounds}\n"
        f"  clients       = {args.clients}\n"
        f"  secure mode   = {args.secure}\n"
        f"  dropout_prob  = {args.dropout_prob}\n"
        f"  seed          = {args.seed}"
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=server_config,
        strategy=strategy,
    )

    # --- Plot metrics and print summary info ---
    plot_metrics(csv_path, save_dir="results")

    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    avg_active = df["active_clients"].mean()
    total_bytes_up = df["bytes_up"].sum()

    print("\n=== Run summary ===")
    print(f"Mode:               {args.secure}")
    print(f"Rounds:             {args.rounds}")
    print(f"Last round index:   {int(last['round'])}")
    print(f"Last-round accuracy:{last['acc']:.4f}")
    print(f"Avg active clients: {avg_active:.2f}")
    print(f"Total uplink bytes: {total_bytes_up:.0f}")
    print(f"Results CSV:        {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--secure",
        type=str,
        default="none",
        choices=["none", "ckks", "mask"],
        help="Security mode: none | ckks | mask (ckks currently aliases to mask).",
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.secure == "ckks":
        print(
            "WARNING: CKKS mode not fully implemented yet; "
            "using noise-masking as placeholder."
        )

    main(args)
