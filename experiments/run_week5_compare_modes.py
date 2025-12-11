"""
run_week5_compare_modes.py

Week-5 experiment:
- Run the full Flower FL pipeline three times:
    1) Baseline  (secure_mode = "none")
    2) Masked    (secure_mode = "mask")
    3) CKKS HE   (secure_mode = "ckks")
- Each run logs per-round metrics to its own CSV in ./results.
- Then we:
    * Plot three curves (none/mask/ckks) for:
        - Accuracy vs rounds
        - Uplink bytes vs rounds
        - Round time vs rounds
    * Compute a summary table with:
        - final accuracy
        - mean training loss
        - mean training accuracy
        - convergence speed (#round to reach >= 0.9 acc)
        - avg client time per round
        - avg server agg_time per round
        - total uplink bytes
        - total communication bytes (up+down)

Outputs (all in ./results):
- results_none.csv
- results_mask.csv
- results_ckks.csv
- week5_acc_three_modes.png
- week5_bytes_up_three_modes.png
- week5_round_time_three_modes.png
- week5_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import flwr as fl
import torch
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402  (after setting backend)

# ---------------------------------------------------------------------
# Make sure project root (with src/) is importable
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generate_synthetic_telemetry
from src.model import MLP
from src.client import get_client_fn
from src.server import make_strategy
from src.utils import set_global_seeds


# ---------------------------------------------------------------------
# 1) Helper: run a single FL experiment for a given mode
# ---------------------------------------------------------------------
def run_single_mode(
        mode: str,
        clients_data,
        rounds: int,
        num_clients: int,
        dropout_prob: float,
        seed: int,
        device: torch.device,
        results_dir: Path,
) -> Path:
    """
    Run one Flower simulation with the given security mode.

    Returns:
        Path to the CSV file with per-round metrics.
    """
    assert mode in ["none", "mask", "ckks"], f"Unknown mode {mode}"

    # Set seeds so that each mode starts from the same initial conditions
    set_global_seeds(seed)

    # Build global validation loader from client 0 (same as Week-3)
    import torch.utils.data as data

    X0, y0 = clients_data[0]
    n0 = len(X0)
    split = int(0.8 * n0)

    val_ds = data.TensorDataset(X0[split:], y0[split:])
    val_loader = data.DataLoader(val_ds, batch_size=64, shuffle=False)

    # Global model
    input_dim = X0.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=32)

    # Results CSV for this mode
    csv_path = results_dir / f"results_{mode}.csv"
    if csv_path.exists():
        csv_path.unlink()

    # Strategy with logging (LoggingFedAvg)
    strategy = make_strategy(
        model=model,
        val_loader=val_loader,
        device=device,
        results_csv=str(csv_path),
        seed=seed,
        secure_mode=mode,
        num_clients=num_clients,
        dropout_prob=dropout_prob,
    )

    # Client function
    # - Mask mode: use small Gaussian noise
    # - None/CKKS: no client-side noise (CKKS is server-side)
    noise_std = 0.01 if mode == "mask" else 0.0

    client_fn = get_client_fn(
        clients_data=clients_data,
        device=device,
        model_cls=MLP,
        local_epochs=1,
        secure_mode=mode,
        noise_std=noise_std,
    )

    print(
        f"\n[Week-5] Starting simulation for mode='{mode}' "
        f"(rounds={rounds}, clients={num_clients}, dropout={dropout_prob})"
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    print(f"[Week-5] Finished mode='{mode}'. CSV -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------
# 2) Helper: plotting three-curve figures
# ---------------------------------------------------------------------
def _plot_three_modes(
        dfs: Dict[str, pd.DataFrame],
        metric_col: str,
        ylabel: str,
        title: str,
        out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))

    for mode, df in dfs.items():
        if metric_col not in df.columns:
            print(f"[Week-5] WARNING: '{metric_col}' not in columns for mode={mode}")
            continue
        plt.plot(df["round"], df[metric_col], marker="o", label=mode)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Week-5] Saved plot: {out_path}")


# ---------------------------------------------------------------------
# 3) Helper: summarise metrics across modes
# ---------------------------------------------------------------------
def summarise_modes(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Build a summary table with the main metrics the professor requested.
    """
    rows: List[Dict] = []

    for mode, df in dfs.items():
        if df.empty:
            continue

        # Final accuracy (last round)
        final_acc = float(df["acc"].iloc[-1])

        # Mean training loss/accuracy if present
        mean_loss = float(df["avg_train_loss"].mean()) if "avg_train_loss" in df.columns else float("nan")
        mean_train_acc = float(df["avg_train_acc"].mean()) if "avg_train_acc" in df.columns else float("nan")

        # Convergence speed: first round where acc >= 0.90 (can adjust threshold)
        conv_thresh = 0.90
        conv_round = float("nan")
        above = df[df["acc"] >= conv_thresh]
        if not above.empty:
            conv_round = int(above["round"].iloc[0])

        # Timing overheads
        if "agg_time" in df.columns:
            avg_server_time = float(df["agg_time"].mean())
            avg_client_time = float((df["round_time"] - df["agg_time"]).mean())
        else:
            avg_server_time = float("nan")
            avg_client_time = float(df["round_time"].mean())

        # Communication overhead
        total_bytes_up = int(df["bytes_up"].sum())
        total_bytes = int(df["bytes_up"].sum() + df["bytes_down"].sum())

        rows.append(
            {
                "mode": mode,
                "final_acc": final_acc,
                "mean_train_loss": mean_loss,
                "mean_train_acc": mean_train_acc,
                "conv_round_acc>=0.90": conv_round,
                "avg_client_time_s": avg_client_time,
                "avg_server_agg_time_s": avg_server_time,
                "total_bytes_up": total_bytes_up,
                "total_bytes_total": total_bytes,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"[Week-5] Summary CSV written to: {out_path}")
    print("\n[Week-5] Summary (per mode):")
    print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------
# 4) Main
# ---------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Week-5] Using device: {device}")
    set_global_seeds(args.seed)

    # Ensure results directory exists
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate synthetic telemetry ONCE so all modes see the same data
    clients_data = generate_synthetic_telemetry(
        num_clients=args.clients,
        samples_per_client=2000,
        window_size=32,
        anomaly_prob=0.05,
        seed=args.seed,
    )

    # Run three modes
    csv_paths: Dict[str, Path] = {}
    for mode in ["none", "mask", "ckks"]:
        csv_paths[mode] = run_single_mode(
            mode=mode,
            clients_data=clients_data,
            rounds=args.rounds,
            num_clients=args.clients,
            dropout_prob=args.dropout_prob,
            seed=args.seed,
            device=device,
            results_dir=results_dir,
        )

    # Load all CSVs
    dfs: Dict[str, pd.DataFrame] = {}
    for mode, path in csv_paths.items():
        dfs[mode] = pd.read_csv(path)

    # Three-curve plots
    _plot_three_modes(
        dfs,
        metric_col="acc",
        ylabel="Accuracy",
        title="Week-5: Accuracy vs Rounds (none vs mask vs ckks)",
        out_path=results_dir / "week5_acc_three_modes.png",
    )

    _plot_three_modes(
        dfs,
        metric_col="bytes_up",
        ylabel="Uplink bytes per round",
        title="Week-5: Uplink Bytes vs Rounds",
        out_path=results_dir / "week5_bytes_up_three_modes.png",
    )

    _plot_three_modes(
        dfs,
        metric_col="round_time",
        ylabel="Round time (s)",
        title="Week-5: Round Time vs Rounds",
        out_path=results_dir / "week5_round_time_three_modes.png",
    )

    # Summary CSV with all primary metrics
    summarise_modes(dfs, out_path=results_dir / "week5_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
