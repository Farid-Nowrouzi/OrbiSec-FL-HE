"""
run_week5_compare_modes.py

Week-5 experiment (4 modes):
- Run the full Flower FL pipeline four times:
    1) Baseline  (secure_mode = "none")
    2) Masked    (secure_mode = "mask")
    3) SecAgg    (secure_mode = "secagg") [CLIENT pairwise mask simulation + server overhead model]
    4) CKKS HE   (secure_mode = "ckks")   [server-side HE aggregation]
- Each run logs per-round metrics to its own CSV in ./results.
- Then we:
    * Plot four curves (none/mask/secagg/ckks) for:
        - Accuracy vs rounds
        - Uplink bytes vs rounds
        - Cumulative uplink bytes vs rounds
        - Round time vs rounds
        - Steady-state round time (rounds >= 2)
    * Compute a summary table.

IMPORTANT:
- Pairwise-mask SecAgg requires NO DROPOUT (all clients each round), otherwise masks won't cancel.
  So we automatically override dropout_prob=0.0 for mode="secagg".
- For bytes plots: some modes may have identical bytes, causing curves to overlap perfectly.
  We apply a plotting-only tiny X-JITTER to make overlapping lines visible without changing y-values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd
import torch

# ---------------------------------------------------------------------
# Make sure project root (with src/) is importable
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.client import get_client_fn
from src.data import generate_synthetic_telemetry
from src.model import MLP
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
    assert mode in ["none", "mask", "secagg", "ckks"], f"Unknown mode {mode}"

    #  IMPORTANT: SecAgg pairwise masking needs NO DROPOUT
    effective_dropout = float(dropout_prob)
    if mode == "secagg" and effective_dropout != 0.0:
        print(
            f"[Week-5] NOTE: overriding dropout_prob={effective_dropout} -> 0.0 for mode='secagg' "
            f"(pairwise-mask SecAgg requires full participation)."
        )
        effective_dropout = 0.0

    # Seed so all modes start comparably
    set_global_seeds(seed)

    # Build global validation loader from client 0
    import torch.utils.data as data

    X0, y0 = clients_data[0]
    n0 = len(X0)
    split = int(0.8 * n0)
    val_ds = data.TensorDataset(X0[split:], y0[split:])
    val_loader = data.DataLoader(val_ds, batch_size=64, shuffle=False)

    # Global model
    model = MLP(input_dim=X0.shape[1], hidden_dim=32)

    # Results CSV
    csv_path = results_dir / f"results_{mode}.csv"
    if csv_path.exists():
        csv_path.unlink()

    # Strategy (server decides secure_mode behavior)
    strategy = make_strategy(
        model=model,
        val_loader=val_loader,
        device=device,
        results_csv=str(csv_path),
        seed=seed,
        secure_mode=mode,
        num_clients=num_clients,
        dropout_prob=effective_dropout,  #  use effective dropout
    )

    # ------------------------------------------------------------
    # Client security configuration
    # ------------------------------------------------------------
    # mask  -> client-side Gaussian noise (baseline obfuscation)
    # secagg-> CLIENT pairwise masking simulation (Bonawitz-style cancellation)
    # ckks  -> server-side HE aggregation; client remains plain
    # none  -> plain
    if mode == "mask":
        client_secure_mode = "mask"
        noise_std = 0.01
    elif mode == "secagg":
        client_secure_mode = "secagg"
        # mask_std for pairwise secagg masks (uses noise_std in your client.py)
        noise_std = 0.01
    else:
        client_secure_mode = "none"
        noise_std = 0.0

    client_fn = get_client_fn(
        clients_data=clients_data,
        device=device,
        model_cls=MLP,
        local_epochs=1,
        secure_mode=client_secure_mode,
        noise_std=noise_std,
    )

    print(
        f"\n[Week-5] Starting simulation for mode='{mode}' "
        f"(rounds={rounds}, clients={num_clients}, dropout={effective_dropout})"
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
# 2) Plotting helpers
# ---------------------------------------------------------------------
def _series_key(values: List[float], ndigits: int = 6) -> Tuple:
    """Create a hashable key for a numeric series (rounded)."""
    return tuple(round(float(v), ndigits) for v in values)


def _compute_overlap_jitter(
        dfs: Dict[str, pd.DataFrame],
        metric_col: str,
        order: List[str],
        *,
        steady_state: bool = False,
        jitter_step: float = 0.03,
) -> Dict[str, float]:
    """
    Detect identical y-series across modes and assign tiny plotting-only X jitter offsets.

    jitter_step is in "round units" (e.g., 0.03 means round 5 becomes 5.03).
    This keeps y-values unchanged and still makes curves visible.
    """
    keys: Dict[str, Tuple] = {}
    for mode in order:
        df = dfs.get(mode)
        if df is None or df.empty:
            continue
        if "round" not in df.columns or metric_col not in df.columns:
            continue

        df_plot = df.copy()
        if steady_state:
            df_plot = df_plot[df_plot["round"] >= 2]

        keys[mode] = _series_key(df_plot[metric_col].tolist(), ndigits=6)

    jitter = {m: 0.0 for m in order}
    group_map: Dict[Tuple, List[str]] = {}
    for mode, k in keys.items():
        group_map.setdefault(k, []).append(mode)

    for _, modes in group_map.items():
        if len(modes) <= 1:
            continue
        for i, m in enumerate(modes):
            jitter[m] = float(i) * float(jitter_step)

    return jitter


def _plot_multi_modes(
        dfs: Dict[str, pd.DataFrame],
        metric_col: str,
        ylabel: str,
        title: str,
        out_path: Path,
        order: List[str],
        *,
        steady_state: bool = False,
        overlap_aware: bool = False,
) -> None:
    """
    Plot curves for the given metric across modes.

    steady_state:
      - if True, plots only rounds >= 2
    overlap_aware:
      - if True, apply plotting-only x-jitter when series overlap perfectly
    """
    plt.figure(figsize=(8.5, 4.8))

    jitter = {m: 0.0 for m in order}
    if overlap_aware:
        jitter = _compute_overlap_jitter(
            dfs,
            metric_col=metric_col,
            order=order,
            steady_state=steady_state,
            jitter_step=0.03,
        )

    for mode in order:
        df = dfs.get(mode)
        if df is None or df.empty:
            print(f"[Week-5] WARNING: empty df for mode={mode}")
            continue
        if "round" not in df.columns:
            print(f"[Week-5] WARNING: 'round' missing for mode={mode}")
            continue
        if metric_col not in df.columns:
            print(f"[Week-5] WARNING: '{metric_col}' missing for mode={mode}")
            continue

        df_plot = df.copy()
        if steady_state:
            df_plot = df_plot[df_plot["round"] >= 2]

        x = df_plot["round"].astype(float).values + float(jitter.get(mode, 0.0))
        y = df_plot[metric_col].astype(float).values
        plt.plot(x, y, marker="o", label=mode)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Week-5] Saved plot: {out_path}")


def _plot_cumulative_uplink(
        dfs: Dict[str, pd.DataFrame],
        out_path: Path,
        order: List[str],
) -> None:
    plt.figure(figsize=(8.5, 4.8))

    keys: Dict[str, Tuple] = {}
    for mode in order:
        df = dfs.get(mode)
        if df is None or df.empty or "round" not in df.columns or "bytes_up" not in df.columns:
            continue
        cum = df["bytes_up"].astype(float).cumsum().tolist()
        keys[mode] = _series_key(cum, ndigits=6)

    jitter = {m: 0.0 for m in order}
    group_map: Dict[Tuple, List[str]] = {}
    for mode, k in keys.items():
        group_map.setdefault(k, []).append(mode)

    for _, modes in group_map.items():
        if len(modes) <= 1:
            continue
        for i, m in enumerate(modes):
            jitter[m] = 0.03 * float(i)

    for mode in order:
        df = dfs.get(mode)
        if df is None or df.empty or "round" not in df.columns or "bytes_up" not in df.columns:
            continue

        x = df["round"].astype(float).values + float(jitter.get(mode, 0.0))
        y = df["bytes_up"].astype(float).cumsum().values
        plt.plot(x, y, marker="o", label=mode)

    plt.xlabel("Round")
    plt.ylabel("Cumulative uplink bytes")
    plt.title("Week-5: Cumulative Uplink Bytes vs Rounds (overlap-aware)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Week-5] Saved plot: {out_path}")


# ---------------------------------------------------------------------
# 3) Summary helper
# ---------------------------------------------------------------------
def summarise_modes(dfs: Dict[str, pd.DataFrame], out_path: Path, order: List[str]) -> None:
    rows: List[Dict] = []
    for mode in order:
        df = dfs.get(mode)
        if df is None or df.empty:
            continue

        final_acc = float(df["acc"].iloc[-1]) if "acc" in df.columns else float("nan")
        mean_loss = float(df["avg_train_loss"].mean()) if "avg_train_loss" in df.columns else float("nan")

        if "avg_train_accuracy" in df.columns:
            mean_train_acc = float(df["avg_train_accuracy"].mean())
        elif "avg_train_acc" in df.columns:
            mean_train_acc = float(df["avg_train_acc"].mean())
        else:
            mean_train_acc = float("nan")

        conv_thresh = 0.90
        conv_round = float("nan")
        if "acc" in df.columns and "round" in df.columns:
            above = df[df["acc"] >= conv_thresh]
            if not above.empty:
                conv_round = int(above["round"].iloc[0])

        df_steady = df[df["round"] >= 2] if "round" in df.columns else df

        if "agg_time" in df.columns and "round_time" in df.columns:
            avg_server_time = float(df["agg_time"].mean())
            avg_client_time = float((df["round_time"] - df["agg_time"]).mean())
            steady_avg_round_time = float(df_steady["round_time"].mean())
            steady_avg_agg_time = float(df_steady["agg_time"].mean())
        else:
            avg_server_time = float("nan")
            avg_client_time = float(df["round_time"].mean()) if "round_time" in df.columns else float("nan")
            steady_avg_round_time = (
                float(df_steady["round_time"].mean()) if "round_time" in df_steady.columns else float("nan")
            )
            steady_avg_agg_time = float("nan")

        total_bytes_up = int(df["bytes_up"].sum()) if "bytes_up" in df.columns else 0
        total_bytes_down = int(df["bytes_down"].sum()) if "bytes_down" in df.columns else 0
        total_bytes = int(total_bytes_up + total_bytes_down)

        rows.append(
            {
                "mode": mode,
                "final_acc": final_acc,
                "mean_train_loss": mean_loss,
                "mean_train_acc": mean_train_acc,
                "conv_round_acc>=0.90": conv_round,
                "avg_client_time_s": avg_client_time,
                "avg_server_agg_time_s": avg_server_time,
                "steady_avg_round_time_s_rounds>=2": steady_avg_round_time,
                "steady_avg_agg_time_s_rounds>=2": steady_avg_agg_time,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Week-5] Using device: {device}")
    set_global_seeds(args.seed)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    clients_data = generate_synthetic_telemetry(
        num_clients=args.clients,
        samples_per_client=2000,
        window_size=32,
        anomaly_prob=0.05,
        seed=args.seed,
    )

    mode_order = ["none", "mask", "secagg", "ckks"]

    # Run all modes
    csv_paths: Dict[str, Path] = {}
    for mode in mode_order:
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

    # Load CSVs
    dfs: Dict[str, pd.DataFrame] = {}
    for mode, path in csv_paths.items():
        try:
            dfs[mode] = pd.read_csv(path)
        except Exception as e:
            print(f"[Week-5] WARNING: failed to read {path}: {e}")
            dfs[mode] = pd.DataFrame()

    # Accuracy
    _plot_multi_modes(
        dfs,
        metric_col="acc",
        ylabel="Accuracy",
        title="Week-5: Accuracy vs Rounds (none vs mask vs secagg vs ckks)",
        out_path=results_dir / "week5_acc_four_modes.png",
        order=mode_order,
        overlap_aware=False,
    )

    # Bytes per round (overlap-aware)
    _plot_multi_modes(
        dfs,
        metric_col="bytes_up",
        ylabel="Uplink bytes per round",
        title="Week-5: Uplink Bytes vs Rounds (overlap-aware)",
        out_path=results_dir / "week5_bytes_up_four_modes.png",
        order=mode_order,
        overlap_aware=True,
    )

    # Cumulative bytes (overlap-aware)
    _plot_cumulative_uplink(
        dfs,
        out_path=results_dir / "week5_bytes_up_cumulative_four_modes.png",
        order=mode_order,
    )

    # Round time
    _plot_multi_modes(
        dfs,
        metric_col="round_time",
        ylabel="Round time (s)",
        title="Week-5: Round Time vs Rounds (none vs mask vs secagg vs ckks)",
        out_path=results_dir / "week5_round_time_four_modes.png",
        order=mode_order,
        overlap_aware=False,
    )

    # Steady-state round time (rounds >= 2)
    _plot_multi_modes(
        dfs,
        metric_col="round_time",
        ylabel="Round time (s)",
        title="Week-5: Steady-State Round Time (Rounds >= 2)",
        out_path=results_dir / "week5_round_time_steady_four_modes.png",
        order=mode_order,
        steady_state=True,
        overlap_aware=False,
    )

    # Summary CSV
    summarise_modes(dfs, out_path=results_dir / "week5_summary.csv", order=mode_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
