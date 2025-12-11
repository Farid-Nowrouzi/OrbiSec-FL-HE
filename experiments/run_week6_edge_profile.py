"""
run_week6_edge_profile.py

Week-6 (A): Edge communication profile for OrbiSec-FL-HE.

Reads per-round logs from:
  - results/results_none.csv
  - results/results_mask.csv
  - results/results_ckks.csv

For each mode, it:
  - Computes uplink/downlink *per active client*.
  - Summarises total and average bytes.
  - Plots uplink/downlink per-client vs rounds.

Outputs:
  - results/week6_uplink_per_client_three_modes.png
  - results/week6_downlink_per_client_three_modes.png
  - results/week6_edge_profile.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = ["none", "mask", "ckks"]


def load_mode_df(results_dir: Path, mode: str) -> pd.DataFrame:
    path = results_dir / f"results_{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    df = pd.read_csv(path)

    # Safety: avoid divide-by-zero if something weird happens
    active = df["active_clients"].clip(lower=1)

    df["mode"] = mode
    df["uplink_per_client"] = df["bytes_up"] / active
    df["downlink_per_client"] = df["bytes_down"] / active
    return df


def make_uplink_plot(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for mode, df in dfs.items():
        plt.plot(
            df["round"],
            df["uplink_per_client"],
            marker="o",
            label=mode,
        )

    plt.xlabel("Round")
    plt.ylabel("Uplink bytes per active client")
    plt.title("Week-6: Uplink bytes per client vs rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Week-6] Uplink plot saved to: {out_path}")


def make_downlink_plot(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for mode, df in dfs.items():
        plt.plot(
            df["round"],
            df["downlink_per_client"],
            marker="o",
            label=mode,
        )

    plt.xlabel("Round")
    plt.ylabel("Downlink bytes per active client")
    plt.title("Week-6: Downlink bytes per client vs rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Week-6] Downlink plot saved to: {out_path}")


def write_edge_summary(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    rows = []
    for mode, df in dfs.items():
        row = {
            "mode": mode,
            "rounds": int(df["round"].max()),
            "final_acc": float(df["acc"].iloc[-1]),
            "mean_acc": float(df["acc"].mean()),
            "total_bytes_up": int(df["bytes_up"].sum()),
            "total_bytes_down": int(df["bytes_down"].sum()),
            "avg_bytes_up_per_round": float(df["bytes_up"].mean()),
            "avg_bytes_down_per_round": float(df["bytes_down"].mean()),
            "avg_uplink_per_client": float(df["uplink_per_client"].mean()),
            "avg_downlink_per_client": float(df["downlink_per_client"].mean()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"[Week-6] Edge profile summary CSV written to: {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load all modes
    dfs = {mode: load_mode_df(results_dir, mode) for mode in MODES}

    # Plots
    make_uplink_plot(
        dfs, results_dir / "week6_uplink_per_client_three_modes.png"
    )
    make_downlink_plot(
        dfs, results_dir / "week6_downlink_per_client_three_modes.png"
    )

    # Summary CSV
    write_edge_summary(dfs, results_dir / "week6_edge_profile.csv")


if __name__ == "__main__":
    main()
