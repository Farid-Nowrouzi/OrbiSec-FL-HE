"""
run_week6_timing_overhead.py

Week-6 (B): Timing / latency overhead for OrbiSec-FL-HE.

Reads per-round logs from:
  - results/results_none.csv
  - results/results_mask.csv
  - results/results_ckks.csv

For each mode, it:
  - Plots per-round 'round_time' curves (server-side round duration).
  - Computes mean and total round_time.
  - Computes overhead vs the 'none' baseline.

Outputs:
  - results/week6_round_time_three_modes.png
  - results/week6_timing_overhead.csv
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
    """Load per-round CSV for a given mode (none/mask/ckks)."""
    path = results_dir / f"results_{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    df = pd.read_csv(path)
    df["mode"] = mode
    return df


def make_round_time_plot(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Plot round_time vs round for all three modes."""
    plt.figure(figsize=(7, 4))
    for mode, df in dfs.items():
        plt.plot(
            df["round"],
            df["round_time"],
            marker="o",
            label=mode,
        )

    plt.xlabel("Round")
    plt.ylabel("Round time (s)")
    plt.title("Week-6: Round time vs rounds (none vs mask vs ckks)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Week-6] Round-time plot saved to: {out_path}")


def write_timing_summary(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Write a small CSV summarising timing overhead vs baseline."""
    base_df = dfs["none"]
    base_mean = float(base_df["round_time"].mean())

    rows = []
    for mode, df in dfs.items():
        mean_rt = float(df["round_time"].mean())
        total_rt = float(df["round_time"].sum())
        overhead_vs_none = mean_rt / base_mean if base_mean > 0 else float("nan")

        row = {
            "mode": mode,
            "rounds": int(df["round"].max()),
            "mean_round_time_s": mean_rt,
            "total_round_time_s": total_rt,
            "overhead_vs_none": overhead_vs_none,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"[Week-6] Timing overhead summary CSV written to: {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load data for all three modes
    dfs: Dict[str, pd.DataFrame] = {
        mode: load_mode_df(results_dir, mode) for mode in MODES
    }

    # Plot round-time curves
    make_round_time_plot(
        dfs, results_dir / "week6_round_time_three_modes.png"
    )

    # Write summary CSV
    write_timing_summary(dfs, results_dir / "week6_timing_overhead.csv")


if __name__ == "__main__":
    main()
