# experiments/run_week6_overall_summary.py

"""
Week-6: Overall summary script.

- Loads:
    * week6_edge_profile.csv
    * week6_timing_overhead.csv
    * week6_membership_attack.csv
- Merges them per mode (none / mask / ckks).
- Writes a single CSV: week6_overall_summary.csv
- Prints a human-readable summary.
- Creates a PNG with three bar charts:
    * Final accuracy
    * Timing overhead vs. none
    * Membership AUC (leakage)
"""

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_csv(path: Path, required_cols: List[str]) -> pd.DataFrame:
    """Load a CSV and check that all required columns are present."""
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def main() -> None:
    print("[Week-6] Building overall summary...")

    # ------------------------------------------------------------------
    # 1) Load per-mode CSVs
    # ------------------------------------------------------------------
    edge_path = RESULTS_DIR / "week6_edge_profile.csv"
    timing_path = RESULTS_DIR / "week6_timing_overhead.csv"
    mia_path = RESULTS_DIR / "week6_membership_attack.csv"

    # Edge / communication profile
    edge_df = load_csv(
        edge_path,
        required_cols=[
            "mode",
            "rounds",
            "final_acc",
            "mean_acc",
            "total_bytes_up",
            "total_bytes_down",
            "avg_bytes_up_per_round",
            "avg_bytes_down_per_round",
        ],
    )

    # Timing / overhead
    timing_df = load_csv(
        timing_path,
        required_cols=[
            "mode",
            "rounds",
            "mean_round_time_s",
            "total_round_time_s",
            "overhead_vs_none",
        ],
    )

    # Membership inference attack
    mia_df = load_csv(
        mia_path,
        required_cols=[
            "mode",
            "auc",
            "attack_acc",
            "mean_train_loss",
            "mean_nontrain_loss",
        ],
    ).rename(
        columns={
            "auc": "MIA_auc",
            "attack_acc": "MIA_attack_acc",
        }
    )

    # ------------------------------------------------------------------
    # 2) Merge everything per mode + rounds
    # ------------------------------------------------------------------
    # edge_df and timing_df both have 'mode' and 'rounds'
    merged = edge_df.merge(
        timing_df,
        on=["mode", "rounds"],
        how="inner",
        suffixes=("", "_timing"),
    )

    # MIA only depends on 'mode'
    merged = merged.merge(
        mia_df,
        on="mode",
        how="inner",
    )

    # Reorder columns to something nice
    ordered_cols = [
        "mode",
        "rounds",
        "final_acc",
        "mean_acc",
        "total_bytes_up",
        "total_bytes_down",
        "avg_bytes_up_per_round",
        "avg_bytes_down_per_round",
        "mean_round_time_s",
        "total_round_time_s",
        "overhead_vs_none",
        "MIA_auc",
        "MIA_attack_acc",
        "mean_train_loss",
        "mean_nontrain_loss",
    ]
    merged = merged[ordered_cols]

    # ------------------------------------------------------------------
    # 3) Save CSV
    # ------------------------------------------------------------------
    out_csv = RESULTS_DIR / "week6_overall_summary.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[Week-6] Overall summary CSV written to: {out_csv}")

    # ------------------------------------------------------------------
    # 4) Print a compact textual summary (for the report / debugging)
    # ------------------------------------------------------------------
    print("\n[Week-6] Overall summary (per mode):\n")

    for _, row in merged.iterrows():
        mode = row["mode"]
        msg = (
            f"Mode = {mode:5s} | "
            f"final_acc={row['final_acc']:.3f}, mean_acc={row['mean_acc']:.3f} | "
            f"total_up={int(row['total_bytes_up'])} B, "
            f"total_down={int(row['total_bytes_down'])} B "
            f"(avg_up/round={row['avg_bytes_up_per_round']:.1f}, "
            f"avg_down/round={row['avg_bytes_down_per_round']:.1f}) | "
            f"mean_round_time={row['mean_round_time_s']:.3f}s, "
            f"total_round_time={row['total_round_time_s']:.3f}s, "
            f"overhead_vs_none={row['overhead_vs_none']:.3f} | "
            f"MIA_AUC={row['MIA_auc']:.3f}, "
            f"MIA_attack_acc={row['MIA_attack_acc']:.3f}, "
            f"mean_train_loss={row['mean_train_loss']:.4f}, "
            f"mean_nontrain_loss={row['mean_nontrain_loss']:.4f}"
        )
        print(msg)

    # ------------------------------------------------------------------
    # 5) Create a PNG with three bar charts:
    #    - final accuracy
    #    - timing overhead (overhead_vs_none)
    #    - membership AUC
    # ------------------------------------------------------------------
    modes = merged["mode"].tolist()
    x = range(len(modes))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    # (a) Final accuracy
    axes[0].bar(x, merged["final_acc"])
    axes[0].set_ylabel("Final acc")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Week-6 Overall Summary")

    # (b) Timing overhead vs none
    axes[1].bar(x, merged["overhead_vs_none"])
    axes[1].set_ylabel("Overhead vs none")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].text(
        -0.3,
        1.01,
        "baseline",
        fontsize=8,
        va="bottom",
        ha="left",
    )

    # (c) Membership inference AUC
    axes[2].bar(x, merged["MIA_auc"])
    axes[2].set_ylabel("MIA AUC")
    axes[2].set_ylim(0.45, 1.05)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(modes)
    axes[2].set_xlabel("Mode (none / mask / ckks)")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_png = RESULTS_DIR / "week6_overall_summary.png"
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"\n[Week-6] Overall summary figure saved to: {out_png}")
    print("\n[Week-6] Overall summary complete.\n")


if __name__ == "__main__":
    main()
