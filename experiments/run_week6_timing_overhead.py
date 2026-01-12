"""
run_week6_timing_overhead.py

Week-6 (B): Timing / latency overhead for OrbiSec-FL-HE.

Reads per-round logs from:
  - results/results_none.csv
  - results/results_mask.csv
  - results/results_ckks_like.csv   (preferred)
    OR results/results_ckks.csv     (fallback)

For each mode, it:
  - Plots per-round 'round_time' curves (server-side round duration).
  - Computes mean and total round_time.
  - Computes overhead vs the 'none' baseline.

Outputs:
  - results/week6_round_time_three_modes.png
  - results/week6_timing_overhead.csv

 (report):
  - results/week6_round_time_overhead_bars.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# We standardize Week-6 naming to match our membership experiment:
#   none, mask, ckks_like
MODES_STD: List[str] = ["none", "mask", "ckks_like"]

# For ckks_like, we will accept either "results_ckks_like.csv" or the older "results_ckks.csv"
MODE_INPUT_CANDIDATES = {
    "none": ["none"],
    "mask": ["mask"],
    "ckks_like": ["ckks_like", "ckks"],
}


def _find_existing_csv(results_dir: Path, candidates: List[str]) -> Tuple[Path, str]:
    """
    Return (path, matched_mode_suffix) for the first existing CSV among candidates.
    Example: candidates=["ckks_like","ckks"] -> picks whichever exists.
    """
    for m in candidates:
        p = results_dir / f"results_{m}.csv"
        if p.exists():
            return p, m
    tried = [str(results_dir / f"results_{m}.csv") for m in candidates]
    raise FileNotFoundError(f"None of the expected CSVs exist. Tried: {tried}")


def load_mode_df(results_dir: Path, mode_std: str) -> pd.DataFrame:
    """
    Load per-round CSV for a given standardized mode name (none/mask/ckks).
    Accepts fallback filenames for ckks_like.
    """
    candidates = MODE_INPUT_CANDIDATES[mode_std]
    path, matched = _find_existing_csv(results_dir, candidates)

    df = pd.read_csv(path)

    # Validate required columns
    required_cols = ["round", "round_time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}. Found: {list(df.columns)}")

    # Standardize mode name for downstream merging (IMPORTANT)
    df["mode"] = mode_std
    df["source_file_mode"] = matched  # for debugging only (won't break merges)

    # Ensure numeric + sorted
    df["round"] = pd.to_numeric(df["round"], errors="raise")
    df["round_time"] = pd.to_numeric(df["round_time"], errors="raise")
    df = df.sort_values("round").reset_index(drop=True)

    return df


def make_round_time_plot(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Plot round_time vs round for all modes (line plot)."""
    plt.figure(figsize=(8, 4.5))

    for mode in MODES_STD:
        df = dfs[mode]
        plt.plot(
            df["round"],
            df["round_time"],
            marker="o",
            linewidth=2,
            label=mode,
        )

    plt.xlabel("Round")
    plt.ylabel("Round time (seconds)")
    plt.title("Week-6: Round time per round (none vs mask vs ckks)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Week-6] Round-time plot saved to: {out_path}")


def make_overhead_bar_plot(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Report-friendly bar plot:
    - mean round time (s)
    - overhead % vs none
    """
    # Ensure mode order
    summary = summary.set_index("mode").loc[MODES_STD].reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    x = range(len(summary))

    # (1) Mean round time
    axes[0].bar(x, summary["mean_round_time_s"])
    axes[0].set_ylabel("Mean round time (s)")
    axes[0].set_title("Week-6 Timing Overhead Summary")

    for i, v in enumerate(summary["mean_round_time_s"].tolist()):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # (2) Overhead percent vs none
    axes[1].bar(x, summary["overhead_pct_vs_none"])
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Overhead vs none (%)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(summary["mode"].tolist())

    for i, v in enumerate(summary["overhead_pct_vs_none"].tolist()):
        axes[1].text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Week-6] Overhead bar plot saved to: {out_path}")


def write_timing_summary(dfs: Dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    """
    Write a small CSV summarising timing overhead vs baseline.
    Returns the summary df for plotting.
    """
    base_df = dfs["none"]
    base_mean = float(base_df["round_time"].mean())

    rows = []
    for mode in MODES_STD:
        df = dfs[mode]

        mean_rt = float(df["round_time"].mean())
        total_rt = float(df["round_time"].sum())

        # IMPORTANT: rounds should represent number of completed rounds, not max round index
        rounds_n = int(df["round"].nunique())

        overhead_ratio = (mean_rt / base_mean) if base_mean > 0 else float("nan")
        overhead_pct = (overhead_ratio - 1.0) * 100.0 if base_mean > 0 else float("nan")

        row = {
            "mode": mode,
            "rounds": rounds_n,
            "mean_round_time_s": mean_rt,
            "total_round_time_s": total_rt,
            "overhead_vs_none": overhead_ratio,          # kept for compatibility with our overall summary script
            "overhead_pct_vs_none": overhead_pct,        # extra, easier to read in report
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"[Week-6] Timing overhead summary CSV written to: {out_path}")

    # Print compact view (nice for debugging)
    for _, r in summary.iterrows():
        print(
            f"[Week-6] mode={r['mode']:8s} rounds={int(r['rounds'])} "
            f"mean={r['mean_round_time_s']:.4f}s total={r['total_round_time_s']:.2f}s "
            f"overhead={r['overhead_vs_none']:.3f} ({r['overhead_pct_vs_none']:+.1f}%)"
        )

    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load data for all standardized modes
    dfs: Dict[str, pd.DataFrame] = {mode: load_mode_df(results_dir, mode) for mode in MODES_STD}

    # Plot per-round timing curves
    make_round_time_plot(dfs, results_dir / "week6_round_time_three_modes.png")

    # Write summary CSV (+ return summary df)
    summary = write_timing_summary(dfs, results_dir / "week6_timing_overhead.csv")

    # Extra: report-friendly bar plot
    make_overhead_bar_plot(summary, results_dir / "week6_round_time_overhead_bars.png")


if __name__ == "__main__":
    main()
