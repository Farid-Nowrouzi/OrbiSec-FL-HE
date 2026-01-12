# experiments/run_week6_overall_summary.py

"""
Week-6: Overall summary script .

Goal:
- Merge Week-6 metrics (edge profile, timing overhead, membership attack) into:
    * results/week6_overall_summary.csv
    * results/week6_overall_summary.png

So we:
1) Load each CSV
2) Normalize mode names to a canonical set:
      none, mask, ckks_like
   where:
      ckks, ckks_proxy, ckks_like -> ckks_like
3) Handle both membership CSV formats:
      - single-run: auc, attack_acc, mean_train_loss, mean_nontrain_loss
      - multi-trial: auc_mean/auc_std, best_attack_acc_mean/std, etc.
4) Make readable plots (zoom + value labels):
      - Final accuracy (zoomed)
      - Timing overhead vs none (%) centered around 0
      - Membership AUC (zoomed, random baseline at 0.5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CANON_MODES = ["none", "mask", "ckks_like"]

# Known aliases from your project history
MODE_ALIASES: Dict[str, str] = {
    "none": "none",
    "mask": "mask",
    "ckks": "ckks_like",
    "ckks_proxy": "ckks_like",
    "ckks-like": "ckks_like",
    "ckks_lite": "ckks_like",
    "ckks_like": "ckks_like",
}


# ----------------------------
# Helpers
# ----------------------------
def load_csv(path: Path, required_cols: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def normalize_modes(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Normalize df['mode'] using MODE_ALIASES.
    Unknown modes raise an error (so we don’t silently corrupt results).
    """
    if "mode" not in df.columns:
        raise ValueError(f"{name}: missing 'mode' column")

    df = df.copy()
    df["mode"] = df["mode"].astype(str)

    unknown = sorted([m for m in df["mode"].unique() if m not in MODE_ALIASES])
    if unknown:
        raise ValueError(
            f"{name}: found unknown mode(s) {unknown}. "
            f"Add them to MODE_ALIASES."
        )

    df["mode"] = df["mode"].map(MODE_ALIASES)
    return df


def ensure_only_canon_modes(df: pd.DataFrame, name: str) -> None:
    modes = sorted(df["mode"].unique().tolist())
    missing = [m for m in CANON_MODES if m not in modes]
    if missing:
        print(f"[Week-6][WARN] {name} missing canonical modes: {missing}. Found={modes}")


def annotate_bars(ax, bars, fmt: str = "{:.3f}", y_offset: float = 0.0) -> None:
    for b in bars:
        h = float(b.get_height())
        x = b.get_x() + b.get_width() / 2
        ax.text(x, h + y_offset, fmt.format(h), ha="center", va="bottom", fontsize=9)


def compute_overhead_pct(overhead_vs_none: np.ndarray) -> np.ndarray:
    """
    Our timing_overhead.csv might store either:
    - ratio around 1.0 (e.g., 1.025)
    - delta around 0.0 (e.g., 0.025)

    We detect and convert to percentage.
    """
    m = float(np.nanmean(overhead_vs_none))
    if m > 0.5:
        # ratio-like
        return (overhead_vs_none - 1.0) * 100.0
    else:
        # delta-like
        return overhead_vs_none * 100.0


def normalize_membership_schema(mia_df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Return a membership df with canonical columns:
        mode, MIA_auc, MIA_auc_std(optional), MIA_attack_acc, MIA_attack_acc_std(optional),
        mean_train_loss(optional), mean_nontrain_loss(optional)
    """
    mia = mia_df_raw.copy()

    # multi-trial format
    if "auc_mean" in mia.columns:
        rename_map = {
            "auc_mean": "MIA_auc",
            "auc_std": "MIA_auc_std",
            "best_attack_acc_mean": "MIA_attack_acc",
            "best_attack_acc_std": "MIA_attack_acc_std",
            "mean_member_loss_mean": "mean_train_loss",
            "mean_nonmember_loss_mean": "mean_nontrain_loss",
        }
        mia = mia.rename(columns=rename_map)

        required = ["mode", "MIA_auc", "MIA_attack_acc"]
        for c in required:
            if c not in mia.columns:
                raise ValueError(f"Membership CSV missing required col: {c}")

    # single-run format
    else:
        required = ["mode", "auc", "attack_acc"]
        for c in required:
            if c not in mia.columns:
                raise ValueError(
                    "Membership CSV schema not recognized. "
                    "Expected either 'auc_mean' format or 'auc' format."
                )

        mia = mia.rename(
            columns={
                "auc": "MIA_auc",
                "attack_acc": "MIA_attack_acc",
            }
        )

        # If present, keep these too
        # (some earlier versions used mean_train_loss / mean_nontrain_loss)
        # No strict requirement.
    return mia


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("[Week-6] Building overall summary...")

    edge_path = RESULTS_DIR / "week6_edge_profile.csv"
    timing_path = RESULTS_DIR / "week6_timing_overhead.csv"
    mia_path = RESULTS_DIR / "week6_membership_attack.csv"

    # ----------------------------
    # 1) Load + normalize
    # ----------------------------
    edge_df = load_csv(
        edge_path,
        required_cols=[
            "mode", "rounds", "final_acc", "mean_acc",
            "total_bytes_up", "total_bytes_down",
            "avg_bytes_up_per_round", "avg_bytes_down_per_round",
        ],
    )
    timing_df = load_csv(
        timing_path,
        required_cols=[
            "mode", "rounds",
            "mean_round_time_s", "total_round_time_s",
            "overhead_vs_none",
        ],
    )
    mia_df_raw = load_csv(mia_path, required_cols=["mode"])

    # Normalize mode strings
    edge_df = normalize_modes(edge_df, "week6_edge_profile.csv")
    timing_df = normalize_modes(timing_df, "week6_timing_overhead.csv")
    mia_df_raw = normalize_modes(mia_df_raw, "week6_membership_attack.csv")

    ensure_only_canon_modes(edge_df, "edge")
    ensure_only_canon_modes(timing_df, "timing")
    ensure_only_canon_modes(mia_df_raw, "membership")

    # Normalize membership schema
    mia_df = normalize_membership_schema(mia_df_raw)

    # I'm Keeping only the canonical modes (don’t crash if extra appears)
    edge_df = edge_df[edge_df["mode"].isin(CANON_MODES)].copy()
    timing_df = timing_df[timing_df["mode"].isin(CANON_MODES)].copy()
    mia_df = mia_df[mia_df["mode"].isin(CANON_MODES)].copy()

    # Stable order
    for df in (edge_df, timing_df, mia_df):
        df["mode"] = pd.Categorical(df["mode"], categories=CANON_MODES, ordered=True)

    edge_df = edge_df.sort_values("mode").reset_index(drop=True)
    timing_df = timing_df.sort_values("mode").reset_index(drop=True)
    mia_df = mia_df.sort_values("mode").reset_index(drop=True)

    # ----------------------------
    # 2) Merge
    # ----------------------------
    merged = edge_df.merge(
        timing_df,
        on=["mode", "rounds"],
        how="inner",
        suffixes=("", "_timing"),
    )

    # Membership depends only on mode
    keep_cols = ["mode", "MIA_auc", "MIA_attack_acc"]
    for optional in ["MIA_auc_std", "MIA_attack_acc_std", "mean_train_loss", "mean_nontrain_loss"]:
        if optional in mia_df.columns:
            keep_cols.append(optional)

    merged = merged.merge(
        mia_df[keep_cols],
        on="mode",
        how="left",
    )

    # Final column order
    ordered_cols = [
        "mode", "rounds",
        "final_acc", "mean_acc",
        "total_bytes_up", "total_bytes_down",
        "avg_bytes_up_per_round", "avg_bytes_down_per_round",
        "mean_round_time_s", "total_round_time_s", "overhead_vs_none",
        "MIA_auc", "MIA_attack_acc",
    ]
    for extra in ["MIA_auc_std", "MIA_attack_acc_std", "mean_train_loss", "mean_nontrain_loss"]:
        if extra in merged.columns and extra not in ordered_cols:
            ordered_cols.append(extra)

    merged = merged[ordered_cols]

    out_csv = RESULTS_DIR / "week6_overall_summary.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[Week-6] Overall summary CSV written to: {out_csv}")

    # ----------------------------
    # 3) Text summary
    # ----------------------------
    print("\n[Week-6] Overall summary (per mode):\n")
    for _, r in merged.iterrows():
        msg = (
            f"Mode={str(r['mode']):8s} | "
            f"final_acc={float(r['final_acc']):.3f}, mean_acc={float(r['mean_acc']):.3f} | "
            f"up={int(r['total_bytes_up'])}B, down={int(r['total_bytes_down'])}B | "
            f"mean_round_time={float(r['mean_round_time_s']):.3f}s | "
            f"overhead_vs_none={float(r['overhead_vs_none']):.4f} | "
            f"MIA_AUC={float(r['MIA_auc']):.3f}, attack_acc={float(r['MIA_attack_acc']):.3f}"
        )
        print(msg)

    # ----------------------------
    # 4) Plot (readable, report-friendly)
    # ----------------------------
    modes = merged["mode"].astype(str).tolist()
    x = np.arange(len(modes))

    final_acc = merged["final_acc"].to_numpy(dtype=float)
    overhead_vs_none = merged["overhead_vs_none"].to_numpy(dtype=float)
    overhead_pct = compute_overhead_pct(overhead_vs_none)
    mia_auc = merged["MIA_auc"].to_numpy(dtype=float)

    mia_auc_std = None
    if "MIA_auc_std" in merged.columns:
        mia_auc_std = merged["MIA_auc_std"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.2), sharex=True)

    # (a) Final accuracy (zoomed)
    ax = axes[0]
    bars = ax.bar(x, final_acc)
    ax.set_title("Week-6 Overall Summary")
    ax.set_ylabel("Final accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ymin = max(0.0, float(np.min(final_acc) - 0.01))
    ymax = min(1.05, float(np.max(final_acc) + 0.005))
    if (ymax - ymin) < 0.02:
        ymax = ymin + 0.02
    ax.set_ylim(ymin, ymax)
    annotate_bars(ax, bars, fmt="{:.3f}", y_offset=(ymax - ymin) * 0.02)

    # (b) Timing overhead vs none (%), centered around 0
    ax = axes[1]
    bars = ax.bar(x, overhead_pct)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Overhead vs none (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    omax = float(np.max(overhead_pct))
    omin = float(np.min(overhead_pct))
    pad = max(1.0, (omax - omin) * 0.25)
    ax.set_ylim(omin - pad, omax + pad)

    for b in bars:
        h = float(b.get_height())
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + (0.15 if h >= 0 else -0.15),
            f"{h:+.1f}%",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9,
            )

    # (c) Membership AUC (zoomed, random baseline)
    ax = axes[2]
    bars = ax.bar(x, mia_auc, yerr=mia_auc_std, capsize=4 if mia_auc_std is not None else 0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.text(-0.35, 0.501, "random (0.5)", fontsize=8, va="bottom", ha="left")
    ax.set_ylabel("MIA AUC")
    ax.set_xlabel("Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ymin = max(0.45, float(np.min(mia_auc) - 0.02))
    ymax = min(1.05, float(np.max(mia_auc) + 0.02))
    if (ymax - ymin) < 0.06:
        ymax = ymin + 0.06
    ax.set_ylim(ymin, ymax)
    annotate_bars(ax, bars, fmt="{:.3f}", y_offset=(ymax - ymin) * 0.02)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(modes)

    fig.tight_layout()
    out_png = RESULTS_DIR / "week6_overall_summary.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"\n[Week-6] Overall summary figure saved to: {out_png}")
    print("[Week-6] Done.\n")


if __name__ == "__main__":
    main()
