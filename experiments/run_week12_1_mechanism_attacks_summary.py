"""
run_week12_1_mechanism_attacks_summary.py

Week-12.1: Mechanism Attacks — Synthesis Summary (Week-4 DLG + Week-6 MIA + Overhead)

What it does :
- Loads:
  - Week-4 DLG metrics   (results/week4_dlg_results.csv)      [if present]
  - Week-6 MIA summary   (results/week6_membership_attack.csv) [required]
  - Week-6 timing        (results/week6_timing_overhead.csv)   [required]
- Produces a clean summary table:
  mode_display, mode, dlg_metric_name, dlg_metric_value, dlg_note, mia_auc, mia_auc_std, overhead_pct, mean_round_time_s
- Generates ONE overview figure with 3 panels:
  (1) DLG metric bar (CKKS shown as "Blocked (ciphertexts only)")
  (2) Timing overhead vs none (%)
  (3) MIA AUC (with random baseline 0.5)

Outputs:
- results/week12_1_mechanism_attacks_summary.csv
- results/week12_1_mechanism_overview.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Naming / display conventions
# -----------------------------
# Internal standardized keys we use in code
MODES_STD = ["none", "mask", "ckks_like"]

# Display aliases for the report (IMPORTANT: show CKKS, not CKKS-like)
MODE_DISPLAY = {
    "none": "none",
    "mask": "mask",
    "ckks_like": "CKKS",
    "ckks": "CKKS",
}

# For CKKS, DLG is not observable under ciphertext-only channel
CKKS_DLG_NOTE = "Not observable under ciphertext-only channel (CKKS)"


# -----------------------------
# File helpers
# -----------------------------
def _first_existing(results_dir: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    for name in candidates:
        p = results_dir / name
        if p.exists():
            return p
    return None


def _normalize_mode(x: str) -> str:
    """Map file modes to our standard modes."""
    s = str(x).strip().lower()
    if s in ("none",):
        return "none"
    if s in ("mask",):
        return "mask"
    if s in ("ckks_like", "ckks"):
        return "ckks_like"
    return s


# -----------------------------
# Load Week-6 MIA summary
# -----------------------------
def load_week6_mia(results_dir: Path) -> pd.DataFrame:
    """
    Expected columns (your Week-6 membership script typically produces):
    - mode
    - attack_auc_mean or mia_auc_mean or auc_mean
    - attack_auc_std  or mia_auc_std  or auc_std
    We will robustly detect.
    """
    p = _first_existing(results_dir, ("week6_membership_attack.csv",))
    if p is None:
        raise FileNotFoundError("Missing required file: results/week6_membership_attack.csv")

    df = pd.read_csv(p)
    if "mode" not in df.columns:
        raise ValueError(f"{p} missing 'mode' column. Found columns: {list(df.columns)}")

    df["mode"] = df["mode"].map(_normalize_mode)

    # Robust column detection
    col_mean = None
    col_std = None
    for cand in ("attack_auc_mean", "mia_auc_mean", "auc_mean", "auc"):
        if cand in df.columns:
            col_mean = cand
            break
    for cand in ("attack_auc_std", "mia_auc_std", "auc_std"):
        if cand in df.columns:
            col_std = cand
            break

    if col_mean is None:
        raise ValueError(
            f"{p} does not include an AUC mean column. "
            f"Tried: attack_auc_mean/mia_auc_mean/auc_mean/auc. Found: {list(df.columns)}"
        )

    # If std missing, set to 0 (still runnable)
    if col_std is None:
        df["mia_auc_std"] = 0.0
    else:
        df["mia_auc_std"] = pd.to_numeric(df[col_std], errors="coerce").fillna(0.0)

    df["mia_auc"] = pd.to_numeric(df[col_mean], errors="coerce")
    return df[["mode", "mia_auc", "mia_auc_std"]]


# -----------------------------
# Load Week-6 Timing overhead
# -----------------------------
def load_week6_timing(results_dir: Path) -> pd.DataFrame:
    p = _first_existing(results_dir, ("week6_timing_overhead.csv",))
    if p is None:
        raise FileNotFoundError("Missing required file: results/week6_timing_overhead.csv")

    df = pd.read_csv(p)
    if "mode" not in df.columns:
        raise ValueError(f"{p} missing 'mode' column. Found columns: {list(df.columns)}")

    df["mode"] = df["mode"].map(_normalize_mode)

    required = ["mean_round_time_s", "overhead_pct_vs_none"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns {missing}. Found: {list(df.columns)}")

    df["mean_round_time_s"] = pd.to_numeric(df["mean_round_time_s"], errors="coerce")
    df["overhead_pct"] = pd.to_numeric(df["overhead_pct_vs_none"], errors="coerce")
    return df[["mode", "mean_round_time_s", "overhead_pct"]]


# -----------------------------
# Load Week-4 DLG metrics (optional)
# -----------------------------
def load_week4_dlg(results_dir: Path) -> Tuple[Optional[str], pd.DataFrame]:
    """
    Optional input: results/week4_dlg_results.csv

    We try to produce a single DLG metric per mode.
    Priority:
      1) psnr (higher better)
      2) ssim (higher better)
      3) mse  (lower better -> convert to -mse as a score)
    """
    p = _first_existing(results_dir, ("week4_dlg_results.csv",))
    if p is None:
        return None, pd.DataFrame(columns=["mode", "dlg_metric_name", "dlg_metric_value"])

    df = pd.read_csv(p)
    if "mode" not in df.columns:
        raise ValueError(f"{p} missing 'mode' column. Found: {list(df.columns)}")

    df["mode"] = df["mode"].map(_normalize_mode)

    # pick metric
    metric_name = None
    if "psnr" in df.columns:
        metric_name = "psnr"
        values = pd.to_numeric(df["psnr"], errors="coerce")
    elif "ssim" in df.columns:
        metric_name = "ssim"
        values = pd.to_numeric(df["ssim"], errors="coerce")
    elif "mse" in df.columns:
        metric_name = "neg_mse"
        values = -pd.to_numeric(df["mse"], errors="coerce")
    else:
        # If the file exists but doesn't have expected columns, just skip gracefully
        return None, pd.DataFrame(columns=["mode", "dlg_metric_name", "dlg_metric_value"])

    out = df.copy()
    out["dlg_metric_name"] = metric_name
    out["dlg_metric_value"] = values

    # If there are multiple rows per mode, take mean (stable summary)
    out = (
        out.groupby(["mode", "dlg_metric_name"], as_index=False)["dlg_metric_value"]
        .mean()
    )
    return metric_name, out[["mode", "dlg_metric_name", "dlg_metric_value"]]


# -----------------------------
# Build merged summary
# -----------------------------
def build_summary(results_dir: Path) -> pd.DataFrame:
    mia = load_week6_mia(results_dir)
    timing = load_week6_timing(results_dir)
    dlg_metric_name, dlg = load_week4_dlg(results_dir)

    merged = pd.DataFrame({"mode": MODES_STD})
    merged = merged.merge(mia, on="mode", how="left")
    merged = merged.merge(timing, on="mode", how="left")

    # DLG is optional
    if not dlg.empty:
        merged = merged.merge(dlg, on="mode", how="left")
    else:
        merged["dlg_metric_name"] = dlg_metric_name if dlg_metric_name else "psnr"
        merged["dlg_metric_value"] = np.nan

    # Human readable display name
    merged["mode_display"] = merged["mode"].map(lambda m: MODE_DISPLAY.get(m, m))

    # Replace CKKS DLG display with a note (keep numeric NaN as-is)
    merged["dlg_note"] = ""
    merged.loc[merged["mode"] == "ckks_like", "dlg_note"] = CKKS_DLG_NOTE

    # If dlg metric exists for ckks_like due to any accidental entry, force it to NaN (threat model)
    merged.loc[merged["mode"] == "ckks_like", "dlg_metric_value"] = np.nan

    # Ensure numeric
    for c in ("mia_auc", "mia_auc_std", "mean_round_time_s", "overhead_pct", "dlg_metric_value"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Keep column order clean
    cols = [
        "mode_display",
        "mode",
        "dlg_metric_name",
        "dlg_metric_value",
        "dlg_note",
        "mia_auc",
        "mia_auc_std",
        "overhead_pct",
        "mean_round_time_s",
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[cols]

    return merged


# -----------------------------
# Plotting (single overview figure)
# -----------------------------
def plot_overview(summary: pd.DataFrame, out_path: Path) -> None:
    # Ensure correct order
    summary = summary.set_index("mode").loc[MODES_STD].reset_index()

    fig = plt.figure(figsize=(9, 8))

    # 1) DLG metric
    ax1 = plt.subplot(3, 1, 1)
    x = np.arange(len(summary))
    y = summary["dlg_metric_value"].values

    # Plot bars for available numeric values
    ax1.bar(x, np.nan_to_num(y, nan=0.0))
    ax1.set_title("Week-12.1 Mechanism Attacks — Synthesis Overview", fontsize=12, weight="bold")
    metric_name = str(summary["dlg_metric_name"].iloc[0]) if "dlg_metric_name" in summary.columns else "DLG metric"
    ax1.set_ylabel(f"DLG metric ({metric_name})")

    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["mode_display"].tolist())

    # Annotate
    for i, (mode, val) in enumerate(zip(summary["mode"].tolist(), y)):
        if mode == "ckks_like":
            ax1.text(i, 0.02, "Blocked\n(ciphertexts only)", ha="center", va="bottom", fontsize=9)
        else:
            if np.isfinite(val):
                ax1.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # 2) Timing overhead
    ax2 = plt.subplot(3, 1, 2)
    overhead = summary["overhead_pct"].values
    ax2.bar(x, np.nan_to_num(overhead, nan=0.0))
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_ylabel("Timing overhead vs none (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary["mode_display"].tolist())

    for i, v in enumerate(overhead):
        if np.isfinite(v):
            ax2.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # 3) MIA AUC
    ax3 = plt.subplot(3, 1, 3)
    auc = summary["mia_auc"].values
    auc_std = summary["mia_auc_std"].values
    ax3.bar(x, np.nan_to_num(auc, nan=0.0), yerr=np.nan_to_num(auc_std, nan=0.0), capsize=4)
    ax3.axhline(0.5, linestyle="--", linewidth=1)
    ax3.set_ylabel("Membership inference AUC")
    ax3.set_xlabel("Mode")
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary["mode_display"].tolist())

    for i, v in enumerate(auc):
        if np.isfinite(v):
            ax3.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    out_csv = results_dir / "week12_1_mechanism_attacks_summary.csv"
    out_png = results_dir / "week12_1_mechanism_overview.png"

    print("[Week-12.1] Building mechanism-attacks synthesis summary...")
    print(f"[Week-12.1] Results dir: {results_dir}")

    summary = build_summary(results_dir)
    summary.to_csv(out_csv, index=False)

    print(f"[Week-12.1] Wrote: {out_csv}")
    print("\n[Week-12.1] Summary table (preview):")
    show_cols = [
        "mode_display",
        "dlg_metric_name",
        "dlg_metric_value",
        "dlg_note",
        "mia_auc",
        "mia_auc_std",
        "overhead_pct",
        "mean_round_time_s",
    ]
    print(summary[show_cols].to_string(index=False))

    plot_overview(summary, out_png)
    print(f"[Week-12.1] Saved: {out_png}")

    print("\n[Week-12.1] Done.")


if __name__ == "__main__":
    main()
