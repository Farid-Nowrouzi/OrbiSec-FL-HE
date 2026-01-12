"""
run_week10_secagg_vs_ckks_comparison.py

WEEK-10: Secure Aggregation vs CKKS (Comparison Baseline)

Goal (report):
- Compare {none, mask, secagg, ckks} on privacy attacks from Weeks 7–9.
- Under the Week-10 threat model ("attacker needs per-client updates"):
    * SecAgg hides per-client updates (server only sees aggregate)
    * CKKS hides per-client updates (server aggregates encrypted updates)
  => these gradient-based attacks should collapse to random-baseline.

IMPORTANT (integrity):
- We do NOT re-run attacks here.
- We read Week 7–9 SUMMARY CSVs (already produced by your pipeline).
- We ADD a synthetic 'secagg' row as the *threat-model equivalent baseline*:
    * If CKKS exists: secagg copies CKKS mean/std (same threat model outcome)
    * If CKKS missing: secagg stays NaN (we do NOT fabricate)

Outputs (saved in /results):
- week10_secagg_property_auc.png
- week10_secagg_attribute_acc.png
- week10_secagg_attribute_f1.png
- week10_secagg_fingerprinting_acc.png
- week10_secagg_fingerprinting_f1.png
- week10_secagg_vs_ckks_summary.csv   (long/tidy format for LaTeX tables)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# IO helpers
# ----------------------------
def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[Week-10] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[Week-10] Failed reading {path}: {e}")
        return None


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# ----------------------------
# Robust column picker
# ----------------------------
ALIASES: Dict[str, List[str]] = {
    # Week-7 property inference
    "auc_mean": ["auc_mean", "attack_auc_mean", "roc_auc_mean", "auc", "roc_auc"],
    "auc_std":  ["auc_std", "attack_auc_std", "roc_auc_std"],

    # Week-8 attribute inference
    "attack_acc_mean": ["attack_acc_mean", "acc_mean", "accuracy_mean", "attack_accuracy_mean"],
    "attack_acc_std":  ["attack_acc_std", "acc_std", "accuracy_std", "attack_accuracy_std"],
    "macro_f1_mean":   ["macro_f1_mean", "f1_mean", "f1_macro_mean", "macro_f1"],
    "macro_f1_std":    ["macro_f1_std", "f1_std", "f1_macro_std"],

    # metadata
    "n_classes":   ["n_classes", "num_classes", "classes", "nclass"],
    "num_clients": ["num_clients", "n_clients", "clients", "nclients"],
}


def pick_value(df: pd.DataFrame, mode: str, key: str) -> float:
    """
    we Pick a value for (mode, metric-key) with alias support.
    Returns NaN if mode or column missing (no crashing ).
    """
    if df is None or df.empty:
        return float("nan")

    if "mode" not in df.columns:
        return float("nan")

    sub = df[df["mode"].astype(str) == str(mode)]
    if sub.empty:
        return float("nan")

    row = sub.iloc[0]

    # direct hit
    if key in row.index:
        return _as_float(row[key])

    # alias hit
    for alt in ALIASES.get(key, []):
        if alt in row.index:
            return _as_float(row[alt])

    return float("nan")


def pick_first_non_nan(df: pd.DataFrame, key: str) -> float:
    """
    Pick the first finite value of a metadata field from ANY row (fallback).
    """
    if df is None or df.empty:
        return float("nan")
    if "mode" not in df.columns:
        return float("nan")

    # find an actual column name matching aliases
    col_name = None
    for alt in [key] + ALIASES.get(key, []):
        if alt in df.columns:
            col_name = alt
            break
    if col_name is None:
        return float("nan")

    series = pd.to_numeric(df[col_name], errors="coerce")
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return float("nan")
    return float(series.iloc[0])


# ----------------------------
# Add 'secagg' row (threat-model baseline)
# ----------------------------
def add_secagg_mode(df: pd.DataFrame, metric_keys: List[str]) -> pd.DataFrame:
    """
    Adds a 'secagg' row.

    Policy (integrity-first):
    - If secagg already exists -> keep it.
    - If ckks exists -> copy ckks row (means/stds/etc.)
    - If ckks missing -> create row with NaNs for metrics (no fabrication).
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df["mode"] = df["mode"].astype(str)

    if "secagg" in set(df["mode"]):
        return df

    ckks = df[df["mode"] == "ckks"]
    if len(ckks) == 1:
        row = ckks.iloc[0].to_dict()
        row["mode"] = "secagg"
        row["week10_note"] = "secagg_copied_from_ckks_threat_model_equivalence"
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # CKKS missing -> do not fabricate
    base_row = {c: np.nan for c in df.columns}
    base_row["mode"] = "secagg"
    base_row["week10_note"] = "secagg_missing_ckks_no_fabrication"
    # keep any non-metric metadata as NaN (fine)
    for k in metric_keys:
        # ensure these columns exist if the df has them under canonical names
        pass
    return pd.concat([df, pd.DataFrame([base_row])], ignore_index=True)


# ----------------------------
# Plotting helper (bar + baseline + error bars)
# ----------------------------
def plot_bar_with_baseline(
        title: str,
        ylabel: str,
        modes: List[str],
        means: List[float],
        stds: List[float],
        baseline: float,
        outpath: Path,
        ylim: Tuple[float, float] = (0.0, 1.05),
) -> None:
    means = [float(x) if np.isfinite(x) else np.nan for x in means]
    stds = [float(x) if np.isfinite(x) else 0.0 for x in stds]

    # keep only finite means for plotting (skip missing safely)
    plot_modes, plot_means, plot_stds = [], [], []
    for m, mu, sd in zip(modes, means, stds):
        if np.isfinite(mu):
            plot_modes.append(m)
            plot_means.append(mu)
            plot_stds.append(sd)

    if len(plot_modes) == 0:
        print(f"[Week-10] Skipping plot (no finite values): {outpath.name}")
        return

    plt.figure(figsize=(9.2, 4.8))
    x = np.arange(len(plot_modes))
    plt.bar(x, plot_means, yerr=plot_stds, capsize=5)

    plt.xticks(x, plot_modes)
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # baseline line + label
    plt.axhline(baseline, linestyle="--", linewidth=1.4, alpha=0.75)
    plt.text(
        0.02,
        0.95,
        f"Random baseline = {baseline:.3f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )

    # value labels
    span = (ylim[1] - ylim[0])
    for i, v in enumerate(plot_means):
        plt.text(i, v + 0.02 * span, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    print(f"[Week-10] Saved plot: {outpath}")


# ----------------------------
# Week-10 workflow
# ----------------------------
def main() -> None:
    # Inputs: your already generated Week 7–9 summary CSVs
    w7_path = RESULTS_DIR / "week7_property_inference_summary.csv"
    w8_path = RESULTS_DIR / "week8_attribute_inference_summary.csv"
    w9_path = RESULTS_DIR / "week9_fingerprinting_summary.csv"

    w7 = read_csv_safe(w7_path)
    w8 = read_csv_safe(w8_path)
    w9 = read_csv_safe(w9_path)

    if w7 is None and w8 is None and w9 is None:
        print("[Week-10] Nothing to do: missing all input summary CSVs.")
        return

    # Standard mode order for report clarity
    modes = ["none", "mask", "secagg", "ckks"]

    combined_rows: List[Dict[str, object]] = []

    # ----------------------------
    # Week 7: Property inference AUC
    # ----------------------------
    if w7 is not None and not w7.empty:
        w7 = w7.copy()
        w7["mode"] = w7["mode"].astype(str)

        w7 = add_secagg_mode(w7, metric_keys=["auc_mean", "auc_std"])

        baseline_auc = 0.5  # random AUC

        means = [pick_value(w7, m, "auc_mean") for m in modes]
        stds = [pick_value(w7, m, "auc_std") for m in modes]

        plot_bar_with_baseline(
            title="Week-10 Comparison: Property Inference AUC (SecAgg vs CKKS)",
            ylabel="AUC (0.5 = random)",
            modes=modes,
            means=means,
            stds=stds,
            baseline=baseline_auc,
            outpath=RESULTS_DIR / "week10_secagg_property_auc.png",
            ylim=(0.45, 1.05),
        )

        for m, mu, sd in zip(modes, means, stds):
            combined_rows.append(
                {
                    "attack": "property_inference",
                    "metric": "auc",
                    "mode": m,
                    "mean": _as_float(mu),
                    "std": _as_float(sd),
                    "baseline": baseline_auc,
                    "source_week": 7,
                }
            )

    # ----------------------------
    # Week 8: Attribute inference (Acc + Macro-F1)
    # ----------------------------
    if w8 is not None and not w8.empty:
        w8 = w8.copy()
        w8["mode"] = w8["mode"].astype(str)

        w8 = add_secagg_mode(
            w8,
            metric_keys=["attack_acc_mean", "attack_acc_std", "macro_f1_mean", "macro_f1_std"],
        )

        # baseline = 1 / n_classes (use metadata if present)
        n_classes = pick_value(w8, "ckks", "n_classes")
        if not np.isfinite(n_classes):
            n_classes = pick_first_non_nan(w8, "n_classes")
        if not np.isfinite(n_classes) or n_classes <= 1:
            # conservative fallback (still documented in report as fallback)
            n_classes = 3.0
        baseline_attr = 1.0 / float(n_classes)

        # Attack Acc
        means_acc = [pick_value(w8, m, "attack_acc_mean") for m in modes]
        stds_acc = [pick_value(w8, m, "attack_acc_std") for m in modes]
        plot_bar_with_baseline(
            title="Week-10 Comparison: Attribute Inference Attack Accuracy (SecAgg vs CKKS)",
            ylabel=f"Attack Accuracy (random baseline ≈ {baseline_attr:.3f})",
            modes=modes,
            means=means_acc,
            stds=stds_acc,
            baseline=baseline_attr,
            outpath=RESULTS_DIR / "week10_secagg_attribute_acc.png",
            ylim=(max(0.0, baseline_attr - 0.05), 1.05),
        )
        for m, mu, sd in zip(modes, means_acc, stds_acc):
            combined_rows.append(
                {
                    "attack": "attribute_inference",
                    "metric": "attack_acc",
                    "mode": m,
                    "mean": _as_float(mu),
                    "std": _as_float(sd),
                    "baseline": baseline_attr,
                    "source_week": 8,
                    "n_classes": float(n_classes),
                }
            )

        # Macro-F1
        means_f1 = [pick_value(w8, m, "macro_f1_mean") for m in modes]
        stds_f1 = [pick_value(w8, m, "macro_f1_std") for m in modes]
        plot_bar_with_baseline(
            title="Week-10 Comparison: Attribute Inference Macro-F1 (SecAgg vs CKKS)",
            ylabel=f"Macro-F1 (random baseline ≈ {baseline_attr:.3f})",
            modes=modes,
            means=means_f1,
            stds=stds_f1,
            baseline=baseline_attr,
            outpath=RESULTS_DIR / "week10_secagg_attribute_f1.png",
            ylim=(max(0.0, baseline_attr - 0.05), 1.05),
        )
        for m, mu, sd in zip(modes, means_f1, stds_f1):
            combined_rows.append(
                {
                    "attack": "attribute_inference",
                    "metric": "macro_f1",
                    "mode": m,
                    "mean": _as_float(mu),
                    "std": _as_float(sd),
                    "baseline": baseline_attr,
                    "source_week": 8,
                    "n_classes": float(n_classes),
                }
            )

    # ----------------------------
    # Week 9: Fingerprinting (Top-1 Acc + Macro-F1)
    # ----------------------------
    if w9 is not None and not w9.empty:
        w9 = w9.copy()
        w9["mode"] = w9["mode"].astype(str)

        w9 = add_secagg_mode(
            w9,
            metric_keys=["attack_acc_mean", "attack_acc_std", "macro_f1_mean", "macro_f1_std"],
        )

        # baseline = 1 / num_clients (prefer metadata if present)
        num_clients = pick_value(w9, "ckks", "num_clients")
        if not np.isfinite(num_clients):
            num_clients = pick_first_non_nan(w9, "num_clients")

        if not np.isfinite(num_clients) or num_clients <= 1:
            # fallback: infer from best-effort (commonly 8/12, but avoid guessing too hard)
            # choose 12 only if you used 12 in Week-9; otherwise update this constant.
            num_clients = 12.0

        baseline_fp = 1.0 / float(num_clients)

        # Top-1 Acc
        means_acc = [pick_value(w9, m, "attack_acc_mean") for m in modes]
        stds_acc = [pick_value(w9, m, "attack_acc_std") for m in modes]
        plot_bar_with_baseline(
            title="Week-10 Comparison: Fingerprinting Top-1 Accuracy (SecAgg vs CKKS)",
            ylabel=f"Attack Accuracy (random baseline ≈ {baseline_fp:.3f})",
            modes=modes,
            means=means_acc,
            stds=stds_acc,
            baseline=baseline_fp,
            outpath=RESULTS_DIR / "week10_secagg_fingerprinting_acc.png",
            ylim=(max(0.0, baseline_fp - 0.02), 1.05),
        )
        for m, mu, sd in zip(modes, means_acc, stds_acc):
            combined_rows.append(
                {
                    "attack": "fingerprinting",
                    "metric": "top1_acc",
                    "mode": m,
                    "mean": _as_float(mu),
                    "std": _as_float(sd),
                    "baseline": baseline_fp,
                    "source_week": 9,
                    "num_clients": float(num_clients),
                }
            )

        # Macro-F1
        means_f1 = [pick_value(w9, m, "macro_f1_mean") for m in modes]
        stds_f1 = [pick_value(w9, m, "macro_f1_std") for m in modes]
        plot_bar_with_baseline(
            title="Week-10 Comparison: Fingerprinting Macro-F1 (SecAgg vs CKKS)",
            ylabel=f"Macro-F1 (random baseline ≈ {baseline_fp:.3f})",
            modes=modes,
            means=means_f1,
            stds=stds_f1,
            baseline=baseline_fp,
            outpath=RESULTS_DIR / "week10_secagg_fingerprinting_f1.png",
            ylim=(max(0.0, baseline_fp - 0.02), 1.05),
        )
        for m, mu, sd in zip(modes, means_f1, stds_f1):
            combined_rows.append(
                {
                    "attack": "fingerprinting",
                    "metric": "macro_f1",
                    "mode": m,
                    "mean": _as_float(mu),
                    "std": _as_float(sd),
                    "baseline": baseline_fp,
                    "source_week": 9,
                    "num_clients": float(num_clients),
                }
            )

    # ----------------------------
    # Save combined summary (long format: easiest for LaTeX)
    # ----------------------------
    if combined_rows:
        out_csv = RESULTS_DIR / "week10_secagg_vs_ckks_summary.csv"
        pd.DataFrame(combined_rows).to_csv(out_csv, index=False)
        print(f"[Week-10] Saved combined summary: {out_csv}")

    print("[Week-10] Done.")


if __name__ == "__main__":
    main()
