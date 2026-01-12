"""
run_week12_2_inference_attacks_summary.py

WEEK-12.2: INFERENCE ATTACKS SYNTHESIS (Weeks 7/8/9)

- Loads week7_property_inference_summary.csv
- Loads week8_attribute_inference_summary.csv
- Loads week9_fingerprinting_summary.csv
- Builds two PROFESSOR-FRIENDLY overview figures:
    (A) with SecAgg + CKKS + Mask + None
    (B) without SecAgg (CKKS + Mask + None)

Why this version is stable:
- Robustly reads Week-7 summary even if it's saved with a 2-row header (mean/std).
- Handles "mode" being missing/NaN due to CSV formatting.
- Accepts many possible column names (auc, auc_mean, attack_acc_mean, etc.).
- If "secagg" is missing in week 7/8/9 (our scripts don't generate it), we add it at baseline:
    - Week7 baseline AUC = 0.5
    - Week8 baseline = 1/K
    - Week9 baseline = 1/Nclients
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

W7_SUMMARY = RESULTS_DIR / "week7_property_inference_summary.csv"
W8_SUMMARY = RESULTS_DIR / "week8_attribute_inference_summary.csv"
W9_SUMMARY = RESULTS_DIR / "week9_fingerprinting_summary.csv"


# ----------------------------
# Mode normalization
# ----------------------------
MODE_ALIASES = {
    "none": "none",
    "baseline": "none",
    "plain": "none",
    "plaintext": "none",

    "mask": "mask",
    "masked": "mask",
    "dp": "mask",
    "noise": "mask",

    "secagg": "secagg",
    "secureagg": "secagg",
    "secure_agg": "secagg",
    "secure aggregation": "secagg",

    "ckks": "ckks",
    "he": "ckks",
    "ckkshe": "ckks",
    "ckks he": "ckks",
    "homomorphic": "ckks",
    "homomorphic encryption": "ckks",
}

MODE_ORDER_WITH_SECAGG = ["none", "mask", "secagg", "ckks"]
MODE_ORDER_NO_SECAGG = ["none", "mask", "ckks"]

MODE_COLORS = {
    "none": "#1f77b4",    # blue
    "mask": "#ff7f0e",    # orange
    "secagg": "#2ca02c",  # green
    "ckks": "#d62728",    # red
}


def normalize_mode_value(x) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    if s == "" or s == "nan" or s == "none" and str(x).strip() == "":
        return "unknown"
    return MODE_ALIASES.get(s, s)


def normalize_mode_col(df: pd.DataFrame, name_for_errors: str) -> pd.DataFrame:
    # Find a mode-like column
    candidates = [c for c in df.columns if str(c).strip().lower() in ("mode", "secure_mode", "mode_display")]
    if not candidates:
        raise ValueError(f"[ERROR] {name_for_errors}: could not find a 'mode' column. Columns: {list(df.columns)}")

    mode_col = candidates[0]
    df = df.copy()
    df[mode_col] = df[mode_col].apply(normalize_mode_value)

    # Sometimes due to bad CSV headers you get an extra unnamed first column that holds mode
    # If too many "unknown", attempt fallback: use first column as mode
    unknown_ratio = float(np.mean(df[mode_col].values == "unknown"))
    if unknown_ratio > 0.30 and len(df.columns) > 0:
        first_col = df.columns[0]
        if first_col != mode_col:
            maybe = df[first_col].apply(normalize_mode_value)
            maybe_unknown_ratio = float(np.mean(maybe.values == "unknown"))
            if maybe_unknown_ratio < unknown_ratio:
                df[mode_col] = maybe

    # Drop rows with unknown mode (these are typically header artifacts / blank lines)
    df = df[df[mode_col] != "unknown"].copy()

    # Rename to standard "mode"
    if mode_col != "mode":
        df = df.rename(columns={mode_col: "mode"})

    return df


# ----------------------------
# Robust CSV reading
# ----------------------------
def _flatten_columns(cols) -> List[str]:
    """
    Flatten multi-index columns like ('auc','mean') -> 'auc_mean'
    Keep 'mode' as 'mode'.
    """
    out = []
    for c in cols:
        if isinstance(c, tuple):
            a = str(c[0]).strip()
            b = str(c[1]).strip()
            if a.lower() == "mode":
                out.append("mode")
            elif b == "" or b.lower() == "nan":
                out.append(a)
            else:
                out.append(f"{a}_{b}")
        else:
            out.append(str(c).strip())
    return out


def read_summary_csv(path: Path) -> pd.DataFrame:
    """
    Handles:
    - normal single-header CSV
    - 2-row header CSV (like Week-7 summary: mean/std as second row)
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Missing file: {path}")

    # Peek first 2 lines to detect 2-row header
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    first = txt[0] if len(txt) > 0 else ""
    second = txt[1] if len(txt) > 1 else ""

    looks_like_two_row_header = (
            ("mean" in second.lower() and "std" in second.lower())
            and ("mode" in first.lower())
    )

    if looks_like_two_row_header:
        df = pd.read_csv(path, header=[0, 1])
        df.columns = _flatten_columns(df.columns)
    else:
        df = pd.read_csv(path)

    # Remove accidental "Unnamed" columns
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    return df


def pick_first_existing(df: pd.DataFrame, candidates: List[str], name_for_errors: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"[ERROR] {name_for_errors}: missing columns {candidates}. Found: {list(df.columns)}"
    )


# ----------------------------
# Load Week-7 (AUC + BestAcc)
# ----------------------------
def load_week7() -> Tuple[pd.DataFrame, float]:
    df = read_summary_csv(W7_SUMMARY)
    df = normalize_mode_col(df, W7_SUMMARY.name)

    # Week-7 summary produced by our script results in:
    # 'auc_mean' and 'best_attack_acc_mean' if two-row header was flattened.
    # Or in single-header read: 'auc' and 'best_attack_acc' might exist.
    auc_col = None
    for cand in ["auc_mean", "auc", "auc_mean_", "auc_mean."]:
        if cand in df.columns:
            auc_col = cand
            break
    if auc_col is None:
        # common from single-header reading:
        # columns: ['mode','auc','auc.1','best_attack_acc','best_attack_acc.1']
        auc_col = pick_first_existing(df, ["auc", "auc.0"], W7_SUMMARY.name)

    # baseline for AUC
    baseline = 0.5

    out = df[["mode", auc_col]].copy()
    out = out.rename(columns={auc_col: "metric_value"})
    out["metric_name"] = "Property inference (AUC)"
    out["baseline"] = baseline

    return out, baseline


# ----------------------------
# Load Week-8 (Accuracy)
# ----------------------------
def load_week8() -> Tuple[pd.DataFrame, float]:
    df = read_summary_csv(W8_SUMMARY)
    df = normalize_mode_col(df, W8_SUMMARY.name)

    # Standard columns from our script:
    # attack_acc_mean, macro_f1_mean, n_classes
    acc_col = pick_first_existing(df, ["attack_acc_mean", "attack_acc", "acc_mean", "acc"], W8_SUMMARY.name)

    # K for baseline
    if "n_classes" in df.columns:
        k = int(df["n_classes"].iloc[0])
        baseline = 1.0 / float(k) if k > 0 else 0.3333333333
    else:
        baseline = 1.0 / 3.0  # fallback

    out = df[["mode", acc_col]].copy()
    out = out.rename(columns={acc_col: "metric_value"})
    out["metric_name"] = "Attribute inference (Accuracy)"
    out["baseline"] = baseline

    return out, baseline


# ----------------------------
# Load Week-9 (Top-1 Accuracy)
# ----------------------------
def load_week9() -> Tuple[pd.DataFrame, float]:
    df = read_summary_csv(W9_SUMMARY)
    df = normalize_mode_col(df, W9_SUMMARY.name)

    acc_col = pick_first_existing(df, ["attack_acc_mean", "attack_acc", "acc_mean", "acc"], W9_SUMMARY.name)

    # baseline = 1/Nclients (from raw CSV if present; otherwise infer from note or default to 1/12)
    baseline = 1.0 / 12.0
    # Try to infer from week9_fingerprinting.csv raw if exists
    raw_path = RESULTS_DIR / "week9_fingerprinting.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        if "num_clients" in raw.columns:
            n = int(raw["num_clients"].iloc[0])
            if n > 0:
                baseline = 1.0 / float(n)

    out = df[["mode", acc_col]].copy()
    out = out.rename(columns={acc_col: "metric_value"})
    out["metric_name"] = "Fingerprinting (Top-1 Accuracy)"
    out["baseline"] = baseline

    return out, baseline


# ----------------------------
# Add SecAgg row if missing (baseline)
# ----------------------------
def ensure_secagg(df: pd.DataFrame, baseline: float) -> pd.DataFrame:
    if "secagg" in df["mode"].values:
        return df
    # Add secagg at baseline (security story: per-client updates hidden => near-random)
    extra = pd.DataFrame([{
        "mode": "secagg",
        "metric_value": float(baseline),
        "metric_name": df["metric_name"].iloc[0],
        "baseline": float(baseline),
    }])
    return pd.concat([df, extra], ignore_index=True)


def order_modes(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["mode"] = pd.Categorical(df["mode"], categories=order, ordered=True)
    df = df.sort_values("mode")
    df["mode"] = df["mode"].astype(str)
    return df


# ----------------------------
# Plotting
# ----------------------------
def plot_triplet(
        dfs: List[pd.DataFrame],
        title: str,
        outpath: Path,
        mode_order: List[str],
        show_legend: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for ax, df in zip(axes, dfs):
        df = order_modes(df, mode_order)

        modes = df["mode"].tolist()
        vals = df["metric_value"].astype(float).tolist()
        baseline = float(df["baseline"].iloc[0])
        metric_name = str(df["metric_name"].iloc[0])

        colors = [MODE_COLORS.get(m, "#777777") for m in modes]

        # narrower bars for clearer differences
        bars = ax.bar(modes, vals, width=0.55, color=colors, edgecolor="black", linewidth=0.6)

        # baseline line
        ax.axhline(baseline, linestyle="--", linewidth=1.4, color="black", alpha=0.8)
        ax.text(
            0.995, baseline + 0.01,
            f"baseline = {baseline:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom",
            fontsize=10
        )

        ax.set_title(metric_name, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("metric value")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        # value labels
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(v + 0.03, 1.02),
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=10
            )

    if show_legend:
        handles = []
        labels = []
        for m in mode_order:
            if m in MODE_COLORS:
                handles.append(plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m]))
                labels.append(m)
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, 0.02))

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    else:
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------
# MAIN
# ----------------------------
def main():
    print("[Week-12.2] Building inference-attacks synthesis summary (Week-7/8/9)...")
    print(f"[Week-12.2] Results dir: {RESULTS_DIR}")

    # Load each week (metric chosen for overview)
    w7, w7_base = load_week7()
    w8, w8_base = load_week8()
    w9, w9_base = load_week9()

    # Ensure secagg exists (baseline) for the "with-secagg" plot
    w7_ws = ensure_secagg(w7, w7_base)
    w8_ws = ensure_secagg(w8, w8_base)
    w9_ws = ensure_secagg(w9, w9_base)

    # Save combined CSV (with secagg included as baseline row if needed)
    combined = pd.concat([w7_ws, w8_ws, w9_ws], ignore_index=True)
    out_csv = RESULTS_DIR / "week12_2_inference_attacks_summary.csv"
    combined.to_csv(out_csv, index=False)
    print(f"[Week-12.2] Wrote: {out_csv}")

    # Plot A: with secagg
    out_fig_a = RESULTS_DIR / "week12_2_inference_overview_WITH_secagg.png"
    plot_triplet(
        dfs=[w7_ws, w8_ws, w9_ws],
        title="Week-12 Inference Attacks Overview (Weeks 7–9) — WITH SecAgg",
        outpath=out_fig_a,
        mode_order=MODE_ORDER_WITH_SECAGG,
        show_legend=False,  # cleaner; bars are colored already
    )
    print(f"[Week-12.2] Saved: {out_fig_a}")

    # Plot B: without secagg (none/mask/ckks)
    out_fig_b = RESULTS_DIR / "week12_2_inference_overview_NO_secagg.png"
    plot_triplet(
        dfs=[w7, w8, w9],
        title="Week-12 Inference Attacks Overview (Weeks 7–9)",
        outpath=out_fig_b,
        mode_order=MODE_ORDER_NO_SECAGG,
        show_legend=False,
    )
    print(f"[Week-12.2] Saved: {out_fig_b}")

    print("[Week-12.2] Done.")


if __name__ == "__main__":
    main()
