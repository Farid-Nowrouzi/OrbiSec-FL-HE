# experiments/run_week11_2_analyze_ckks_optimizations.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config (visual + behavior)
# =========================
RESULTS_DIR = "./results"

ROUND_CSV_DEFAULT = os.path.join(RESULTS_DIR, "week11_ckks_optimizations_roundstats.csv")
SUMMARY_CSV_OUT = os.path.join(RESULTS_DIR, "week11_ckks_optimizations_summary.csv")

# Clean visuals by default
SHOW_ERROR_BARS = False   # set True if we want std error bars
DPI = 220

# If deltas are basically 0, we show an explanatory "constant comm" plot instead of empty delta charts
DELTA_EPS_BYTES = 1.0          # treat <= 1 byte difference as "same"
DELTA_EPS_PERCENT = 1e-6       # treat tiny percent as same


# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates):
    """Pick the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist: {tuple(candidates)}\nAvailable columns: {list(df.columns)}")


def fmt_bytes(n: float) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "NA"
    n = float(n)
    units = ["B", "KB", "MB", "GB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{n:.0f} {units[i]}"
    return f"{n:.2f} {units[i]}"


def fmt_seconds(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.3f}s"


def fmt_float(x: float, digits=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{digits}f}"


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def barplot_clean(
        labels,
        values,
        title,
        ylabel,
        outpath,
        y_zoom=None,
        annotate_fmt=None,
        y_is_lower_better=False,
        stds=None
):
    """
    Clean bar plot.
    - y_zoom: tuple(min,max) to zoom y axis
    - annotate_fmt: function(v)->str
    - stds: list/array of std values (only drawn if SHOW_ERROR_BARS=True)
    """
    x = np.arange(len(labels))
    fig_w = max(10, 1.6 * len(labels))
    plt.figure(figsize=(fig_w, 5.5))

    if SHOW_ERROR_BARS and stds is not None:
        plt.bar(x, values, yerr=stds, capsize=6)
        plt.figtext(0.99, 0.01, "Error bars = std across FL rounds", ha="right", fontsize=9)
    else:
        plt.bar(x, values)

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(x, labels, rotation=15, ha="right")

    plt.grid(axis="y", alpha=0.25)

    # Zoom logic for “meaningful story”
    if y_zoom is not None:
        plt.ylim(y_zoom[0], y_zoom[1])
    else:
        # add a nice margin automatically
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-12):
            # flat chart -> small symmetric padding so it doesn't look broken
            pad = 0.05 * (abs(vmax) if abs(vmax) > 1e-12 else 1.0)
            plt.ylim(vmin - pad, vmax + pad)
        else:
            pad = 0.08 * (vmax - vmin)
            plt.ylim(vmin - pad, vmax + pad)

    # annotate values
    if annotate_fmt is not None:
        for i, v in enumerate(values):
            plt.text(i, v, annotate_fmt(v), ha="center", va="bottom", fontsize=10)

    # little hint about “direction”
    if y_is_lower_better:
        plt.figtext(0.01, 0.01, "Lower is better", ha="left", fontsize=9)

    savefig(outpath)


def constant_metric_explain_plot(labels, values, title, ylabel, outpath, explain_text):
    """
    When a metric is constant across variants (e.g., CKKS bytes),
    produce a nice plot that looks intentional and tells the story.
    """
    x = np.arange(len(labels))
    fig_w = max(10, 1.6 * len(labels))
    plt.figure(figsize=(fig_w, 5.5))

    plt.bar(x, values)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)

    # small padding so bars aren't glued to the top
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = 0.08 * (vmax if vmax != 0 else 1.0)
    plt.ylim(0, vmax + pad)

    # annotate bars (bytes)
    for i, v in enumerate(values):
        plt.text(i, v, fmt_bytes(v), ha="center", va="bottom", fontsize=10)

    # big explanation box
    plt.figtext(
        0.5, -0.12,
        explain_text,
        ha="center",
        fontsize=10
    )

    savefig(outpath)


def make_tradeoff_scatter(xvals, yvals, labels, title, xlabel, ylabel, outpath):
    plt.figure(figsize=(10.5, 6))
    plt.scatter(xvals, yvals)

    for x, y, lab in zip(xvals, yvals, labels):
        plt.text(x, y, f"  {lab}", fontsize=10, va="bottom")

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(alpha=0.25)
    savefig(outpath)


# =========================
# Main analysis
# =========================
def main():
    ensure_dir(RESULTS_DIR)

    if not os.path.exists(ROUND_CSV_DEFAULT):
        raise FileNotFoundError(
            f"Could not find round CSV: {ROUND_CSV_DEFAULT}\n"
            f"Check that the file exists in ./results."
        )

    df = pd.read_csv(ROUND_CSV_DEFAULT)

    # Accept your actual schema (you had 'variant', not 'mode')
    col_variant = pick_col(df, ["variant", "mode", "setting", "run", "name"])
    col_acc = pick_col(df, ["acc", "accuracy", "test_acc", "val_acc"])
    col_up = pick_col(df, ["bytes_up", "uplink_bytes", "client_bytes_up", "avg_client_bytes_up"])
    col_down = pick_col(df, ["bytes_down", "downlink_bytes", "server_bytes_down", "avg_server_bytes_down"])
    col_rt = pick_col(df, ["round_time", "round_time_sec", "sec_round", "time_round"])

    # Group
    g = df.groupby(col_variant, sort=False)

    summary = pd.DataFrame({
        "variant": g.size().index,
        "n_rounds": g.size().values,
        "mean_acc": g[col_acc].mean().values,
        "std_acc": g[col_acc].std(ddof=0).values,
        "final_acc": g[col_acc].last().values,  # last round
        "mean_uplink_bytes": g[col_up].mean().values,
        "std_uplink_bytes": g[col_up].std(ddof=0).values,
        "mean_downlink_bytes": g[col_down].mean().values,
        "std_downlink_bytes": g[col_down].std(ddof=0).values,
        "mean_round_time": g[col_rt].mean().values,
        "std_round_time": g[col_rt].std(ddof=0).values,
    })

    # Save summary
    summary.to_csv(SUMMARY_CSV_OUT, index=False)

    # Order (nice fixed order if present)
    desired_order = [
        "ckks_baseline",
        "ckks_clip",
        "ckks_quant",
        "ckks_clip_quant",
        "ckks_clip_quant_comp"
    ]
    if set(desired_order).issubset(set(summary["variant"].tolist())):
        summary["__order"] = summary["variant"].apply(lambda v: desired_order.index(v))
        summary = summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    labels = summary["variant"].tolist()

    # =========================
    # Plots: Accuracy (zoomed)
    # =========================
    final_acc = summary["final_acc"].to_numpy()
    mean_acc = summary["mean_acc"].to_numpy()

    # Zoomed y-limits to tell the story, but safe auto if weird ranges appear
    fa_min, fa_max = float(np.min(final_acc)), float(np.max(final_acc))
    ma_min, ma_max = float(np.min(mean_acc)), float(np.max(mean_acc))

    final_zoom = None
    if fa_max <= 1.0 and fa_min >= 0.0:
        # zoom tightly near the observed region
        pad = max(0.002, 0.15 * (fa_max - fa_min))
        final_zoom = (max(0.0, fa_min - pad), min(1.01, fa_max + pad))

    mean_zoom = None
    if ma_max <= 1.0 and ma_min >= 0.0:
        pad = max(0.005, 0.15 * (ma_max - ma_min))
        mean_zoom = (max(0.0, ma_min - pad), min(1.01, ma_max + pad))

    barplot_clean(
        labels,
        final_acc,
        "Week-11 CKKS Optimizations: Final Accuracy",
        "Final accuracy (higher is better)",
        os.path.join(RESULTS_DIR, "week11_ckks_final_acc.png"),
        y_zoom=final_zoom,
        annotate_fmt=lambda v: f"{v:.3f}",
        stds=summary["std_acc"].to_numpy()
    )

    barplot_clean(
        labels,
        mean_acc,
        "Week-11 CKKS Optimizations: Mean Accuracy",
        "Mean accuracy across rounds (higher is better)",
        os.path.join(RESULTS_DIR, "week11_ckks_mean_acc.png"),
        y_zoom=mean_zoom,
        annotate_fmt=lambda v: f"{v:.3f}",
        stds=summary["std_acc"].to_numpy()
    )

    # =========================
    # Plot: Round time
    # =========================
    mean_rt = summary["mean_round_time"].to_numpy()
    barplot_clean(
        labels,
        mean_rt,
        "Week-11 CKKS Optimizations: Mean Round Time",
        "Round time (sec) — lower is better",
        os.path.join(RESULTS_DIR, "week11_ckks_round_time.png"),
        annotate_fmt=lambda v: fmt_seconds(v),
        y_is_lower_better=True,
        stds=summary["std_round_time"].to_numpy()
    )

    # =========================
    # Plots: Uplink/Downlink bytes
    # =========================
    mean_up = summary["mean_uplink_bytes"].to_numpy()
    mean_down = summary["mean_downlink_bytes"].to_numpy()

    # If constant: produce “intentional” constant communication plot
    up_range = float(np.max(mean_up) - np.min(mean_up))
    down_range = float(np.max(mean_down) - np.min(mean_down))

    if up_range <= DELTA_EPS_BYTES:
        constant_metric_explain_plot(
            labels,
            mean_up,
            "Week-11 CKKS Optimizations: Mean Uplink Bytes (Constant)",
            "Uplink bytes per round (lower is better)",
            os.path.join(RESULTS_DIR, "week11_ckks_uplink_bytes.png"),
            explain_text=(
                "Uplink bytes are effectively constant across variants.\n"
                "Reason: CKKS ciphertext size dominates communication; clip/quant affects computation/accuracy, not ciphertext length."
            )
        )
    else:
        barplot_clean(
            labels,
            mean_up,
            "Week-11 CKKS Optimizations: Mean Uplink Bytes",
            "Uplink bytes per round (lower is better)",
            os.path.join(RESULTS_DIR, "week11_ckks_uplink_bytes.png"),
            annotate_fmt=lambda v: fmt_bytes(v),
            y_is_lower_better=True,
            stds=summary["std_uplink_bytes"].to_numpy()
        )

    if down_range <= DELTA_EPS_BYTES:
        constant_metric_explain_plot(
            labels,
            mean_down,
            "Week-11 CKKS Optimizations: Mean Downlink Bytes (Constant)",
            "Downlink bytes per round (lower is better)",
            os.path.join(RESULTS_DIR, "week11_ckks_downlink_bytes.png"),
            explain_text=(
                "Downlink bytes are effectively constant across variants.\n"
                "Reason: CKKS ciphertext size dominates communication; clip/quant affects computation/accuracy, not ciphertext length."
            )
        )
    else:
        barplot_clean(
            labels,
            mean_down,
            "Week-11 CKKS Optimizations: Mean Downlink Bytes",
            "Downlink bytes per round (lower is better)",
            os.path.join(RESULTS_DIR, "week11_ckks_downlink_bytes.png"),
            annotate_fmt=lambda v: fmt_bytes(v),
            y_is_lower_better=True,
            stds=summary["std_downlink_bytes"].to_numpy()
        )

    # =========================
    # Delta / Percent vs baseline (ONLY if non-trivial)
    # =========================
    if "ckks_baseline" in labels:
        bidx = labels.index("ckks_baseline")

        # --- Uplink delta/pct
        base_up = float(mean_up[bidx])
        delta_up = mean_up - base_up
        pct_up = (delta_up / base_up) * 100.0 if base_up != 0 else np.zeros_like(delta_up)

        if float(np.max(np.abs(delta_up))) > DELTA_EPS_BYTES:
            barplot_clean(
                labels,
                delta_up,
                "Week-11 CKKS Optimizations: Uplink Δ vs ckks_baseline",
                "Δ uplink bytes per round (vs baseline)",
                os.path.join(RESULTS_DIR, "week11_ckks_uplink_delta_vs_baseline.png"),
                annotate_fmt=lambda v: f"{v:+.0f} B"
            )
        # else: don’t generate misleading empty plot

        if float(np.max(np.abs(pct_up))) > DELTA_EPS_PERCENT:
            barplot_clean(
                labels,
                pct_up,
                "Week-11 CKKS Optimizations: Uplink % Change vs ckks_baseline",
                "% uplink change per round (vs baseline)",
                os.path.join(RESULTS_DIR, "week11_ckks_uplink_pct_vs_baseline.png"),
                annotate_fmt=lambda v: f"{v:+.3f}%"
            )

        # --- Downlink delta/pct
        base_down = float(mean_down[bidx])
        delta_down = mean_down - base_down
        pct_down = (delta_down / base_down) * 100.0 if base_down != 0 else np.zeros_like(delta_down)

        if float(np.max(np.abs(delta_down))) > DELTA_EPS_BYTES:
            barplot_clean(
                labels,
                delta_down,
                "Week-11 CKKS Optimizations: Downlink Δ vs ckks_baseline",
                "Δ downlink bytes per round (vs baseline)",
                os.path.join(RESULTS_DIR, "week11_ckks_downlink_delta_vs_baseline.png"),
                annotate_fmt=lambda v: f"{v:+.0f} B"
            )

        if float(np.max(np.abs(pct_down))) > DELTA_EPS_PERCENT:
            barplot_clean(
                labels,
                pct_down,
                "Week-11 CKKS Optimizations: Downlink % Change vs ckks_baseline",
                "% downlink change per round (vs baseline)",
                os.path.join(RESULTS_DIR, "week11_ckks_downlink_pct_vs_baseline.png"),
                annotate_fmt=lambda v: f"{v:+.3f}%"
            )

    # =========================
    # Trade-off scatter plots
    # =========================
    make_tradeoff_scatter(
        summary["mean_round_time"].to_numpy(),
        summary["final_acc"].to_numpy(),
        labels,
        "Week-11 Utility Trade-off: Final Accuracy vs Mean Round Time",
        "Mean round time (sec) — lower is better",
        "Final accuracy — higher is better",
        os.path.join(RESULTS_DIR, "week11_tradeoff_finalacc_vs_time.png")
    )

    make_tradeoff_scatter(
        summary["mean_uplink_bytes"].to_numpy(),
        summary["final_acc"].to_numpy(),
        labels,
        "Week-11 Utility Trade-off: Final Accuracy vs Mean Uplink Bytes",
        "Mean uplink bytes per round — lower is better",
        "Final accuracy — higher is better",
        os.path.join(RESULTS_DIR, "week11_tradeoff_finalacc_vs_uplink.png")
    )

    make_tradeoff_scatter(
        summary["mean_round_time"].to_numpy(),
        summary["mean_acc"].to_numpy(),
        labels,
        "Week-11 Utility Trade-off: Mean Accuracy vs Mean Round Time",
        "Mean round time (sec) — lower is better",
        "Mean accuracy — higher is better",
        os.path.join(RESULTS_DIR, "week11_tradeoff_meanacc_vs_time.png")
    )

    make_tradeoff_scatter(
        summary["mean_uplink_bytes"].to_numpy(),
        summary["mean_acc"].to_numpy(),
        labels,
        "Week-11 Utility Trade-off: Mean Accuracy vs Mean Uplink Bytes",
        "Mean uplink bytes per round — lower is better",
        "Mean accuracy — higher is better",
        os.path.join(RESULTS_DIR, "week11_tradeoff_meanacc_vs_uplink.png")
    )

    print("[Week-11] Analysis done.")
    print(f"[Week-11] Round-level CSV: {ROUND_CSV_DEFAULT}")
    print(f"[Week-11] Summary CSV:     {SUMMARY_CSV_OUT}")
    print(f"[Week-11] Plots saved in:  {RESULTS_DIR}")
    if not SHOW_ERROR_BARS:
        print("[Week-11] Note: Error bars are OFF (cleanest). Set SHOW_ERROR_BARS=True if needed.")


if __name__ == "__main__":
    main()
