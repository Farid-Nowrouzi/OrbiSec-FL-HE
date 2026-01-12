"""
run_week12_4_ckks_optimizations_summary.py

Week-12.4: CKKS Optimization Summary (Final Decision + Clean Plots)

What this script does:
- Loads a CKKS optimization summary CSV (baseline/clip/quant/clip+quant/clip+quant+comp).
- Robustly handles different column names from upstream logs (bytes vs KB, time vs time_sec).
- Computes:
  - Pareto front (Final Acc vs Mean Round Time)
  - Regret plots (accuracy regret + time regret)
  - Normalized heatmap (final acc, mean acc, speed score)
  - Utility score bars (two weight profiles)
  - A clean scorecard table (rounded numbers)
  - A larger final recommendation card (readable in reports)

Outputs are saved into ./results with week12_4_* filenames.
"""

from __future__ import annotations

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
RESULTS_DIR = "results"

DEFAULT_INPUT_CSV = os.path.join(RESULTS_DIR, "week12_4_ckks_optimization_summary.csv")

PLOT_PARETO = os.path.join(RESULTS_DIR, "week12_4_ckks_pareto_finalacc_vs_time.png")
PLOT_REGRET_ACC = os.path.join(RESULTS_DIR, "week12_4_ckks_regret_accuracy.png")
PLOT_REGRET_TIME = os.path.join(RESULTS_DIR, "week12_4_ckks_regret_time.png")
PLOT_HEATMAP = os.path.join(RESULTS_DIR, "week12_4_ckks_score_heatmap.png")
PLOT_SCORECARD = os.path.join(RESULTS_DIR, "week12_4_ckks_scorecard.png")
PLOT_UTILITY_ACC = os.path.join(RESULTS_DIR, "week12_4_ckks_utility_score_accuracy_focused.png")
PLOT_UTILITY_LAT = os.path.join(RESULTS_DIR, "week12_4_ckks_utility_score_latency_focused.png")
PLOT_FINAL_CARD = os.path.join(RESULTS_DIR, "week12_4_ckks_final_recommendation_card.png")


# ----------------------------
# Helpers: column resolution
# ----------------------------
def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept multiple upstream naming schemes and standardize to:

    Required standardized fields:
      - variant
      - final_acc
      - mean_acc
      - mean_round_time_sec
      - uplink_kb
      - downlink_kb
    """
    # variant
    if "variant" not in df.columns:
        raise ValueError("CSV must contain a 'variant' column.")

    # accuracy columns
    c_final = _first_existing(df, ["final_acc", "final_accuracy", "acc_final"])
    c_mean = _first_existing(df, ["mean_acc", "mean_accuracy", "acc_mean"])
    if c_final is None or c_mean is None:
        raise ValueError(
            "CSV missing accuracy columns. Expected one of:\n"
            " - final_acc / final_accuracy / acc_final\n"
            " - mean_acc  / mean_accuracy  / acc_mean"
        )

    # time columns (seconds)
    c_time = _first_existing(df, ["mean_round_time_sec", "mean_round_time", "round_time_sec_mean", "mean_time_sec"])
    if c_time is None:
        raise ValueError(
            "CSV missing time column. Expected one of:\n"
            " - mean_round_time_sec\n"
            " - mean_round_time\n"
            " - round_time_sec_mean\n"
            " - mean_time_sec"
        )

    # comm columns: prefer KB directly, else bytes -> KB
    c_uplink_kb = _first_existing(df, ["uplink_kb", "mean_uplink_kb"])
    c_downlink_kb = _first_existing(df, ["downlink_kb", "mean_downlink_kb"])

    c_uplink_bytes = _first_existing(df, ["mean_uplink_bytes", "uplink_bytes"])
    c_downlink_bytes = _first_existing(df, ["mean_downlink_bytes", "downlink_bytes"])

    # build standardized frame
    out = df.copy()

    out["final_acc"] = out[c_final].astype(float)
    out["mean_acc"] = out[c_mean].astype(float)
    out["mean_round_time_sec"] = out[c_time].astype(float)

    if c_uplink_kb is not None and c_downlink_kb is not None:
        out["uplink_kb"] = out[c_uplink_kb].astype(float)
        out["downlink_kb"] = out[c_downlink_kb].astype(float)
    elif c_uplink_bytes is not None and c_downlink_bytes is not None:
        out["uplink_kb"] = out[c_uplink_bytes].astype(float) / 1024.0
        out["downlink_kb"] = out[c_downlink_bytes].astype(float) / 1024.0
    else:
        # If missing, we keep but fill with NaN so we can still plot main decisions
        out["uplink_kb"] = np.nan
        out["downlink_kb"] = np.nan

    # keep only relevant columns + any extras
    return out


# ----------------------------
# Pareto + scoring utilities
# ----------------------------
def pareto_mask_finalacc_vs_time(df: pd.DataFrame) -> np.ndarray:
    """
    Pareto optimal under:
      - maximize final_acc
      - minimize mean_round_time_sec
    A point is dominated if exists another point with:
      final_acc >= and time <= (and at least one strict)
    """
    acc = df["final_acc"].to_numpy()
    t = df["mean_round_time_sec"].to_numpy()

    n = len(df)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = (acc[j] >= acc[i]) and (t[j] <= t[i])
            strictly_better = (acc[j] > acc[i]) or (t[j] < t[i])
            if better_or_equal and strictly_better:
                is_pareto[i] = False
                break
    return is_pareto


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if math.isclose(mx, mn):
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def safe_round(x: float, nd: int) -> float:
    if pd.isna(x):
        return np.nan
    return float(np.round(x, nd))


# ----------------------------
# Plot helpers (layout fixes)
# ----------------------------
def finalize_save(fig: plt.Figure, path: str, tight: bool = True, pad: float = 0.4) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout(pad=pad)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pareto(df: pd.DataFrame, is_pareto: np.ndarray) -> None:
    fig = plt.figure(figsize=(11.5, 6.8))
    ax = fig.add_subplot(111)

    ax.scatter(df["mean_round_time_sec"], df["final_acc"])

    # circle pareto points
    for _, r in df[is_pareto].iterrows():
        ax.scatter(r["mean_round_time_sec"], r["final_acc"], s=140, facecolors="none", edgecolors="black", linewidths=2)

    # labels
    for _, r in df.iterrows():
        ax.annotate(
            r["variant"],
            (r["mean_round_time_sec"], r["final_acc"]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=10,
        )

    ax.set_title("Week-12.4 CKKS Decision View: Pareto Frontier (Final Acc vs Mean Round Time)")
    ax.set_xlabel("Mean round time (sec) — lower is better")
    ax.set_ylabel("Final accuracy — higher is better")

    # bottom note
    note = (
        "Circled points = Pareto-optimal: no other variant is BOTH faster AND more accurate.\n"
        "we Pick a Pareto point that matches your latency/accuracy priorities."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9)

    finalize_save(fig, PLOT_PARETO, tight=False)
    # manual spacing so the bottom note never overlaps
    # (avoid tight_layout squashing the note)
    # Re-open a new fig style: we already saved above.


def plot_regrets(df: pd.DataFrame) -> None:
    best_final = float(df["final_acc"].max())
    best_time = float(df["mean_round_time_sec"].min())

    # regret definitions (lower is better)
    # accuracy_regret = best_final_acc - final_acc (best accuracy -> 0)
    # time_regret = mean_round_time_sec - best_time (fastest -> 0)
    df = df.copy()
    df["accuracy_regret"] = best_final - df["final_acc"]
    df["time_regret"] = df["mean_round_time_sec"] - best_time

    # --- accuracy regret
    fig = plt.figure(figsize=(11.5, 6.6))
    ax = fig.add_subplot(111)
    ax.bar(df["variant"], df["accuracy_regret"])
    ax.set_title("Week-12.4 CKKS: Accuracy Regret vs Best (lower is better)")
    ax.set_ylabel("best_final_acc − final_acc")
    ax.set_xlabel("")  # remove 'Variant' to prevent overlap with formula
    ax.tick_params(axis="x", rotation=15)

    for i, v in enumerate(df["accuracy_regret"].to_numpy()):
        ax.text(i, v + (max(df["accuracy_regret"]) * 0.02 + 1e-6), f"{v:.4f}", ha="center", fontsize=10)

    fig.text(
        0.5,
        0.02,
        "Definition: accuracy_regret = best_final_acc − final_acc. Best accuracy ⇒ regret = 0.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.18)
    finalize_save(fig, PLOT_REGRET_ACC, tight=False)

    # --- time regret
    fig = plt.figure(figsize=(11.5, 6.6))
    ax = fig.add_subplot(111)
    ax.bar(df["variant"], df["time_regret"])
    ax.set_title("Week-12.4 CKKS: Time Regret vs Best (lower is better)")
    ax.set_ylabel("mean_round_time_sec − best_round_time_sec")
    ax.set_xlabel("")  # remove 'Variant' to prevent overlap with formula
    ax.tick_params(axis="x", rotation=15)

    for i, v in enumerate(df["time_regret"].to_numpy()):
        ax.text(i, v + (max(df["time_regret"]) * 0.02 + 1e-6), f"{v:.3f}s", ha="center", fontsize=10)

    fig.text(
        0.5,
        0.02,
        "Definition: time_regret = mean_round_time_sec − best_round_time_sec. Fastest method ⇒ regret = 0.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.18)
    finalize_save(fig, PLOT_REGRET_TIME, tight=False)


def plot_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap columns:
      - final_acc (norm)
      - mean_acc  (norm)
      - speed_score (fast->high): 1 - norm(mean_round_time_sec)
    """
    df = df.copy()

    final_n = minmax_norm(df["final_acc"].to_numpy())
    mean_n = minmax_norm(df["mean_acc"].to_numpy())
    time_n = minmax_norm(df["mean_round_time_sec"].to_numpy())
    speed_score = 1.0 - time_n

    mat = np.vstack([final_n, mean_n, speed_score]).T
    col_labels = ["final_acc (norm)", "mean_acc (norm)", "speed_score (fast→high)"]

    fig = plt.figure(figsize=(11.8, 6.8))
    ax = fig.add_subplot(111)

    im = ax.imshow(mat, aspect="auto")
    ax.set_title("Week-12.4 CKKS: Normalized Score Heatmap (Higher = Better)")

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["variant"])

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)  # keep readable (no overlap)

    # annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized score")

    fig.text(
        0.5,
        0.02,
        "All columns are min-max normalized to [0,1] across variants.\n"
        "For time, we invert: speed_score = 1 − norm(mean_round_time_sec), so faster ⇒ higher score.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.20)
    finalize_save(fig, PLOT_HEATMAP, tight=False)


def plot_utility(df: pd.DataFrame, w_acc: float, w_time: float, out_path: str) -> pd.DataFrame:
    """
    Utility uses normalized accuracy score and speed score.
      acc_score = mean( norm(final_acc), norm(mean_acc) )
      speed_score = 1 - norm(mean_round_time_sec)
      utility = w_acc*acc_score + w_time*speed_score
    """
    df = df.copy()

    final_n = minmax_norm(df["final_acc"].to_numpy())
    mean_n = minmax_norm(df["mean_acc"].to_numpy())
    time_n = minmax_norm(df["mean_round_time_sec"].to_numpy())
    speed_score = 1.0 - time_n

    acc_score = 0.5 * (final_n + mean_n)
    utility = w_acc * acc_score + w_time * speed_score

    df["acc_score"] = acc_score
    df["speed_score"] = speed_score
    df["utility"] = utility

    # plot
    fig = plt.figure(figsize=(11.5, 6.8))
    ax = fig.add_subplot(111)

    ax.bar(df["variant"], df["utility"])
    ax.set_title(f"Week-12.4 CKKS: Utility Score ({w_acc:.2f} accuracy / {w_time:.2f} time)")
    ax.set_ylabel("Utility score (higher is better)")

    # KEY FIX: remove xlabel and create more bottom space so nothing overlaps
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=18)

    for i, v in enumerate(df["utility"].to_numpy()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    formula = (
        "Utility uses normalized scores in [0,1].\n"
        "acc_score = mean(norm(final_acc), norm(mean_acc))\n"
        "speed_score = 1 − norm(mean_round_time_sec)\n"
        f"utility = {w_acc:.2f}·acc_score + {w_time:.2f}·speed_score"
    )
    fig.text(0.5, 0.02, formula, ha="center", fontsize=9)

    fig.subplots_adjust(bottom=0.25)
    finalize_save(fig, out_path, tight=False)

    return df


def plot_scorecard(df: pd.DataFrame, is_pareto: np.ndarray) -> None:
    """
    Table with rounded/short numbers so it looks clean in the report.
    """
    df = df.copy()
    df["Pareto"] = np.where(is_pareto, "YES", "NO")

    # rank by final_acc (desc) and time (asc)
    df["Rank (Acc)"] = df["final_acc"].rank(ascending=False, method="min").astype(int)
    df["Rank (Lat)"] = df["mean_round_time_sec"].rank(ascending=True, method="min").astype(int)

    # Round for readability:
    # - Acc: 4 decimals
    # - Time: 3 decimals
    # - KB: 1 decimal (or blank if NaN)
    score = pd.DataFrame({
        "Variant": df["variant"],
        "Final Acc": df["final_acc"].map(lambda x: safe_round(x, 4)),
        "Mean Acc": df["mean_acc"].map(lambda x: safe_round(x, 4)),
        "Mean Time (s)": df["mean_round_time_sec"].map(lambda x: safe_round(x, 3)),
        "Uplink (KB)": df["uplink_kb"].map(lambda x: safe_round(x, 1) if not pd.isna(x) else np.nan),
        "Downlink (KB)": df["downlink_kb"].map(lambda x: safe_round(x, 1) if not pd.isna(x) else np.nan),
        "Pareto": df["Pareto"],
        "Rank (Acc)": df["Rank (Acc)"],
        "Rank (Lat)": df["Rank (Lat)"],
    })

    # plot as table
    fig = plt.figure(figsize=(14.0, 4.8))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.set_title("Week-12.4 CKKS Optimization Scorecard (Pareto + Utility Rankings)", pad=12)

    table = ax.table(
        cellText=score.values,
        colLabels=score.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.08, 1.25)

    finalize_save(fig, PLOT_SCORECARD, tight=True, pad=0.6)


def plot_final_recommendation_card(df: pd.DataFrame, is_pareto: np.ndarray) -> None:
    """
    Bigger final card (readable).
    """
    df = df.copy()
    pareto_variants = df.loc[is_pareto, "variant"].tolist()

    # recommended by accuracy-focused utility + latency-focused utility
    df_acc = plot_utility(df, 0.70, 0.30, out_path=PLOT_UTILITY_ACC)
    df_lat = plot_utility(df, 0.55, 0.45, out_path=PLOT_UTILITY_LAT)

    best_acc = df_acc.sort_values("utility", ascending=False).iloc[0]
    best_lat = df_lat.sort_values("utility", ascending=False).iloc[0]

    # observation about comm stability
    comm_constant = False
    if df["uplink_kb"].notna().all() and df["downlink_kb"].notna().all():
        u_span = float(df["uplink_kb"].max() - df["uplink_kb"].min())
        d_span = float(df["downlink_kb"].max() - df["downlink_kb"].min())
        comm_constant = (u_span < 1e-6) and (d_span < 1e-6)

    fig = plt.figure(figsize=(14.5, 8.2))  # BIGGER
    ax = fig.add_subplot(111)
    ax.axis("off")

    title = "Week-12.4 — CKKS Optimization: Final Decision Summary"
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=18, fontweight="bold")

    lines = []
    lines.append("What we optimized:")
    lines.append(" • We compared CKKS optimization variants (baseline / clip / quant / clip+quant / clip+quant+comp).")
    lines.append(" • Metrics: final accuracy, mean accuracy, mean round time, and communication (uplink/downlink).")
    lines.append("")
    lines.append("Key observation (communication):")
    if comm_constant:
        lines.append(" • Uplink and downlink are effectively constant across CKKS variants (ciphertext size dominates bandwidth).")
        lines.append("   → Clip/quant mainly affect compute + accuracy, not ciphertext length.")
    else:
        lines.append(" • Communication varies slightly across variants (check HE packing/ciphertext sizing).")
    lines.append("")
    lines.append("Pareto decision rule:")
    lines.append(" • A variant is Pareto-optimal if no other variant is BOTH faster AND more accurate.")
    lines.append(f" • Pareto set: {', '.join(pareto_variants) if pareto_variants else '(none)'}")
    lines.append("")
    lines.append("Regret definitions (lower is better):")
    lines.append(" • accuracy_regret = best_final_acc − final_acc   (best accuracy ⇒ 0)")
    lines.append(" • time_regret     = mean_round_time_sec − best_time (fastest ⇒ 0)")
    lines.append("")
    lines.append("Recommended choices:")
    lines.append(f" • Accuracy-focused (0.70 acc / 0.30 time): {best_acc['variant']}")
    lines.append(f"   - final_acc={best_acc['final_acc']:.4f}, mean_acc={best_acc['mean_acc']:.4f}, time={best_acc['mean_round_time_sec']:.3f}s")
    lines.append(f" • Latency-sensitive (0.55 acc / 0.45 time): {best_lat['variant']}")
    lines.append(f"   - final_acc={best_lat['final_acc']:.4f}, mean_acc={best_lat['mean_acc']:.4f}, time={best_lat['mean_round_time_sec']:.3f}s")
    lines.append("")
    lines.append("How to read the heatmap:")
    lines.append(" • Values are min-max normalized to [0,1] across variants.")
    lines.append(" • 0.00 means “worst among these variants” for that metric (not literal zero).")
    lines.append(" • speed_score is inverted time: faster ⇒ higher score.")

    # render lines bigger
    y = 0.88
    for s in lines:
        if s == "":
            y -= 0.035
            continue
        ax.text(0.06, y, s, ha="left", va="top", fontsize=12)
        y -= 0.035

    finalize_save(fig, PLOT_FINAL_CARD, tight=True, pad=0.8)


# ----------------------------
# Main
# ----------------------------
def main(input_csv: str = DEFAULT_INPUT_CSV) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df_raw = pd.read_csv(input_csv)
    df = resolve_and_standardize_columns(df_raw)

    # Pareto
    is_pareto = pareto_mask_finalacc_vs_time(df)

    # Plots
    plot_pareto(df, is_pareto)
    plot_regrets(df)
    plot_heatmap(df)

    # Utility plots are generated inside final card (and saved)
    plot_scorecard(df, is_pareto)
    plot_final_recommendation_card(df, is_pareto)

    # quick console summary
    print("\n Week-12.4 CKKS optimization summary complete.")
    print("Saved plots in ./results:")
    for p in [
        PLOT_PARETO,
        PLOT_REGRET_ACC,
        PLOT_REGRET_TIME,
        PLOT_HEATMAP,
        PLOT_UTILITY_ACC,
        PLOT_UTILITY_LAT,
        PLOT_SCORECARD,
        PLOT_FINAL_CARD,
    ]:
        print(" -", os.path.basename(p))

    # communication note
    if df["uplink_kb"].notna().all() and df["downlink_kb"].notna().all():
        u_span = float(df["uplink_kb"].max() - df["uplink_kb"].min())
        d_span = float(df["downlink_kb"].max() - df["downlink_kb"].min())
        if (u_span < 1e-6) and (d_span < 1e-6):
            print("\nNote: Communication is constant across variants (CKKS ciphertext size dominates).")
        else:
            print("\nNote: Communication varies slightly across variants (check packing/ciphertext sizing).")
    else:
        print("\nNote: Communication columns missing or partial — plots still generated.")


if __name__ == "__main__":
    main()
