"""
run_week12_3_utility_tradeoffs_summary.py

WEEK-12.3: Utility trade-offs summary across modes (from Week-5 per-round logs).

Goals :
- Read Week-5 per-round logs (results_{mode}.csv) for modes: none, mask, secagg, ckks
- Compute clean utility + cost summaries (NO fabrication):
    * final_acc (last available round)
    * mean_acc (across rounds)
    * mean_round_time (sec)
    * mean_bytes_up, mean_bytes_down, mean_total_comm (= up + down)
    * convergence_round_acc>=THRESH (first round reaching threshold, if any)
- Produce report-friendly figures:
    (1) Overhead bars: mean_round_time vs none (×baseline + absolute labels)
    (2) Overhead bars: mean_total_comm vs none (×baseline + absolute labels)
    (3) Pareto: mean_acc vs mean_round_time (labels with anti-overlap)
    (4) Pareto: mean_acc vs mean_total_comm (labels with anti-overlap)
    (5) Optional Pareto for final_acc ONLY if final_acc varies (otherwise skip)
    (6) Utility heatmap (normalized; higher=better; costs inverted) + raw values shown

Outputs in /results:
- week12_3_utility_raw_summary.csv
- week12_3_utility_final_table.csv
- week12_3_overhead_round_time.png
- week12_3_overhead_total_comm.png
- week12_3_pareto_meanacc_vs_time.png
- week12_3_pareto_meanacc_vs_comm.png
- week12_3_pareto_finalacc_vs_time.png (optional)
- week12_3_pareto_finalacc_vs_comm.png (optional)
- week12_3_utility_matrix_normalized.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # experiments/ -> project root
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODE_ORDER = ["none", "mask", "secagg", "ckks"]

# Convergence threshold: pick something we used / can defend.
# If our task often saturates at ~1.0, 0.99 is reasonable.
CONV_THRESH = 0.99

# Numeric tolerance to decide "final accuracy is basically constant"
FINAL_ACC_FLAT_EPS = 1e-4


# ------------------------------------------------------------
# Helpers: robust CSV loading + column picking + numeric coercion
# ------------------------------------------------------------
def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    # exact match
    for cand in candidates:
        cl = cand.lower()
        if cl in cols_lower:
            return cols[cols_lower.index(cl)]

    # contains match
    for i, c in enumerate(cols_lower):
        for cand in candidates:
            if cand.lower() in c:
                return cols[i]

    return None


def _coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype("string")
    s = s.replace({"NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA})
    s = s.str.replace(",", "", regex=False)
    # handle "x +/- y" and "x±y"
    s = s.str.replace("±", "+/-", regex=False)
    first = s.str.split("+/-", n=1, regex=False).str[0].str.strip()
    return pd.to_numeric(first, errors="coerce")


def _mean_std(vals: pd.Series) -> Tuple[float, float, int]:
    v = _coerce_numeric(vals).dropna()
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    mean = float(v.mean())
    std = float(v.std(ddof=1) if len(v) > 1 else 0.0)
    return mean, std, int(len(v))


def _candidate_csvs_for_mode(mode: str) -> List[Path]:
    # Week-5 logs typically:
    # results_none.csv, results_mask.csv, results_secagg.csv, results_ckks.csv
    return [
        RESULTS_DIR / f"results_{mode}.csv",
        RESULTS_DIR / f"week5_results_{mode}.csv",
        RESULTS_DIR / f"week10_{mode}.csv",
        RESULTS_DIR / f"week10_{mode}_results.csv",
        ]


def _format_bytes(x: float) -> str:
    if not np.isfinite(x):
        return "NaN"
    # Keep it simple and report-friendly
    if x >= 1e9:
        return f"{x/1e9:.2f} GB"
    if x >= 1e6:
        return f"{x/1e6:.2f} MB"
    if x >= 1e3:
        return f"{x/1e3:.2f} KB"
    return f"{x:.0f} B"


# ------------------------------------------------------------
# Core: load one mode summary from per-round logs
# ------------------------------------------------------------
def load_mode_summary(mode: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "mode": mode,
        "source_csv": "",
        "note": "",

        "final_acc": np.nan,
        "mean_acc": np.nan,
        "mean_acc_std": np.nan,
        "mean_acc_n": 0,

        "mean_round_time": np.nan,
        "mean_round_time_std": np.nan,
        "mean_round_time_n": 0,

        "mean_bytes_up": np.nan,
        "mean_bytes_up_std": np.nan,
        "mean_bytes_up_n": 0,

        "mean_bytes_down": np.nan,
        "mean_bytes_down_std": np.nan,
        "mean_bytes_down_n": 0,

        "mean_total_comm": np.nan,

        f"conv_round_acc>={CONV_THRESH:.2f}": np.nan,
    }

    csv_path = _find_first_existing(_candidate_csvs_for_mode(mode))
    if csv_path is None:
        out["note"] = "missing_source_csv"
        return out

    out["source_csv"] = str(csv_path)
    df = _read_csv_safe(csv_path)

    # Column picks (robust):
    round_col = _pick_column(df, ["round", "rnd", "federated_round"])
    acc_col = _pick_column(df, ["acc", "accuracy", "val_acc", "val_accuracy"])
    round_time_col = _pick_column(df, ["round_time", "round_time_s", "round_time_sec"])
    bytes_up_col = _pick_column(df, ["bytes_up", "uplink", "uplink_bytes"])
    bytes_down_col = _pick_column(df, ["bytes_down", "downlink", "downlink_bytes"])

    # Accuracy
    if acc_col is not None:
        acc_numeric = _coerce_numeric(df[acc_col]).dropna()
        if acc_numeric.empty:
            out["note"] += f"|acc_no_numeric({acc_col})"
        else:
            out["final_acc"] = float(acc_numeric.iloc[-1])
            m, s, n = _mean_std(df[acc_col])
            out["mean_acc"], out["mean_acc_std"], out["mean_acc_n"] = m, s, n
    else:
        out["note"] += "|acc_col_missing"

    # Round time
    if round_time_col is not None:
        m, s, n = _mean_std(df[round_time_col])
        out["mean_round_time"], out["mean_round_time_std"], out["mean_round_time_n"] = m, s, n
    else:
        out["note"] += "|round_time_col_missing"

    # Bytes up/down
    if bytes_up_col is not None:
        m, s, n = _mean_std(df[bytes_up_col])
        out["mean_bytes_up"], out["mean_bytes_up_std"], out["mean_bytes_up_n"] = m, s, n
    else:
        out["note"] += "|bytes_up_col_missing"

    if bytes_down_col is not None:
        m, s, n = _mean_std(df[bytes_down_col])
        out["mean_bytes_down"], out["mean_bytes_down_std"], out["mean_bytes_down_n"] = m, s, n
    else:
        out["note"] += "|bytes_down_col_missing"

    # Total comm
    if np.isfinite(out["mean_bytes_up"]) and np.isfinite(out["mean_bytes_down"]):
        out["mean_total_comm"] = float(out["mean_bytes_up"]) + float(out["mean_bytes_down"])

    # Convergence: first round where acc >= threshold
    # If round column missing, we use index+1 as round label (still honest, not fabricated).
    if acc_col is not None:
        acc_all = _coerce_numeric(df[acc_col])
        if round_col is not None:
            r_all = _coerce_numeric(df[round_col])
        else:
            r_all = pd.Series(np.arange(1, len(df) + 1), index=df.index)

        ok = (acc_all >= CONV_THRESH) & np.isfinite(acc_all) & np.isfinite(r_all)
        if ok.any():
            first_idx = ok.idxmax()  # first True index
            out[f"conv_round_acc>={CONV_THRESH:.2f}"] = float(r_all.loc[first_idx])

    return out


# ------------------------------------------------------------
# Plotting helpers (cleaner labels + anti-overlap)
# ------------------------------------------------------------
def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close()
    print(f"[Week-12.3] Saved: {path}")


def plot_overhead_bar(
        df: pd.DataFrame,
        metric_col: str,
        out: Path,
        title: str,
        ylabel: str,
        abs_fmt_fn,
) -> None:
    d = df.copy()
    d = d[d["mode"].isin(MODE_ORDER)]
    d["mode"] = pd.Categorical(d["mode"], categories=MODE_ORDER, ordered=True)
    d = d.sort_values("mode")

    base = d.loc[d["mode"] == "none", metric_col]
    if len(base) != 1 or not np.isfinite(float(base.iloc[0])):
        print(f"[Week-12.3] WARN: Can't compute overhead for {metric_col} (missing none baseline).")
        return
    base_val = float(base.iloc[0])

    ratios = []
    abs_vals = []
    for _, row in d.iterrows():
        v = float(row[metric_col]) if np.isfinite(row[metric_col]) else np.nan
        abs_vals.append(v)
        if np.isfinite(v) and base_val != 0:
            ratios.append(v / base_val)
        else:
            ratios.append(np.nan)

    plt.figure(figsize=(10.5, 5.4))
    x = np.arange(len(d))
    plt.bar(x, ratios)

    plt.axhline(1.0, linestyle="--", linewidth=1.3, alpha=0.7)
    plt.text(0.01, 0.92, "baseline = 1.0×", transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.25", alpha=0.15))

    plt.xticks(x, d["mode"].astype(str).tolist())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    # Labels: show ratio and absolute value
    for xi, r, v in zip(x, ratios, abs_vals):
        if np.isfinite(r) and np.isfinite(v):
            plt.text(xi, r + 0.02, f"{r:.3f}×\n({abs_fmt_fn(v)})", ha="center", va="bottom", fontsize=9)
        elif np.isfinite(r):
            plt.text(xi, r + 0.02, f"{r:.3f}×", ha="center", va="bottom", fontsize=9)

    _save_fig(out)


def _auto_axis_pad(vals: np.ndarray, pad_frac: float = 0.18) -> Tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return 0.0, 1.0
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx == mn:
        span = abs(mx) if mx != 0 else 1.0
        return mn - pad_frac * span, mx + pad_frac * span
    span = mx - mn
    return mn - pad_frac * span, mx + pad_frac * span


def plot_pareto_scatter(
        df: pd.DataFrame,
        xcol: str,
        ycol: str,
        out: Path,
        title: str,
        xlabel: str,
        ylabel: str,
        show_xy: bool = True,
) -> None:
    d = df.copy()
    d = d[d["mode"].isin(MODE_ORDER)]
    d["mode"] = pd.Categorical(d["mode"], categories=MODE_ORDER, ordered=True)
    d = d.sort_values("mode")

    d = d[np.isfinite(d[xcol].values) & np.isfinite(d[ycol].values)]
    if len(d) == 0:
        print(f"[Week-12.3] WARN: Not enough data for {title} (missing {xcol} or {ycol}).")
        return

    x = d[xcol].values.astype(float)
    y = d[ycol].values.astype(float)
    labels = d["mode"].astype(str).tolist()

    plt.figure(figsize=(10.8, 5.8))
    plt.scatter(x, y)

    xmin, xmax = _auto_axis_pad(x, 0.22)
    ymin, ymax = _auto_axis_pad(y, 0.18)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)

    # ------------------------------------------------------------
    # Label placement (robust for "final_acc is flat" plots)
    # ------------------------------------------------------------
    def _fmt_x(val: float) -> str:
        if "comm" in xcol.lower() or "bytes" in xcol.lower():
            return f"{val/1e3:.1f} KB"
        return f"{val:.3f}"

    def _fmt_y(val: float) -> str:
        return f"{val:.4f}".rstrip("0").rstrip(".")

    ax = plt.gca()
    order = np.argsort(x)

    y_span = float(np.nanmax(y) - np.nanmin(y))
    flat_y = ("final_acc" in ycol.lower()) or (y_span < 1e-6)

    if flat_y:
        # When y is (almost) constant, pixel offsets still collide.
        # So we place label text in DATA coordinates with vertical jitter
        # based on current axis height -> guaranteed separation.
        x_span_ax = float(xmax - xmin) if xmax != xmin else 1.0
        y_span_ax = float(ymax - ymin) if ymax != ymin else 1.0

        # Deterministic “stacking” pattern
        dy_pattern = [1, -1, 2, -2]
        for k, idx in enumerate(order):
            xi, yi, lab = x[idx], y[idx], labels[idx]

            # shorter text for the flat case = cleaner and avoids overlap
            if show_xy:
                text = f"{lab}\n({_fmt_x(xi)} s, {_fmt_y(yi)})" if "time" in xcol.lower() else f"{lab}\n({_fmt_x(xi)}, {_fmt_y(yi)})"
            else:
                text = lab

            # small horizontal shift (data units) + stronger vertical jitter (data units)
            dx_data = (0.03 * x_span_ax) * (1 if (k % 2 == 0) else -1)
            dy_data = (0.07 * y_span_ax) * dy_pattern[k % len(dy_pattern)]

            x_text = xi + dx_data
            y_text = yi + dy_data

            plt.annotate(
                text,
                xy=(xi, yi),
                xytext=(x_text, y_text),
                textcoords="data",
                ha="left" if dx_data > 0 else "right",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
                arrowprops=dict(arrowstyle="-", alpha=0.35),
            )
    else:
        # Normal case: light repel in display coords
        placed = []
        for k, idx in enumerate(order):
            xi, yi, lab = x[idx], y[idx], labels[idx]

            text = lab
            if show_xy:
                text = f"{lab} ({_fmt_x(xi)}, {_fmt_y(yi)})"

            # base offsets (points)
            dx = 14 if (k % 2 == 0) else -80
            dy = 16 if (k % 3 != 0) else -16

            # display coords for collision checks
            x_disp, y_disp = ax.transData.transform((xi, yi))

            push = 0
            for (px, py) in placed:
                if abs((x_disp + dx) - px) < 85 and abs((y_disp + dy) - py) < 22:
                    push += 22

            dy = dy + push if dy >= 0 else dy - push
            placed.append((x_disp + dx, y_disp + dy))

            plt.annotate(
                text,
                xy=(xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
                arrowprops=dict(arrowstyle="-", alpha=0.35),
            )

    _save_fig(out)




# ------------------------------------------------------------
# Heatmap: normalized matrix (higher=better; costs inverted)
# ------------------------------------------------------------
def _minmax_norm(values: np.ndarray) -> np.ndarray:
    v = values.astype(float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.full_like(v, np.nan, dtype=float)

    mn = float(np.min(v[finite]))
    mx = float(np.max(v[finite]))


    if abs(mx - mn) < 1e-12:
        out = np.full_like(v, 1.0, dtype=float)
        out[~finite] = np.nan
        return out

    out = (v - mn) / (mx - mn)
    out[~finite] = np.nan
    return out


def plot_utility_heatmap(
        df: pd.DataFrame,
        out: Path,
) -> None:
    """
    Build a matrix with:
      - mean_acc (benefit)
      - mean_round_time (cost -> inverted)
      - mean_total_comm (cost -> inverted)
      - convergence_round (cost -> inverted)  [optional if present]
      - final_acc (benefit) only if not flat, otherwise we still show it but it will normalize to 1.0 and annotate const.
    """
    d = df.copy()
    d = d[d["mode"].isin(MODE_ORDER)]
    d["mode"] = pd.Categorical(d["mode"], categories=MODE_ORDER, ordered=True)
    d = d.sort_values("mode").reset_index(drop=True)

    conv_col = f"conv_round_acc>={CONV_THRESH:.2f}"
    has_conv = conv_col in d.columns and np.isfinite(d[conv_col].values).any()

    cols = ["final_acc", "mean_acc", "mean_round_time", "mean_total_comm"]
    if has_conv:
        cols.append(conv_col)

    raw = d[cols].to_numpy(dtype=float)

    # Determine which are costs
    cost_cols = set(["mean_round_time", "mean_total_comm", conv_col])

    # Normalize each column independently
    norm = np.zeros_like(raw, dtype=float)
    const_mask = np.zeros_like(raw, dtype=bool)

    for j, c in enumerate(cols):
        v = raw[:, j].copy()

        # invert costs (lower is better):
        if c in cost_cols:
            # Inversion done AFTER min-max makes interpretation easier:
            # score = 1 - norm(cost)
            n = _minmax_norm(v)
            # detect constant column (min=max) => _minmax_norm returns 1.0
            finite = np.isfinite(v)
            if finite.any() and abs(np.nanmax(v) - np.nanmin(v)) < 1e-12:
                const_mask[:, j] = True
            n = 1.0 - n
            norm[:, j] = n
        else:
            n = _minmax_norm(v)
            finite = np.isfinite(v)
            if finite.any() and abs(np.nanmax(v) - np.nanmin(v)) < 1e-12:
                const_mask[:, j] = True
            norm[:, j] = n

    # Plot
    plt.figure(figsize=(12.2, 4.3))
    im = plt.imshow(norm, aspect="auto")

    plt.yticks(np.arange(len(d)), d["mode"].astype(str).tolist())
    plt.xticks(np.arange(len(cols)), cols, rotation=20, ha="right")

    plt.title("Week-12.3 Utility Matrix (normalized; higher=better; costs inverted)")
    cbar = plt.colorbar(im)
    cbar.set_label("normalized (higher = better)")

    # annotate cells with: normalized and raw in parentheses; mark const
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            nv = norm[i, j]
            rv = raw[i, j]
            if not np.isfinite(nv) or not np.isfinite(rv):
                txt = "NA"
            else:
                # Pretty raw formatting
                if "comm" in cols[j]:
                    rv_txt = f"{rv:.2e}"
                elif "time" in cols[j]:
                    rv_txt = f"{rv:.3f}"
                else:
                    rv_txt = f"{rv:.4f}".rstrip("0").rstrip(".")
                suffix = " const" if const_mask[i, j] else ""
                txt = f"{nv:.2f}\n({rv_txt}){suffix}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    _save_fig(out)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    print("[Week-12.3] Building utility trade-off summary from Week-5 logs...")

    rows = [load_mode_summary(m) for m in MODE_ORDER]
    df = pd.DataFrame(rows)

    # Save raw summary (full)
    raw_csv = RESULTS_DIR / "week12_3_utility_raw_summary.csv"
    df.to_csv(raw_csv, index=False)
    print(f"[Week-12.3] Wrote: {raw_csv}")

    # Add overhead ratios vs none (report-friendly table)
    base_time = df.loc[df["mode"] == "none", "mean_round_time"]
    base_comm = df.loc[df["mode"] == "none", "mean_total_comm"]

    base_time_val = float(base_time.iloc[0]) if len(base_time) == 1 and np.isfinite(float(base_time.iloc[0])) else np.nan
    base_comm_val = float(base_comm.iloc[0]) if len(base_comm) == 1 and np.isfinite(float(base_comm.iloc[0])) else np.nan

    df["overhead_time_x"] = np.nan
    df["overhead_comm_x"] = np.nan

    if np.isfinite(base_time_val) and base_time_val != 0:
        df["overhead_time_x"] = df["mean_round_time"] / base_time_val

    if np.isfinite(base_comm_val) and base_comm_val != 0:
        df["overhead_comm_x"] = df["mean_total_comm"] / base_comm_val

    # Final table (compact columns)
    conv_col = f"conv_round_acc>={CONV_THRESH:.2f}"
    final_cols = [
        "mode",
        "final_acc",
        "mean_acc",
        "mean_round_time",
        "mean_total_comm",
        "overhead_time_x",
        "overhead_comm_x",
        conv_col,
        "note",
        "source_csv",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    final_table = df[final_cols].copy()

    final_csv = RESULTS_DIR / "week12_3_utility_final_table.csv"
    final_table.to_csv(final_csv, index=False)
    print(f"[Week-12.3] Wrote: {final_csv}")

    # --------------------------
    # Plots: overhead bars
    # --------------------------
    plot_overhead_bar(
        df=df,
        metric_col="mean_round_time",
        out=RESULTS_DIR / "week12_3_overhead_round_time.png",
        title="Week-12.3 Utility Overhead: Mean Round Time (vs none)",
        ylabel="Time overhead (× baseline none)",
        abs_fmt_fn=lambda v: f"{v:.3f} s",
    )

    plot_overhead_bar(
        df=df,
        metric_col="mean_total_comm",
        out=RESULTS_DIR / "week12_3_overhead_total_comm.png",
        title="Week-12.3 Utility Overhead: Mean Total Communication per Round (vs none)",
        ylabel="Communication overhead (× baseline none)",
        abs_fmt_fn=_format_bytes,
    )

    # --------------------------
    # Pareto plots (use mean_acc as primary utility)
    # --------------------------
    plot_pareto_scatter(
        df=df,
        xcol="mean_round_time",
        ycol="mean_acc",
        out=RESULTS_DIR / "week12_3_pareto_meanacc_vs_time.png",
        title="Week-12.3 Pareto: Mean Accuracy vs Mean Round Time",
        xlabel="Mean round time (sec) — lower is better",
        ylabel="Mean accuracy across rounds — higher is better",
        show_xy=True,
    )

    plot_pareto_scatter(
        df=df,
        xcol="mean_total_comm",
        ycol="mean_acc",
        out=RESULTS_DIR / "week12_3_pareto_meanacc_vs_comm.png",
        title="Week-12.3 Pareto: Mean Accuracy vs Mean Total Communication",
        xlabel="Mean total comm per round (bytes_up + bytes_down) — lower is better",
        ylabel="Mean accuracy across rounds — higher is better",
        show_xy=True,
    )

    # Final-acc Pareto (ALWAYS generate for the report, even if saturated)
    finite_final = df["final_acc"].to_numpy(dtype=float)
    finite_final = finite_final[np.isfinite(finite_final)]
    if len(finite_final) >= 2 and (float(np.max(finite_final)) - float(np.min(finite_final))) <= FINAL_ACC_FLAT_EPS:
        print("[Week-12.3] NOTE: final_acc is saturated/flat across modes -> plot will show overlap (handled by label stacking).")

    plot_pareto_scatter(
        df=df,
        xcol="mean_round_time",
        ycol="final_acc",
        out=RESULTS_DIR / "week12_3_pareto_finalacc_vs_time.png",
        title="Week-12.3 Pareto: Final Accuracy vs Mean Round Time",
        xlabel="Mean round time (sec) — lower is better",
        ylabel="Final accuracy — higher is better",
        show_xy=True,
    )

    plot_pareto_scatter(
        df=df,
        xcol="mean_total_comm",
        ycol="final_acc",
        out=RESULTS_DIR / "week12_3_pareto_finalacc_vs_comm.png",
        title="Week-12.3 Pareto: Final Accuracy vs Mean Total Communication",
        xlabel="Mean total comm per round (bytes_up + bytes_down) — lower is better",
        ylabel="Final accuracy — higher is better",
        show_xy=True,
    )


    # --------------------------
    # Heatmap (normalized + raw)
    # --------------------------
    plot_utility_heatmap(
        df=df,
        out=RESULTS_DIR / "week12_3_utility_matrix_normalized.png",
    )

    # Console summary (quick sanity check)
    show = [
        "mode",
        "final_acc",
        "mean_acc",
        "mean_round_time",
        "mean_total_comm",
        "overhead_time_x",
        "overhead_comm_x",
        f"conv_round_acc>={CONV_THRESH:.2f}",
        "note",
    ]
    show = [c for c in show if c in df.columns]
    print("\n[Week-12.3] Summary (sanity check):")
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
