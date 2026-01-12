# experiments/run_week6_membership_attack.py
"""
Week-6: Membership Inference Attack (MIA) — loss-based,  consistent with the project's MLP.

Key fixes vs old version:
- Uses src/model.py MLP (binary classifier: Sigmoid output).
- Creates a real binary classification task from synthetic telemetry windows.
- Encourages overfitting in "none" so membership signal exists (members tend to have lower loss).
- "mask" adds small parameter noise during training to reduce memorization.
- "ckks_like" simulates strong protection / reduced observability via:
    - fewer epochs (early stopping effect),
    - stronger weight decay,
    - optional label smoothing,
    - smaller hidden dimension.

Outputs (in ./results):
- week6_membership_attack.csv
- week6_membership_auc.png
- week6_membership_loss_distributions.png
- week6_membership_roc.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import sys
import argparse

# ---------------------------------------------------------------------
# Ensure project root is importable (src.*)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.model import MLP  # <- our project model (Sigmoid binary classifier)


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODES = ["none", "mask", "ckks_like"]


# ---------------------------------------------------------------------
# Repro helpers
# ---------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Synthetic telemetry -> binary label task
# ---------------------------------------------------------------------
def generate_windows(n: int, length: int, seed: int) -> np.ndarray:
    """
    Generate synthetic telemetry windows as noisy sinusoids with random phase/amplitude.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, length)

    windows = []
    for _ in range(n):
        amp = rng.uniform(0.8, 1.4)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0.0, 0.15, size=length)
        w = amp * np.sin(x + phase) + noise
        windows.append(w.astype(np.float32))
    return np.stack(windows, axis=0)


def make_labels(windows: np.ndarray, seed: int) -> np.ndarray:
    """
    Create binary labels (0/1) based on a simple "anomaly score":
    label = 1 if max(|window|) > threshold; else 0

    This yields a non-trivial classification problem with both classes present.
    """
    rng = np.random.default_rng(seed)
    scores = np.max(np.abs(windows), axis=1)
    # Set threshold at ~60th percentile -> keeps both classes balanced-ish
    thr = float(np.quantile(scores, 0.60))
    y = (scores > thr).astype(np.float32)

    # Small label noise (optional but helps realism)
    flip_prob = 0.03
    flips = rng.random(len(y)) < flip_prob
    y[flips] = 1.0 - y[flips]

    return y


# ---------------------------------------------------------------------
# MIA utilities: per-sample loss, ROC, AUC, best accuracy
# ---------------------------------------------------------------------
def per_sample_bce_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """
    Compute per-sample BCE loss (no reduction).
    """
    model.eval()
    with torch.no_grad():
        pred = model(x).view(-1)  # shape (N,)
        y = y.view(-1)
        loss = nn.BCELoss(reduction="none")(pred, y)
    return loss.detach().cpu().numpy()


def roc_curve_from_scores(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC curve points (FPR, TPR) without sklearn.
    labels: 1=member, 0=non-member
    scores: higher => more likely member
    """
    order = np.argsort(scores)[::-1]  # descending
    y = labels[order]
    s = scores[order]

    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tpr = [0.0]
    fpr = [0.0]

    tp = 0
    fp = 0
    prev_score = None

    for yi, si in zip(y, s):
        if prev_score is None or si != prev_score:
            tpr.append(tp / P)
            fpr.append(fp / N)
            prev_score = si

        if yi == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(tp / P)
    fpr.append(fp / N)

    return np.array(fpr), np.array(tpr)


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Trapezoidal AUC.
    """
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    return float(np.trapz(tpr, fpr))


def best_attack_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Best threshold accuracy: predict member if score >= t.
    """
    thresholds = np.unique(scores)
    best = 0.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        acc = float(np.mean(pred == labels))
        if acc > best:
            best = acc
    return best


# ---------------------------------------------------------------------
# Training per mode
# ---------------------------------------------------------------------
def train_for_mode(
        mode: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        device: torch.device,
        *,
        seed: int,
        hidden_dim_none: int = 64,
        hidden_dim_ckks: int = 16,
        epochs_none: int = 40,
        epochs_mask: int = 40,
        epochs_ckks: int = 8,
) -> nn.Module:
    """
    Train MLP in a way that creates a membership signal for 'none' by encouraging overfitting.

    - none: higher capacity + more epochs + low regularization
    - mask: same as none but adds small parameter noise each step (reduces memorization)
    - ckks_like: reduced capacity + fewer epochs + stronger weight decay + label smoothing
    """
    assert mode in MODES

    torch.manual_seed(seed)

    input_dim = x_train.shape[1]

    if mode == "none":
        hidden_dim = hidden_dim_none
        epochs = epochs_none
        weight_decay = 0.0
        noise_std = 0.0
        label_smoothing = 0.0

    elif mode == "mask":
        hidden_dim = hidden_dim_none
        epochs = epochs_mask
        weight_decay = 0.0
        noise_std = 0.01  # parameter noise
        label_smoothing = 0.0

    else:  # ckks_like
        hidden_dim = hidden_dim_ckks
        epochs = epochs_ckks
        weight_decay = 2e-3
        noise_std = 0.0
        label_smoothing = 0.05  # soften targets a bit

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1)

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x).view(-1)

        # label smoothing (only for ckks_like)
        if label_smoothing > 0.0:
            y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            y_smooth = y

        loss = loss_fn(pred, y_smooth)
        loss.backward()
        opt.step()

        # mask mode: add parameter noise after each update
        if noise_std > 0.0:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.normal(0.0, noise_std, size=p.shape, device=device))

    return model


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_members", type=int, default=180)      # smaller => easier to overfit
    parser.add_argument("--n_nonmembers", type=int, default=400)
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Week-6] Device: {device}")

    all_rows: List[Dict] = []

    # We also keep one representative trial's curves for plotting ROC + loss distributions
    rep_data: Dict[str, Dict[str, np.ndarray]] = {}

    for t in range(args.trials):
        trial_seed = args.seed + 1000 * t
        set_seeds(trial_seed)

        # Generate members/non-members from same generator but different seeds
        x_mem = generate_windows(args.n_members, args.length, seed=trial_seed)
        y_mem = make_labels(x_mem, seed=trial_seed + 1)

        x_non = generate_windows(args.n_nonmembers, args.length, seed=trial_seed + 999)
        y_non = make_labels(x_non, seed=trial_seed + 1001)

        for mode in MODES:
            model = train_for_mode(
                mode,
                x_mem,
                y_mem,
                device,
                seed=trial_seed,
            )

            x_mem_t = torch.tensor(x_mem, dtype=torch.float32, device=device)
            y_mem_t = torch.tensor(y_mem, dtype=torch.float32, device=device)

            x_non_t = torch.tensor(x_non, dtype=torch.float32, device=device)
            y_non_t = torch.tensor(y_non, dtype=torch.float32, device=device)

            # Loss-based membership signal:
            # members tend to have LOWER loss => use score = -loss so higher score => more member-like
            mem_losses = per_sample_bce_loss(model, x_mem_t, y_mem_t)
            non_losses = per_sample_bce_loss(model, x_non_t, y_non_t)

            labels = np.concatenate([np.ones_like(mem_losses, dtype=int), np.zeros_like(non_losses, dtype=int)])
            scores = np.concatenate([-mem_losses, -non_losses])

            fpr, tpr = roc_curve_from_scores(labels, scores)
            auc = auc_trapz(fpr, tpr)
            best_acc = best_attack_accuracy(labels, scores)

            row = {
                "trial": t,
                "mode": mode,
                "auc": float(auc),
                "best_attack_acc": float(best_acc),
                "mean_member_loss": float(mem_losses.mean()),
                "mean_nonmember_loss": float(non_losses.mean()),
            }
            all_rows.append(row)

            # Save representative trial (t=0) for plots
            if t == 0:
                rep_data[mode] = {
                    "mem_losses": mem_losses,
                    "non_losses": non_losses,
                    "labels": labels,
                    "scores": scores,
                }

            print(
                f"[Week-6][trial={t}] mode={mode:9s} AUC={auc:.3f} best_acc={best_acc:.3f} "
                f"mem_loss={mem_losses.mean():.4f} nonmem_loss={non_losses.mean():.4f}"
            )

    df = pd.DataFrame(all_rows)

    # Aggregate (mean ± std)
    agg = (
        df.groupby("mode")[["auc", "best_attack_acc", "mean_member_loss", "mean_nonmember_loss"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten columns
    agg.columns = ["mode"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[1:]]
    out_csv = RESULTS_DIR / "week6_membership_attack.csv"
    agg.to_csv(out_csv, index=False)
    print(f"[Week-6] Wrote: {out_csv}")
    print(agg.to_string(index=False))

    # ----------------------------
    # Plot 1: AUC bar (mean ± std)
    # ----------------------------
    plt.figure(figsize=(7.5, 4.2))
    x = np.arange(len(MODES))
    auc_means = [float(agg.loc[agg["mode"] == m, "auc_mean"].values[0]) for m in MODES]
    auc_stds = [float(agg.loc[agg["mode"] == m, "auc_std"].values[0]) for m in MODES]
    plt.bar(MODES, auc_means, yerr=auc_stds, capsize=4)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0.45, 1.02)
    plt.title("Week-6: Membership Inference AUC (loss-based)")
    plt.ylabel("AUC (0.5 = random guessing)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(auc_means):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_auc = RESULTS_DIR / "week6_membership_auc.png"
    plt.savefig(out_auc, dpi=170)
    plt.close()
    print(f"[Week-6] Saved: {out_auc}")

    # ----------------------------
    # Plot 2: Loss distributions (rep trial)
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    # consistent x-limits across all subplots
    all_losses = np.concatenate(
        [rep_data[m]["mem_losses"] for m in MODES] + [rep_data[m]["non_losses"] for m in MODES]
    )
    xmin, xmax = float(np.min(all_losses)), float(np.max(all_losses))

    for ax, mode in zip(axes, MODES):
        mem_losses = rep_data[mode]["mem_losses"]
        non_losses = rep_data[mode]["non_losses"]
        ax.hist(mem_losses, bins=18, alpha=0.7, density=True, label="member")
        ax.hist(non_losses, bins=18, alpha=0.7, density=True, label="non-member")
        ax.set_title(mode)
        ax.set_xlabel("Per-sample BCE loss")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Week-6: Member vs Non-member Loss Distributions (MIA signal)", y=1.03)
    plt.tight_layout()
    out_hist = RESULTS_DIR / "week6_membership_loss_distributions.png"
    plt.savefig(out_hist, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[Week-6] Saved: {out_hist}")

    # ----------------------------
    # Plot 3: ROC curves (rep trial)
    # ----------------------------
    plt.figure(figsize=(7.0, 5.0))
    for mode in MODES:
        labels = rep_data[mode]["labels"]
        scores = rep_data[mode]["scores"]
        fpr, tpr = roc_curve_from_scores(labels, scores)
        auc = auc_trapz(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{mode} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Week-6: ROC Curves (Membership Inference via loss)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_roc = RESULTS_DIR / "week6_membership_roc.png"
    plt.savefig(out_roc, dpi=170)
    plt.close()
    print(f"[Week-6] Saved: {out_roc}")

    print("[Week-6] Done.")


if __name__ == "__main__":
    main()
