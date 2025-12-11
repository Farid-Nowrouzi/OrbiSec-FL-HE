# experiments/run_week6_membership_attack.py

"""
Week-6: Membership Inference Attack (MIA) mini-experiment.

We simulate three modes:
- "none": standard training (high leakage)
- "mask": parameters get small Gaussian noise after each step (reduced leakage)
- "ckks": very underfitted model (few epochs) to mimic strong protection

We:
1) Generate synthetic telemetry windows (train + non-train).
2) Train one MLP per mode.
3) Run a membership inference attack using per-sample reconstruction loss.
4) Compute AUC and best attack accuracy.
5) Save CSV and a bar-plot of AUC across modes.
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

# ---------------------------------------------------------------------
# Make sure we can import the project modules (src.*)
# ---------------------------------------------------------------------
# Project root = parent of "experiments" directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from src.model import MLP  # uses your existing MLP definition


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODES = ["none", "mask", "ckks"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_windows(n: int = 200, length: int = 32) -> np.ndarray:
    """
    Simulate telemetry windows (same length as your MLP input).

    We use a noisy sinusoid to mimic a smooth telemetry curve.
    """
    x = np.linspace(0, 4 * np.pi, length)
    windows = []
    for _ in range(n):
        noise = np.random.normal(0.0, 0.15, size=length)
        windows.append((np.sin(x) + noise).astype(np.float32))
    return np.stack(windows, axis=0)


def train_model_for_mode(
        mode: str,
        train_windows: np.ndarray,
        device: torch.device,
) -> nn.Module:
    """
    Train a small MLP autoencoder-ish model for a given mode.

    - input: telemetry window (size 32)
    - target: same window (reconstruction)
    """

    model = MLP(input_dim=train_windows.shape[1], hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x_train = torch.tensor(train_windows, dtype=torch.float32).to(device)

    # Training schedule per mode
    if mode == "none":
        epochs = 15
        noise_std = 0.0
    elif mode == "mask":
        epochs = 15
        noise_std = 0.01  # small parameter noise after each step
    elif mode == "ckks":
        epochs = 4       # underfit on purpose -> harder for MIA
        noise_std = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(x_train)
        loss = loss_fn(preds, x_train)
        loss.backward()
        optimizer.step()

        # Add small Gaussian noise to parameters for "mask" mode
        if noise_std > 0.0:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.normal(0.0, noise_std, size=p.shape).to(device))

    return model


def compute_per_sample_losses(
        model: nn.Module, data: np.ndarray, device: torch.device
) -> np.ndarray:
    """
    Compute MSE reconstruction loss for each sample.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).to(device)
        preds = model(x)
        # Mean over feature dimension => shape (N,)
        losses = ((preds - x) ** 2).mean(dim=1).cpu().numpy()
    return losses


def auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute AUC using rank-based formula (no sklearn needed).

    labels: 1 = member, 0 = non-member
    scores: larger score => more likely member
    """
    # Sort by scores ascending
    order = np.argsort(scores)
    sorted_labels = labels[order]

    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Ranks are 1..N
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks_sum = np.sum(ranks[sorted_labels == 1])

    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_attack_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Best possible attack accuracy over all possible thresholds.

    We classify as member if score >= threshold.
    """
    thresholds = np.unique(scores)
    best_acc = 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
    return float(best_acc)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:
    print("[Week-6] Running membership inference experiment...")
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Week-6] Using device: {device}")

    # 1) Create members (train) and non-members (nontrain)
    n_members = 200
    n_nonmembers = 200
    train_windows = generate_synthetic_windows(n=n_members, length=32)
    nontrain_windows = generate_synthetic_windows(n=n_nonmembers, length=32)

    summary_rows = []

    for mode in MODES:
        print(f"[Week-6] Training model for mode: {mode}")
        model = train_model_for_mode(mode, train_windows, device)

        # 2) Compute losses for members & non-members
        train_losses = compute_per_sample_losses(model, train_windows, device)
        nontrain_losses = compute_per_sample_losses(model, nontrain_windows, device)

        # Membership inference: lower loss => more likely member
        labels = np.concatenate(
            [
                np.ones_like(train_losses, dtype=int),
                np.zeros_like(nontrain_losses, dtype=int),
            ]
        )
        scores = np.concatenate(
            [-train_losses, -nontrain_losses]
        )  # higher = more "member-like"

        auc = auc_from_scores(labels, scores)
        acc = best_attack_accuracy(labels, scores)

        summary_rows.append(
            {
                "mode": mode,
                "auc": auc,
                "attack_acc": acc,
                "mean_train_loss": float(train_losses.mean()),
                "mean_nontrain_loss": float(nontrain_losses.mean()),
            }
        )

        print(
            f"[Week-6] Mode={mode:5s}  AUC={auc:.3f}  best_acc={acc:.3f}  "
            f"mean_train_loss={train_losses.mean():.4f}  "
            f"mean_nontrain_loss={nontrain_losses.mean():.4f}"
        )

    # 3) Save CSV
    df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "week6_membership_attack.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Week-6] Membership attack summary written to: {csv_path}")

    # 4) Plot AUC bar chart
    plt.figure(figsize=(7, 4))
    plt.bar(df["mode"], df["auc"])
    plt.title("Week-6 Membership Inference AUC (none vs mask vs ckks)")
    plt.ylabel("AUC (1 = strong leakage, 0.5 = random guessing)")
    plt.ylim(0.45, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    out_plot = RESULTS_DIR / "week6_membership_auc.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=160)
    plt.close()

    print(f"[Week-6] Membership AUC plot saved to: {out_plot}")


if __name__ == "__main__":
    main()
