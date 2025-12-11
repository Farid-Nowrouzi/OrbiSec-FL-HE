import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List


def set_global_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_accuracy(preds, labels):
    preds = (preds > 0.5).float()
    return (preds == labels).float().mean().item()


def estimate_bytes(tensor_dict):
    """Estimate bytes used by a state_dict (PyTorch tensors)."""
    total = 0
    for _, v in tensor_dict.items():
        total += v.nelement() * v.element_size()
    return total


def params_nbytes(params: List[np.ndarray]) -> int:
    """Total bytes for a list of NumPy parameter arrays."""
    return sum(p.nbytes for p in params)


def append_to_csv(path, row_dict):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", index=False, header=False)


def plot_metrics(csv_path, save_dir="results"):
    import matplotlib
    matplotlib.use("Agg")  # headless

    df = pd.read_csv(csv_path)

    # Accuracy vs rounds
    plt.figure(figsize=(6, 4))
    plt.plot(df["round"], df["acc"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "acc_vs_rounds.png"))
    plt.close()

    # Bytes vs rounds
    plt.figure(figsize=(6, 4))
    plt.plot(df["round"], df["bytes_up"], label="uplink")
    plt.plot(df["round"], df["bytes_down"], label="downlink")
    plt.xlabel("Round")
    plt.ylabel("Bytes")
    plt.legend()
    plt.title("Bytes vs Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "bytes_vs_rounds.png"))
    plt.close()

# === Week-4: additional metric helpers (MSE, PSNR, SSIM-1D) ===

import math
from typing import Tuple


def mse_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error between two 1D arrays."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    return float(np.mean((x - y) ** 2))


def psnr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio for 1D signals.

    We use the dynamic range of x as MAX_I.
    """
    err = mse_1d(x, y)
    if err == 0.0:
        return float("inf")
    max_i = float(np.max(np.abs(x)))
    if max_i == 0.0:
        max_i = 1.0
    return 10.0 * math.log10((max_i * max_i) / err)


def ssim_1d(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simple 1D SSIM (Structural Similarity Index) implementation.

    This is a simplified, global SSIM (no sliding window), which is
    sufficient to compare reconstructed telemetry windows.
    """
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()

    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov_xy = np.mean((x - mu_x) * (y - mu_y))

    num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)
    if den == 0.0:
        return 1.0
    return float(num / den)
