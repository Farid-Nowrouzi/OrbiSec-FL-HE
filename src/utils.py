import os
import math
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------
# Reproducibility
# ----------------------------
def set_global_seeds(seed: int = 42) -> None:
    """Set global RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (safer for reproducible results)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ----------------------------
# Metrics
# ----------------------------
def binary_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Binary accuracy.
    - If preds are logits (outside [0,1]), apply sigmoid automatically.
    - Ensures shapes are compatible.
    """
    if preds is None or labels is None:
        return float("nan")

    preds = preds.detach()
    labels = labels.detach()

    # Make sure labels are float 0/1
    if labels.dtype != torch.float32:
        labels = labels.float()

    # If preds look like logits, convert to probabilities
    # (common for BCEWithLogitsLoss models)
    if preds.min().item() < 0.0 or preds.max().item() > 1.0:
        preds = torch.sigmoid(preds)

    # Flatten safely
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Threshold
    preds_bin = (preds > 0.5).float()

    return (preds_bin == labels).float().mean().item()


# ----------------------------
# Bytes / sizes
# ----------------------------
def estimate_bytes(tensor_dict: Dict[str, Any]) -> int:
    """
    Estimate bytes used by a state_dict-like mapping.

    Works for:
    - model.state_dict()
    - any dict-like object with tensor values

    Ignores non-tensors safely.
    """
    total = 0
    for _, v in tensor_dict.items():
        if torch.is_tensor(v):
            total += int(v.nelement()) * int(v.element_size())
    return int(total)


def params_nbytes(params: List[np.ndarray]) -> int:
    """Total bytes for a list of NumPy parameter arrays."""
    return int(sum(int(p.nbytes) for p in params))


# ----------------------------
# CSV logging (FIXED)
# ----------------------------
def _read_csv_header_cols(path: str) -> List[str]:
    """Read just the first line (header) columns from an existing CSV."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    if not header:
        return []
    return [c.strip() for c in header.split(",")]


def append_to_csv(path: str, row_dict: Dict[str, Any], columns_order: Optional[List[str]] = None) -> None:
    """
    Append one row to a CSV file robustly.

     Fixes the classic bug:
       - Existing file has a header column order
       - New rows must be appended in EXACTLY that order
       - Otherwise pandas will append misordered values -> corrupted CSV.

    Behavior:
    - If file doesn't exist: write with header (stable column order)
    - If file exists: align row to header columns before append
    - If row contains new columns not in header: expand schema safely
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.exists(path):
        # Create new file: choose a stable order
        cols = columns_order if columns_order else list(row_dict.keys())
        df = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in cols}])
        df.to_csv(path, index=False)
        return

    # File exists: align to header
    header_cols = _read_csv_header_cols(path)

    # If header is empty/corrupt, recreate cleanly
    if not header_cols:
        cols = columns_order if columns_order else list(row_dict.keys())
        df = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in cols}])
        df.to_csv(path, index=False)
        return

    row_cols = set(row_dict.keys())
    header_set = set(header_cols)

    # If we have new columns not in header, expand schema
    new_cols = [c for c in row_dict.keys() if c not in header_set]
    if new_cols:
        # Read existing CSV, add new columns, rewrite, then append
        old_df = pd.read_csv(path)
        for c in new_cols:
            old_df[c] = np.nan
        # Keep original order + new cols at end
        new_header = list(old_df.columns)
        old_df.to_csv(path, index=False)

        header_cols = new_header  # update header_cols

    # Align row exactly to header order
    aligned_row = {c: row_dict.get(c, np.nan) for c in header_cols}
    df = pd.DataFrame([aligned_row])

    # Append without header
    df.to_csv(path, mode="a", index=False, header=False)


# ----------------------------
# Plot helper (early weeks)
# ----------------------------
def plot_metrics(csv_path: str, save_dir: str = "results") -> None:
    """Basic utility plotting function (used in early weeks)."""
    import matplotlib
    matplotlib.use("Agg")  # headless

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Accuracy vs rounds
    plt.figure(figsize=(6, 4))
    plt.plot(df["round"], df["acc"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Rounds")
    plt.grid(True)
    plt.tight_layout()
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
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bytes_vs_rounds.png"))
    plt.close()


# ----------------------------
# Week-4: additional metric helpers (MSE, PSNR, SSIM-1D)
# ----------------------------
def mse_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error between two 1D arrays."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    return float(np.mean((x - y) ** 2))


def psnr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio for 1D signals (MAX_I = max(|x|))."""
    err = mse_1d(x, y)
    if err == 0.0:
        return float("inf")
    max_i = float(np.max(np.abs(x)))
    if max_i == 0.0:
        max_i = 1.0
    return 10.0 * math.log10((max_i * max_i) / err)


def ssim_1d(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Simple global 1D SSIM (no sliding window)."""
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
