"""
run_week4_dlg_minitest.py

Week-4 experiment: DLG-style gradient leakage mini-test on synthetic telemetry.

We:
  1. Generate synthetic telemetry for one client.
  2. Take a single window x_real (shape [1, 32]) and label y_real.
  3. Compute the gradients of the model parameters for this sample.
  4. Run a DLG-style reconstruction to recover x from gradients.

We repeat this for three "modes":
  - baseline: attacker sees true gradients
  - mask:     attacker sees noisy gradients (Gaussian mask)
  - ckks:     attacker does NOT see gradients in plaintext (not applicable)

Outputs:
  - results/results_dlg.csv (one row per mode)
  - results/dlg_reconstructions.png (original vs baseline vs mask)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Make sure we can import from src/
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.data import generate_synthetic_telemetry
from src.model import MLP
from src.utils import (
    set_global_seeds,
    mse_1d,
    psnr_1d,
    ssim_1d,
)


# ---------------------------------------------------------------------
# Core DLG-style reconstruction
# ---------------------------------------------------------------------
def compute_gradients(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute gradients of the loss w.r.t. model parameters for (x, y)."""
    model.zero_grad()
    criterion = nn.BCELoss()
    preds = model(x)
    loss = criterion(preds, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [g.detach().clone() for g in grads]


def dlg_reconstruct(
        model: nn.Module,
        target_grads: List[torch.Tensor],
        y: torch.Tensor,
        num_steps: int = 300,
        lr: float = 0.1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Given target gradients (list of tensors), reconstruct input x_hat such that
    gradients of the loss at x_hat match the target gradients.

    Only x_hat is optimized; model parameters are fixed.
    """
    model = model.to(device)
    model.eval()

    # Start from random noise with the same shape as the true sample
    # (batch_size=1, window_size=32)
    x_hat = torch.randn((1, model.model[0].in_features), device=device, requires_grad=True)

    opt = torch.optim.Adam([x_hat], lr=lr)
    criterion = nn.BCELoss()

    target_grads = [tg.to(device) for tg in target_grads]

    for _ in range(num_steps):
        opt.zero_grad()

        # Forward + loss
        preds_hat = model(x_hat)
        loss_hat = criterion(preds_hat, y.to(device))

        # Gradients w.r.t. model parameters (for x_hat)
        grads_hat = torch.autograd.grad(loss_hat, model.parameters(), create_graph=True)

        # Gradient-matching loss
        grad_loss = 0.0
        for gh, gt in zip(grads_hat, target_grads):
            grad_loss = grad_loss + ((gh - gt) ** 2).sum()

        grad_loss.backward()
        opt.step()

    return x_hat.detach().cpu()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def run_week4(seed: int = 123, noise_std: float = 0.05) -> None:
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Week-4] Using device: {device}")

    # 1) Generate data for a single client and pick one sample
    clients_data = generate_synthetic_telemetry(
        num_clients=1,
        samples_per_client=2000,
        window_size=32,
        anomaly_prob=0.05,
        seed=seed,
    )
    X, y = clients_data[0]
    x_real = X[0].unsqueeze(0)  # shape [1, 32]
    y_real = y[0].unsqueeze(0)  # shape [1, 1]

    # 2) Model (fresh MLP, same as FL)
    model = MLP(input_dim=x_real.shape[1], hidden_dim=32)

    # 3) Compute "victim" gradients
    grads_true = compute_gradients(model, x_real, y_real)

    # 4) Prepare gradient variants for the three modes
    grads_masked = [
        g + noise_std * torch.randn_like(g) for g in grads_true
    ]

    # CKKS mode: no plaintext gradients available → we do not create grads_ckks

    # 5) Run DLG-style reconstruction for baseline and mask
    print("[Week-4] Running DLG reconstruction (baseline)...")
    x_rec_baseline = dlg_reconstruct(
        model=model,
        target_grads=grads_true,
        y=y_real,
        num_steps=300,
        lr=0.1,
        device=device,
    )

    print("[Week-4] Running DLG reconstruction (mask)...")
    x_rec_mask = dlg_reconstruct(
        model=model,
        target_grads=grads_masked,
        y=y_real,
        num_steps=300,
        lr=0.1,
        device=device,
    )

    # 6) Compute metrics (MSE, PSNR, SSIM)
    x_real_np = x_real.squeeze(0).numpy()
    x_rec_baseline_np = x_rec_baseline.squeeze(0).numpy()
    x_rec_mask_np = x_rec_mask.squeeze(0).numpy()

    results: Dict[str, Dict[str, float]] = {}

    results["baseline"] = {
        "mse": mse_1d(x_real_np, x_rec_baseline_np),
        "psnr": psnr_1d(x_real_np, x_rec_baseline_np),
        "ssim": ssim_1d(x_real_np, x_rec_baseline_np),
    }

    results["mask"] = {
        "mse": mse_1d(x_real_np, x_rec_mask_np),
        "psnr": psnr_1d(x_real_np, x_rec_mask_np),
        "ssim": ssim_1d(x_real_np, x_rec_mask_np),
    }

    # CKKS: attacker cannot access gradients in plaintext → not applicable.
    # We record NaN to emphasise that DLG cannot even be started.
    results["ckks"] = {
        "mse": float("nan"),
        "psnr": float("nan"),
        "ssim": float("nan"),
    }

    # 7) Save CSV
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "results_dlg.csv"

    import csv

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "mse", "psnr", "ssim"])
        for mode, metrics in results.items():
            writer.writerow([mode, metrics["mse"], metrics["psnr"], metrics["ssim"]])

    print(f"[Week-4] Metrics written to: {csv_path}")

    # 8) Plot original vs reconstructions (baseline & mask)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    t = np.arange(len(x_real_np))
    plt.plot(t, x_real_np, label="original", linewidth=2)
    plt.plot(t, x_rec_baseline_np, label="baseline (DLG)", linestyle="--")
    plt.plot(t, x_rec_mask_np, label="mask (DLG)", linestyle=":")
    plt.xlabel("Time index")
    plt.ylabel("Telemetry value")
    plt.title("Week-4 DLG Reconstruction (Baseline vs Mask)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = results_dir / "dlg_reconstructions.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"[Week-4] Reconstruction plot saved to: {fig_path}")

    # 9) Pretty-print summary
    print("\n[Week-4] Summary (DLG leakage metrics)")
    for mode in ["baseline", "mask", "ckks"]:
        m = results[mode]
        print(
            f"  {mode:8s} -> MSE={m['mse']:.4e}, "
            f"PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}"
        )


if __name__ == "__main__":
    run_week4()
