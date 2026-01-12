"""
run_week4_dlg_minitest.py

Week-4 experiment: DLG-style gradient leakage mini-test on synthetic telemetry.

Modes (aligned with Weeks 7–9):
  - none  : attacker sees true gradients -> DLG works
  - mask  : attacker sees noisy gradients -> DLG degrades
  - ckks  : attacker cannot see plaintext gradients -> NOT OBSERVABLE (random / N/A)

Outputs (in /results):
  - week4_dlg_results.csv
  - week4_dlg_reconstructions.png      (original vs none vs mask)
  - week4_dlg_metrics.png              (bar chart: none vs mask vs ckks-not-observable)
"""

from __future__ import annotations

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

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generate_synthetic_telemetry
from src.model import MLP
from src.utils import (
    set_global_seeds,
    mse_1d,
    psnr_1d,
    ssim_1d,
)


# ---------------------------------------------------------------------
# DLG helpers
# ---------------------------------------------------------------------
def compute_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
    model.zero_grad(set_to_none=True)
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
    model = model.to(device)
    model.eval()

    # input dimension
    try:
        in_dim = model.model[0].in_features
    except Exception:
        in_dim = 32

    x_hat = torch.randn((1, in_dim), device=device, requires_grad=True)
    opt = torch.optim.Adam([x_hat], lr=lr)
    criterion = nn.BCELoss()

    target_grads = [tg.to(device) for tg in target_grads]
    y = y.to(device)

    for _ in range(num_steps):
        opt.zero_grad(set_to_none=True)

        preds_hat = model(x_hat)
        loss_hat = criterion(preds_hat, y)

        grads_hat = torch.autograd.grad(loss_hat, model.parameters(), create_graph=True)

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

    # 1) Data for one client; pick one sample
    clients_data = generate_synthetic_telemetry(
        num_clients=1,
        samples_per_client=2000,
        window_size=32,
        anomaly_prob=0.05,
        seed=seed,
    )
    X, y = clients_data[0]
    x_real = X[0].unsqueeze(0)  # [1, 32]
    y_real = y[0].unsqueeze(0)  # [1, 1]

    # 2) Model
    model = MLP(input_dim=x_real.shape[1], hidden_dim=32)

    # 3) Victim gradients
    grads_true = compute_gradients(model, x_real, y_real)

    # 4) Masked gradients
    grads_masked = [g + noise_std * torch.randn_like(g) for g in grads_true]

    # 5) DLG reconstructions (none & mask only)
    print("[Week-4] Running DLG reconstruction (none)...")
    x_rec_none = dlg_reconstruct(model, grads_true, y_real, num_steps=300, lr=0.1, device=device)

    print("[Week-4] Running DLG reconstruction (mask)...")
    x_rec_mask = dlg_reconstruct(model, grads_masked, y_real, num_steps=300, lr=0.1, device=device)

    # 6) Metrics
    x_real_np = x_real.squeeze(0).numpy()
    x_none_np = x_rec_none.squeeze(0).numpy()
    x_mask_np = x_rec_mask.squeeze(0).numpy()

    results: Dict[str, Dict[str, object]] = {}

    results["none"] = {
        "mse": float(mse_1d(x_real_np, x_none_np)),
        "psnr": float(psnr_1d(x_real_np, x_none_np)),
        "ssim": float(ssim_1d(x_real_np, x_none_np)),
        "note": "",
    }
    results["mask"] = {
        "mse": float(mse_1d(x_real_np, x_mask_np)),
        "psnr": float(psnr_1d(x_real_np, x_mask_np)),
        "ssim": float(ssim_1d(x_real_np, x_mask_np)),
        "note": "",
    }

    # CKKS: attacker cannot observe plaintext gradients -> DLG not observable
    results["ckks"] = {
        "mse": "N/A",
        "psnr": "N/A",
        "ssim": "N/A",
        "note": "Not observable under CKKS encrypted-channel threat model.",
    }

    # 7) Save CSV
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "week4_dlg_results.csv"

    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "mse", "psnr", "ssim", "note"])
        for mode in ["none", "mask", "ckks"]:
            r = results[mode]
            writer.writerow([mode, r["mse"], r["psnr"], r["ssim"], r["note"]])

    print(f"[Week-4] Metrics written to: {csv_path}")

    # 8) Plot reconstructions (original vs none vs mask)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recon_path = results_dir / "week4_dlg_reconstructions.png"
    plt.figure(figsize=(7.5, 4))
    t = np.arange(len(x_real_np))
    plt.plot(t, x_real_np, label="original", linewidth=2)
    plt.plot(t, x_none_np, label="none (DLG)", linestyle="--")
    plt.plot(t, x_mask_np, label="mask (DLG)", linestyle=":")
    plt.xlabel("Time index")
    plt.ylabel("Telemetry value")
    plt.title("Week-4 DLG Reconstruction (none vs mask)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(recon_path, dpi=170)
    plt.close()
    print(f"[Week-4] Reconstruction plot saved to: {recon_path}")

    # 9) Metrics bar plot (clean for report; CKKS shown as 'N/A')
    metrics_path = results_dir / "week4_dlg_metrics.png"

    def bar_metric(metric_name: str, ylabel: str):
        vals = []
        labels = ["none", "mask", "ckks"]
        for m in labels:
            v = results[m][metric_name]
            vals.append(0.0 if isinstance(v, str) else float(v))

        plt.figure(figsize=(8, 4))
        plt.bar(labels, vals)
        plt.ylabel(ylabel)
        plt.title(f"Week-4 DLG Leakage Metric: {metric_name.upper()} (CKKS = Not observable)")
        plt.grid(axis="y", linestyle="--", alpha=0.35)

        for i, m in enumerate(labels):
            v = results[m][metric_name]
            if isinstance(v, str):
                plt.text(i, 0.02, "N/A", ha="center", va="bottom")
            else:
                plt.text(i, float(v) + (max(vals) * 0.03 + 1e-6), f"{float(v):.3g}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(results_dir / f"week4_dlg_{metric_name}.png", dpi=170)
        plt.close()

    bar_metric("mse", "MSE (lower reconstruction error)")
    bar_metric("psnr", "PSNR (higher = closer reconstruction)")
    bar_metric("ssim", "SSIM (higher = more similar)")

    # 10) Print summary
    print("\n[Week-4] Summary (DLG leakage metrics)")
    for mode in ["none", "mask", "ckks"]:
        r = results[mode]
        if mode == "ckks":
            print(f"  {mode:6s} -> N/A (not observable)")
        else:
            print(f"  {mode:6s} -> MSE={r['mse']:.4e}, PSNR={r['psnr']:.2f}, SSIM={r['ssim']:.4f}")

    print("\n[Week-4] Done.")


if __name__ == "__main__":
    run_week4()
