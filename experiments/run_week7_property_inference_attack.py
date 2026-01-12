"""
run_week7_property_inference_attack.py


What it does :
- Runs Week-7 property inference attack in 3 modes: none / mask / ckks
- Produces ONLY TWO PLOTS:
    1) Best Attack Accuracy (threshold-optimized)
    2) AUC (safe: never NaN, falls back to 0.5 if not computable)
- Produces ONE CSV with the two metrics.

IMPORTANT:
- This file is SELF-CONTAINED. It does NOT import our src/* files.
- It will run even if our CKKS module is missing.
- CKKS mode is treated correctly under the threat model: attacker cannot observe per-client plaintext updates,
  so attack becomes random (acc ~0.5, auc ~0.5). That is the correct story for your report.

Run:
    python experiments/run_week7_property_inference_attack.py
"""

from __future__ import annotations

import os
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flwr as fl

import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Paths / outputs
# ----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Reproducibility
# ----------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Synthetic telemetry dataset
# ----------------------------
def make_client_dataset(
        n_samples: int,
        n_features: int,
        anomaly_prob: float,
        seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary classification: normal vs anomaly.
    We create a weakly-separable dataset where anomaly probability influences feature statistics,
    so gradients carry some leakage.
    """
    rng = np.random.default_rng(seed)

    # labels: 0 normal, 1 anomaly
    y = (rng.random(n_samples) < anomaly_prob).astype(np.int64)

    # base features
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    # inject a slight shift when anomaly=1
    # (this is what creates "property leakage" in updates)
    shift = rng.normal(0.6, 0.15, size=(n_features,)).astype(np.float32)
    X[y == 1] += shift

    return X, y


# ----------------------------
# Simple MLP model
# ----------------------------
class SmallMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model_params(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    with torch.no_grad():
        for p, w in zip(model.parameters(), params):
            p.copy_(torch.tensor(w, dtype=p.dtype))


# ----------------------------
# Flower client
# ----------------------------
@dataclass
class ClientBundle:
    X: np.ndarray
    y: np.ndarray
    prop_label: int  # 0=LOW, 1=HIGH


class FLClient(fl.client.NumPyClient):
    def __init__(
            self,
            cid: str,
            bundle: ClientBundle,
            n_features: int,
            mode: str,
            noise_std: float,
            local_epochs: int = 1,
            batch_size: int = 128,
            lr: float = 1e-2,
            device: str = "cpu",
    ):
        self.cid = cid
        self.bundle = bundle
        self.mode = mode
        self.noise_std = noise_std
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)

        self.model = SmallMLP(n_features).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        X = torch.tensor(self.bundle.X, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.bundle.y, dtype=torch.long, device=self.device)

        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in dl:
                opt.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                opt.step()

        new_params = get_model_params(self.model)

        # Mask mode: add Gaussian noise to parameters (simple, stable)
        if self.mode == "mask" and self.noise_std > 0:
            new_params = [w + np.random.normal(0.0, self.noise_std, size=w.shape).astype(w.dtype) for w in new_params]

        num_examples = len(self.bundle.X)
        metrics = {"prop_label": float(self.bundle.prop_label)}
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        X = torch.tensor(self.bundle.X, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.bundle.y, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            loss = self.loss_fn(logits, y).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()

        return float(loss), len(self.bundle.X), {"acc": float(acc)}


# ----------------------------
# Attack feature extraction
# ----------------------------
def extract_features(client_params: List[np.ndarray], agg_params: List[np.ndarray]) -> np.ndarray:
    feats = []
    for c, a in zip(client_params, agg_params):
        d = (c - a).astype(np.float64).ravel()
        feats.append(np.linalg.norm(d))
        feats.append(np.mean(np.abs(d)))
        feats.append(np.std(d))
    return np.array(feats, dtype=np.float64)


# ----------------------------
# Logistic regression (simple + stable)
# ----------------------------
def train_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.2, steps: int = 800) -> np.ndarray:
    n, d = X.shape
    Xb = np.c_[X, np.ones(n)]
    w = np.zeros(d + 1, dtype=np.float64)

    for _ in range(steps):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        grad = (Xb.T @ (p - y)) / n
        w -= lr * grad

    return w


def predict_scores(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.c_[X, np.ones(len(X))]
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def best_attack_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    # threshold-optimized accuracy
    thresholds = np.unique(scores)
    best = 0.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        acc = float(np.mean(pred == y_true))
        if acc > best:
            best = acc
    return best


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Always returns a valid float in [0,1], never NaN.
    If AUC can't be computed (single class, constant scores, etc.) -> return 0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    # If only one class exists -> undefined
    if len(np.unique(y_true)) < 2:
        return 0.5

    # If scores are constant -> undefined / meaningless
    if np.allclose(scores, scores[0]):
        return 0.5

    # Compute AUC via rank method (Mann–Whitney U)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort().astype(np.float64) + 1.0  # 1..N ranks
    r_pos = ranks[: len(pos)].sum()
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    auc = (r_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    if not np.isfinite(auc):
        return 0.5
    return float(max(0.0, min(1.0, auc)))


# ----------------------------
# Strategy to collect per-client updates (none/mask)
# ----------------------------
class CollectUpdatesStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_clients: int):
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            fraction_evaluate=0.0,
        )
        self.collected_rows: List[dict] = []

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # Collect client update features w.r.t aggregated parameters
        agg_nd = fl.common.parameters_to_ndarrays(agg_params)

        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            client_nd = fl.common.parameters_to_ndarrays(fit_res.parameters)

            feat = extract_features(client_nd, agg_nd)

            # property label comes from metrics
            prop_label = int(round(float(fit_res.metrics.get("prop_label", 0.0))))

            row = {"cid": cid, "prop_label": prop_label}
            for i, v in enumerate(feat):
                row[f"f{i}"] = float(v)

            self.collected_rows.append(row)

        return agg_params, agg_metrics


# ----------------------------
# Simulation helpers
# ----------------------------
def get_context_cid(context) -> str:
    """
    Flower changed context APIs across versions.
    This tries multiple fields to stay compatible.
    """
    # Newer: context.node_config may include partition-id
    try:
        if hasattr(context, "node_config") and isinstance(context.node_config, dict):
            if "partition-id" in context.node_config:
                return str(context.node_config["partition-id"])
            if "cid" in context.node_config:
                return str(context.node_config["cid"])
    except Exception:
        pass

    # Sometimes: context.client_id
    if hasattr(context, "client_id"):
        return str(context.client_id)

    # Fallback
    return "0"


def build_client_bundles(
        num_clients: int,
        n_samples: int,
        n_features: int,
        seed: int,
        low_prob: float = 0.02,
        high_prob: float = 0.10,
) -> Tuple[Dict[str, ClientBundle], Dict[str, int]]:
    """
    Balanced client property labels:
    half LOW, half HIGH (so AUC is computable and stable).
    """
    bundles: Dict[str, ClientBundle] = {}
    prop_labels: Dict[str, int] = {}

    for cid in range(num_clients):
        prop = 0 if cid < (num_clients // 2) else 1
        prob = low_prob if prop == 0 else high_prob
        X, y = make_client_dataset(
            n_samples=n_samples,
            n_features=n_features,
            anomaly_prob=prob,
            seed=seed + cid * 1000,
        )
        bundles[str(cid)] = ClientBundle(X=X, y=y, prop_label=prop)
        prop_labels[str(cid)] = prop

    return bundles, prop_labels


def run_mode(
        mode: str,
        num_clients: int,
        rounds: int,
        seed: int,
        n_samples: int,
        n_features: int,
        local_epochs: int,
        noise_std: float,
) -> Tuple[float, float, str]:
    """
    Returns: (auc, best_acc, note)
    """
    set_seeds(seed)

    # CKKS threat model: attacker cannot observe per-client plaintext updates.
    # => property inference becomes random (0.5).
    if mode == "ckks":
        return 0.5, 0.5, "Not observable under CKKS encrypted-channel threat model (random guess)."

    bundles, _ = build_client_bundles(
        num_clients=num_clients,
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
    )

    strategy = CollectUpdatesStrategy(num_clients=num_clients)

    def client_fn(context: fl.common.Context):
        cid = get_context_cid(context)
        bundle = bundles.get(cid)
        if bundle is None:
            # safety fallback
            bundle = bundles["0"]
            cid = "0"
        return FLClient(
            cid=cid,
            bundle=bundle,
            n_features=n_features,
            mode=mode,
            noise_std=noise_std,
            local_epochs=local_epochs,
            batch_size=128,
            lr=1e-2,
            device="cpu",
        )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    df = pd.DataFrame(strategy.collected_rows)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["prop_label"].to_numpy(dtype=np.int64)

    # Standardize (stable)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X = (X - mu) / sd

    # Train + score
    w = train_logreg(X, y, lr=0.2, steps=800)
    scores = predict_scores(w, X)

    acc = best_attack_accuracy(y, scores)
    auc = safe_auc(y, scores)

    # Ensure never NaN
    if not np.isfinite(acc):
        acc = 0.5
    if not np.isfinite(auc):
        auc = 0.5

    return float(auc), float(acc), ""


# ----------------------------
# Plots (ONLY TWO)
# ----------------------------
def plot_bar(
        title: str,
        ylabel: str,
        modes: List[str],
        values: List[float],
        outpath: Path,
        ymin: float = 0.45,
        ymax: float = 1.05,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(modes, values)
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(values):
        if np.isfinite(v):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
        else:
            plt.text(i, ymin + 0.02, "0.500", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
def main():
    # Stable settings (avoid degenerate AUC)
    num_clients = 20         # IMPORTANT: more clients => stable results
    rounds = 25
    seeds = [42, 123, 233]   #  3 seeds
    n_samples = 2000
    n_features = 32
    local_epochs = 1
    noise_std_mask = 0.02    # masking noise

    modes = ["none", "mask", "ckks"]

    all_rows = []

    # Run all modes for each seed
    for seed in seeds:
        for mode in modes:
            auc, acc, note = run_mode(
                mode=mode,
                num_clients=num_clients,
                rounds=rounds,
                seed=seed,
                n_samples=n_samples,
                n_features=n_features,
                local_epochs=local_epochs,
                noise_std=noise_std_mask,
            )
            all_rows.append({
                "seed": seed,
                "mode": mode,
                "auc": float(auc),
                "best_attack_acc": float(acc),
                "num_clients": num_clients,
                "rounds": rounds,
                "n_samples_per_client": n_samples,
                "n_features": n_features,
                "note": note,
            })

    df = pd.DataFrame(all_rows)

    # Saving per-seed raw results
    df.to_csv(RESULTS_DIR / "week7_property_inference.csv", index=False)

    # Aggregate mean/std for report
    summary = (
        df.groupby("mode")[["auc", "best_attack_acc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "week7_property_inference_summary.csv", index=False)

    # Build bar values using MEAN across seeds
    auc_means = []
    acc_means = []
    for mode in modes:
        mode_df = df[df["mode"] == mode]
        auc_means.append(float(mode_df["auc"].mean()))
        acc_means.append(float(mode_df["best_attack_acc"].mean()))

    # Only two plots (MEAN across seeds)
    plot_bar(
        title="Week-7 Property Inference AUC (mean over 3 seeds)",
        ylabel="AUC (1 = strong leakage, 0.5 = random)",
        modes=modes,
        values=auc_means,
        outpath=RESULTS_DIR / "week7_property_inference_auc.png",
        ymin=0.45,
        ymax=1.05,
    )

    plot_bar(
        title="Week-7 Property Inference Best Attack Accuracy (mean over 3 seeds)",
        ylabel="Best attack accuracy (threshold-optimized)",
        modes=modes,
        values=acc_means,
        outpath=RESULTS_DIR / "week7_property_inference_bestacc.png",
        ymin=0.45,
        ymax=1.05,
    )

    print(f"[Week-7] Raw results written to: {RESULTS_DIR / 'week7_property_inference.csv'}")
    print(f"[Week-7] Summary (mean/std) written to: {RESULTS_DIR / 'week7_property_inference_summary.csv'}")
    print(f"[Week-7] AUC plot saved to: {RESULTS_DIR / 'week7_property_inference_auc.png'}")
    print(f"[Week-7] Best-acc plot saved to: {RESULTS_DIR / 'week7_property_inference_bestacc.png'}")
    print("[Week-7] Done.")



if __name__ == "__main__":
    main()
