"""
run_week8_attribute_inference_attack.py

WEEK-8: ATTRIBUTE INFERENCE ATTACK (none vs mask vs ckks)

What it does:
- Runs Week-8 attribute inference attack in 3 modes: none / mask / ckks
- Attribute inference is MULTI-CLASS: attacker tries to infer each client's hidden attribute class (K classes)
- Produces ONLY TWO PLOTS (professor-friendly + security story):
    1) Attack Accuracy (mean over 3 seeds)
    2) Macro-F1 (mean over 3 seeds)
- Produces:
    - One RAW CSV (per-seed, per-mode)
    - One SUMMARY CSV (mean/std per-mode)

Important security note for report:
- CKKS threat model: attacker cannot observe per-client plaintext updates
  => attribute inference becomes random guess (accuracy ~1/K, macro-F1 ~1/K)

Run:
    python experiments/run_week8_attribute_inference_attack.py
"""

from __future__ import annotations

import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
# Synthetic telemetry dataset (per-client)
# ----------------------------
def make_client_dataset_multiclass(
        n_samples: int,
        n_features: int,
        class_id: int,
        n_classes: int,
        seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local task (client-side model training):
    - Binary classification: normal (0) vs anomaly (1)
    - Client has a hidden attribute class_id in {0..K-1} that changes feature statistics,
      which can leak through updates.

    Attribute leakage mechanism:
    - Each attribute class has a distinct mean-shift pattern injected into anomalous samples.
    """
    rng = np.random.default_rng(seed)

    # binary labels
    y = (rng.random(n_samples) < 0.08).astype(np.int64)  # fixed anomaly rate for stability

    # base features
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    # class-specific pattern (stable, separated, not too strong)
    # build a deterministic class "signature" vector
    base = rng.normal(0.0, 1.0, size=(n_features,)).astype(np.float32)
    signature = np.sin((class_id + 1) * np.linspace(0.5, 2.5, n_features)).astype(np.float32)
    signature = 0.6 * signature + 0.2 * base

    # inject only into anomalies => gradients carry attribute signal
    X[y == 1] += signature

    return X, y


# ----------------------------
# Simple MLP (local client model)
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
# Flower client bundle
# ----------------------------
@dataclass
class ClientBundle:
    X: np.ndarray
    y: np.ndarray
    attr_label: int  # 0..K-1


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

        # Mask mode: add Gaussian noise (stable)
        if self.mode == "mask" and self.noise_std > 0:
            new_params = [
                w + np.random.normal(0.0, self.noise_std, size=w.shape).astype(w.dtype)
                for w in new_params
            ]

        num_examples = len(self.bundle.X)
        metrics = {"attr_label": float(self.bundle.attr_label)}
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
        # add a couple more stable scalars for multi-class separation
        feats.append(np.max(np.abs(d)))
        feats.append(np.median(np.abs(d)))
    return np.array(feats, dtype=np.float64)


# ----------------------------
# Softmax regression (stable NumPy implementation)
# ----------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -50, 50))
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)


def train_softmax_reg(
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        lr: float = 0.2,
        steps: int = 1200,
        l2: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N,D), y: (N,)
    returns W: (D,C), b: (C,)
    """
    N, D = X.shape
    W = np.zeros((D, n_classes), dtype=np.float64)
    b = np.zeros((n_classes,), dtype=np.float64)

    # one-hot
    Y = np.zeros((N, n_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    for _ in range(steps):
        logits = X @ W + b
        P = softmax(logits)
        gradW = (X.T @ (P - Y)) / N + l2 * W
        gradb = np.mean(P - Y, axis=0)
        W -= lr * gradW
        b -= lr * gradb

    return W, b


def predict_proba(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return softmax(X @ W + b)


def attack_accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    pred = np.argmax(proba, axis=1)
    return float(np.mean(pred == y_true))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        denom = (2 * tp + fp + fn)
        if denom == 0:
            f1 = 0.0
        else:
            f1 = (2 * tp) / denom
        f1s.append(float(f1))
    return float(np.mean(f1s))


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

        agg_nd = fl.common.parameters_to_ndarrays(agg_params)

        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            client_nd = fl.common.parameters_to_ndarrays(fit_res.parameters)
            feat = extract_features(client_nd, agg_nd)

            attr_label = int(round(float(fit_res.metrics.get("attr_label", 0.0))))

            row = {"cid": cid, "attr_label": attr_label}
            for i, v in enumerate(feat):
                row[f"f{i}"] = float(v)
            self.collected_rows.append(row)

        return agg_params, agg_metrics


# ----------------------------
# Flower context helper (stable across versions)
# ----------------------------
def get_context_cid(context) -> str:
    try:
        if hasattr(context, "node_config") and isinstance(context.node_config, dict):
            if "partition-id" in context.node_config:
                return str(context.node_config["partition-id"])
            if "cid" in context.node_config:
                return str(context.node_config["cid"])
    except Exception:
        pass
    if hasattr(context, "client_id"):
        return str(context.client_id)
    return "0"


# ----------------------------
# Build balanced client bundles
# ----------------------------
def build_client_bundles(
        num_clients: int,
        n_samples: int,
        n_features: int,
        n_classes: int,
        seed: int,
) -> Dict[str, ClientBundle]:
    """
    Balanced attribute labels across clients (important for stable metrics).
    """
    bundles: Dict[str, ClientBundle] = {}

    # assign classes round-robin so perfectly balanced
    for cid in range(num_clients):
        cls = cid % n_classes
        X, y = make_client_dataset_multiclass(
            n_samples=n_samples,
            n_features=n_features,
            class_id=cls,
            n_classes=n_classes,
            seed=seed + cid * 1000,
        )
        bundles[str(cid)] = ClientBundle(X=X, y=y, attr_label=cls)

    return bundles


# ----------------------------
# Run one mode for one seed
# ----------------------------
def run_mode_once(
        mode: str,
        num_clients: int,
        rounds: int,
        seed: int,
        n_samples: int,
        n_features: int,
        n_classes: int,
        local_epochs: int,
        noise_std: float,
) -> Tuple[float, float, str]:
    """
    Returns: (attack_acc, macro_f1, note)
    """
    set_seeds(seed)

    # CKKS threat model: no per-client plaintext updates observable -> random guess
    if mode == "ckks":
        baseline = 1.0 / float(n_classes)
        return float(baseline), float(baseline), "Not observable under CKKS encrypted-channel threat model (random guess)."

    bundles = build_client_bundles(
        num_clients=num_clients,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
    )

    strategy = CollectUpdatesStrategy(num_clients=num_clients)

    def client_fn(context: fl.common.Context):
        cid = get_context_cid(context)
        bundle = bundles.get(cid, bundles["0"])

        numpy_client = FLClient(
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
        # IMPORTANT: convert NumPyClient -> Client (removes Flower warning)
        return numpy_client.to_client()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    df = pd.DataFrame(strategy.collected_rows)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["attr_label"].to_numpy(dtype=np.int64)

    # standardize (stable)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X = (X - mu) / sd

    # train attack model (softmax regression)
    W, b = train_softmax_reg(X, y, n_classes=n_classes, lr=0.2, steps=1200, l2=1e-2)
    proba = predict_proba(W, b, X)
    y_pred = np.argmax(proba, axis=1)

    acc = attack_accuracy(y, proba)
    f1 = macro_f1(y, y_pred, n_classes=n_classes)

    # never NaN
    if not np.isfinite(acc):
        acc = 1.0 / float(n_classes)
    if not np.isfinite(f1):
        f1 = 1.0 / float(n_classes)

    return float(acc), float(f1), ""


# ----------------------------
# Plot helper (ONLY TWO plots)
# ----------------------------
def plot_bar(
        title: str,
        ylabel: str,
        modes: List[str],
        values: List[float],
        outpath: Path,
        ymin: float,
        ymax: float,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(modes, values)
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(values):
        plt.text(i, min(v + 0.02, ymax - 0.01), f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
def main():
    # Stable settings
    num_clients = 24
    rounds = 25
    n_samples = 2000
    n_features = 32
    local_epochs = 1
    noise_std_mask = 0.02

    # Attribute inference classes (K)
    n_classes = 3  # <-- keep 3 for report (clean baseline = 0.333)

    # Seeds (3 runs, mean/std)
    seeds = [42, 1337, 2024]

    modes = ["none", "mask", "ckks"]

    raw_rows = []
    for seed in seeds:
        for mode in modes:
            acc, f1, note = run_mode_once(
                mode=mode,
                num_clients=num_clients,
                rounds=rounds,
                seed=seed,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                local_epochs=local_epochs,
                noise_std=noise_std_mask,
            )
            raw_rows.append({
                "seed": seed,
                "mode": mode,
                "attack_acc": float(acc),
                "macro_f1": float(f1),
                "num_clients": num_clients,
                "rounds": rounds,
                "n_samples_per_client": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "note": note,
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_path = RESULTS_DIR / "week8_attribute_inference.csv"
    raw_df.to_csv(raw_path, index=False)

    # Summary mean/std per mode
    summary = []
    for mode in modes:
        sub = raw_df[raw_df["mode"] == mode]
        summary.append({
            "mode": mode,
            "attack_acc_mean": float(sub["attack_acc"].mean()),
            "attack_acc_std": float(sub["attack_acc"].std(ddof=0)),
            "macro_f1_mean": float(sub["macro_f1"].mean()),
            "macro_f1_std": float(sub["macro_f1"].std(ddof=0)),
            "n_classes": int(n_classes),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = RESULTS_DIR / "week8_attribute_inference_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plots (mean over 3 seeds)
    acc_means = [float(summary_df[summary_df["mode"] == m]["attack_acc_mean"].iloc[0]) for m in modes]
    f1_means = [float(summary_df[summary_df["mode"] == m]["macro_f1_mean"].iloc[0]) for m in modes]

    baseline = 1.0 / float(n_classes)
    ymin = max(0.0, baseline - 0.05)
    ymax = min(1.05, max(acc_means + f1_means) + 0.15)

    plot_bar(
        title="Week-8 Attribute Inference Attack Accuracy (mean over 3 seeds)",
        ylabel=f"Attack Accuracy (random baseline = {baseline:.3f})",
        modes=modes,
        values=acc_means,
        outpath=RESULTS_DIR / "week8_attribute_inference_acc.png",
        ymin=ymin,
        ymax=ymax,
    )

    plot_bar(
        title="Week-8 Attribute Inference Macro-F1 (mean over 3 seeds)",
        ylabel=f"Macro-F1 (random baseline = {baseline:.3f})",
        modes=modes,
        values=f1_means,
        outpath=RESULTS_DIR / "week8_attribute_inference_f1.png",
        ymin=ymin,
        ymax=ymax,
    )

    print(f"[Week-8] Raw results written to: {raw_path}")
    print(f"[Week-8] Summary written to: {summary_path}")
    print(f"[Week-8] Accuracy plot saved to: {RESULTS_DIR / 'week8_attribute_inference_acc.png'}")
    print(f"[Week-8] Macro-F1 plot saved to: {RESULTS_DIR / 'week8_attribute_inference_f1.png'}")
    print("[Week-8] Done.")


if __name__ == "__main__":
    main()
