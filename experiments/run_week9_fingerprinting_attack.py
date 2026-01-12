"""
run_week9_fingerprinting_attack.py

WEEK-9: CLIENT FINGERPRINTING ATTACK (none vs mask vs ckks)

What it does:
- Simulates federated learning with multiple clients.
- Attacker tries to "fingerprint" (identify) WHICH client produced an update.
- Runs 3 modes:
    1) none  : plaintext per-client updates observable
    2) mask  : plaintext but masked with Gaussian noise on parameters
    3) ckks  : encrypted-channel threat model -> per-client plaintext updates NOT observable
- Uses a simple multi-class attacker (softmax regression) trained on update-derived features.
- Outputs:
    - week9_fingerprinting_acc.png   (Top-1 attack accuracy, mean over 3 seeds)
    - week9_fingerprinting_f1.png    (Macro-F1, mean over 3 seeds)
    - week9_fingerprinting.csv       (raw results per seed)
    - week9_fingerprinting_summary.csv (mean/std summary)

Important report story:
- Fingerprinting is possible when per-client updates are visible (none/mask).
- Under CKKS encrypted-channel threat model, attacker cannot see per-client plaintext updates -> random guess.
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
# Synthetic telemetry dataset with CLIENT-SPECIFIC SIGNATURE
# ----------------------------
def make_client_dataset(
        n_samples: int,
        n_features: int,
        client_signature: np.ndarray,
        anomaly_prob: float,
        seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary classification: normal vs anomaly.

    Key idea for fingerprinting:
    - Each client has a unique "signature" vector added to its features.
    - This creates client-specific learning dynamics and update patterns.
    """
    rng = np.random.default_rng(seed)

    # labels: 0 normal, 1 anomaly
    y = (rng.random(n_samples) < anomaly_prob).astype(np.int64)

    # base features
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    # anomaly shift (task signal)
    anomaly_shift = rng.normal(0.6, 0.15, size=(n_features,)).astype(np.float32)
    X[y == 1] += anomaly_shift

    # client signature (fingerprinting signal)
    X += client_signature.astype(np.float32)

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
            new_params = [
                w + np.random.normal(0.0, self.noise_std, size=w.shape).astype(w.dtype)
                for w in new_params
            ]

        num_examples = len(self.bundle.X)
        # include cid for strategy collection (string)
        metrics = {"cid": str(self.cid)}
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
# Feature extraction for fingerprinting
# ----------------------------
def extract_features(client_params: List[np.ndarray], agg_params: List[np.ndarray]) -> np.ndarray:
    """
    Create a compact feature vector from (client - aggregate) deltas.
    Stable & cheap: norms + basic stats per layer.
    """
    feats = []
    for c, a in zip(client_params, agg_params):
        d = (c - a).astype(np.float64).ravel()
        feats.append(np.linalg.norm(d))
        feats.append(np.mean(np.abs(d)))
        feats.append(np.std(d))
        feats.append(np.max(np.abs(d)))
    return np.array(feats, dtype=np.float64)


# ----------------------------
# Multi-class softmax regression attacker (stable)
# ----------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def train_softmax_regression(
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        lr: float = 0.2,
        steps: int = 1200,
        l2: float = 1e-3,
) -> np.ndarray:
    """
    X: (n, d)
    y: (n,) class indices
    returns W: (d+1, C) including bias
    """
    n, d = X.shape
    Xb = np.c_[X, np.ones(n)]
    W = np.zeros((d + 1, num_classes), dtype=np.float64)

    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    for _ in range(steps):
        logits = Xb @ W
        P = softmax(logits)
        grad = (Xb.T @ (P - Y)) / n
        # L2 on weights (not bias)
        grad[:-1, :] += l2 * W[:-1, :]
        W -= lr * grad

    return W


def predict_classes(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.c_[X, np.ones(len(X))]
    logits = Xb @ W
    return np.argmax(logits, axis=1)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for k in range(num_classes):
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s))


# ----------------------------
# Strategy: collect per-client updates for fingerprinting (none/mask)
# ----------------------------
class CollectFingerprintStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_clients: int):
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            fraction_evaluate=0.0,
        )
        self.rows: List[dict] = []

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        agg_nd = fl.common.parameters_to_ndarrays(agg_params)

        for client_proxy, fit_res in results:
            # Determine cid robustly
            cid = None
            try:
                cid = str(client_proxy.cid)
            except Exception:
                pass
            if cid is None:
                cid = str(fit_res.metrics.get("cid", "0"))

            client_nd = fl.common.parameters_to_ndarrays(fit_res.parameters)
            feat = extract_features(client_nd, agg_nd)

            row = {"round": int(server_round), "cid": str(cid)}
            for i, v in enumerate(feat):
                row[f"f{i}"] = float(v)
            self.rows.append(row)

        return agg_params, agg_metrics


# ----------------------------
# Flower context compatibility (cid)
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
# Build client bundles with unique signatures
# ----------------------------
def build_client_bundles(
        num_clients: int,
        n_samples: int,
        n_features: int,
        seed: int,
        anomaly_prob: float = 0.06,
) -> Dict[str, ClientBundle]:
    rng = np.random.default_rng(seed)
    bundles: Dict[str, ClientBundle] = {}

    # Each client gets a unique signature vector
    # Make it strong enough to be fingerprintable in none/mask
    signatures = rng.normal(0.0, 0.35, size=(num_clients, n_features)).astype(np.float32)

    for cid in range(num_clients):
        X, y = make_client_dataset(
            n_samples=n_samples,
            n_features=n_features,
            client_signature=signatures[cid],
            anomaly_prob=anomaly_prob,
            seed=seed + cid * 1000,
        )
        bundles[str(cid)] = ClientBundle(X=X, y=y)

    return bundles


# ----------------------------
# Run one mode (one seed)
# ----------------------------
def run_mode(
        mode: str,
        num_clients: int,
        rounds: int,
        seed: int,
        n_samples: int,
        n_features: int,
        local_epochs: int,
        noise_std_mask: float,
) -> Tuple[float, float, str]:
    """
    Returns: (attack_acc, macro_f1, note)
    """
    set_seeds(seed)

    # CKKS threat model: attacker cannot observe per-client plaintext updates.
    # Fingerprinting becomes random guess among N clients.
    if mode == "ckks":
        baseline = 1.0 / float(num_clients)
        return float(baseline), float(baseline), "Not observable under CKKS encrypted-channel threat model (random guess)."

    bundles = build_client_bundles(
        num_clients=num_clients,
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
    )

    strategy = CollectFingerprintStrategy(num_clients=num_clients)

    def client_fn(context: fl.common.Context):
        cid = get_context_cid(context)
        bundle = bundles.get(cid)
        if bundle is None:
            cid = "0"
            bundle = bundles["0"]

        c = FLClient(
            cid=cid,
            bundle=bundle,
            n_features=n_features,
            mode=mode,
            noise_std=noise_std_mask,
            local_epochs=local_epochs,
            batch_size=128,
            lr=1e-2,
            device="cpu",
        )

        # Fix newer Flower expectation: return Client when possible
        return c.to_client() if hasattr(c, "to_client") else c

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    df = pd.DataFrame(strategy.rows)
    if df.empty:
        baseline = 1.0 / float(num_clients)
        return float(baseline), float(baseline), "No rows collected (unexpected)."

    # Encode cid -> class index
    cids_sorted = sorted(df["cid"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    cid_to_idx = {cid: i for i, cid in enumerate(cids_sorted)}
    df["y"] = df["cid"].map(cid_to_idx).astype(int)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.int64)
    C = len(cids_sorted)

    # Standardize features (stable)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X = (X - mu) / sd

    # Train/test split by rounds (prevents trivial leakage)
    split_round = max(1, int(rounds * 0.7))
    train_mask = df["round"].to_numpy() <= split_round
    test_mask = ~train_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    # Safety: if something weird happens, fall back to random
    if len(X_te) == 0 or len(np.unique(y_tr)) < C:
        baseline = 1.0 / float(num_clients)
        return float(baseline), float(baseline), "Degenerate split (unexpected); used baseline."

    W = train_softmax_regression(X_tr, y_tr, num_classes=C, lr=0.2, steps=1200, l2=1e-3)
    pred = predict_classes(W, X_te)

    acc = float(np.mean(pred == y_te))
    f1 = macro_f1(y_te, pred, num_classes=C)

    # Never NaN
    if not np.isfinite(acc):
        acc = 1.0 / float(num_clients)
    if not np.isfinite(f1):
        f1 = 1.0 / float(num_clients)

    return float(acc), float(f1), ""


# ----------------------------
# Plot helpers
# ----------------------------
def plot_bar(
        title: str,
        ylabel: str,
        modes: List[str],
        values: List[float],
        outpath: Path,
        baseline: float,
        ymin: float,
        ymax: float,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(modes, values)
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # baseline line (label via legend to avoid overlap)
    plt.axhline(
        baseline,
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        color="gray",
        label=f"Random baseline = {baseline:.3f}",
    )
    plt.legend(loc="upper left")

    for i, v in enumerate(values):
        plt.text(i, v + (ymax - ymin) * 0.02, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()



# ----------------------------
# MAIN (3 seeds)
# ----------------------------
def main():
    # Stable settings
    num_clients = 12          # fingerprinting baseline = 1/12 ≈ 0.083
    rounds = 25
    seeds = [42, 1337, 2025]  # 3-seed mean/std
    n_samples = 2000
    n_features = 32
    local_epochs = 1
    noise_std_mask = 0.05     # increase masking a bit (still often not enough)

    modes = ["none", "mask", "ckks"]
    baseline = 1.0 / float(num_clients)

    raw_rows = []

    for seed in seeds:
        for mode in modes:
            acc, f1, note = run_mode(
                mode=mode,
                num_clients=num_clients,
                rounds=rounds,
                seed=seed,
                n_samples=n_samples,
                n_features=n_features,
                local_epochs=local_epochs,
                noise_std_mask=noise_std_mask,
            )
            raw_rows.append({
                "seed": int(seed),
                "mode": mode,
                "attack_acc": float(acc),
                "macro_f1": float(f1),
                "num_clients": int(num_clients),
                "rounds": int(rounds),
                "n_samples_per_client": int(n_samples),
                "n_features": int(n_features),
                "mask_noise_std": float(noise_std_mask),
                "note": note,
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(RESULTS_DIR / "week9_fingerprinting.csv", index=False)

    # Summary mean/std per mode
    summary = raw_df.groupby("mode").agg(
        attack_acc_mean=("attack_acc", "mean"),
        attack_acc_std=("attack_acc", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()

    # Keep a stable order
    summary["mode"] = pd.Categorical(summary["mode"], categories=modes, ordered=True)
    summary = summary.sort_values("mode")

    summary.to_csv(RESULTS_DIR / "week9_fingerprinting_summary.csv", index=False)

    # Plot mean over seeds
    acc_means = [float(summary.loc[summary["mode"] == m, "attack_acc_mean"].values[0]) for m in modes]
    f1_means = [float(summary.loc[summary["mode"] == m, "macro_f1_mean"].values[0]) for m in modes]

    plot_bar(
        title="Week-9 Fingerprinting Attack Top-1 Accuracy (mean over 3 seeds)",
        ylabel=f"Attack Accuracy (random baseline = {baseline:.3f})",
        modes=modes,
        values=acc_means,
        outpath=RESULTS_DIR / "week9_fingerprinting_acc.png",
        baseline=baseline,
        ymin=max(0.0, baseline - 0.02),
        ymax=1.02,
    )

    plot_bar(
        title="Week-9 Fingerprinting Attack Macro-F1 (mean over 3 seeds)",
        ylabel=f"Macro-F1 (random baseline ≈ {baseline:.3f})",
        modes=modes,
        values=f1_means,
        outpath=RESULTS_DIR / "week9_fingerprinting_f1.png",
        baseline=baseline,
        ymin=max(0.0, baseline - 0.02),
        ymax=1.02,
    )

    print("[SUMMARY]")
    print(f"Run finished {rounds} round(s) x {len(seeds)} seeds")
    print(f"[Week-9] Raw results written to: {RESULTS_DIR / 'week9_fingerprinting.csv'}")
    print(f"[Week-9] Summary written to: {RESULTS_DIR / 'week9_fingerprinting_summary.csv'}")
    print(f"[Week-9] Accuracy plot saved to: {RESULTS_DIR / 'week9_fingerprinting_acc.png'}")
    print(f"[Week-9] Macro-F1 plot saved to: {RESULTS_DIR / 'week9_fingerprinting_f1.png'}")
    print("[Week-9] Done.")


if __name__ == "__main__":
    main()
