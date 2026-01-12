# experiments/run_week11_1_ckks_optimizations.py
from __future__ import annotations

import os
import sys
import time
import inspect
from typing import Any, Callable, Dict

import pandas as pd
import torch
import flwr as fl

# ---------------------------------------------------------------------
# Project-root import support (so running from /experiments works)
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Week-11 config (variants + naming)
# ---------------------------------------------------------------------
from src.week11_config import (
    Week11RunConfig,
    build_week11_variants,
    variant_order,
    week11_csv_name,
)

# ---------------------------------------------------------------------
# our real project modules
# ---------------------------------------------------------------------
from src.data import generate_synthetic_telemetry, get_dataloaders_for_client
from src.model import MLP
from src.client import get_client_fn
from src.server import make_strategy


# ---------------------------------------------------------------------
# Utility: call a function using only parameters it supports
# (keeps Week-11 runner stable even if server.py changes slightly)
# ---------------------------------------------------------------------
def _call_with_signature_filter(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """
    Calls fn(**kwargs) but only passes keys that exist in fn signature.
    If fn has **kwargs, everything is passed.

    This makes experiment runners resilient to small refactors in server/client code.
    """
    sig = inspect.signature(fn)

    # If **kwargs exists, pass everything safely
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)

    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


# ---------------------------------------------------------------------
# Build a pooled validation loader (server-side evaluation)
# ---------------------------------------------------------------------
def _build_val_loader_from_clients(
        clients_data,
        batch_size: int = 64,
        val_ratio: float = 0.2,
):
    """
    Builds a single pooled validation DataLoader by taking the last `val_ratio`
    split from each client and concatenating them.

    This keeps the evaluation story consistent across variants:
    - same validation set distribution
    - comparable accuracy curves in the CSV outputs
    """
    val_X_all = []
    val_y_all = []

    for (X, y) in clients_data:
        n = int(X.shape[0])
        n_val = max(1, int(n * float(val_ratio)))
        split = max(1, n - n_val)

        X_val = X[split:]
        y_val = y[split:]

        val_X_all.append(X_val)
        val_y_all.append(y_val)

    Xv = torch.cat(val_X_all, dim=0)
    yv = torch.cat(val_y_all, dim=0)

    # NOTE: your get_dataloaders_for_client returns a DataLoader for a given (X, y)
    val_loader = get_dataloaders_for_client(Xv, yv, batch_size=int(batch_size))
    return val_loader


# ---------------------------------------------------------------------
# Run ONE Week-11 variant
# ---------------------------------------------------------------------
def run_one_variant(cfg: Week11RunConfig, variant_name: str, optim_spec) -> pd.DataFrame:
    """
    Runs exactly one Week-11 CKKS optimization variant and returns the resulting per-round DataFrame.

    Design notes for report:
    - Secure mode is fixed to CKKS for Week-11.
    - The only controlled change across runs is the OptimSpec variant.
    - The output is always saved as a CSV per variant for later analysis (Week-11.2).
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, week11_csv_name(variant_name))

    # -----------------------------
    # 1) Device + model
    # -----------------------------
    device_str = getattr(cfg, "device", "cpu")
    device = torch.device(device_str)

    input_dim = int(getattr(cfg, "input_dim", 32))
    hidden_dim = int(getattr(cfg, "hidden_dim", 32))
    dropout_prob = float(getattr(cfg, "dropout_prob", 0.0))

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # -----------------------------
    # 2) Synthetic clients data
    # -----------------------------
    samples_per_client = int(getattr(cfg, "samples_per_client", 2000))
    window_size = int(getattr(cfg, "window_size", 32))
    anomaly_prob = float(getattr(cfg, "anomaly_prob", 0.05))

    clients_data = generate_synthetic_telemetry(
        num_clients=int(cfg.num_clients_total),
        samples_per_client=samples_per_client,
        window_size=window_size,
        anomaly_prob=anomaly_prob,
        seed=int(cfg.seed),
    )

    # -----------------------------
    # 3) Server-side validation loader
    # -----------------------------
    val_ratio = float(getattr(cfg, "val_ratio", 0.2))
    val_batch_size = int(getattr(cfg, "val_batch_size", 64))
    val_loader = _build_val_loader_from_clients(
        clients_data=clients_data,
        batch_size=val_batch_size,
        val_ratio=val_ratio,
    )

    # -----------------------------
    # 4) Flower client_fn (our existing factory)
    # -----------------------------
    local_epochs = int(getattr(cfg, "local_epochs", 1))
    noise_std = float(getattr(cfg, "noise_std", 0.01))

    # IMPORTANT:
    # our src.client.get_client_fn signature is:
    #   get_client_fn(clients_data, device, model_cls, local_epochs, secure_mode, noise_std)
    client_fn = get_client_fn(
        clients_data=clients_data,
        device=device,
        model_cls=MLP,
        local_epochs=local_epochs,
        secure_mode="ckks",
        noise_std=noise_std,
    )

    # -----------------------------
    # 5) Strategy (server side)
    # -----------------------------
    # This must match your real src.server.make_strategy() requirements.
    # From the errors: it requires at least:
    #   model, val_loader, device, results_csv, num_clients, dropout_prob, seed
    strategy_kwargs = dict(
        model=model,
        val_loader=val_loader,
        device=device,
        results_csv=out_csv,
        num_clients=int(cfg.num_clients_total),
        dropout_prob=float(dropout_prob),
        seed=int(cfg.seed),

        # Week-11 experiment metadata / controls:
        secure_mode="ckks",
        he_scheme="ckks",

        # Optimization object (passed only if our server supports it)
        optim_spec=optim_spec,
        optim=optim_spec,
        optimization=optim_spec,

        # If our server supports per-round participation knobs:
        num_clients_per_round=int(cfg.num_clients_per_round),
    )

    strategy = _call_with_signature_filter(make_strategy, strategy_kwargs)

    # -----------------------------
    # 6) Simulation run
    # -----------------------------
    t0 = time.time()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(cfg.num_clients_total),
        config=fl.server.ServerConfig(num_rounds=int(cfg.rounds)),
        strategy=strategy,
    )

    elapsed = float(time.time() - t0)

    # -----------------------------
    # 7) Load CSV (single source of truth)
    # -----------------------------
    if not os.path.exists(out_csv):
        # Safety fallback: create a minimal CSV so Week-11.2 does not crash
        df = pd.DataFrame([{
            "round": int(cfg.rounds),
            "acc": float("nan"),
            "round_time": elapsed / max(int(cfg.rounds), 1),
            "bytes_up": float("nan"),
            "bytes_down": float("nan"),
            "variant": variant_name,
        }])
        df.to_csv(out_csv, index=False)
        return df

    df = pd.read_csv(out_csv)

    # Add variant label (useful for downstream groupby)
    if "variant" not in df.columns:
        df["variant"] = variant_name

    # If round_time is missing, estimate it from wall-clock (better than nulls)
    if "round_time" not in df.columns:
        df["round_time"] = elapsed / max(len(df), 1)

    return df


# ---------------------------------------------------------------------
# Main: run 5 variants => produce 5 CSVs
# ---------------------------------------------------------------------
def main() -> None:
    cfg = Week11RunConfig()
    variants = build_week11_variants()

    print("\n[Week-11] CKKS optimizations — running 5 variants (will produce 5 CSVs)\n")

    for name in variant_order():
        if name not in variants:
            raise RuntimeError(
                f"[Week-11] Config mismatch: '{name}' not found in build_week11_variants()."
            )

        spec = variants[name]
        print(f"[Week-11] Running: {name}")

        df = run_one_variant(cfg, name, spec)

        out_csv = os.path.join(cfg.results_dir, week11_csv_name(name))
        print(f"[Week-11] Saved: {out_csv} (rows={len(df)})\n")

    print("[Week-11] Done. Next step:")
    print("  python experiments/run_week11_2_analyze_ckks_optimizations.py\n")


if __name__ == "__main__":
    main()
