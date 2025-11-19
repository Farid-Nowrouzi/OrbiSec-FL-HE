"""
run_week2_he_demo.py

Week-2 experiment:
- Simulate federated client updates as numpy arrays.
- Compute plain FedAvg in the clear.
- Compute FedAvg with CKKS homomorphic encryption (TenSEAL).
- Compare the two (MSE error) and log the result to results/results_he.csv.

This does not change our Week-1 Flower pipeline; it is a separate HE demo
that we can describe in the report as:
    "secure aggregation of model updates using CKKS, verified against
     plain FedAvg."
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np

from src.secure_he import CKKSSecureAggregator, FlattenInfo


def simulate_client_updates(
    num_clients: int,
    num_layers: int,
    layer_dim: int,
    noise_scale: float = 0.01,
) -> List[List[np.ndarray]]:
    """
    Create synthetic "updates" for each client.

    We start from a common base vector and add small Gaussian noise per client
    (so that averaging makes sense).
    """
    updates: List[List[np.ndarray]] = []

    base = [
        np.random.randn(layer_dim).astype(np.float64) for _ in range(num_layers)
    ]

    for _ in range(num_clients):
        client_layers: List[np.ndarray] = []
        for layer in base:
            noise = np.random.normal(
                loc=0.0, scale=noise_scale, size=layer.shape
            ).astype(np.float64)
            client_layers.append(layer + noise)
        updates.append(client_layers)

    return updates


def plain_fedavg(
    client_updates: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Standard FedAvg on numpy arrays (no encryption).
    """
    num_clients = len(client_updates)
    num_layers = len(client_updates[0])

    avg: List[np.ndarray] = []
    for layer_idx in range(num_layers):
        stacked = np.stack(
            [client_updates[c][layer_idx] for c in range(num_clients)],
            axis=0,
        )
        avg.append(stacked.mean(axis=0))
    return avg


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Simulate client updates
    # ------------------------------------------------------------------
    num_clients = 8
    num_layers = 3
    layer_dim = 256

    rng = np.random.default_rng(seed=21780)
    np.random.seed(21780)  # just to be extra deterministic

    client_updates = simulate_client_updates(
        num_clients=num_clients,
        num_layers=num_layers,
        layer_dim=layer_dim,
        noise_scale=0.01,
    )

    # ------------------------------------------------------------------
    # 2) Plain FedAvg (baseline, in the clear)
    # ------------------------------------------------------------------
    plain_avg = plain_fedavg(client_updates)

    # ------------------------------------------------------------------
    # 3) HE FedAvg with CKKS
    # ------------------------------------------------------------------
    aggregator = CKKSSecureAggregator()

    # Flatten all client updates
    flat_per_client, info = aggregator.flatten_updates(client_updates)
    assert isinstance(info, FlattenInfo)

    # Encrypt each client's flat vector
    ciphertexts = aggregator.encrypt_updates(flat_per_client)

    # Aggregate homomorphically (encrypted sum)
    enc_sum = aggregator.aggregate_encrypted(ciphertexts)

    # Decrypt and reconstruct layer-wise tensors (including division by N)
    he_avg_layers = aggregator.decrypt_aggregate(enc_sum, info, num_clients)

    # ------------------------------------------------------------------
    # 4) Compare plain vs HE average
    # ------------------------------------------------------------------
    layer_mse = [mse(p, h) for p, h in zip(plain_avg, he_avg_layers)]
    total_mse = float(np.mean(layer_mse))

    print("=== Week-2 CKKS HE demo ===")
    print(f"Clients          : {num_clients}")
    print(f"Layers           : {num_layers}")
    print(f"Layer dimension  : {layer_dim}")
    print(f"MSE per layer    : {layer_mse}")
    print(f"Mean MSE (overall): {total_mse:.4e}")

    # ------------------------------------------------------------------
    # 5) Save a small CSV in results/
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "results_he.csv"

    header = [
        "num_clients",
        "num_layers",
        "layer_dim",
        "mean_mse",
        "layer_mse_0",
        "layer_mse_1",
        "layer_mse_2",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(
            [
                num_clients,
                num_layers,
                layer_dim,
                total_mse,
                layer_mse[0],
                layer_mse[1],
                layer_mse[2],
            ]
        )

    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
