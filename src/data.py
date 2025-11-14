import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic_telemetry(
    num_clients=8,
    samples_per_client=2000,
    window_size=32,
    anomaly_prob=0.05,
    seed=42
):
    """
    Synthetic telemetry per client:
    sinusoid + noise + random anomaly spikes.
    Returns: list of (X_tensor, y_tensor) per client.
    """
    rng = np.random.default_rng(seed)
    clients_data = []

    for cid in range(num_clients):
        t = np.linspace(0, 50, samples_per_client)
        base = np.sin(t) + 0.05 * rng.normal(size=samples_per_client)

        anomalies = rng.random(samples_per_client) < anomaly_prob
        base[anomalies] += rng.normal(3.0, 1.0, size=np.sum(anomalies))
        base = base.astype(np.float32)

        X, y = [], []
        for i in range(samples_per_client - window_size):
            window = base[i:i+window_size]
            label = 1.0 if np.any(anomalies[i:i+window_size]) else 0.0
            X.append(window)
            y.append(label)

        X = torch.tensor(np.stack(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        clients_data.append((X, y))

    return clients_data

def get_dataloaders_for_client(X, y, batch_size=32):
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader
