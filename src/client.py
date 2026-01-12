import time
from typing import List, Dict, Any, Optional

import flwr as fl
import numpy as np
import torch

from src.utils import binary_accuracy, params_nbytes
from src.secure import apply_security_to_params

# Flower serialization helpers (for true payload byte measurement)
from flwr.common import ndarrays_to_parameters


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy arrays."""
    state_dict = model.state_dict()
    for (name, _), array in zip(state_dict.items(), parameters):
        state_dict[name] = torch.tensor(array)
    model.load_state_dict(state_dict)


def _pair_seed(base_seed: int, server_round: int, a: int, b: int) -> int:
    """
    Deterministically derive a per-(round, pair) seed.

    We keep it simple and stable across Python runs by avoiding hash().
    """
    lo, hi = (a, b) if a < b else (b, a)
    x = np.uint64(base_seed)
    x ^= np.uint64(server_round + 1) * np.uint64(0x9E3779B97F4A7C15)
    x ^= np.uint64(lo + 1) * np.uint64(0xBF58476D1CE4E5B9)
    x ^= np.uint64(hi + 1) * np.uint64(0x94D049BB133111EB)
    return int(x % np.uint64(2**32 - 1))


def _secagg_pairwise_mask_params(
        params: List[np.ndarray],
        client_id: int,
        num_clients: int,
        server_round: int,
        base_seed: int,
        mask_std: float,
) -> List[np.ndarray]:
    """
    Simulated Bonawitz-style Secure Aggregation via pairwise masks.

    IMPORTANT LIMITATION:
    - This simple simulation assumes NO DROPOUT for secagg runs.
      If some clients do not participate in a round, pairwise masks won't cancel,
      and the aggregate becomes corrupted.
    """
    if num_clients <= 1:
        return params

    masked = [np.array(p, copy=True) for p in params]

    for j in range(num_clients):
        if j == client_id:
            continue

        seed_ij = _pair_seed(base_seed=base_seed, server_round=server_round, a=client_id, b=j)
        rng_ij = np.random.default_rng(seed_ij)

        # If i < j => +noise, else -noise, so that sums cancel across full participation
        sign = 1.0 if client_id < j else -1.0

        for k, p in enumerate(masked):
            noise = rng_ij.normal(loc=0.0, scale=mask_std, size=p.shape).astype(p.dtype, copy=False)
            masked[k] = p + (sign * noise)

    return masked


def flower_payload_nbytes(params: List[np.ndarray]) -> int:
    """
    Measure the actual bytes that would be shipped in the Flower message payload,
    i.e., after serialization to `flwr.common.Parameters`.

    This is usually the right metric for "uplink bytes" in FL experiments.
    """
    parameters = ndarrays_to_parameters(params)
    return int(sum(len(t) for t in parameters.tensors))


class TelemetryClient(fl.client.NumPyClient):
    """
    Flower NumPyClient wrapping a PyTorch model and its local train/val loaders.

    secure_mode:
      - "none"
      - "mask"
      - "secagg"
      - "ckks"  (server-side aggregation in this project)
    """

    def __init__(
            self,
            model: torch.nn.Module,
            train_loader,
            val_loader,
            device: torch.device,
            client_id: int,
            secure_mode: str = "none",
            noise_std: float = 0.01,
            local_epochs: int = 1,
            secagg_seed: int = 12345,
            num_clients: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = int(local_epochs)

        self.client_id = int(client_id)
        self.secure_mode = (secure_mode or "none").lower().strip()
        self.noise_std = float(noise_std)

        self.secagg_seed = int(secagg_seed)
        self.num_clients = int(num_clients) if num_clients is not None else None

        self.rng = np.random.default_rng()

    # --- Flower NumPyClient API ---

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Load global parameters
        set_parameters(self.model, parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        # -----------------------------
        # 1) Local training time
        # -----------------------------
        train_start = time.time()

        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        train_end = time.time()
        train_time = float(train_end - train_start)

        # -----------------------------
        # 2) Compute train metrics
        # -----------------------------
        self.model.eval()
        all_preds, all_labels = [], []
        train_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb)
                loss = criterion(preds, yb)

                train_loss += loss.item() * xb.size(0)
                num_examples += xb.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())

        all_preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty((0,))
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,))
        acc = binary_accuracy(all_preds, all_labels) if num_examples > 0 else float("nan")
        avg_train_loss = float(train_loss / max(1, num_examples))

        # -----------------------------
        # 3) Apply security + measure time and bytes
        # -----------------------------
        sec_start = time.time()

        params = get_parameters(self.model)
        mode = (self.secure_mode or "none").lower().strip()

        if mode == "mask":
            params = apply_security_to_params(
                params,
                mode="mask",
                noise_std=self.noise_std,
                rng=self.rng,
            )

        elif mode == "secagg":
            # These should be provided by server via configure_fit, but we also fall back
            server_round = int(config.get("server_round", 1)) if isinstance(config, dict) else 1
            num_clients = (
                int(config.get("num_clients")) if isinstance(config, dict) and "num_clients" in config
                else self.num_clients
            )
            secagg_seed = (
                int(config.get("secagg_seed")) if isinstance(config, dict) and "secagg_seed" in config
                else self.secagg_seed
            )

            if num_clients is None:
                raise ValueError(
                    "secagg requires 'num_clients' (send via server fit config or pass into TelemetryClient)."
                )

            # HARD GUARD: If dropout is happening, SecAgg simulation will break.
            # The server should force full participation for secagg.
            if isinstance(config, dict) and "expected_active_clients" in config:
                expected_active = int(config["expected_active_clients"])
                if expected_active != int(num_clients):
                    raise RuntimeError(
                        f"[SECAGG] Dropout detected (expected_active_clients={expected_active}, "
                        f"num_clients={num_clients}). Pairwise-mask SecAgg simulation requires full participation. "
                        f"Fix server strategy to use fraction_fit=1.0 and min_fit_clients=num_clients for secagg."
                    )

            params = _secagg_pairwise_mask_params(
                params=params,
                client_id=self.client_id,
                num_clients=int(num_clients),
                server_round=int(server_round),
                base_seed=int(secagg_seed),
                mask_std=self.noise_std,
            )

        elif mode == "ckks":
            # CKKS is server-side in this repo: no client-side transform
            pass

        elif mode == "none":
            pass

        else:
            raise ValueError(
                f"Unknown secure_mode='{self.secure_mode}'. Expected one of: none, mask, secagg, ckks"
            )

        sec_end = time.time()
        sec_time = float(sec_end - sec_start)

        # -----------------------------
        # 4) Bytes measurement
        # -----------------------------
        # (A) Raw array bytes (shape*dtype), useful but not "network payload"
        model_nbytes = int(params_nbytes(params))

        # (B) True Flower payload bytes (serialized tensors actually sent)
        payload_nbytes = int(flower_payload_nbytes(params))

        # (C) Optional overhead bytes per round (protocol/handshake/etc.)
        overhead_nbytes = 0
        if isinstance(config, dict):
            overhead_nbytes = int(config.get("uplink_overhead_bytes", 0))

        total_uplink_nbytes = int(payload_nbytes + overhead_nbytes)

        metrics: Dict[str, Any] = {
            "train_loss": float(avg_train_loss),
            "train_accuracy": float(acc),
            "train_time": float(train_time),
            "sec_time": float(sec_time),

            # Backwards compatible metric used by server aggregation logger
            "client_bytes_up": float(total_uplink_nbytes),

            # Explicit metrics (recommended for report + plots)
            "uplink_model_nbytes": float(model_nbytes),
            "uplink_payload_nbytes": float(payload_nbytes),
            "uplink_overhead_nbytes": float(overhead_nbytes),
            "uplink_total_nbytes": float(total_uplink_nbytes),

            "secure_mode": str(mode),
        }

        return params, int(num_examples), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

        criterion = torch.nn.BCELoss()
        val_loss = 0.0
        num_examples = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb)
                loss = criterion(preds, yb)

                val_loss += loss.item() * xb.size(0)
                num_examples += xb.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())

        all_preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty((0,))
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,))
        acc = binary_accuracy(all_preds, all_labels) if num_examples > 0 else float("nan")

        return float(val_loss / max(1, num_examples)), int(num_examples), {"accuracy": float(acc)}


def get_client_fn(
        clients_data,
        device: torch.device,
        model_cls,
        local_epochs: int = 1,
        secure_mode: str = "none",
        noise_std: float = 0.01,
):
    """
    Factory that returns a Flower client_fn for flwr.simulation.start_simulation.
    """
    import torch.utils.data as data

    def client_fn(cid: str):
        idx = int(cid)
        X, y = clients_data[idx]
        n = len(X)
        split = int(0.8 * n)

        train_ds = data.TensorDataset(X[:split], y[:split])
        val_ds = data.TensorDataset(X[split:], y[split:])

        train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = data.DataLoader(val_ds, batch_size=64, shuffle=False)

        model = model_cls(input_dim=X.shape[1], hidden_dim=32)

        numpy_client = TelemetryClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            client_id=idx,
            secure_mode=secure_mode,
            noise_std=noise_std,
            local_epochs=local_epochs,
            #  IMPORTANT: always pass num_clients so secagg doesn't depend on server config
            num_clients=len(clients_data),
        )

        return numpy_client.to_client()

    return client_fn
