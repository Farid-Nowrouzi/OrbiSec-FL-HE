import time
from typing import List

import flwr as fl
import numpy as np
import torch

from src.utils import binary_accuracy, params_nbytes
from src.secure import apply_security_to_params


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy arrays."""
    state_dict = model.state_dict()
    for (name, _), array in zip(state_dict.items(), parameters):
        state_dict[name] = torch.tensor(array)
    model.load_state_dict(state_dict)


class TelemetryClient(fl.client.NumPyClient):
    """
    Flower NumPyClient wrapping a PyTorch model and its local train/val loaders.

    secure_mode: "none" | "mask" | "ckks".
    """

    def __init__(
            self,
            model: torch.nn.Module,
            train_loader,
            val_loader,
            device: torch.device,
            secure_mode: str = "none",
            noise_std: float = 0.01,
            local_epochs: int = 1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.secure_mode = secure_mode
        self.noise_std = noise_std
        self.rng = np.random.default_rng()

    # --- Flower NumPyClient API ---

    def get_parameters(self, config):
        params = get_parameters(self.model)
        # We do not secure parameters here; security is applied after local training.
        return params

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
        train_time = train_end - train_start

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

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = binary_accuracy(all_preds, all_labels)
        avg_train_loss = train_loss / max(1, num_examples)

        # -----------------------------
        # 3) Apply security + measure time and bytes
        # -----------------------------
        sec_start = time.time()

        params = get_parameters(self.model)
        params = apply_security_to_params(
            params,
            mode=self.secure_mode,
            noise_std=self.noise_std,
            rng=self.rng,
        )

        sec_end = time.time()
        sec_time = sec_end - sec_start

        client_bytes_up = params_nbytes(params)

        metrics = {
            "train_loss": avg_train_loss,
            "train_accuracy": acc,
            "train_time": train_time,
            "sec_time": sec_time,
            "client_bytes_up": float(client_bytes_up),
        }

        return params, num_examples, metrics

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

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = binary_accuracy(all_preds, all_labels)

        return float(val_loss / max(1, num_examples)), num_examples, {"accuracy": acc}


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

    clients_data: list of (X_tensor, y_tensor) per client.
    model_cls:    callable that returns a new model instance (e.g., MLP).
    """

    import torch.utils.data as data

    def client_fn(cid: str):
        # cid is a string; convert to integer index
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
            secure_mode=secure_mode,
            noise_std=noise_std,
            local_epochs=local_epochs,
        )

        # IMPORTANT: convert NumPyClient -> Client (removes the Flower warning)
        return numpy_client.to_client()

    return client_fn
