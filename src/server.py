import time
from typing import Dict, Optional, Tuple

import flwr as fl
import torch
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from src.utils import estimate_bytes, append_to_csv, binary_accuracy
from src.secure_he import CKKSSecureAggregator


def _set_model_parameters_from_fl(
        model: torch.nn.Module, parameters: Parameters
) -> None:
    """Load Flower Parameters into a PyTorch model."""
    ndarrays = parameters_to_ndarrays(parameters)
    state_dict = model.state_dict()
    for (name, _), array in zip(state_dict.items(), ndarrays):
        state_dict[name] = torch.tensor(array)
    model.load_state_dict(state_dict)


def _evaluate_model(model: torch.nn.Module, val_loader, device: torch.device) -> float:
    """Evaluate accuracy of the global model on a validation loader."""
    model.eval()
    model.to(device)
    preds_all, labels_all = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            preds_all.append(preds.cpu())
            labels_all.append(yb.cpu())

    if not preds_all:
        return float("nan")

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return binary_accuracy(preds_all, labels_all)


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg strategy that logs per-round metrics to CSV:
    round, acc, bytes_up, bytes_down, round_time, active_clients, seed, config.

    If secure_mode == "ckks", the aggregation is performed using the
    CKKSSecureAggregator instead of the standard FedAvg averaging.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            val_loader,
            device: torch.device,
            results_csv: str,
            seed: int,
            secure_mode: str,
            num_clients: int,
            dropout_prob: float,
    ):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.results_csv = results_csv
        self.seed = seed
        self.secure_mode = secure_mode
        self.num_clients = num_clients
        self.dropout_prob = dropout_prob
        self._last_time = time.time()

        # CKKS secure aggregator (only used when secure_mode == "ckks")
        self._ckks_agg: Optional[CKKSSecureAggregator] = None

        # Approximate update size per client (for byte logging)
        self.param_bytes = estimate_bytes(self.model.state_dict())

        fraction_fit = 1.0 - dropout_prob
        fraction_eval = 1.0 - dropout_prob
        min_fit_clients = max(1, int(num_clients * fraction_fit))

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_available_clients=num_clients,
        )

    # ------------------------------------------------------------------
    # CKKS aggregation helper
    # ------------------------------------------------------------------

    def _aggregate_fit_ckks(self, results) -> Optional[Parameters]:
        """Aggregate client updates using CKKS homomorphic encryption.

        This mirrors the Week-2 demo:
        1. Collect client parameter ndarrays.
        2. Flatten all updates.
        3. Encrypt, homomorphically sum, and decrypt.
        4. Reconstruct layer-shaped ndarrays and wrap as Flower Parameters.
        """
        if not results:
            return None

        # Lazily initialise CKKS aggregator
        if self._ckks_agg is None:
            self._ckks_agg = CKKSSecureAggregator()

        # 1) Collect parameter ndarrays from each client
        client_updates = []
        for _, fit_res in results:
            if fit_res.parameters is None:
                continue
            nds = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(nds)

        if not client_updates:
            return None

        # 2) Flatten updates
        flat_updates, flatten_info = self._ckks_agg.flatten_updates(client_updates)

        # 3) Encrypt and aggregate in the encrypted domain
        enc_updates = self._ckks_agg.encrypt_updates(flat_updates)
        enc_sum = self._ckks_agg.aggregate_encrypted(enc_updates)

        # 4) Decrypt and reconstruct averaged update
        avg_update_ndarrays = self._ckks_agg.decrypt_aggregate(
            enc_sum, flatten_info, num_clients=len(client_updates)
        )

        # Convert back to Flower Parameters
        return ndarrays_to_parameters(avg_update_ndarrays)

    # ------------------------------------------------------------------
    # Override aggregate_fit to plug in logging (and optionally CKKS)
    # ------------------------------------------------------------------

    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        start_time = self._last_time

        # Use CKKS-based aggregation only when requested
        if self.secure_mode.lower() == "ckks":
            aggregated_parameters = self._aggregate_fit_ckks(results)
            # We ignore client metrics here (can be added later if needed)
        else:
            # Standard FedAvg aggregation
            aggregated_parameters, _ = super().aggregate_fit(
                server_round, results, failures
            )

        end_time = time.time()
        self._last_time = end_time

        round_time = end_time - start_time
        active_clients = len(results)
        bytes_up = self.param_bytes * active_clients
        bytes_down = self.param_bytes * active_clients

        # Evaluate global model on validation loader
        if aggregated_parameters is not None:
            _set_model_parameters_from_fl(self.model, aggregated_parameters)
            acc = _evaluate_model(self.model, self.val_loader, self.device)
        else:
            acc = float("nan")

        row = {
            "round": server_round,
            "acc": acc,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "round_time": round_time,
            "active_clients": active_clients,
            "seed": self.seed,
            "secure_mode": self.secure_mode,
            "num_clients": self.num_clients,
            "dropout_prob": self.dropout_prob,
        }
        append_to_csv(self.results_csv, row)

        # We don't currently use aggregated metrics, but we can return an empty dict
        metrics: Dict[str, Scalar] = {}
        return aggregated_parameters, metrics


def make_strategy(
        model: torch.nn.Module,
        val_loader,
        device: torch.device,
        results_csv: str,
        seed: int,
        secure_mode: str,
        num_clients: int,
        dropout_prob: float,
) -> LoggingFedAvg:
    """Factory to create the LoggingFedAvg strategy."""
    return LoggingFedAvg(
        model=model,
        val_loader=val_loader,
        device=device,
        results_csv=results_csv,
        seed=seed,
        secure_mode=secure_mode,
        num_clients=num_clients,
        dropout_prob=dropout_prob,
    )
