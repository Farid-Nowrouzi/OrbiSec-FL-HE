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


def _set_model_parameters_from_fl(model: torch.nn.Module, parameters: Parameters) -> None:
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
    FedAvg strategy that logs per-round metrics to CSV.

    Supported secure_mode:
      - "none"   : standard FedAvg
      - "mask"   : standard FedAvg (masking/noise happens client-side)
      - "ckks"   : aggregation performed using CKKSSecureAggregator (server-side HE)
      - "secagg" : FedAvg aggregation outcome + overhead model; client uses pairwise masks

    IMPORTANT:
    - This project does NOT implement full cryptographic SecAgg end-to-end.
      "secagg" here is a simulation + overhead model for comparison & reporting.
    - Pairwise-mask SecAgg simulation requires FULL participation (no dropout),
      otherwise masks won’t cancel and the aggregated update becomes corrupted.
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
        self.seed = int(seed)
        self.secure_mode = (secure_mode or "none").lower().strip()
        self.num_clients = int(num_clients)
        self.dropout_prob = float(dropout_prob)

        # Used to compute per-round time deltas
        self._last_time = time.time()

        # CKKS secure aggregator (only used when secure_mode == "ckks")
        self._ckks_agg: Optional[CKKSSecureAggregator] = None

        # Approximate parameter bytes (one model copy)
        self.param_bytes = int(estimate_bytes(self.model.state_dict()))

        mode = self.secure_mode

        #  IMPORTANT: secagg must be full participation
        if mode == "secagg":
            fraction_fit = 1.0
            fraction_eval = 1.0
            min_fit_clients = self.num_clients
        else:
            fraction_fit = 1.0 - self.dropout_prob
            fraction_eval = 1.0 - self.dropout_prob
            min_fit_clients = max(1, int(self.num_clients * fraction_fit))

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_available_clients=self.num_clients,
        )

    # ------------------------------------------------------------------
    # Send round config to clients (SecAgg needs it)
    # ------------------------------------------------------------------
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Safely inject SecAgg-related config (server_round, num_clients, seed)
        into whatever FitIns the base FedAvg creates.
        """
        instructions = super().configure_fit(server_round, parameters, client_manager)

        active_expected = len(instructions)

        base_cfg = {
            "server_round": int(server_round),
            "num_clients": int(self.num_clients),
            "secagg_seed": int(self.seed),

            # Lets client detect dropout mismatch in secagg mode
            "expected_active_clients": int(active_expected),
        }

        # Optional: uplink overhead bytes (for client metrics accounting)
        # We only inject overhead for secagg mode since other modes do not need it.
        if (self.secure_mode or "none").lower().strip() == "secagg":
            base_cfg["uplink_overhead_bytes"] = int(self._secagg_overhead_bytes(active_expected))
        else:
            base_cfg["uplink_overhead_bytes"] = 0

        updated = []
        for client, fit_ins in instructions:
            cfg = dict(getattr(fit_ins, "config", {}) or {})
            cfg.update(base_cfg)

            # Keep optional user config hook if used elsewhere
            if self.on_fit_config_fn is not None:
                user_cfg = self.on_fit_config_fn(server_round)
                if isinstance(user_cfg, dict):
                    cfg.update(user_cfg)

            updated.append((client, fl.common.FitIns(fit_ins.parameters, cfg)))

        return updated

    # ------------------------------------------------------------------
    # CKKS aggregation helper
    # ------------------------------------------------------------------
    def _aggregate_fit_ckks(self, results) -> Optional[Parameters]:
        """Aggregate client updates using CKKS homomorphic encryption (server-side demo)."""
        if not results:
            return None

        if self._ckks_agg is None:
            self._ckks_agg = CKKSSecureAggregator()

        client_updates = []
        for _, fit_res in results:
            if fit_res.parameters is None:
                continue
            nds = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(nds)

        if not client_updates:
            return None

        flat_updates, flatten_info = self._ckks_agg.flatten_updates(client_updates)
        enc_updates = self._ckks_agg.encrypt_updates(flat_updates)
        enc_sum = self._ckks_agg.aggregate_encrypted(enc_updates)

        avg_update_ndarrays = self._ckks_agg.decrypt_aggregate(
            enc_sum, flatten_info, num_clients=len(client_updates)
        )

        return ndarrays_to_parameters(avg_update_ndarrays)

    # ------------------------------------------------------------------
    # SecAgg overhead model (small + stable)
    # ------------------------------------------------------------------
    @staticmethod
    def _secagg_overhead_seconds(active_clients: int) -> float:
        return 0.003 * float(max(active_clients, 1))

    def _secagg_overhead_bytes(self, active_clients: int) -> int:
        base = int(self.param_bytes * max(active_clients, 1))
        overhead = int(0.02 * base) + int(512 * max(active_clients, 1))
        return overhead

    # ------------------------------------------------------------------
    # Override aggregate_fit to plug in logging + CKKS/SecAgg
    # ------------------------------------------------------------------
    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        start_time = self._last_time
        mode = (self.secure_mode or "none").lower().strip()

        # -----------------------------
        # 1) Aggregation + time
        # -----------------------------
        agg_start = time.time()

        if mode == "ckks":
            aggregated_parameters = self._aggregate_fit_ckks(results)
        else:
            aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        agg_end = time.time()
        agg_time = float(agg_end - agg_start)

        # -----------------------------
        # 2) Round timing
        # -----------------------------
        end_time = time.time()
        self._last_time = end_time
        round_time = float(end_time - start_time)

        # -----------------------------
        # 3) Aggregate client-side metrics
        # -----------------------------
        active_clients = len(results)
        total_examples = sum(fit_res.num_examples for _, fit_res in results) or 1

        def weighted_avg(key: str) -> float:
            num = 0.0
            for _, fit_res in results:
                if not fit_res.metrics:
                    continue
                val = fit_res.metrics.get(key)
                if val is not None:
                    num += float(val) * fit_res.num_examples
            return num / float(total_examples)

        def sum_metric(key: str, default_each: float) -> float:
            s = 0.0
            for _, fit_res in results:
                if not fit_res.metrics:
                    s += float(default_each)
                else:
                    s += float(fit_res.metrics.get(key, default_each))
            return s

        if active_clients > 0:
            avg_train_loss = weighted_avg("train_loss")
            avg_train_accuracy = weighted_avg("train_accuracy")
            avg_train_time = weighted_avg("train_time")
            avg_sec_time = weighted_avg("sec_time")
            avg_client_bytes_up = weighted_avg("client_bytes_up")
        else:
            avg_train_loss = float("nan")
            avg_train_accuracy = float("nan")
            avg_train_time = float("nan")
            avg_sec_time = float("nan")
            avg_client_bytes_up = float("nan")

        # -----------------------------
        # 4) Communication accounting
        # -----------------------------
        # uplink: what clients actually report (payload + overhead if any)
        bytes_up = int(sum_metric("client_bytes_up", float(self.param_bytes)))

        # downlink: server sends global params to each active client
        bytes_down = int(self.param_bytes * active_clients)

        # SecAgg overhead ONLY for mode == "secagg" (server-side accounting)
        if mode == "secagg":
            extra_bytes = int(self._secagg_overhead_bytes(active_clients))
            bytes_down += extra_bytes  # server->clients also carries protocol overhead

            extra_t = float(self._secagg_overhead_seconds(active_clients))
            agg_time += extra_t
            round_time += extra_t

            # keep internal clock consistent
            self._last_time = self._last_time + extra_t

        # -----------------------------
        # 5) Evaluate global model
        # -----------------------------
        if aggregated_parameters is not None:
            _set_model_parameters_from_fl(self.model, aggregated_parameters)
            acc = _evaluate_model(self.model, self.val_loader, self.device)
        else:
            acc = float("nan")

        # -----------------------------
        # 6) Log to CSV
        # -----------------------------
        row = {
            "round": int(server_round),
            "acc": float(acc),
            "bytes_up": int(bytes_up),
            "bytes_down": int(bytes_down),
            "round_time": float(round_time),
            "agg_time": float(agg_time),
            "active_clients": int(active_clients),
            "seed": int(self.seed),
            "secure_mode": str(self.secure_mode),
            "num_clients": int(self.num_clients),
            "dropout_prob": float(self.dropout_prob),
            "avg_train_loss": float(avg_train_loss),
            "avg_train_accuracy": float(avg_train_accuracy),
            "avg_train_time": float(avg_train_time),
            "avg_sec_time": float(avg_sec_time),
            "avg_client_bytes_up": float(avg_client_bytes_up),
        }
        append_to_csv(self.results_csv, row)

        return aggregated_parameters, {}


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


# ============================
# Week-11 helper entrypoint
# ============================
def run_simulation(
        secure_mode: str = "ckks",
        rounds: int = 20,
        num_clients_total: int = 10,
        num_clients_per_round: int = 5,
        seed: int = 42,
        optim=None,
        results_csv_path: str | None = None,
        **kwargs,
):
    """
    Server-side simulation entrypoint used by Week-11 experiment runners.

    Rationale:
      - Some experiment scripts expect src.server to expose a runnable entrypoint
        (e.g., run_simulation / run_server / main).
      - The project already exposes make_strategy(...), so this wrapper simply
        constructs the strategy and launches Flower simulation consistently.

    Parameters:
      secure_mode: "none", "mask", "secagg", "ckks" (Week-11 uses ckks)
      rounds: number of FL rounds
      num_clients_total: total simulated clients
      num_clients_per_round: participating clients per round
      seed: reproducibility seed
      optim: optional OptimSpec (Week-11 variants)
      results_csv_path: optional CSV path for per-round metrics logging
    """
    import random
    import numpy as np

    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    random.seed(seed)
    np.random.seed(seed)

    import flwr as fl
    import inspect

    # ----------------------------
    # Resolve client_fn from src.client
    # ----------------------------
    from src import client as client_mod

    client_fn = None
    # Common naming patterns (we try several to stay robust)
    for name in ["client_fn", "get_client_fn", "make_client_fn", "build_client_fn"]:
        if hasattr(client_mod, name):
            client_fn = getattr(client_mod, name)
            break

    # If the module exposes a Client class builder instead, wrap it
    if client_fn is None:
        for name in ["make_client", "build_client", "create_client", "get_client"]:
            if hasattr(client_mod, name):
                make_client = getattr(client_mod, name)

                def client_fn(cid: str):
                    return make_client(cid)

                break

    if client_fn is None:
        raise RuntimeError(
            "Could not resolve a client factory in src.client. "
            "Expected one of: client_fn/get_client_fn/make_client/build_client/etc."
        )

    # ----------------------------
    # Build strategy using existing make_strategy(...)
    # ----------------------------
    # We pass only parameters that make_strategy actually accepts (signature-safe)
    sig = inspect.signature(make_strategy)
    params = sig.parameters

    strategy_kwargs = {}
    if "secure_mode" in params:
        strategy_kwargs["secure_mode"] = secure_mode
    if "optim" in params:
        strategy_kwargs["optim"] = optim
    if "optim_spec" in params:
        strategy_kwargs["optim_spec"] = optim
    if "results_csv_path" in params:
        strategy_kwargs["results_csv_path"] = results_csv_path
    if "out_csv" in params:
        strategy_kwargs["out_csv"] = results_csv_path
    if "seed" in params:
        strategy_kwargs["seed"] = seed

    strategy = make_strategy(**strategy_kwargs)

    # ----------------------------
    # Start Flower simulation
    # ----------------------------
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients_total,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    return hist
