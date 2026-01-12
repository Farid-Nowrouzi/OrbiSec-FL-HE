# src/week11_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.optimizations import OptimSpec

# =============================================================================
# Week-11: CKKS Optimizations Study (Clipping + Quantization + Compression Model)
# =============================================================================
# Objective:
#   This week evaluates “utility vs. cost” when using CKKS-based secure aggregation.
#   The model + dataset + FL loop stay aligned with earlier weeks to preserve fairness.
#
# Variants:
#   - CKKS baseline
#   - CKKS + clipping
#   - CKKS + quantization   (accounting-only under CKKS)
#   - CKKS + clipping + quantization
#   - CKKS + clipping + quantization + compression-model
#
# Important modeling choice:
#   With CKKS, updates are encrypted as ciphertext; typical compression/quantization
#   does not apply directly to ciphertext without additional encoding/packing design.
#   Therefore, in this project we treat quantization/compression as *accounting-only*
#   (communication cost model) while the training pipeline remains CKKS-correct.
# =============================================================================


# ----------------------------
# Core run settings
# ----------------------------
@dataclass(frozen=True)
class Week11RunConfig:
    """
    Week-11 run configuration.

    These values are kept consistent with prior weeks so that differences in
    results can be attributed to the Week-11 “optimization variant” settings.
    """
    rounds: int = 20

    # Total simulated clients and how many participate per round
    num_clients_total: int = 10
    num_clients_per_round: int = 5

    # Reproducibility
    seed: int = 42

    # Output directory for CSVs / plots
    results_dir: str = "./results"

    # Local training settings (kept consistent with prior runs unless explicitly changed)
    local_epochs: int = 1
    noise_std: float = 0.01  # only relevant if non-ckks modes are used; kept for completeness

    # This parameter is used by make_strategy(...) in src/server.py
    # It represents the probability that a client is not participating in a round.
    @property
    def dropout_prob(self) -> float:
        return 1.0 - (float(self.num_clients_per_round) / float(self.num_clients_total))


# ----------------------------
# Optimization variants
# ----------------------------
def build_week11_variants(
        clip_norm: float = 1.0,
        quant_bits: int = 8,
        assumed_compression_ratio: Optional[float] = 0.65,
) -> Dict[str, OptimSpec]:
    """
    Returns:
        dict: variant_name -> OptimSpec

    Notes:
      - accounting_only=True is used throughout Week-11 to avoid breaking CKKS correctness.
      - compression is treated as a modeled saving factor, not literal ciphertext compression.
    """

    ckks_baseline = OptimSpec(
        enable_clipping=False,
        enable_quantization=False,
        enable_compression=False,
        accounting_only=True,
    )

    ckks_clip = OptimSpec(
        enable_clipping=True,
        clip_norm=float(clip_norm),
        enable_quantization=False,
        enable_compression=False,
        accounting_only=True,
    )

    ckks_quant = OptimSpec(
        enable_clipping=False,
        enable_quantization=True,
        quant_bits=int(quant_bits),
        quant_symmetric=True,
        quant_per_tensor=True,
        enable_compression=False,
        accounting_only=True,
    )

    ckks_clip_quant = OptimSpec(
        enable_clipping=True,
        clip_norm=float(clip_norm),
        enable_quantization=True,
        quant_bits=int(quant_bits),
        quant_symmetric=True,
        quant_per_tensor=True,
        enable_compression=False,
        accounting_only=True,
    )

    ckks_clip_quant_comp = OptimSpec(
        enable_clipping=True,
        clip_norm=float(clip_norm),
        enable_quantization=True,
        quant_bits=int(quant_bits),
        quant_symmetric=True,
        quant_per_tensor=True,
        enable_compression=True,
        compression="zlib",          # kept for completeness
        compression_level=6,         # kept for completeness
        assumed_compression_ratio=float(assumed_compression_ratio)
        if assumed_compression_ratio is not None
        else 0.65,
        accounting_only=True,
    )

    return {
        "ckks_baseline": ckks_baseline,
        "ckks_clip": ckks_clip,
        "ckks_quant": ckks_quant,
        "ckks_clip_quant": ckks_clip_quant,
        "ckks_clip_quant_comp": ckks_clip_quant_comp,
    }


# Backward-friendly name used by experiment scripts
def make_week11_variants(
        clip_norm: float = 1.0,
        quant_bits: int = 8,
        assumed_compression_ratio: Optional[float] = 0.65,
) -> Dict[str, OptimSpec]:
    """Alias wrapper (kept to avoid import churn in experiment scripts)."""
    return build_week11_variants(
        clip_norm=clip_norm,
        quant_bits=quant_bits,
        assumed_compression_ratio=assumed_compression_ratio,
    )


def variant_order() -> List[str]:
    """Stable ordering for tables/plots."""
    return [
        "ckks_baseline",
        "ckks_clip",
        "ckks_quant",
        "ckks_clip_quant",
        "ckks_clip_quant_comp",
    ]


def pretty_label(variant_name: str) -> str:
    """Human-readable labels used in plot legends and report tables."""
    mapping = {
        "ckks_baseline": "CKKS",
        "ckks_clip": "CKKS + Clip",
        "ckks_quant": "CKKS + Quant",
        "ckks_clip_quant": "CKKS + Clip + Quant",
        "ckks_clip_quant_comp": "CKKS + Clip + Quant + Comp(model)",
    }
    return mapping.get(variant_name, variant_name)


def week11_csv_name(variant_name: str) -> str:
    """Per-variant CSV output name."""
    return f"results_week11_{variant_name}.csv"


def week11_summary_csv_name() -> str:
    """Aggregated Week-11 summary CSV output name."""
    return "week11_ckks_optimizations_summary.csv"


def week11_plot_name(stem: str) -> str:
    """Plot naming helper."""
    return f"week11_{stem}.png"
