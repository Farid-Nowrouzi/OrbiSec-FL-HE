"""
secure_he.py

Week-2: CKKS homomorphic encryption demo for secure federated aggregation.

- Uses TenSEAL CKKS to encrypt flattened model updates.
- Aggregates them homomorphically (FedAvg in the encrypted domain).
- Decrypts and compares with plain FedAvg to verify correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tenseal as ts


@dataclass
class FlattenInfo:
    """Metadata to reconstruct tensors from a flat vector."""
    sizes: List[int]
    shapes: List[Tuple[int, ...]]


class CKKSSecureAggregator:
    """
    CKKS-based secure aggregator.

    Workflow:
        1) flatten_updates(list_of_arrays_per_client)
        2) encrypt_updates(flat_updates)
        3) aggregate_encrypted(ciphertexts) -> encrypted sum
        4) decrypt_aggregate(enc_sum, flatten_info, num_clients)
    """

    def __init__(
        self,
        poly_mod_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] | None = None,
        global_scale: float = 2**40,
    ) -> None:
        if coeff_mod_bit_sizes is None:
            # Reasonable default for small experiments
            coeff_mod_bit_sizes = [40, 20, 40]

        # Create CKKS context (note: argument name is poly_modulus_degree)
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        self.context.generate_galois_keys()
        self.context.global_scale = global_scale

    # ------------------------------------------------------------------
    # Flatten / unflatten helpers
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_updates(
        client_updates: List[List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], FlattenInfo]:
        """
        Flatten each client's list of arrays into a single 1D vector.

        Args:
            client_updates: List over clients, each is a list of numpy arrays
                            (e.g., model parameter tensors).

        Returns:
            flat_per_client: list of 1D numpy arrays (one per client)
            info: FlattenInfo describing how to unflatten.
        """
        if not client_updates:
            raise ValueError("client_updates must not be empty")

        # We assume all clients have the same parameter shapes
        example = client_updates[0]
        sizes = [p.size for p in example]
        shapes = [p.shape for p in example]

        flat_per_client: List[np.ndarray] = []
        for client in client_updates:
            flat_list = [p.astype(np.float64).reshape(-1) for p in client]
            flat_vec = np.concatenate(flat_list)
            flat_per_client.append(flat_vec)

        return flat_per_client, FlattenInfo(sizes=sizes, shapes=shapes)

    @staticmethod
    def unflatten_update(
        flat_vec: np.ndarray,
        info: FlattenInfo,
    ) -> List[np.ndarray]:
        """
        Inverse of `flatten_updates` for a single vector.
        """
        arrays: List[np.ndarray] = []
        idx = 0
        for size, shape in zip(info.sizes, info.shapes):
            chunk = flat_vec[idx: idx + size]
            arrays.append(chunk.reshape(shape))
            idx += size
        return arrays

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------
    def encrypt_updates(
        self, flat_per_client: List[np.ndarray]
    ) -> List[ts.CKKSVector]:
        """
        Encrypt each client's flat update as a CKKS vector.
        """
        ciphertexts: List[ts.CKKSVector] = []
        for flat in flat_per_client:
            # TenSEAL expects a Python list
            ckks_vec = ts.ckks_vector(self.context, flat.tolist())
            ciphertexts.append(ckks_vec)
        return ciphertexts

    def aggregate_encrypted(
        self, ciphertexts: List[ts.CKKSVector]
    ) -> ts.CKKSVector:
        """
        Aggregate encrypted vectors by summing them:

            enc_sum = enc_1 + ... + enc_N

        We *do not* divide by N here to avoid scale issues.
        The division is done after decryption.
        """
        if not ciphertexts:
            raise ValueError("ciphertexts must not be empty")

        agg = ciphertexts[0]
        for ct in ciphertexts[1:]:
            agg = agg + ct

        return agg

    def decrypt_aggregate(
        self,
        enc_sum: ts.CKKSVector,
        info: FlattenInfo,
        num_clients: int,
    ) -> List[np.ndarray]:
        """
        Decrypt the aggregated ciphertext, divide by number of clients,
        and unflatten back into tensors.
        """
        flat = np.array(enc_sum.decrypt(), dtype=np.float64)

        # Convert sum -> average in plaintext (avoids scale problems)
        flat = flat / float(num_clients)

        return self.unflatten_update(flat, info)
