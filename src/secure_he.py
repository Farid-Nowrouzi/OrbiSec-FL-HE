"""
secure_he.py

Week-2: CKKS homomorphic encryption demo for secure federated aggregation.

- Uses TenSEAL CKKS to encrypt flattened model updates.
- Aggregates them homomorphically (FedAvg in the encrypted domain).
- Decrypts and compares with plain FedAvg to verify correctness.

NOTE (important for Week-5 plots):
- If we want CKKS to have different "uplink bytes", we MUST measure
  the ciphertext bytes (serialized size) instead of plaintext float arrays.
  This file now provides ciphertext_nbytes(...) for that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Union

import numpy as np
import tenseal as ts


@dataclass
class FlattenInfo:
    """Metadata to reconstruct tensors from a flat vector."""
    sizes: List[int]
    shapes: List[Tuple[int, ...]]
    total_len: int

    # Optional chunk metadata (used only if chunking is enabled)
    chunk_sizes: Optional[List[int]] = None  # sizes of each chunk in flat space


class CKKSSecureAggregator:
    """
    CKKS-based secure aggregator.

    Workflow (simple, single-vector):
        1) flatten_updates(list_of_arrays_per_client)
        2) encrypt_updates(flat_updates)
        3) aggregate_encrypted(ciphertexts) -> encrypted sum
        4) decrypt_aggregate(enc_sum, flatten_info, num_clients)

    Workflow (recommended, chunked for real usage + correct byte measurement):
        1) flatten_updates(...)
        2) encrypt_updates_chunked(...)
        3) aggregate_encrypted_chunked(...)
        4) decrypt_aggregate_chunked(...)
    """

    def __init__(
            self,
            poly_mod_degree: int = 8192,
            coeff_mod_bit_sizes: Optional[List[int]] = None,
            global_scale: float = 2**40,
    ) -> None:
        if coeff_mod_bit_sizes is None:
            # Reasonable default for small experiments
            coeff_mod_bit_sizes = [40, 20, 40]

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        self.context.generate_galois_keys()
        self.context.global_scale = global_scale

        # CKKS slot capacity is approximately poly_mod_degree/2
        self.poly_mod_degree = int(poly_mod_degree)
        self.slot_capacity = int(poly_mod_degree // 2)

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

        example = client_updates[0]
        sizes = [p.size for p in example]
        shapes = [p.shape for p in example]
        total_len = int(sum(sizes))

        flat_per_client: List[np.ndarray] = []
        for client in client_updates:
            flat_list = [p.astype(np.float64).reshape(-1) for p in client]
            flat_vec = np.concatenate(flat_list)
            flat_per_client.append(flat_vec)

        info = FlattenInfo(sizes=sizes, shapes=shapes, total_len=total_len, chunk_sizes=None)
        return flat_per_client, info

    @staticmethod
    def unflatten_update(flat_vec: np.ndarray, info: FlattenInfo) -> List[np.ndarray]:
        """Inverse of `flatten_updates` for a single vector."""
        arrays: List[np.ndarray] = []
        idx = 0
        for size, shape in zip(info.sizes, info.shapes):
            chunk = flat_vec[idx: idx + size]
            arrays.append(chunk.reshape(shape))
            idx += size
        return arrays

    @staticmethod
    def _chunk_flat_vector(flat: np.ndarray, chunk_size: int) -> Tuple[List[np.ndarray], List[int]]:
        """Split a flat vector into chunks of at most chunk_size."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        chunks: List[np.ndarray] = []
        chunk_sizes: List[int] = []

        n = int(flat.shape[0])
        start = 0
        while start < n:
            end = min(n, start + chunk_size)
            part = flat[start:end]
            chunks.append(part)
            chunk_sizes.append(int(part.shape[0]))
            start = end

        return chunks, chunk_sizes

    # ------------------------------------------------------------------
    # Ciphertext size (bytes) helpers (for Week-5 "uplink bytes")
    # ------------------------------------------------------------------
    @staticmethod
    def ciphertext_nbytes(ct: ts.CKKSVector) -> int:
        """
        True ciphertext payload size in bytes (serialized).
        This is what we should use for "uplink bytes" if clients send ciphertexts.
        """
        # TenSEAL returns bytes from serialize()
        return int(len(ct.serialize()))

    def ciphertexts_nbytes(self, cts: Sequence[ts.CKKSVector]) -> int:
        """Sum of serialized ciphertext sizes."""
        return int(sum(self.ciphertext_nbytes(ct) for ct in cts))

    # ------------------------------------------------------------------
    # Encryption helpers (single-vector)
    # ------------------------------------------------------------------
    def encrypt_updates(self, flat_per_client: List[np.ndarray]) -> List[ts.CKKSVector]:
        """Encrypt each client's flat update as a single CKKS vector."""
        ciphertexts: List[ts.CKKSVector] = []
        for flat in flat_per_client:
            ckks_vec = ts.ckks_vector(self.context, flat.tolist())
            ciphertexts.append(ckks_vec)
        return ciphertexts

    def aggregate_encrypted(self, ciphertexts: List[ts.CKKSVector]) -> ts.CKKSVector:
        """
        Aggregate encrypted vectors by summing them:
            enc_sum = enc_1 + ... + enc_N

        We do NOT divide by N here; division is done after decryption.
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
        flat = flat / float(num_clients)
        return self.unflatten_update(flat, info)

    # ------------------------------------------------------------------
    # Encryption helpers (chunked, recommended for real usage)
    # ------------------------------------------------------------------
    def encrypt_updates_chunked(
            self,
            flat_per_client: List[np.ndarray],
            info: FlattenInfo,
            chunk_size: Optional[int] = None,
    ) -> List[List[ts.CKKSVector]]:
        """
        Encrypt each client's flat update as a LIST of CKKS vectors (chunks).

        Why chunking:
        - CKKSVector has a slot capacity ~ poly_mod_degree/2.
        - If your flattened model is larger than capacity, you MUST chunk.

        Args:
            flat_per_client: list of 1D arrays (one per client)
            info: FlattenInfo (will be updated with chunk_sizes)
            chunk_size: max values per ciphertext. Default: slot_capacity.

        Returns:
            enc_per_client: List over clients -> list of CKKSVector chunks.
        """
        if chunk_size is None:
            chunk_size = self.slot_capacity
        chunk_size = int(chunk_size)

        enc_per_client: List[List[ts.CKKSVector]] = []
        inferred_chunk_sizes: Optional[List[int]] = None

        for flat in flat_per_client:
            flat = flat.astype(np.float64, copy=False)
            chunks, chunk_sizes = self._chunk_flat_vector(flat, chunk_size)

            if inferred_chunk_sizes is None:
                inferred_chunk_sizes = chunk_sizes
            else:
                # Sanity: all clients must have same chunking layout
                if chunk_sizes != inferred_chunk_sizes:
                    raise ValueError("Chunking mismatch across clients (unexpected).")

            enc_chunks: List[ts.CKKSVector] = []
            for part in chunks:
                enc_chunks.append(ts.ckks_vector(self.context, part.tolist()))
            enc_per_client.append(enc_chunks)

        info.chunk_sizes = inferred_chunk_sizes
        return enc_per_client

    def aggregate_encrypted_chunked(
            self,
            enc_per_client: List[List[ts.CKKSVector]],
    ) -> List[ts.CKKSVector]:
        """
        Aggregate chunked ciphertexts:
        For each chunk index k:
            agg_k = sum_i enc_i[k]
        Returns list of aggregated chunks.
        """
        if not enc_per_client:
            raise ValueError("enc_per_client must not be empty")

        num_clients = len(enc_per_client)
        num_chunks = len(enc_per_client[0])

        for c in enc_per_client:
            if len(c) != num_chunks:
                raise ValueError("All clients must have the same number of ciphertext chunks.")

        agg_chunks: List[ts.CKKSVector] = []
        for k in range(num_chunks):
            agg = enc_per_client[0][k]
            for i in range(1, num_clients):
                agg = agg + enc_per_client[i][k]
            agg_chunks.append(agg)

        return agg_chunks

    def decrypt_aggregate_chunked(
            self,
            agg_chunks: List[ts.CKKSVector],
            info: FlattenInfo,
            num_clients: int,
    ) -> List[np.ndarray]:
        """
        Decrypt chunked aggregated ciphertexts, divide by num_clients,
        stitch back into one flat vector, then unflatten.
        """
        if info.chunk_sizes is None:
            raise ValueError("info.chunk_sizes is None. Did you use encrypt_updates_chunked()?")

        if len(agg_chunks) != len(info.chunk_sizes):
            raise ValueError("Mismatch: number of agg_chunks != number of chunk_sizes in FlattenInfo")

        flat_parts: List[np.ndarray] = []
        for ct, size in zip(agg_chunks, info.chunk_sizes):
            part = np.array(ct.decrypt(), dtype=np.float64)
            part = part[:size]  # safety trim
            part = part / float(num_clients)
            flat_parts.append(part)

        flat = np.concatenate(flat_parts)
        flat = flat[: info.total_len]  # safety trim

        return self.unflatten_update(flat, info)
