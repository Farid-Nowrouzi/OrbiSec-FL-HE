# src/optimizations.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import numpy as np

ArrayLike = Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]


# ----------------------------
# Config container (Week-11)
# ----------------------------
@dataclass
class OptimSpec:
    """
    Optimization spec used in Week-11.

    Notes:
    - In CKKS pipelines, you typically MUST keep float arrays to encrypt them.
      So we support "accounting_only=True" to estimate bandwidth reduction without
      changing the payload type.
    - If we use non-HE or simulated payloads, we can set accounting_only=False
      to actually quantize + pack to bytes.
    """

    # Clipping
    enable_clipping: bool = False
    clip_norm: float = 1.0  # L2-norm clipping threshold

    # Quantization
    enable_quantization: bool = False
    quant_bits: int = 8  # 8 or 16 recommended
    quant_symmetric: bool = True  # symmetric around 0
    quant_per_tensor: bool = True  # per-tensor scaling (simple & stable)

    # Compression model (lightweight, practical)
    enable_compression: bool = False
    compression: str = "zlib"  # "zlib" | "none"
    compression_level: int = 6  # zlib level 1..9
    # If you don’t want real compression, you can instead set an assumed ratio:
    assumed_compression_ratio: Optional[float] = None  # e.g., 0.6 means 40% saving

    # Mode
    accounting_only: bool = True  # recommended for CKKS pipelines


# ----------------------------
# Utilities
# ----------------------------
def _as_list(payload: ArrayLike) -> List[np.ndarray]:
    if isinstance(payload, np.ndarray):
        return [payload]
    return list(payload)


def _same_container(original: ArrayLike, arrays: List[np.ndarray]) -> ArrayLike:
    if isinstance(original, np.ndarray):
        return arrays[0]
    if isinstance(original, tuple):
        return tuple(arrays)
    return arrays


def nbytes_of_payload(payload: ArrayLike) -> int:
    """True raw bytes in memory for a payload of numpy arrays."""
    arrays = _as_list(payload)
    return int(sum(a.nbytes for a in arrays))


# ----------------------------
# 1) L2-norm clipping
# ----------------------------
def l2_clip(payload: ArrayLike, clip_norm: float) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Clips the full payload (treated as one big vector) to L2 norm <= clip_norm.
    Returns (clipped_payload, meta).
    """
    if clip_norm <= 0:
        return payload, {"clipped": False, "reason": "clip_norm<=0"}

    arrays = _as_list(payload)

    # Flatten & compute global norm
    sq_sum = 0.0
    for a in arrays:
        x = a.astype(np.float64, copy=False).ravel()
        sq_sum += float(np.dot(x, x))
    norm = math.sqrt(sq_sum)

    if norm <= clip_norm or norm == 0.0:
        return payload, {"clipped": False, "l2_norm": norm, "scale": 1.0}

    scale = clip_norm / (norm + 1e-12)
    clipped = [a * scale for a in arrays]
    return _same_container(payload, clipped), {"clipped": True, "l2_norm": norm, "scale": scale}


# ----------------------------
# 2) Quantization (per-tensor symmetric)
# ----------------------------
def quantize_tensor(
        x: np.ndarray,
        bits: int = 8,
        symmetric: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantize a float tensor into int8/int16.

    Returns:
      q: integer tensor
      meta: contains scale and dtype info for reconstruction
    """
    if bits not in (8, 16):
        raise ValueError(f"quant_bits must be 8 or 16, got {bits}")

    x_f = x.astype(np.float32, copy=False)

    if bits == 8:
        qdtype = np.int8
        qmax = 127
    else:
        qdtype = np.int16
        qmax = 32767

    if symmetric:
        max_abs = float(np.max(np.abs(x_f))) if x_f.size else 0.0
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / qmax

        q = np.round(x_f / (scale + 1e-12)).astype(np.int64)
        q = np.clip(q, -qmax, qmax).astype(qdtype)

        meta = {
            "scheme": "sym",
            "bits": bits,
            "scale": float(scale),
            "qdtype": str(qdtype),
            "shape": x.shape,
        }
        return q, meta

    # (Optional extension) Asymmetric quantization could be added here.
    raise NotImplementedError("Only symmetric quantization is implemented (stable + simple).")


def dequantize_tensor(q: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct float tensor from quantized integer tensor using meta.
    """
    scheme = meta.get("scheme", "sym")
    if scheme != "sym":
        raise NotImplementedError("Only symmetric dequantization implemented.")
    scale = float(meta["scale"])
    return (q.astype(np.float32) * scale).reshape(meta["shape"])


def quantize_payload(
        payload: ArrayLike,
        bits: int = 8,
        symmetric: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Quantize each tensor in payload. Returns a list of q-arrays + meta.
    """
    arrays = _as_list(payload)
    q_list: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for a in arrays:
        q, m = quantize_tensor(a, bits=bits, symmetric=symmetric)
        q_list.append(q)
        metas.append(m)

    return q_list, {"per_tensor_meta": metas, "bits": bits, "symmetric": symmetric}


def dequantize_payload(q_list: Sequence[np.ndarray], meta: Dict[str, Any]) -> List[np.ndarray]:
    metas = meta["per_tensor_meta"]
    out: List[np.ndarray] = []
    for q, m in zip(q_list, metas):
        out.append(dequantize_tensor(q, m))
    return out


# ----------------------------
# 3) Packing + compression (optional)
# ----------------------------
def pack_int_payload(q_list: Sequence[np.ndarray]) -> bytes:
    """
    Packs a list of int arrays into raw bytes (concatenated).
    Metadata (shapes, scales, etc.) must be stored separately.
    """
    return b"".join(np.ascontiguousarray(q).tobytes() for q in q_list)


def compress_bytes(data: bytes, method: str = "zlib", level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
    """
    Real compression for non-HE payloads.
    For CKKS, you typically shouldn't compress ciphertext here; use accounting_only instead.
    """
    method = (method or "none").lower()
    if method == "none":
        return data, {"method": "none", "ratio": 1.0, "in_bytes": len(data), "out_bytes": len(data)}

    if method == "zlib":
        import zlib
        out = zlib.compress(data, level=max(1, min(9, int(level))))
        ratio = (len(out) / len(data)) if len(data) > 0 else 1.0
        return out, {"method": "zlib", "level": int(level), "ratio": ratio, "in_bytes": len(data), "out_bytes": len(out)}

    raise ValueError(f"Unknown compression method: {method}")


# ----------------------------
# Effective-bytes estimation (great for CKKS)
# ----------------------------
def estimate_effective_bytes(
        raw_bytes: int,
        spec: OptimSpec,
        quantized_bits: Optional[int] = None,
) -> int:
    """
    Estimate 'effective bytes on the wire' after optimizations.

    - Quantization: assume float32 -> int{bits}
      (If our payload is float64/float32, we can refine this, but this is good enough
       for a clean Week-11 comparison.)
    - Compression: either assumed ratio or real ratio model.

    Returns an integer bytes estimate.
    """
    effective = float(raw_bytes)

    # Quantization factor (if enabled)
    if spec.enable_quantization:
        bits = quantized_bits or spec.quant_bits
        # Baseline assumption: float32 (32 bits)
        q_factor = bits / 32.0
        effective *= q_factor

    # Compression factor
    if spec.enable_compression:
        if spec.assumed_compression_ratio is not None:
            effective *= float(spec.assumed_compression_ratio)
        else:
            # If no assumed ratio, apply a conservative default (mild benefit)
            # (Real compression depends heavily on data entropy.)
            effective *= 0.85

    return int(max(1, round(effective)))


# ----------------------------
# Main entrypoint for Week-11
# ----------------------------
def apply_optimizations(
        payload: ArrayLike,
        spec: OptimSpec,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Applying clipping/quantization/compression based on spec.

    Returns:
      optimized_payload: either same-type arrays (CKKS-safe) or bytes (if packing)
      meta: info for reporting + potential reconstruction

    Behavior:
    - If spec.accounting_only=True:
        payload is returned unchanged (except clipping if enabled),
        but meta includes effective byte estimates for uplink/downlink accounting.
    - If spec.accounting_only=False:
        quantize -> pack -> compress can return bytes, with meta for reconstruction.
    """
    meta: Dict[str, Any] = {"spec": spec}

    # 1) Clipping (safe for both CKKS and non-CKKS)
    out: ArrayLike = payload
    if spec.enable_clipping:
        out, m_clip = l2_clip(out, spec.clip_norm)
        meta["clipping"] = m_clip
    else:
        meta["clipping"] = {"enabled": False}

    # Raw bytes after clipping (actual in-memory size of floats)
    raw_bytes = nbytes_of_payload(out)
    meta["raw_bytes"] = raw_bytes

    # 2) CKKS-safe path: do NOT change dtype to ints/bytes
    if spec.accounting_only:
        meta["accounting_only"] = True
        meta["effective_bytes_estimate"] = estimate_effective_bytes(raw_bytes, spec)
        return out, meta

    # 3) Real transform path (non-HE / explicit packing)
    meta["accounting_only"] = False

    if spec.enable_quantization:
        q_list, q_meta = quantize_payload(out, bits=spec.quant_bits, symmetric=spec.quant_symmetric)
        meta["quantization"] = q_meta
        packed = pack_int_payload(q_list)
        meta["packed_bytes"] = len(packed)
    else:
        # If no quantization, we pack float bytes directly (still possible)
        arrays = _as_list(out)
        packed = b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
        meta["packed_bytes"] = len(packed)
        meta["quantization"] = {"enabled": False}

    if spec.enable_compression:
        packed2, c_meta = compress_bytes(packed, method=spec.compression, level=spec.compression_level)
        meta["compression"] = c_meta
        meta["final_bytes"] = len(packed2)
        return packed2, meta

    meta["compression"] = {"enabled": False}
    meta["final_bytes"] = len(packed)
    return packed, meta


def reconstruct_from_bytes(
        packed: bytes,
        quant_meta: Dict[str, Any],
) -> List[np.ndarray]:
    """
    If we used apply_optimizations(..., accounting_only=False) with quantization=True
    and no compression (or after decompression), we can reconstruct floats here.

    IMPORTANT:
    - This requires you to know how you split packed bytes back into tensors.
      We keep this utility for completeness, but most Week-11 CKKS runs should use accounting_only=True.
    """
    metas = quant_meta["per_tensor_meta"]
    bits = int(quant_meta["bits"])

    # determine dtype and element size
    if bits == 8:
        dtype = np.int8
        elem_size = 1
    else:
        dtype = np.int16
        elem_size = 2

    out_q: List[np.ndarray] = []
    offset = 0
    for m in metas:
        shape = tuple(m["shape"])
        n_elems = int(np.prod(shape))
        n_bytes = n_elems * elem_size
        chunk = packed[offset : offset + n_bytes]
        q = np.frombuffer(chunk, dtype=dtype).reshape(shape)
        out_q.append(q.copy())
        offset += n_bytes

    # Dequantize
    return dequantize_payload(out_q, quant_meta)
