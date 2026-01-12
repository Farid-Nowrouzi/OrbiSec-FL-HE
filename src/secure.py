import numpy as np
from typing import List, Optional
from numpy.random import Generator, default_rng


def apply_security_to_params(
        params: List[np.ndarray],
        mode: str = "none",
        noise_std: float = 0.01,
        rng: Optional[Generator] = None,
) -> List[np.ndarray]:
    """
    Apply a simple security transformation to model parameters.

    DESIGN INTENT (IMPORTANT):
    --------------------------
    This helper is intentionally SIMPLE.

    In this project:
      - "mask"  : lightweight baseline (heuristic Gaussian noise)
      - "ckks"  : handled entirely SERVER-SIDE (homomorphic aggregation)
      - "dp"    : handled CLIENT-SIDE on the UPDATE (delta), NOT here

    This file MUST NOT implement DP logic to avoid double-noising.

    Supported modes:
      - "none" : return parameters unchanged
      - "mask" : add Gaussian noise to each parameter tensor
      - "ckks" : return parameters unchanged (encryption is server-side)
      - "dp"   : return parameters unchanged (DP handled in client.py)

    This separation is CRITICAL for:
      - clean experiments
      - correct threat modeling
      - a defensible system-security narrative
    """

    if rng is None:
        rng = default_rng()

    mode = (mode or "none").lower().strip()

    # --------------------------------------------------
    # No protection (baseline)
    # --------------------------------------------------
    if mode == "none":
        return params

    # --------------------------------------------------
    # CKKS: encryption + aggregation happens server-side
    # --------------------------------------------------
    if mode == "ckks":
        return params

    # --------------------------------------------------
    # DP is explicitly NOT applied here
    # (to avoid double-noising)
    # --------------------------------------------------
    if mode == "dp":
        return params

    # --------------------------------------------------
    # Masking: lightweight Gaussian noise baseline
    # --------------------------------------------------
    if mode == "mask":
        noisy: List[np.ndarray] = []
        for p in params:
            noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=p.shape,
            ).astype(p.dtype)
            noisy.append(p + noise)
        return noisy

    # --------------------------------------------------
    # Fallback (should never happen)
    # --------------------------------------------------
    print(f"[secure] Unknown mode '{mode}', falling back to 'none'.")
    return params
