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

    - mode = "none": return parameters unchanged.
    - mode = "mask": add Gaussian noise (additive masking) to each parameter
      tensor (lightweight privacy / secure aggregation).
    - mode = "ckks": return parameters unchanged; secure aggregation is handled
      on the server side via homomorphic encryption (CKKS).

    NOTE: A full end-to-end CKKS pipeline (client-side encryption +
    server-side aggregation) is deferred to future work. In this project,
    CKKS is used to protect the *aggregation step* on the server.
    """
    if rng is None:
        rng = default_rng()

    # Plain baseline: no transformation
    if mode == "none":
        return params

    # CKKS: no client-side noise; encryption happens in the server aggregator
    if mode == "ckks":
        return params

    # Masking: lightweight Gaussian noise
    if mode == "mask":
        noisy: List[np.ndarray] = []
        for p in params:
            noise = rng.normal(loc=0.0, scale=noise_std, size=p.shape).astype(p.dtype)
            noisy.append(p + noise)
        return noisy

    # Unknown mode: warn and fall back to no protection
    print(f"[secure] Unknown mode '{mode}', using 'none'.")
    return params
