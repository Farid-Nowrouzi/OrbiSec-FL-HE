import numpy as np
from typing import List


def apply_security_to_params(
    params: List[np.ndarray],
    mode: str = "none",
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """
    Apply a simple security transformation to model parameters.

    mode="none": return parameters unchanged
    mode="mask": add Gaussian noise (additive masking) to each parameter tensor

    NOTE: For now, mode="ckks" falls back to "mask" with a warning. A real CKKS
    HE pipeline will be integrated later.
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == "none":
        return params

    if mode == "ckks":
        # Placeholder: behave like mask for now
        print("[secure] CKKS mode not wired yet, falling back to noise-masking.")
        mode = "mask"

    if mode == "mask":
        noisy = []
        for p in params:
            noise = rng.normal(loc=0.0, scale=noise_std, size=p.shape).astype(p.dtype)
            noisy.append(p + noise)
        return noisy

    # Unknown mode: just return params and warn
    print(f"[secure] Unknown mode '{mode}', using 'none'.")
    return params
