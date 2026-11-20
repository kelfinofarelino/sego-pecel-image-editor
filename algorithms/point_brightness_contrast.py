import numpy as np


def apply_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    img = image.astype(np.float64)
    out = alpha * (img - 128.0) + 128.0 + beta
    return np.clip(out, 0, 255).astype(np.uint8)