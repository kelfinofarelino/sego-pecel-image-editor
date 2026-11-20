import numpy as np


def apply_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    img = image.astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return np.stack([binary, binary, binary], axis=-1)