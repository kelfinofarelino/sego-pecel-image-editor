import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng()
    H, W = image.shape[:2]
    probs = rng.random((H, W))
    out = image.copy()

    out[probs < (amount / 2.0)] = 0
    out[probs > (1.0 - amount / 2.0)] = 255
    return out


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, image.shape)
    out = image.astype(np.float64) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Median Filter non-linear manual dengan mengambil nilai tengah pada jendela k x k.
    """
    pad = ksize // 2
    out = np.zeros_like(image, dtype=np.uint8)

    for c in range(image.shape[2]):
        padded = np.pad(image[:, :, c], pad, mode="edge")
        windows = sliding_window_view(padded, (ksize, ksize))
        out[:, :, c] = np.median(windows, axis=(-2, -1)).astype(np.uint8)

    return out