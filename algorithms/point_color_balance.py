import numpy as np


def apply_color_balance(
    image: np.ndarray,
    temperature: float = 0.0,
    w_r: float = 1.0,
    w_g: float = 1.0,
    w_b: float = 1.0,
) -> np.ndarray:
    """
    temperature: -100 (Cooler) hingga +100 (Warmer)
    w_r, w_g, w_b: pengali RGB Channel Mixer
    """
    img = image.astype(np.float64)
    temp_shift = temperature * 0.6

    img[:, :, 0] = (img[:, :, 0] + temp_shift) * w_r
    img[:, :, 1] = img[:, :, 1] * w_g
    img[:, :, 2] = (img[:, :, 2] - temp_shift) * w_b

    return np.clip(img, 0, 255).astype(np.uint8)