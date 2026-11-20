import numpy as np


def apply_grayscale(image: np.ndarray, mode: str = "luminosity") -> np.ndarray:
    img = image.astype(np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    if mode == "average":
        gray = (r + g + b) / 3.0
    else:
        gray = 0.299 * r + 0.587 * g + 0.114 * b

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)