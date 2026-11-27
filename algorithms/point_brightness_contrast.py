import numpy as np

# Mengatur kecerahan (brightness) dan kontras (contrast) citra berwarna dengan mengubah nilai piksel berdasarkan faktor alpha (kontras) dan beta (kecerahan) yang diberikan. Rumus yang digunakan adalah: out = alpha * (img - 128) + 128 + beta

def apply_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    img = image.astype(np.float64)
    out = alpha * (img - 128.0) + 128.0 + beta
    return np.clip(out, 0, 255).astype(np.uint8)