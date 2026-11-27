import numpy as np

# Mengatur kepekatan warna (saturation) citra berwarna dengan mengubah nilai piksel berdasarkan faktor saturasi yang diberikan (C = L + S * (C - L), di mana C adalah nilai piksel asli, L adalah luminansi, dan S adalah faktor saturasi)

def apply_saturation(image: np.ndarray, sat_factor: float = 1.0) -> np.ndarray:
    img = image.astype(np.float64)
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    lum_3d = lum[:, :, np.newaxis]
    out = lum_3d + sat_factor * (img - lum_3d)
    return np.clip(out, 0, 255).astype(np.uint8)