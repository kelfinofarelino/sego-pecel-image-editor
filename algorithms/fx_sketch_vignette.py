import numpy as np
from algorithms.spatial_gaussian_blur import apply_gaussian_blur

### Mengubah citra berwarna menjadi citra sketsa pensil (pencil sketch) dan menambahkan efek vignette (gelap di tepi) dengan menggunakan konversi ke grayscale, inversi, Gaussian blur, dan rumus Color Dodge.

# Rumus Color Dodge: min(255, (I * 256) / (255 - B_inv + 1)), di mana I adalah citra grayscale asli dan B_inv adalah citra negatif yang telah diburamkan.
def apply_pencil_sketch(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    inv = 255.0 - gray

    # Blur citra negatif memakai konvolusi Gaussian manual 5x5
    blurred_inv, _ = apply_gaussian_blur(inv, ksize=5)
    blurred_inv, _ = apply_gaussian_blur(blurred_inv, ksize=5)
    b_float = blurred_inv.astype(np.float64)

    # Rumus Color Dodge: min(255, (I * 256) / (255 - B_inv + 1))
    dodge = np.minimum(255.0, (gray * 256.0) / (255.0 - b_float + 1.0))
    sketch_u8 = np.clip(dodge, 0, 255).astype(np.uint8)
    return np.stack([sketch_u8, sketch_u8, sketch_u8], axis=-1)

# Menambahkan efek vignette (gelap di tepi) pada citra berwarna dengan membuat masker Gaussian berbentuk lingkaran dan mengalikan citra asli dengan rumus: out = image * exp(-strength * 2.5 * (norm_dist^2)), di mana norm_dist adalah jarak normalisasi dari pusat citra ke setiap piksel.
def apply_vignette(image: np.ndarray, strength: float = 0.65) -> np.ndarray:
    H, W = image.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    yy, xx = np.indices((H, W), dtype=np.float64)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    norm_dist = dist / max(max_dist, 1e-6)

    # Masker jatuh cahaya Gaussian
    mask = np.exp(-strength * 2.5 * (norm_dist**2))
    out = image.astype(np.float64) * mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)