import numpy as np

### Mengubah citra berwarna menjadi citra pixelated (kotak-kotak) dengan mengelompokkan piksel dalam blok berukuran tertentu dan mengganti nilai piksel dalam blok tersebut dengan rata-rata warna blok tersebut.

def apply_pixelate(image: np.ndarray, block_size: int = 12) -> np.ndarray:
    k = max(2, int(block_size))
    H, W, C = image.shape
    out = image.copy()

    for y in range(0, H, k):
        for x in range(0, W, k):
            block = image[y : y + k, x : x + k]
            mean_color = np.mean(block, axis=(0, 1))
            out[y : y + k, x : x + k] = mean_color.astype(np.uint8)

    return out