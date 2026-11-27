import numpy as np
from core.convolution_engine import convolve2d_manual

## Mengaburkan (blur) citra berwarna menggunakan filter box blur (3x3) dengan kernel yang diisi dengan nilai 1/9. Rumus yang digunakan adalah: out = convolve2d_manual(image, kernel), di mana kernel adalah matriks 3x3 dengan semua elemen bernilai 1/9.

def apply_box_blur(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    result = convolve2d_manual(image, kernel, clip=True)
    return result, kernel