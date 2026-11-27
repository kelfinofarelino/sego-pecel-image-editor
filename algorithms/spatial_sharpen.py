import numpy as np
from core.convolution_engine import convolve2d_manual

### Mengasah (sharpen) citra berwarna menggunakan filter sharpen (3x3) dengan kernel yang sesuai. Rumus yang digunakan adalah: out = convolve2d_manual(image, kernel), di mana kernel adalah matriks 3x3 dengan elemen [[0, -1, 0], [-1, 5, -1], [0, -1, 0]].

def apply_sharpen(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    result = convolve2d_manual(image, kernel, clip=True)
    return result, kernel