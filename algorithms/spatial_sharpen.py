import numpy as np
from core.convolution_engine import convolve2d_manual


def apply_sharpen(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    result = convolve2d_manual(image, kernel, clip=True)
    return result, kernel