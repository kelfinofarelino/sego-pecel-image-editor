import numpy as np
from core.convolution_engine import convolve2d_manual


def apply_box_blur(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    result = convolve2d_manual(image, kernel, clip=True)
    return result, kernel