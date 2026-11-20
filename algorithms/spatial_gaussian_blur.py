import numpy as np
from core.convolution_engine import convolve2d_manual


def apply_gaussian_blur(
    image: np.ndarray, ksize: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    if ksize == 5:
        kernel = (
            np.array(
                [
                    [1, 4, 7, 4, 1],
                    [4, 16, 26, 16, 4],
                    [7, 26, 41, 26, 7],
                    [4, 16, 26, 16, 4],
                    [1, 4, 7, 4, 1],
                ],
                dtype=np.float64,
            )
            / 273.0
        )
    else:
        kernel = (
            np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        )

    result = convolve2d_manual(image, kernel, clip=True)
    return result, kernel