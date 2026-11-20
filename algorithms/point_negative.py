import numpy as np


def apply_negative(image: np.ndarray) -> np.ndarray:
    return (255 - image.astype(np.int16)).clip(0, 255).astype(np.uint8)