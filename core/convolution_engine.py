import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def convolve2d_manual(
    image: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "edge",
    clip: bool = True,
) -> np.ndarray:
    """
    Konvolusi 2D manual menggunakan padding array dan sliding window dot-product.
    Mendukung citra 2D (grayscale) maupun 3D (RGB).
    """
    img_float = image.astype(np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    kh, kw = k.shape
    pad_h, pad_w = kh // 2, kw // 2

    if img_float.ndim == 2:
        padded = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)
        windows = sliding_window_view(padded, (kh, kw))
        output = np.sum(windows * k, axis=(-2, -1))
    else:
        channels = []
        for c in range(img_float.shape[2]):
            padded = np.pad(
                img_float[:, :, c], ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode
            )
            windows = sliding_window_view(padded, (kh, kw))
            conv_c = np.sum(windows * k, axis=(-2, -1))
            channels.append(conv_c)
        output = np.stack(channels, axis=-1)

    if clip:
        return np.clip(output, 0, 255).astype(np.uint8)
    return output