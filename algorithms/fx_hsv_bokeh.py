import numpy as np
from algorithms.spatial_gaussian_blur import apply_gaussian_blur


def rgb_to_hsv_manual(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Konversi manual ruang warna RGB [0..255] ke HSV:
    H in [0, 360), S in [0, 1], V in [0, 1].
    """
    rgb = image.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    h = np.zeros_like(c_max)
    mask_nz = delta > 1e-6

    r_max = mask_nz & (c_max == r)
    g_max = mask_nz & (c_max == g)
    b_max = mask_nz & (c_max == b)

    h[r_max] = (60.0 * (((g[r_max] - b[r_max]) / delta[r_max]) % 6.0))
    h[g_max] = (60.0 * (((b[g_max] - r[g_max]) / delta[g_max]) + 2.0))
    h[b_max] = (60.0 * (((r[b_max] - g[b_max]) / delta[b_max]) + 4.0))
    h = h % 360.0

    s = np.where(c_max > 1e-6, delta / np.maximum(c_max, 1e-6), 0.0)
    v = c_max
    return h, s, v


def apply_hsv_bokeh(
    image: np.ndarray,
    target_hue: float = 0.0,
    hue_tolerance: float = 25.0,
    min_saturation: float = 0.25,
) -> np.ndarray:
    """
    Segmentasi warna target pada ruang HSV lalu memburamkan latar belakang (Bokeh).
    """
    h, s, _ = rgb_to_hsv_manual(image)
    diff = np.abs(h - target_hue)
    circular_diff = np.minimum(diff, 360.0 - diff)

    fg_mask = (circular_diff <= hue_tolerance) & (s >= min_saturation)

    # Blur latar belakang menggunakan Gaussian Blur manual 5x5 multi-pass
    bg_blur, _ = apply_gaussian_blur(image, ksize=5)
    bg_blur, _ = apply_gaussian_blur(bg_blur, ksize=5)
    bg_blur, _ = apply_gaussian_blur(bg_blur, ksize=5)

    return np.where(fg_mask[:, :, np.newaxis], image, bg_blur).astype(np.uint8)