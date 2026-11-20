import numpy as np


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1, :].copy()


def flip_vertical(image: np.ndarray) -> np.ndarray:
    return image[::-1, :, :].copy()


def rotate_90(image: np.ndarray) -> np.ndarray:
    # Transposisi sumbu H dan W lalu balik kolom (90 derajat searah jarum jam)
    return np.transpose(image, (1, 0, 2))[:, ::-1, :].copy()


def rotate_180(image: np.ndarray) -> np.ndarray:
    return image[::-1, ::-1, :].copy()


def rotate_270(image: np.ndarray) -> np.ndarray:
    # Transposisi lalu balik baris (270 derajat searah jarum jam / 90 CCW)
    return np.transpose(image, (1, 0, 2))[::-1, :, :].copy()


def rotate_free(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotasi sudut bebas menggunakan pemetaan invers matriks rotasi
    dan interpolasi bilinear manual.
    """
    if angle_deg % 360 == 0:
        return image.copy()

    H, W, C = image.shape
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    new_W = int(np.ceil(abs(W * cos_t) + abs(H * sin_t)))
    new_H = int(np.ceil(abs(W * sin_t) + abs(H * cos_t)))

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ncx, ncy = (new_W - 1) / 2.0, (new_H - 1) / 2.0

    yy, xx = np.indices((new_H, new_W), dtype=np.float64)
    dx_out = xx - ncx
    dy_out = yy - ncy

    # Invers rotasi R(-theta)
    src_x = cos_t * dx_out + sin_t * dy_out + cx
    src_y = -sin_t * dx_out + cos_t * dy_out + cy

    valid = (src_x >= 0) & (src_x <= W - 1) & (src_y >= 0) & (src_y <= H - 1)
    out = np.zeros((new_H, new_W, C), dtype=np.uint8)

    vx = src_x[valid]
    vy = src_y[valid]

    x0 = np.floor(vx).astype(np.int32)
    y0 = np.floor(vy).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = (vx - x0)[:, np.newaxis]
    wy = (vy - y0)[:, np.newaxis]

    Ia = image[y0, x0].astype(np.float64)
    Ib = image[y0, x1].astype(np.float64)
    Ic = image[y1, x0].astype(np.float64)
    Id = image[y1, x1].astype(np.float64)

    interp = (
        Ia * (1 - wx) * (1 - wy)
        + Ib * wx * (1 - wy)
        + Ic * (1 - wx) * wy
        + Id * wx * wy
    )
    out[valid] = np.clip(interp, 0, 255).astype(np.uint8)
    return out