import numpy as np


def resize_manual(
    image: np.ndarray, scale_percent: int = 100, method: str = "bilinear"
) -> np.ndarray:
    H, W, C = image.shape
    new_W = max(1, int(round(W * scale_percent / 100.0)))
    new_H = max(1, int(round(H * scale_percent / 100.0)))

    if new_W == W and new_H == H:
        return image.copy()

    if method == "nearest":
        row_idx = np.minimum(
            np.floor(np.arange(new_H) * (H / new_H)).astype(np.int32), H - 1
        )
        col_idx = np.minimum(
            np.floor(np.arange(new_W) * (W / new_W)).astype(np.int32), W - 1
        )
        return image[row_idx[:, None], col_idx[None, :]].copy()

    # Bilinear Interpolation manual
    yy = np.linspace(0, H - 1, new_H)
    xx = np.linspace(0, W - 1, new_W)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")

    x0 = np.floor(grid_x).astype(np.int32)
    y0 = np.floor(grid_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = (grid_x - x0)[:, :, np.newaxis]
    wy = (grid_y - y0)[:, :, np.newaxis]

    Ia = image[y0, x0].astype(np.float64)
    Ib = image[y0, x1].astype(np.float64)
    Ic = image[y1, x0].astype(np.float64)
    Id = image[y1, x1].astype(np.float64)

    out = (
        Ia * (1 - wx) * (1 - wy)
        + Ib * wx * (1 - wy)
        + Ic * (1 - wx) * wy
        + Id * wx * wy
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_manual(
    image: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    H, W = image.shape[:2]
    top = max(0, min(top, H - 1))
    bottom = max(top + 1, min(bottom, H))
    left = max(0, min(left, W - 1))
    right = max(left + 1, min(right, W))
    return image[top:bottom, left:right].copy()