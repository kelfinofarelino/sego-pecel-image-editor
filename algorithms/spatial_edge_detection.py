import numpy as np
from core.convolution_engine import convolve2d_manual


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float64)
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def _canny_manual(
    gray: np.ndarray, low_thresh: float = 50.0, high_thresh: float = 140.0
) -> tuple[np.ndarray, np.ndarray]:
    # Tahap 1: Gaussian Smoothing 5x5
    g_kernel = (
        np.array(
            [
                [2, 4, 5, 4, 2],
                [4, 9, 12, 9, 4],
                [5, 12, 15, 12, 5],
                [4, 9, 12, 9, 4],
                [2, 4, 5, 4, 2],
            ],
            dtype=np.float64,
        )
        / 159.0
    )
    smoothed = convolve2d_manual(gray, g_kernel, clip=False)

    # Tahap 2: Gradien Sobel Gx, Gy, Magnitudo & Sudut
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve2d_manual(smoothed, kx, clip=False)
    gy = convolve2d_manual(smoothed, ky, clip=False)

    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = (mag / mag.max()) * 255.0
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    # Tahap 3: Non-Maximum Suppression
    H, W = mag.shape
    nms = np.zeros((H, W), dtype=np.float64)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            ang = angle[i, j]
            q, r = 255.0, 255.0
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q, r = mag[i, j + 1], mag[i, j - 1]
            elif 22.5 <= ang < 67.5:
                q, r = mag[i + 1, j - 1], mag[i - 1, j + 1]
            elif 67.5 <= ang < 112.5:
                q, r = mag[i + 1, j], mag[i - 1, j]
            elif 112.5 <= ang < 157.5:
                q, r = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if mag[i, j] >= q and mag[i, j] >= r:
                nms[i, j] = mag[i, j]

    # Tahap 4: Double Thresholding & Hysteresis
    strong = 255
    weak = 75
    res = np.zeros((H, W), dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= high_thresh)
    weak_i, weak_j = np.where((nms <= high_thresh) & (nms >= low_thresh))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Pelacakan histeresis 8-ketetanggaan
    changed = True
    while changed:
        changed = False
        padded = np.pad(res, 1, mode="constant")
        weak_mask = res == weak
        has_strong_neighbor = (
            (padded[:-2, :-2] == strong)
            | (padded[:-2, 1:-1] == strong)
            | (padded[:-2, 2:] == strong)
            | (padded[1:-1, :-2] == strong)
            | (padded[1:-1, 2:] == strong)
            | (padded[2:, :-2] == strong)
            | (padded[2:, 1:-1] == strong)
            | (padded[2:, 2:] == strong)
        )
        promote = weak_mask & has_strong_neighbor
        if np.any(promote):
            res[promote] = strong
            changed = True

    res[res != strong] = 0
    return res, g_kernel


def apply_edge_detection(
    image: np.ndarray,
    method: str = "sobel",
    low_thresh: float = 50.0,
    high_thresh: float = 140.0,
) -> tuple[np.ndarray, np.ndarray]:
    gray = _to_gray_float(image)
    method = method.lower()

    if method == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        gx = convolve2d_manual(gray, kx, clip=False)
        gy = convolve2d_manual(gray, ky, clip=False)
        mag = np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)
        kernel_shown = kx
    elif method == "laplacian":
        kernel_shown = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        resp = convolve2d_manual(gray, kernel_shown, clip=False)
        mag = np.clip(np.abs(resp), 0, 255).astype(np.uint8)
    elif method == "canny":
        mag, kernel_shown = _canny_manual(gray, low_thresh, high_thresh)
    else:  # sobel
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        gx = convolve2d_manual(gray, kx, clip=False)
        gy = convolve2d_manual(gray, ky, clip=False)
        mag = np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)
        kernel_shown = kx

    rgb_out = np.stack([mag, mag, mag], axis=-1)
    return rgb_out, kernel_shown