import numpy as np

# Mengubah citra berwarna menjadi citra biner (hitam-putih) berdasarkan ambang batas (threshold) yang diberikan. Piksel dengan nilai luminansi di atas ambang akan menjadi putih (255), sedangkan di bawah ambang akan menjadi hitam (0).

def apply_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    img = image.astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return np.stack([binary, binary, binary], axis=-1)