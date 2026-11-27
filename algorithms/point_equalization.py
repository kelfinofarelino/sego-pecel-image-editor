import numpy as np

# Mengubah citra berwarna menjadi citra dengan histogram equalization manual menggunakan perhitungan PDF dan CDF per kanal. Dengan rumus: lut = round(255 * CDF), di mana CDF adalah cumulative distribution function dari PDF (probability density function) histogram citra.

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Histogram Equalization manual menggunakan perhitungan PDF dan CDF per kanal.
    """
    out = np.zeros_like(image, dtype=np.uint8)
    total_pixels = image.shape[0] * image.shape[1]

    for c in range(3):
        channel = image[:, :, c]
        hist = np.bincount(channel.ravel(), minlength=256)
        pdf = hist.astype(np.float64) / total_pixels
        cdf = np.cumsum(pdf)
        lut = np.round(255.0 * cdf).astype(np.uint8)
        out[:, :, c] = lut[channel]

    return out