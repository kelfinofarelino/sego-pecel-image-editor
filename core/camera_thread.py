import time
from typing import Callable
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, float)
    camera_error = pyqtSignal(str)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self._running = False
        self.live_filter_enabled = False
        self.filter_fn: Callable[[np.ndarray], np.ndarray] | None = None
        self.latest_raw_rgb: np.ndarray | None = None

    def set_filter(self, fn: Callable[[np.ndarray], np.ndarray] | None) -> None:
        self.filter_fn = fn

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            self.camera_error.emit("Kamera tidak terdeteksi atau sedang digunakan aplikasi lain.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._running = True
        prev_time = time.perf_counter()

        while self._running:
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            # Konversi BGR ke RGB via pembalikan kanal array
            before_rgb = frame_bgr[:, :, ::-1].copy()
            self.latest_raw_rgb = before_rgb

            if self.live_filter_enabled and self.filter_fn is not None:
                try:
                    after_rgb = self.filter_fn(before_rgb)
                except Exception:
                    after_rgb = before_rgb.copy()
            else:
                after_rgb = before_rgb.copy()

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            fps = 1.0 / dt
            prev_time = now

            self.frame_ready.emit(before_rgb, after_rgb, fps)

        cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait(1500)