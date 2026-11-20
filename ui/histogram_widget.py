import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class HistogramWidget(QWidget):
    def __init__(self, title: str, enable_ghost_toggle: bool = False):
        super().__init__()
        self.setMinimumHeight(130)
        self.setMaximumHeight(150)
        self.hist_rgb: list[np.ndarray] | None = None
        self.ghost_rgb: list[np.ndarray] | None = None
        self.show_ghost = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(2)

        header = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700; letter-spacing: 0.5px;")
        header.addWidget(lbl)
        header.addStretch()

        if enable_ghost_toggle:
            self.ghost_chk = QCheckBox("Compare Ghost Before")
            self.ghost_chk.setStyleSheet("color: #fa7516; font-size: 11px; font-weight: 600;")
            self.ghost_chk.toggled.connect(self._toggle_ghost)
            header.addWidget(self.ghost_chk)

        layout.addLayout(header)
        layout.addStretch()

    def _toggle_ghost(self, checked: bool) -> None:
        self.show_ghost = checked
        self.update()

    @staticmethod
    def _compute_rgb_hist(image: np.ndarray) -> list[np.ndarray]:
        hists = []
        for c in range(3):
            counts = np.bincount(image[:, :, c].ravel(), minlength=256).astype(np.float64)
            if counts.max() > 0:
                counts /= counts.max()
            hists.append(counts)
        return hists

    def update_histogram(
        self, image: np.ndarray | None, ghost_image: np.ndarray | None = None
    ) -> None:
        self.hist_rgb = self._compute_rgb_hist(image) if image is not None else None
        self.ghost_rgb = (
            self._compute_rgb_hist(ghost_image) if ghost_image is not None else None
        )
        self.update()

    def _draw_curves(
        self, painter: QPainter, hists: list[np.ndarray], alpha_line: int, alpha_fill: int
    ) -> None:
        x0, y0 = 10, 26
        w = max(10, self.width() - 20)
        h = max(10, self.height() - 34)
        colors = [
            QColor(239, 68, 68, alpha_line),
            QColor(34, 197, 94, alpha_line),
            QColor(59, 130, 246, alpha_line),
        ]
        fills = [
            QColor(239, 68, 68, alpha_fill),
            QColor(34, 197, 94, alpha_fill),
            QColor(59, 130, 246, alpha_fill),
        ]

        for c in range(3):
            vals = hists[c]
            path = QPainterPath()
            path.moveTo(x0, y0 + h)
            for i in range(256):
                px = x0 + (i / 255.0) * w
                py = y0 + h - (vals[i] * h)
                path.lineTo(px, py)
            path.lineTo(x0 + w, y0 + h)
            path.closeSubpath()

            painter.fillPath(path, fills[c])
            painter.setPen(QPen(colors[c], 1.3))
            painter.drawPath(path)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#15171c"))
        painter.setPen(QPen(QColor("#262a34"), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if self.show_ghost and self.ghost_rgb is not None:
            self._draw_curves(painter, self.ghost_rgb, alpha_line=90, alpha_fill=22)

        if self.hist_rgb is not None:
            self._draw_curves(painter, self.hist_rgb, alpha_line=235, alpha_fill=50)

        painter.end()