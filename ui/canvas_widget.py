import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget


def numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    contig = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, ch = contig.shape
    qimg = QImage(contig.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class DualCanvasWidget(QWidget):
    pixel_inspected = pyqtSignal(int, int, tuple, tuple)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(620, 380)

        self.before_img: np.ndarray | None = None
        self.after_img: np.ndarray | None = None
        self.before_pix: QPixmap | None = None
        self.after_pix: QPixmap | None = None
        self.is_preview_state = False

        self.split_mode = False
        self.split_ratio = 0.5
        self.dragging_split = False

        self.crop_mode = False
        self.crop_start: QPoint | None = None
        self.crop_end: QPoint | None = None
        self.dragging_crop = False
        self.last_after_rect = QRect()

        self.live_fps: float | None = None

    def set_images(
        self,
        before: np.ndarray | None,
        after: np.ndarray | None,
        is_preview: bool = False,
    ) -> None:
        self.before_img = before
        self.after_img = after
        self.is_preview_state = is_preview
        self.before_pix = numpy_to_pixmap(before) if before is not None else None
        self.after_pix = numpy_to_pixmap(after) if after is not None else None
        self.update()

    def set_split_mode(self, enabled: bool) -> None:
        self.split_mode = enabled
        self.update()

    def set_crop_mode(self, enabled: bool) -> None:
        self.crop_mode = enabled
        self.crop_start = None
        self.crop_end = None
        self.update()

    def get_crop_coordinates(self) -> tuple[int, int, int, int] | None:
        if (
            not self.crop_mode
            or self.crop_start is None
            or self.crop_end is None
            or self.after_img is None
            or self.last_after_rect.width() <= 0
        ):
            return None

        sel = QRect(self.crop_start, self.crop_end).normalized().intersected(self.last_after_rect)
        if sel.width() < 5 or sel.height() < 5:
            return None

        H, W = self.after_img.shape[:2]
        rx = W / self.last_after_rect.width()
        ry = H / self.last_after_rect.height()

        left = int((sel.left() - self.last_after_rect.left()) * rx)
        right = int((sel.right() - self.last_after_rect.left()) * rx)
        top = int((sel.top() - self.last_after_rect.top()) * ry)
        bottom = int((sel.bottom() - self.last_after_rect.top()) * ry)
        return (top, bottom, left, right)

    def _fit_rect(self, pix: QPixmap, bounds: QRect) -> QRect:
        scaled = pix.size().scaled(bounds.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = bounds.x() + (bounds.width() - scaled.width()) // 2
        y = bounds.y() + (bounds.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.crop_mode and self.last_after_rect.contains(event.pos()):
                self.crop_start = event.pos()
                self.crop_end = event.pos()
                self.dragging_crop = True
                self.update()
            elif self.split_mode and self.last_after_rect.contains(event.pos()):
                self.dragging_split = True
                self._update_split(event.pos().x())

    def mouseMoveEvent(self, event) -> None:
        pos = event.pos()
        if self.dragging_crop and self.crop_mode:
            self.crop_end = pos
            self.update()
            return

        if self.dragging_split and self.split_mode:
            self._update_split(pos.x())
            return

        if self.after_img is not None and self.last_after_rect.contains(pos):
            H, W = self.after_img.shape[:2]
            ix = int(
                (pos.x() - self.last_after_rect.left())
                * W
                / max(1, self.last_after_rect.width())
            )
            iy = int(
                (pos.y() - self.last_after_rect.top())
                * H
                / max(1, self.last_after_rect.height())
            )
            ix = max(0, min(ix, W - 1))
            iy = max(0, min(iy, H - 1))

            proc_rgb = tuple(int(v) for v in self.after_img[iy, ix])
            if (
                self.before_img is not None
                and iy < self.before_img.shape[0]
                and ix < self.before_img.shape[1]
            ):
                orig_rgb = tuple(int(v) for v in self.before_img[iy, ix])
            else:
                orig_rgb = proc_rgb
            self.pixel_inspected.emit(ix, iy, orig_rgb, proc_rgb)

    def mouseReleaseEvent(self, event) -> None:
        self.dragging_crop = False
        self.dragging_split = False

    def _update_split(self, mouse_x: int) -> None:
        if self.last_after_rect.width() > 0:
            ratio = (mouse_x - self.last_after_rect.left()) / self.last_after_rect.width()
            self.split_ratio = max(0.05, min(0.95, ratio))
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#101115"))
        painter.setPen(QPen(QColor("#262a34"), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if self.before_pix is None or self.after_pix is None:
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Click 'Open Image' or 'Start Camera' to load a source frame",
            )
            painter.end()
            return

        margin = 12
        header_h = 24
        status_tag = "PREVIEW" if self.is_preview_state else "PROCESSED"

        if self.split_mode:
            area = QRect(
                margin,
                margin + header_h,
                self.width() - 2 * margin,
                self.height() - 2 * margin - header_h,
            )
            target = self._fit_rect(self.after_pix, area)
            self.last_after_rect = target

            painter.drawPixmap(target, self.after_pix)
            split_w = int(target.width() * self.split_ratio)
            src_split_w = int(self.before_pix.width() * self.split_ratio)

            painter.drawPixmap(
                QRect(target.left(), target.top(), split_w, target.height()),
                self.before_pix,
                QRect(0, 0, src_split_w, self.before_pix.height()),
            )
            split_x = target.left() + split_w
            painter.setPen(QPen(QColor("#fa7516"), 2))
            painter.drawLine(split_x, target.top(), split_x, target.bottom())
            painter.drawText(
                margin, 20, f"SPLIT VIEW | BEFORE (LEFT) / AFTER - {status_tag} (RIGHT)"
            )
        else:
            half_w = (self.width() - 3 * margin) // 2
            left_area = QRect(
                margin, margin + header_h, half_w, self.height() - 2 * margin - header_h
            )
            right_area = QRect(
                2 * margin + half_w,
                margin + header_h,
                half_w,
                self.height() - 2 * margin - header_h,
            )

            r_before = self._fit_rect(self.before_pix, left_area)
            r_after = self._fit_rect(self.after_pix, right_area)
            self.last_after_rect = r_after

            painter.setPen(QColor("#94a3b8"))
            painter.drawText(
                margin,
                20,
                f"BEFORE (ORIGINAL) | {self.before_img.shape[1]}x{self.before_img.shape[0]}",
            )
            painter.setPen(QColor("#38bdf8" if self.is_preview_state else "#fa7516"))
            painter.drawText(
                2 * margin + half_w,
                20,
                f"AFTER ({status_tag}) | {self.after_img.shape[1]}x{self.after_img.shape[0]}",
            )

            painter.drawPixmap(r_before, self.before_pix)
            painter.drawPixmap(r_after, self.after_pix)

            if self.crop_mode and self.crop_start and self.crop_end:
                sel = QRect(self.crop_start, self.crop_end).normalized().intersected(r_after)
                dim = QColor(0, 0, 0, 170)
                painter.fillRect(
                    QRect(r_after.left(), r_after.top(), r_after.width(), sel.top() - r_after.top()),
                    dim,
                )
                painter.fillRect(
                    QRect(
                        r_after.left(),
                        sel.bottom(),
                        r_after.width(),
                        r_after.bottom() - sel.bottom(),
                    ),
                    dim,
                )
                painter.fillRect(
                    QRect(r_after.left(), sel.top(), sel.left() - r_after.left(), sel.height()),
                    dim,
                )
                painter.fillRect(
                    QRect(sel.right(), sel.top(), r_after.right() - sel.right(), sel.height()),
                    dim,
                )
                painter.setPen(QPen(QColor("#38bdf8"), 2, Qt.PenStyle.DashLine))
                painter.drawRect(sel)

        if self.live_fps is not None:
            hud_rect = QRect(self.width() - 230, 6, 218, 22)
            painter.fillRect(hud_rect, QColor(18, 20, 26, 225))
            painter.setBrush(QColor("#22c55e"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(hud_rect.left() + 10, hud_rect.top() + 7, 8, 8)
            painter.setPen(QColor("#e2e8f0"))
            painter.drawText(
                hud_rect.adjusted(22, 0, 0, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"LIVE | {self.live_fps:.1f} FPS (Manual NumPy)",
            )

        painter.end()