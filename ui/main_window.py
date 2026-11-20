from typing import Callable
import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from algorithms.fx_hsv_bokeh import apply_hsv_bokeh
from algorithms.fx_noise_denoise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_median_filter,
)
from algorithms.fx_pixelate import apply_pixelate
from algorithms.fx_sketch_vignette import apply_pencil_sketch, apply_vignette
from algorithms.geo_resize_crop import crop_manual, resize_manual
from algorithms.geo_rotate_flip import (
    flip_horizontal,
    flip_vertical,
    rotate_90,
    rotate_180,
    rotate_270,
    rotate_free,
)
from algorithms.point_brightness_contrast import apply_brightness_contrast
from algorithms.point_color_balance import apply_color_balance
from algorithms.point_equalization import apply_histogram_equalization
from algorithms.point_grayscale import apply_grayscale
from algorithms.point_negative import apply_negative
from algorithms.point_saturation import apply_saturation
from algorithms.point_threshold import apply_threshold
from algorithms.spatial_box_blur import apply_box_blur
from algorithms.spatial_edge_detection import apply_edge_detection
from algorithms.spatial_gaussian_blur import apply_gaussian_blur
from algorithms.spatial_sharpen import apply_sharpen
from core.camera_thread import CameraThread
from core.history_manager import HistoryManager
from ui.canvas_widget import DualCanvasWidget
from ui.histogram_widget import HistogramWidget
from ui.inspector_panel import InspectorPanel

EFFECT_CATALOG = [
    (
        "1. Point Operations",
        [
            ("grayscale", "Grayscale Conversion", "I = 0.299R + 0.587G + 0.114B | (R+G+B)/3"),
            ("threshold", "Binary Thresholding", "g(x,y) = 255 if f(x,y) >= T else 0"),
            ("negative", "Image Negative", "g(x,y) = 255 - f(x,y)"),
            ("brightness_contrast", "Brightness & Contrast", "g = clip(alpha*(f - 128) + 128 + beta)"),
            ("saturation", "Color Saturation", "C' = clip(L + S * (C - L), 0, 255)"),
            ("color_balance", "Color Temperature & RGB Mixer", "Channel scaling (wR, wG, wB) + temp shift"),
            ("equalization", "Histogram Equalization", "s_k = round(255 * CDF(r_k))"),
        ],
    ),
    (
        "2. Spatial Operations (2D Conv)",
        [
            ("box_blur", "Box Blur (Averaging)", "2D Convolution with 3x3 uniform kernel (1/9)"),
            ("gaussian_blur", "Gaussian Blur", "2D Convolution with Gaussian kernel (3x3 / 5x5)"),
            ("sharpen", "Sharpening", "High-boost center kernel [0,-1,0; -1,5,-1; 0,-1,0]"),
            ("edge_detection", "Edge Detection", "Gradient magnitude G = sqrt(Gx^2 + Gy^2) / Canny"),
        ],
    ),
    (
        "3. Geometric Transformations",
        [
            ("rotate_fixed", "Rotate", "Matrix transposition & axis reversal (90 / 180 / 270 deg)"),
            ("rotate_free", "Free Angle Rotation", "Inverse rotation matrix + bilinear interpolation"),
            ("mirror", "Mirror / Flip", "Horizontal or vertical index inversion"),
            ("resize", "Scale / Resize", "Bilinear or Nearest-Neighbor coordinate mapping"),
            ("crop", "Manual Crop", "Array slice img[top:bottom, left:right]"),
        ],
    ),
    (
        "4. Creative & Noise Filters",
        [
            ("pencil_sketch", "Pencil Sketch", "Color Dodge: min(255, (I * 256) / (255 - B_inv + 1))"),
            ("vignette", "Vignette", "Gaussian radial falloff mask from center"),
            ("pixelate", "Pixelation (Mosaic)", "Block-wise mean intensity over k x k tiles"),
            ("noise_gen", "Noise Generator", "Salt & Pepper impulse or Gaussian N(0, sigma^2)"),
            ("median_denoise", "Denoising (Median Filter)", "Non-linear spatial median over k x k window"),
        ],
    ),
    (
        "5. Segmentation & Optics",
        [
            ("hsv_bokeh", "HSV Color Segmentation & Bokeh", "Manual RGB->HSV Hue mask + Gaussian blur"),
        ],
    ),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Sego Pecel Image Editor - Studi Eksplorasi Grayscale dan Optik untuk Pengolahan Citra dan Elemen piksel"
        )
        self.resize(1420, 860)
        self.history = HistoryManager(max_steps=20)
        self.camera_thread: CameraThread | None = None

        self.selected_effect_id: str | None = None
        self.staged_image: np.ndarray | None = None
        self.staged_effect_label: str = ""
        self.active_live_fn: Callable[[np.ndarray], np.ndarray] | None = None

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #111215;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #1e2128;
                border: 1px solid #2e3440;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                border-color: #fa7516;
                background-color: #262a33;
            }
            QPushButton:checked {
                background-color: #fa7516;
                color: #111215;
                border-color: #fa7516;
            }
            QComboBox {
                background-color: #1e2128;
                border: 1px solid #2e3440;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2b303c;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #fa7516;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 8, 10, 10)
        root_layout.setSpacing(8)

        # 1. TOP TOOLBAR
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        btn_open = QPushButton("Open Image")
        btn_open.clicked.connect(self.open_image_dialog)

        self.btn_cam = QPushButton("Start Camera")
        self.btn_cam.clicked.connect(self.toggle_camera)

        self.btn_capture = QPushButton("Capture Frame")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.capture_camera_frame)

        self.btn_live_fx = QPushButton("Live Filter: OFF")
        self.btn_live_fx.setEnabled(False)
        self.btn_live_fx.clicked.connect(self.toggle_live_filter)

        self.btn_split = QPushButton("Split View")
        self.btn_split.setCheckable(True)
        self.btn_split.toggled.connect(self._toggle_split_view)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #2e3440;")

        self.top_btn_undo = QPushButton("Undo (0)")
        self.top_btn_undo.setEnabled(False)
        self.top_btn_undo.clicked.connect(self.undo_action)

        self.top_btn_redo = QPushButton("Redo (0)")
        self.top_btn_redo.setEnabled(False)
        self.top_btn_redo.clicked.connect(self.redo_action)

        self.top_btn_reset = QPushButton("Reset Original")
        self.top_btn_reset.setEnabled(False)
        self.top_btn_reset.clicked.connect(self.reset_action)

        btn_save = QPushButton("Export Image")
        btn_save.setStyleSheet(
            "QPushButton { background-color: #fa7516; color: #111215; border: none; font-weight: 700; }"
            "QPushButton:hover { background-color: #ff8933; }"
        )
        btn_save.clicked.connect(self.save_image_dialog)

        for w in (
            btn_open,
            self.btn_cam,
            self.btn_capture,
            self.btn_live_fx,
            self.btn_split,
            sep,
            self.top_btn_undo,
            self.top_btn_redo,
            self.top_btn_reset,
        ):
            top_bar.addWidget(w)
        top_bar.addStretch()
        top_bar.addWidget(btn_save)
        root_layout.addLayout(top_bar)

        # 2. MAIN WORKSPACE (Left Effects List | Center Monitors + Histograms | Right Inspector + Control Panel)
        workspace_row = QHBoxLayout()
        workspace_row.setSpacing(8)

        # LEFT: EFFECTS LIST
        left_frame = QFrame()
        left_frame.setFixedWidth(250)
        left_frame.setStyleSheet(
            "QFrame { background-color: #14161b; border: 1px solid #282d38; border-radius: 6px; }"
        )
        left_lay = QVBoxLayout(left_frame)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)

        cat_header = QLabel("EFFECTS CATALOG")
        cat_header.setStyleSheet(
            "color: #94a3b8; font-size: 10px; font-weight: 700; letter-spacing: 0.8px; border: none;"
        )
        left_lay.addWidget(cat_header)

        self.effect_tree = QTreeWidget()
        self.effect_tree.setHeaderHidden(True)
        self.effect_tree.setIndentation(14)
        self.effect_tree.setStyleSheet(
            """
            QTreeWidget {
                background-color: #181b22;
                border: 1px solid #252a34;
                border-radius: 4px;
                font-size: 11px;
                outline: none;
            }
            QTreeWidget::item {
                padding: 5px 4px;
                border-radius: 3px;
            }
            QTreeWidget::item:hover {
                background-color: #232834;
            }
            QTreeWidget::item:selected {
                background-color: #fa7516;
                color: #111215;
                font-weight: 700;
            }
            """
        )

        for category_name, items in EFFECT_CATALOG:
            cat_item = QTreeWidgetItem([category_name])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font_cat = cat_item.font(0)
            font_cat.setBold(True)
            cat_item.setFont(0, font_cat)
            self.effect_tree.addTopLevelItem(cat_item)

            for eff_id, eff_name, eff_formula in items:
                child = QTreeWidgetItem([eff_name])
                child.setData(0, Qt.ItemDataRole.UserRole, (eff_id, eff_name, eff_formula))
                cat_item.addChild(child)

        self.effect_tree.expandAll()
        self.effect_tree.itemClicked.connect(self._on_effect_selected)
        left_lay.addWidget(self.effect_tree)
        workspace_row.addWidget(left_frame)

        # CENTER: DUAL MONITOR CANVAS + SYNCED HISTOGRAMS
        center_col = QVBoxLayout()
        center_col.setSpacing(8)

        self.canvas = DualCanvasWidget()
        center_col.addWidget(self.canvas, stretch=4)

        hist_row = QHBoxLayout()
        hist_row.setSpacing(8)
        self.hist_before = HistogramWidget(
            "RGB HISTOGRAM - BEFORE (ORIGINAL)", enable_ghost_toggle=False
        )
        self.hist_after = HistogramWidget(
            "RGB HISTOGRAM - AFTER (PROCESSED)", enable_ghost_toggle=True
        )
        hist_row.addWidget(self.hist_before)
        hist_row.addWidget(self.hist_after)
        center_col.addLayout(hist_row, stretch=1)

        workspace_row.addLayout(center_col, stretch=1)

        # RIGHT: PIXEL INSPECTOR + KERNEL + HISTORY + EFFECT CONTROL PANEL
        self.inspector = InspectorPanel()
        self.canvas.pixel_inspected.connect(self.inspector.update_pixel)
        self.inspector.btn_undo.clicked.connect(self.undo_action)
        self.inspector.btn_redo.clicked.connect(self.redo_action)
        self.inspector.btn_reset.clicked.connect(self.reset_action)
        self.inspector.apply_clicked.connect(self.commit_staged_effect)
        self.inspector.discard_preview_clicked.connect(self.discard_staged_preview)

        workspace_row.addWidget(self.inspector)
        root_layout.addLayout(workspace_row, stretch=1)

        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_action)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo_action)

    # --- HELPER PEMBUAT SLIDER DI CONTROL PANEL ---
    def _add_slider_control(
        self,
        label_text: str,
        min_val: int,
        max_val: int,
        default_val: int,
        on_change: Callable[[int], None],
        formatter: Callable[[int], str] | None = None,
    ) -> QSlider:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 11px; color: #cbd5e1;")
        val_lbl = QLabel(formatter(default_val) if formatter else str(default_val))
        val_lbl.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 11px; color: #fa7516; font-weight: 700;"
        )
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(val_lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)

        def handle_slide(v: int) -> None:
            val_lbl.setText(formatter(v) if formatter else str(v))
            on_change(v)

        slider.valueChanged.connect(handle_slide)
        self.inspector.params_layout.addLayout(row)
        self.inspector.params_layout.addWidget(slider)
        return slider

    def _make_transform_icon(self, kind: str) -> QIcon:
        pix = QPixmap(20, 20)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor("#e2e8f0"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        if kind == "none":
            painter.drawEllipse(QRectF(4.0, 4.0, 12.0, 12.0))
            painter.drawLine(QPointF(6.0, 14.0), QPointF(14.0, 6.0))

        elif kind == "rot_90":
            # Panah melingkar searah jarum jam (90 CW)
            painter.drawArc(QRectF(3.5, 3.5, 13.0, 13.0), 135 * 16, -250 * 16)
            painter.setBrush(QColor("#e2e8f0"))
            arrow = QPolygonF([QPointF(15.5, 11.5), QPointF(12.5, 9.5), QPointF(16.8, 8.2)])
            painter.drawPolygon(arrow)

        elif kind == "rot_180":
            # Panah putar balik 180 derajat
            painter.drawArc(QRectF(4.0, 3.5, 12.0, 13.0), 90 * 16, -180 * 16)
            painter.drawLine(QPointF(6.0, 3.5), QPointF(10.0, 3.5))
            painter.drawLine(QPointF(6.0, 16.5), QPointF(10.0, 16.5))
            painter.setBrush(QColor("#e2e8f0"))
            arrow = QPolygonF([QPointF(5.5, 16.5), QPointF(8.5, 14.5), QPointF(8.5, 18.5)])
            painter.drawPolygon(arrow)

        elif kind == "rot_270":
            # Panah melingkar berlawanan jarum jam (270 CW / 90 CCW)
            painter.drawArc(QRectF(3.5, 3.5, 13.0, 13.0), 45 * 16, 250 * 16)
            painter.setBrush(QColor("#e2e8f0"))
            arrow = QPolygonF([QPointF(4.5, 11.5), QPointF(7.5, 9.5), QPointF(3.2, 8.2)])
            painter.drawPolygon(arrow)

        elif kind == "flip_h":
            # Garis sumbu vertikal + dua segitiga berhadapan horizontal
            dash_pen = QPen(QColor("#fa7516"), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            painter.drawLine(QPointF(10.0, 2.5), QPointF(10.0, 17.5))

            painter.setPen(pen)
            painter.setBrush(QColor("#e2e8f0"))
            left_tri = QPolygonF([QPointF(3.0, 10.0), QPointF(7.5, 6.0), QPointF(7.5, 14.0)])
            right_tri = QPolygonF([QPointF(17.0, 10.0), QPointF(12.5, 6.0), QPointF(12.5, 14.0)])
            painter.drawPolygon(left_tri)
            painter.drawPolygon(right_tri)

        elif kind == "flip_v":
            # Garis sumbu horizontal + dua segitiga berhadapan vertikal
            dash_pen = QPen(QColor("#fa7516"), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            painter.drawLine(QPointF(2.5, 10.0), QPointF(17.5, 10.0))

            painter.setPen(pen)
            painter.setBrush(QColor("#e2e8f0"))
            top_tri = QPolygonF([QPointF(10.0, 3.0), QPointF(6.0, 7.5), QPointF(14.0, 7.5)])
            bot_tri = QPolygonF([QPointF(10.0, 17.0), QPointF(6.0, 12.5), QPointF(14.0, 12.5)])
            painter.drawPolygon(top_tri)
            painter.drawPolygon(bot_tri)

        painter.end()
        return QIcon(pix)

    def _clear_pending_preview(self, status_msg: str) -> None:
        self.staged_image = None
        self.staged_effect_label = ""
        self.active_live_fn = None
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_filter(None)
        elif self.history.current_image is not None:
            self.canvas.set_images(
                self.history.original_image, self.history.current_image, is_preview=False
            )
            self.hist_after.update_histogram(
                self.history.current_image, ghost_image=self.history.original_image
            )
        self.inspector.set_status_info(status_msg)

    # --- PEMILIHAN EFEK & PEMBANGUNAN CONTROL PANEL ---
    def _on_effect_selected(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        eff_id, eff_name, eff_formula = data
        self.selected_effect_id = eff_id
        self.canvas.set_crop_mode(False)
        self.inspector.set_control_header(eff_name, eff_formula)
        self.inspector.clear_parameters()
        self.inspector.update_kernel(eff_name, None)

        if eff_id == "grayscale":
            cb = QComboBox()
            cb.addItems(["Luminosity (0.299R + 0.587G + 0.114B)", "Average ((R+G+B)/3)"])
            self.inspector.params_layout.addWidget(QLabel("Conversion Method:"))
            self.inspector.params_layout.addWidget(cb)

            def update():
                mode = "luminosity" if cb.currentIndex() == 0 else "average"
                self._stage_point_op(
                    lambda im, m=mode: apply_grayscale(im, m),
                    f"Grayscale ({mode})",
                )

            cb.currentIndexChanged.connect(lambda _: update())
            update()

        elif eff_id == "threshold":
            sl = self._add_slider_control(
                "Threshold (T):",
                0,
                255,
                128,
                lambda v: self._stage_point_op(
                    lambda im, t=v: apply_threshold(im, t), f"Binary Threshold (T={v})"
                ),
            )
            self._stage_point_op(
                lambda im, t=sl.value(): apply_threshold(im, t),
                f"Binary Threshold (T={sl.value()})",
            )

        elif eff_id == "negative":
            info = QLabel("Inverts every channel intensity: 255 - pixel.")
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 11px; color: #94a3b8;")
            self.inspector.params_layout.addWidget(info)
            self._stage_point_op(apply_negative, "Image Negative")

        elif eff_id == "brightness_contrast":
            sl_b = self._add_slider_control("Brightness (beta):", -100, 100, 0, lambda _: update())
            sl_c = self._add_slider_control(
                "Contrast (alpha):",
                20,
                250,
                100,
                lambda _: update(),
                formatter=lambda v: f"{v / 100.0:.2f}x",
            )

            def update():
                a = sl_c.value() / 100.0
                b = sl_b.value()
                self._stage_point_op(
                    lambda im, al=a, bt=b: apply_brightness_contrast(im, al, bt),
                    f"Brightness ({b}) & Contrast ({a:.2f}x)",
                )

            update()

        elif eff_id == "saturation":
            sl_s = self._add_slider_control(
                "Saturation Factor (S):",
                0,
                250,
                135,
                lambda v: self._stage_point_op(
                    lambda im, s=v / 100.0: apply_saturation(im, s),
                    f"Saturation ({v / 100.0:.2f}x)",
                ),
                formatter=lambda v: f"{v / 100.0:.2f}x",
            )
            self._stage_point_op(
                lambda im, s=sl_s.value() / 100.0: apply_saturation(im, s),
                f"Saturation ({sl_s.value() / 100.0:.2f}x)",
            )

        elif eff_id == "color_balance":
            sl_t = self._add_slider_control("Temperature (-Cool / +Warm):", -80, 80, 0, lambda _: update())
            sl_r = self._add_slider_control(
                "Red Gain (wR):", 50, 180, 100, lambda _: update(), formatter=lambda v: f"{v/100:.2f}"
            )
            sl_g = self._add_slider_control(
                "Green Gain (wG):", 50, 180, 100, lambda _: update(), formatter=lambda v: f"{v/100:.2f}"
            )
            sl_b = self._add_slider_control(
                "Blue Gain (wB):", 50, 180, 100, lambda _: update(), formatter=lambda v: f"{v/100:.2f}"
            )

            def update():
                t = sl_t.value()
                wr, wg, wb = sl_r.value() / 100.0, sl_g.value() / 100.0, sl_b.value() / 100.0
                self._stage_point_op(
                    lambda im: apply_color_balance(im, t, wr, wg, wb),
                    f"Color Balance (Temp={t}, RGB={wr:.2f}/{wg:.2f}/{wb:.2f})",
                )

            update()

        elif eff_id == "equalization":
            info = QLabel("Equalizes tonal distribution per channel via manual PDF & CDF.")
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 11px; color: #94a3b8;")
            self.inspector.params_layout.addWidget(info)
            self._stage_point_op(apply_histogram_equalization, "Histogram Equalization")

        elif eff_id == "box_blur":
            self._stage_spatial_op(apply_box_blur, "Box Blur 3x3")

        elif eff_id == "gaussian_blur":
            cb = QComboBox()
            cb.addItems(["3x3 Kernel (sigma ~ 0.85)", "5x5 Kernel (sigma ~ 1.4)"])
            self.inspector.params_layout.addWidget(QLabel("Kernel Size:"))
            self.inspector.params_layout.addWidget(cb)

            def update():
                k = 3 if cb.currentIndex() == 0 else 5
                self._stage_spatial_op(lambda im, ks=k: apply_gaussian_blur(im, ks), f"Gaussian {k}x{k}")

            cb.currentIndexChanged.connect(lambda _: update())
            update()

        elif eff_id == "sharpen":
            self._stage_spatial_op(apply_sharpen, "Sharpen 3x3")

        elif eff_id == "edge_detection":
            cb = QComboBox()
            cb.addItems(["Sobel Operator", "Prewitt Operator", "Laplacian (2nd Derivative)", "Canny (4-Stage)"])
            self.inspector.params_layout.addWidget(QLabel("Operator:"))
            self.inspector.params_layout.addWidget(cb)

            sl_low = self._add_slider_control("Canny Low Threshold:", 10, 120, 50, lambda _: update())
            sl_high = self._add_slider_control("Canny High Threshold:", 80, 240, 140, lambda _: update())

            def update():
                methods = ["sobel", "prewitt", "laplacian", "canny"]
                m = methods[cb.currentIndex()]
                is_canny = m == "canny"
                sl_low.setEnabled(is_canny)
                sl_high.setEnabled(is_canny)
                lt, ht = float(sl_low.value()), float(sl_high.value())
                self._stage_spatial_op(
                    lambda im: apply_edge_detection(im, m, lt, ht),
                    f"Edge Detection ({m.upper()})",
                )

            cb.currentIndexChanged.connect(lambda _: update())
            update()

        elif eff_id == "rotate_fixed":
            lbl_info = QLabel("Select Rotation Angle:")
            lbl_info.setStyleSheet("font-size: 11px; color: #cbd5e1;")
            self.inspector.params_layout.addWidget(lbl_info)

            grid = QGridLayout()
            grid.setSpacing(6)
            self._rot_btn_group = QButtonGroup(self)
            self._rot_btn_group.setExclusive(True)

            options = [
                (0, "0 deg (None)", "none", None, ""),
                (1, "90 deg CW", "rot_90", rotate_90, "Rotate 90 deg CW"),
                (2, "180 deg", "rot_180", rotate_180, "Rotate 180 deg"),
                (3, "270 deg CW", "rot_270", rotate_270, "Rotate 270 deg CW"),
            ]

            btn_style = (
                "QPushButton { background: #1e2128; border: 1px solid #343a48; "
                "padding: 7px 6px; border-radius: 5px; text-align: left; font-size: 11px; }"
                "QPushButton:hover { border-color: #fa7516; }"
                "QPushButton:checked { background: #2b211a; border: 1px solid #fa7516; color: #fa7516; }"
            )

            for idx, text, icon_kind, fn, label in options:
                btn = QPushButton(f" {text}")
                btn.setIcon(self._make_transform_icon(icon_kind))
                btn.setCheckable(True)
                btn.setStyleSheet(btn_style)
                if idx == 0:
                    btn.setChecked(True)
                self._rot_btn_group.addButton(btn, idx)
                grid.addWidget(btn, idx // 2, idx % 2)

            def on_rot_chosen(btn_id: int) -> None:
                _, _, _, chosen_fn, chosen_lbl = options[btn_id]
                if chosen_fn is None:
                    self._clear_pending_preview("Select a rotation angle to preview.")
                else:
                    self._stage_point_op(chosen_fn, chosen_lbl)

            self._rot_btn_group.idClicked.connect(on_rot_chosen)
            self.inspector.params_layout.addLayout(grid)
            self._clear_pending_preview("Select a rotation angle, then click 'Apply Effect'.")

        elif eff_id == "rotate_free":
            def on_free_angle(deg: int) -> None:
                if deg == 0:
                    self._clear_pending_preview("Angle at 0 deg. Move slider to preview rotation.")
                else:
                    self._stage_point_op(
                        lambda im, a=deg: rotate_free(im, a), f"Free Rotation ({deg} deg)"
                    )

            self._add_slider_control(
                "Angle (Degrees):",
                -180,
                180,
                0,
                on_free_angle,
                formatter=lambda v: f"{v} deg",
            )
            self._clear_pending_preview("Angle at 0 deg. Adjust slider, then click 'Apply Effect'.")


        elif eff_id == "mirror":
            lbl_info = QLabel("Select Flip Direction:")
            lbl_info.setStyleSheet("font-size: 11px; color: #cbd5e1;")
            self.inspector.params_layout.addWidget(lbl_info)

            col = QVBoxLayout()
            col.setSpacing(6)
            self._flip_btn_group = QButtonGroup(self)
            self._flip_btn_group.setExclusive(True)

            flip_options = [
                (0, "None (Original)", "none", None, ""),
                (1, "Horizontal Flip (Left / Right)", "flip_h", flip_horizontal, "Horizontal Flip"),
                (2, "Vertical Flip (Top / Bottom)", "flip_v", flip_vertical, "Vertical Flip"),
            ]

            btn_style = (
                "QPushButton { background: #1e2128; border: 1px solid #343a48; "
                "padding: 7px 10px; border-radius: 5px; text-align: left; font-size: 11px; }"
                "QPushButton:hover { border-color: #fa7516; }"
                "QPushButton:checked { background: #2b211a; border: 1px solid #fa7516; color: #fa7516; }"
            )

            for idx, text, icon_kind, fn, label in flip_options:
                btn = QPushButton(f"  {text}")
                btn.setIcon(self._make_transform_icon(icon_kind))
                btn.setCheckable(True)
                btn.setStyleSheet(btn_style)
                if idx == 0:
                    btn.setChecked(True)
                self._flip_btn_group.addButton(btn, idx)
                col.addWidget(btn)

            def on_flip_chosen(btn_id: int) -> None:
                _, _, _, chosen_fn, chosen_lbl = flip_options[btn_id]
                if chosen_fn is None:
                    self._clear_pending_preview("Select Horizontal or Vertical Flip to preview.")
                else:
                    self._stage_point_op(chosen_fn, chosen_lbl)

            self._flip_btn_group.idClicked.connect(on_flip_chosen)
            self.inspector.params_layout.addLayout(col)
            self._clear_pending_preview("Select a flip direction, then click 'Apply Effect'.")

        elif eff_id == "resize":
            cb = QComboBox()
            cb.addItems(["Bilinear Interpolation", "Nearest Neighbor"])
            self.inspector.params_layout.addWidget(QLabel("Interpolation:"))
            self.inspector.params_layout.addWidget(cb)
            sl = self._add_slider_control("Scale Percentage:", 25, 200, 75, lambda _: update(), lambda v: f"{v}%")

            def update():
                m = "bilinear" if cb.currentIndex() == 0 else "nearest"
                sc = sl.value()
                self._stage_point_op(
                    lambda im: resize_manual(im, sc, m), f"Resize {sc}% ({m})"
                )

            cb.currentIndexChanged.connect(lambda _: update())
            update()

        elif eff_id == "crop":
            info = QLabel(
                "Drag directly over the AFTER monitor to define the crop region, then click 'Apply Effect'."
            )
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 11px; color: #38bdf8;")
            self.inspector.params_layout.addWidget(info)
            self.canvas.set_crop_mode(True)
            self.staged_image = None
            self.staged_effect_label = "Manual Crop"
            self.inspector.set_status_info("Crop overlay active on AFTER monitor.")

        elif eff_id == "pencil_sketch":
            self._stage_point_op(apply_pencil_sketch, "Pencil Sketch")

        elif eff_id == "vignette":
            sl = self._add_slider_control(
                "Falloff Strength:",
                10,
                150,
                75,
                lambda v: self._stage_point_op(
                    lambda im, s=v / 100.0: apply_vignette(im, s), f"Vignette ({v/100:.2f})"
                ),
                formatter=lambda v: f"{v / 100.0:.2f}",
            )
            self._stage_point_op(
                lambda im, s=sl.value() / 100.0: apply_vignette(im, s),
                f"Vignette ({sl.value() / 100.0:.2f})",
            )

        elif eff_id == "pixelate":
            sl = self._add_slider_control(
                "Block Size (px):",
                4,
                36,
                12,
                lambda v: self._stage_point_op(
                    lambda im, k=v: apply_pixelate(im, k), f"Pixelation ({v}x{v} px)"
                ),
                formatter=lambda v: f"{v} px",
            )
            self._stage_point_op(
                lambda im, k=sl.value(): apply_pixelate(im, k),
                f"Pixelation ({sl.value()}x{sl.value()} px)",
            )

        elif eff_id == "noise_gen":
            cb = QComboBox()
            cb.addItems(["Salt & Pepper Noise", "Gaussian Noise"])
            self.inspector.params_layout.addWidget(QLabel("Noise Distribution:"))
            self.inspector.params_layout.addWidget(cb)
            sl = self._add_slider_control("Intensity:", 1, 50, 10, lambda _: update())

            def update():
                val = sl.value()
                if cb.currentIndex() == 0:
                    amt = val / 200.0
                    self._stage_point_op(
                        lambda im, a=amt: add_salt_pepper_noise(im, a),
                        f"Salt & Pepper Noise ({amt * 100:.1f}%)",
                    )
                else:
                    sig = float(val * 1.5)
                    self._stage_point_op(
                        lambda im, s=sig: add_gaussian_noise(im, s),
                        f"Gaussian Noise (sigma={sig:.1f})",
                    )

            cb.currentIndexChanged.connect(lambda _: update())
            update()

        elif eff_id == "median_denoise":
            cb = QComboBox()
            cb.addItems(["3x3 Window", "5x5 Window"])
            self.inspector.params_layout.addWidget(QLabel("Neighborhood Window:"))
            self.inspector.params_layout.addWidget(cb)

            def update():
                k = 3 if cb.currentIndex() == 0 else 5
                self._stage_point_op(
                    lambda im, ks=k: apply_median_filter(im, ks), f"Median Filter ({k}x{k})"
                )

            cb.currentIndexChanged.connect(lambda _: update())
            update()

        elif eff_id == "hsv_bokeh":
            sl_h = self._add_slider_control(
                "Target Hue (0-359 deg):", 0, 359, 0, lambda _: update(), lambda v: f"{v} deg"
            )
            sl_tol = self._add_slider_control(
                "Hue Tolerance:", 5, 60, 25, lambda _: update(), lambda v: f"+/- {v} deg"
            )

            def update():
                h, tol = sl_h.value(), sl_tol.value()
                self._stage_point_op(
                    lambda im: apply_hsv_bokeh(im, target_hue=h, hue_tolerance=tol),
                    f"HSV Bokeh (Hue={h} deg, Tol={tol})",
                )

            update()

    # --- STAGING PREVIEW & COMMIT KE HISTORY ---
    def _stage_point_op(self, fn: Callable[[np.ndarray], np.ndarray], label: str) -> None:
        self.active_live_fn = fn
        self.staged_effect_label = label

        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_filter(fn)
            self.inspector.set_status_info(f"Live camera filter set: {label}")
            return

        if self.history.current_image is None:
            self.inspector.set_status_info("Load an image first to preview effects.")
            return

        self.staged_image = fn(self.history.current_image)
        self.canvas.set_images(self.history.original_image, self.staged_image, is_preview=True)
        self.hist_after.update_histogram(self.staged_image, ghost_image=self.history.original_image)
        self.inspector.set_status_info(f"Previewing: {label}. Click 'Apply Effect' to commit.")

    def _stage_spatial_op(
        self,
        fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        label: str,
    ) -> None:
        self.active_live_fn = lambda im: fn(im)[0]
        self.staged_effect_label = label

        _, sample_kernel = fn(np.zeros((8, 8, 3), dtype=np.uint8))
        self.inspector.update_kernel(label, sample_kernel)

        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_filter(self.active_live_fn)
            self.inspector.set_status_info(f"Live camera filter set: {label}")
            return

        if self.history.current_image is None:
            self.inspector.set_status_info("Load an image first to preview convolution.")
            return

        result, kernel = fn(self.history.current_image)
        self.staged_image = result
        self.inspector.update_kernel(label, kernel)
        self.canvas.set_images(self.history.original_image, self.staged_image, is_preview=True)
        self.hist_after.update_histogram(self.staged_image, ghost_image=self.history.original_image)
        self.inspector.set_status_info(f"Previewing: {label}. Click 'Apply Effect' to commit.")

    def commit_staged_effect(self) -> None:
        if self.history.current_image is None:
            QMessageBox.information(self, "No Image", "Open an image or capture a frame first.")
            return

        if self.selected_effect_id == "crop":
            coords = self.canvas.get_crop_coordinates()
            if coords is None:
                QMessageBox.warning(
                    self, "Crop Selection", "Drag a selection box over the AFTER monitor first."
                )
                return
            top, bottom, left, right = coords
            crop_label = f"Manual Crop [{top}:{bottom}, {left}:{right}]"
            cropped = crop_manual(self.history.current_image, top, bottom, left, right)
            self.history.push_state(cropped, crop_label)
            self.canvas.set_crop_mode(False)
            self.staged_image = None
            self._refresh_committed_views(f"Applied: {crop_label}")
            return

        if self.staged_image is None:
            QMessageBox.information(
                self, "No Pending Preview", "Select an effect from the catalog first."
            )
            return

        applied_name = self.staged_effect_label
        self.history.push_state(self.staged_image, applied_name)
        self.staged_image = None
        self._refresh_committed_views(
            f"Applied: {applied_name} (Step {len(self.history.undo_stack)})"
        )

    def discard_staged_preview(self) -> None:
        self.staged_image = None
        self.canvas.set_crop_mode(False)
        if self.history.current_image is not None:
            self._refresh_committed_views("Preview cancelled. Showing last committed state.")

    def _sync_history_buttons(self) -> None:
        undo_count = len(self.history.undo_stack)
        redo_count = len(self.history.redo_stack)
        has_image = self.history.original_image is not None

        for btn in (self.top_btn_undo, self.inspector.btn_undo):
            btn.setText(f"Undo ({undo_count})")
            btn.setEnabled(undo_count > 0)

        for btn in (self.top_btn_redo, self.inspector.btn_redo):
            btn.setText(f"Redo ({redo_count})")
            btn.setEnabled(redo_count > 0)

        self.top_btn_reset.setEnabled(has_image and undo_count > 0)
        self.inspector.btn_reset.setEnabled(has_image and undo_count > 0)

        entries, active_idx = self.history.get_history_entries()
        self.inspector.update_history_list(entries, active_idx)

    def _refresh_committed_views(self, status_msg: str | None = None) -> None:
        orig = self.history.original_image
        curr = self.history.current_image
        self.canvas.set_images(orig, curr, is_preview=False)
        self.hist_before.update_histogram(orig)
        self.hist_after.update_histogram(curr, ghost_image=orig)
        self._sync_history_buttons()
        if status_msg:
            self.inspector.set_status_info(status_msg)

    def undo_action(self) -> None:
        self.staged_image = None
        self.canvas.set_crop_mode(False)
        if self.history.undo() is not None:
            self._refresh_committed_views(
                f"Undo performed (Active: {self.history.current_label})"
            )

    def redo_action(self) -> None:
        self.staged_image = None
        self.canvas.set_crop_mode(False)
        if self.history.redo() is not None:
            self._refresh_committed_views(
                f"Redo performed (Active: {self.history.current_label})"
            )

    def reset_action(self) -> None:
        self.staged_image = None
        self.canvas.set_crop_mode(False)
        if self.history.reset() is not None:
            self._refresh_committed_views("Image reset to original state.")

    def _toggle_split_view(self, checked: bool) -> None:
        self.canvas.set_split_mode(checked)

    # --- FILE & CAMERA HANDLERS ---
    def open_image_dialog(self) -> None:
        self.stop_camera_if_running()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.critical(self, "Load Error", "Unable to read the selected image file.")
            return
        rgb = bgr[:, :, ::-1].copy()
        self.staged_image = None
        self.history.load_image(rgb, f"Original Image ({rgb.shape[1]}x{rgb.shape[0]})")
        self._refresh_committed_views(f"Loaded image ({rgb.shape[1]}x{rgb.shape[0]}).")

    def save_image_dialog(self) -> None:
        target = self.staged_image if self.staged_image is not None else self.history.current_image
        if target is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "sego_pecel_output.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        cv2.imwrite(path, target[:, :, ::-1])
        QMessageBox.information(self, "Export Complete", f"Saved image to:\n{path}")

    def toggle_camera(self) -> None:
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera_if_running()
            return

        self.camera_thread = CameraThread(0)
        self.camera_thread.set_filter(self.active_live_fn)
        self.camera_thread.frame_ready.connect(self._on_camera_frame)
        self.camera_thread.camera_error.connect(
            lambda msg: QMessageBox.warning(self, "Camera Error", msg)
        )
        self.camera_thread.start()

        self.btn_cam.setText("Stop Camera")
        self.btn_capture.setEnabled(True)
        self.btn_live_fx.setEnabled(True)

    def stop_camera_if_running(self) -> None:
        if self.camera_thread is not None:
            try:
                self.camera_thread.frame_ready.disconnect(self._on_camera_frame)
            except TypeError:
                pass
            if self.camera_thread.isRunning():
                self.camera_thread.stop()
        self.camera_thread = None
        self.canvas.live_fps = None
        self.btn_cam.setText("Start Camera")
        self.btn_capture.setEnabled(False)
        self.btn_live_fx.setEnabled(False)
        self.btn_live_fx.setText("Live Filter: OFF")

    def toggle_live_filter(self) -> None:
        if not self.camera_thread:
            return
        state = not self.camera_thread.live_filter_enabled
        self.camera_thread.live_filter_enabled = state
        self.btn_live_fx.setText(f"Live Filter: {'ON' if state else 'OFF'}")

    def capture_camera_frame(self) -> None:
        if self.camera_thread and self.camera_thread.latest_raw_rgb is not None:
            captured = self.camera_thread.latest_raw_rgb.copy()
            self.stop_camera_if_running()
            self.staged_image = None
            self.history.load_image(captured, f"Captured Frame ({captured.shape[1]}x{captured.shape[0]})")
            self._refresh_committed_views("Captured frame from camera.")

    def _on_camera_frame(
        self, before_rgb: np.ndarray, after_rgb: np.ndarray, fps: float
    ) -> None:
        if self.camera_thread is None or not self.camera_thread.isRunning():
            return

        self.canvas.live_fps = fps
        self.canvas.set_images(
            before_rgb,
            after_rgb,
            is_preview=self.camera_thread.live_filter_enabled,
        )
        self.hist_before.update_histogram(before_rgb)
        self.hist_after.update_histogram(after_rgb, ghost_image=before_rgb)

    def closeEvent(self, event) -> None:
        self.stop_camera_if_running()
        super().closeEvent(event)