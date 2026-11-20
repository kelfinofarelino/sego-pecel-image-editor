import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class InspectorPanel(QWidget):
    apply_clicked = pyqtSignal()
    discard_preview_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(315)
        self.setStyleSheet(
            """
            QWidget { background-color: #14161b; color: #e2e8f0; }
            QLabel { background: transparent; }
            QFrame#Card {
                background-color: #1a1d24;
                border: 1px solid #282d38;
                border-radius: 6px;
            }
            QLabel#SectionTitle {
                color: #94a3b8;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 1. PIXEL INSPECTOR
        pix_card = QFrame()
        pix_card.setObjectName("Card")
        pix_lay = QVBoxLayout(pix_card)
        pix_lay.setContentsMargins(10, 8, 10, 10)
        pix_lay.setSpacing(4)

        t_pix = QLabel("PIXEL INSPECTOR")
        t_pix.setObjectName("SectionTitle")
        self.coord_lbl = QLabel("Coordinates (x, y): (-, -)")
        self.orig_rgb_lbl = QLabel("Before (R, G, B): (-, -, -)")
        self.proc_rgb_lbl = QLabel("After  (R', G', B'): (-, -, -)")
        for lbl in (self.coord_lbl, self.orig_rgb_lbl, self.proc_rgb_lbl):
            lbl.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")

        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(6)
        self.swatch_before = QLabel("Before")
        self.swatch_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.swatch_before.setFixedHeight(20)
        self.swatch_before.setStyleSheet(
            "background: #252932; border-radius: 4px; font-size: 10px; font-weight: 600;"
        )

        self.swatch_after = QLabel("After")
        self.swatch_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.swatch_after.setFixedHeight(20)
        self.swatch_after.setStyleSheet(
            "background: #252932; border-radius: 4px; font-size: 10px; font-weight: 600;"
        )
        swatch_row.addWidget(self.swatch_before)
        swatch_row.addWidget(self.swatch_after)

        pix_lay.addWidget(t_pix)
        pix_lay.addWidget(self.coord_lbl)
        pix_lay.addWidget(self.orig_rgb_lbl)
        pix_lay.addWidget(self.proc_rgb_lbl)
        pix_lay.addLayout(swatch_row)

        # 2. CONVOLUTION KERNEL MATRIX
        ker_card = QFrame()
        ker_card.setObjectName("Card")
        ker_lay = QVBoxLayout(ker_card)
        ker_lay.setContentsMargins(10, 8, 10, 10)
        ker_lay.setSpacing(4)

        t_ker = QLabel("CONVOLUTION KERNEL MATRIX")
        t_ker.setObjectName("SectionTitle")
        self.kernel_desc = QLabel("No spatial convolution active.")
        self.kernel_desc.setWordWrap(True)
        self.kernel_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")

        self.kernel_grid_widget = QWidget()
        self.kernel_grid = QGridLayout(self.kernel_grid_widget)
        self.kernel_grid.setContentsMargins(0, 0, 0, 0)
        self.kernel_grid.setSpacing(3)

        ker_lay.addWidget(t_ker)
        ker_lay.addWidget(self.kernel_desc)
        ker_lay.addWidget(self.kernel_grid_widget)

        # 3. HISTORY & PIPELINE STATUS (Dengan Applied Effects Log)
        hist_card = QFrame()
        hist_card.setObjectName("Card")
        hist_lay = QVBoxLayout(hist_card)
        hist_lay.setContentsMargins(10, 8, 10, 10)
        hist_lay.setSpacing(6)

        t_hist = QLabel("HISTORY & APPLIED PIPELINE")
        t_hist.setObjectName("SectionTitle")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_undo = QPushButton("Undo (0)")
        self.btn_undo.setToolTip("Undo last applied effect (Ctrl+Z)")
        self.btn_redo = QPushButton("Redo (0)")
        self.btn_redo.setToolTip("Redo undone effect (Ctrl+Y)")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Restore original loaded image")
        for b in (self.btn_undo, self.btn_redo, self.btn_reset):
            b.setEnabled(False)
            b.setStyleSheet(
                "QPushButton { background: #242832; border: 1px solid #343a48; "
                "padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }"
                "QPushButton:hover:enabled { border-color: #fa7516; }"
                "QPushButton:disabled { color: #525b6e; border-color: #222630; }"
            )
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_redo)
        btn_row.addWidget(self.btn_reset)

        self.history_list = QListWidget()
        self.history_list.setFixedHeight(96)
        self.history_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.history_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.history_list.setStyleSheet(
            """
            QListWidget {
                background-color: #121419;
                border: 1px solid #252a34;
                border-radius: 4px;
                padding: 2px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 3px 5px;
                border-radius: 2px;
            }
            """
        )

        self.status_lbl = QLabel("Ready. Select an effect from the catalog.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")

        hist_lay.addWidget(t_hist)
        hist_lay.addLayout(btn_row)
        hist_lay.addWidget(self.history_list)
        hist_lay.addWidget(self.status_lbl)

        # 4. EFFECT CONTROL PANEL
        ctrl_card = QFrame()
        ctrl_card.setObjectName("Card")
        ctrl_lay = QVBoxLayout(ctrl_card)
        ctrl_lay.setContentsMargins(10, 8, 10, 10)
        ctrl_lay.setSpacing(6)

        t_ctrl = QLabel("EFFECT CONTROL PANEL")
        t_ctrl.setObjectName("SectionTitle")

        self.selected_effect_lbl = QLabel("No Effect Selected")
        self.selected_effect_lbl.setStyleSheet(
            "color: #fa7516; font-size: 13px; font-weight: 700;"
        )

        self.formula_lbl = QLabel("Select an operation from the left list to adjust parameters.")
        self.formula_lbl.setWordWrap(True)
        self.formula_lbl.setStyleSheet(
            "color: #94a3b8; font-family: 'Consolas', monospace; font-size: 10px;"
        )

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")

        self.params_container = QWidget()
        self.params_container.setStyleSheet("background: transparent;")
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 6, 4, 4)
        self.params_layout.setSpacing(6)
        self.params_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(self.params_container)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.btn_discard = QPushButton("Cancel Preview")
        self.btn_discard.setStyleSheet(
            "QPushButton { background: #242832; border: 1px solid #343a48; "
            "padding: 7px; border-radius: 4px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { border-color: #94a3b8; }"
        )
        self.btn_discard.clicked.connect(self.discard_preview_clicked.emit)

        self.btn_apply = QPushButton("Apply Effect")
        self.btn_apply.setStyleSheet(
            "QPushButton { background: #fa7516; color: #111215; border: none; "
            "padding: 7px; border-radius: 4px; font-weight: 700; font-size: 11px; }"
            "QPushButton:hover { background: #ff8933; }"
        )
        self.btn_apply.clicked.connect(self.apply_clicked.emit)

        action_row.addWidget(self.btn_discard)
        action_row.addWidget(self.btn_apply)

        ctrl_lay.addWidget(t_ctrl)
        ctrl_lay.addWidget(self.selected_effect_lbl)
        ctrl_lay.addWidget(self.formula_lbl)
        ctrl_lay.addWidget(scroll_area, stretch=1)
        ctrl_lay.addLayout(action_row)

        layout.addWidget(pix_card)
        layout.addWidget(ker_card)
        layout.addWidget(hist_card)
        layout.addWidget(ctrl_card, stretch=1)

    def update_history_list(self, entries: list[str], active_index: int) -> None:
        self.history_list.clear()
        active_item: QListWidgetItem | None = None

        for idx, label in enumerate(entries):
            prefix = ">" if idx == active_index else " "
            item = QListWidgetItem(f"{prefix} {idx:02d}. {label}")
            font = QFont("Consolas", 9)

            if idx == active_index:
                font.setBold(True)
                item.setForeground(QColor("#fa7516"))
                item.setBackground(QColor("#1e232e"))
                active_item = item
            elif idx < active_index:
                item.setForeground(QColor("#cbd5e1"))
            else:
                # Langkah yang sudah di-Undo (berada di redo_stack)
                font.setItalic(True)
                item.setForeground(QColor("#525b6e"))

            item.setFont(font)
            self.history_list.addItem(item)

        if active_item is not None:
            self.history_list.scrollToItem(active_item)

    def set_control_header(self, effect_name: str, formula_text: str) -> None:
        self.selected_effect_lbl.setText(effect_name)
        self.formula_lbl.setText(formula_text)

    def clear_parameters(self) -> None:
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_sublayout(item.layout())

    def _clear_sublayout(self, sublayout) -> None:
        while sublayout.count():
            child = sublayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_sublayout(child.layout())

    def update_pixel(self, x: int, y: int, orig: tuple, proc: tuple) -> None:
        self.coord_lbl.setText(f"Coordinates (x, y): ({x}, {y})")
        self.orig_rgb_lbl.setText(f"Before (R, G, B): {orig}")
        self.proc_rgb_lbl.setText(f"After  (R', G', B'): {proc}")
        self.swatch_before.setStyleSheet(
            f"background-color: rgb{orig}; color: #fff; border-radius: 4px; font-size: 10px; font-weight: 600;"
        )
        self.swatch_after.setStyleSheet(
            f"background-color: rgb{proc}; color: #fff; border-radius: 4px; font-size: 10px; font-weight: 600;"
        )

    def update_kernel(self, name: str, kernel: np.ndarray | None) -> None:
        while self.kernel_grid.count():
            item = self.kernel_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if kernel is None:
            self.kernel_desc.setText("Point or geometric operation (no 2D kernel).")
            return

        self.kernel_desc.setText(f"{name} ({kernel.shape[0]}x{kernel.shape[1]}):")
        for r in range(kernel.shape[0]):
            for c in range(kernel.shape[1]):
                val = kernel[r, c]
                cell = QLabel(f"{val:.3f}" if abs(val) < 1 and val != 0 else f"{val:.1f}")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setStyleSheet(
                    "background: #111318; border: 1px solid #334155; "
                    "border-radius: 3px; padding: 2px; font-family: 'Consolas', monospace; "
                    "font-size: 10px; color: #38bdf8;"
                )
                self.kernel_grid.addWidget(cell, r, c)

    def set_status_info(self, text: str) -> None:
        self.status_lbl.setText(text)