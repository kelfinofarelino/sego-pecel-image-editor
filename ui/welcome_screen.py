import os
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

TEAM_MEMBERS = [
    {"nama": "Muhammad Farelino Kelfin Ramadhani", "nim": "123240205", "foto": "assets/anggota_1.jpg"},
    {"nama": "Naufal Alwi Rizqillah", "nim": "123240215", "foto": "assets/anggota_2.jpg"},
    {"nama": "Muhammad Fariz Syahada", "nim": "123240158", "foto": "assets/anggota_3.jpg"},
    {"nama": "Muhamamd Wildan Hanif", "nim": "123230215", "foto": "assets/anggota_4.jpg"},
]


def _get_gta_font(pixel_size: int, fallback_bold: bool = True) -> QFont:
    font_path = os.path.join("assets", "pricedown.ttf")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            f = QFont(families[0])
            f.setPixelSize(pixel_size)
            return f

    f = QFont("Impact")
    if not f.exactMatch():
        f = QFont("Arial Black")
    f.setPixelSize(pixel_size)
    f.setBold(fallback_bold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -1.0)
    return f


class OutlinedTextBanner(QWidget):
    def __init__(
        self,
        text: str,
        font: QFont,
        stroke_width: float = 5.0,
        fixed_height: int = 72,
    ):
        super().__init__()
        self.text = text
        self.custom_font = font
        self.stroke_width = stroke_width
        self.setFixedHeight(fixed_height)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.custom_font)

        metrics = painter.fontMetrics()
        lines = self.text.split("\n")
        line_h = metrics.height()
        total_h = line_h * len(lines)
        start_y = (self.height() - total_h) / 2.0 + metrics.ascent()

        path = QPainterPath()
        for idx, line in enumerate(lines):
            text_w = metrics.horizontalAdvance(line)
            x = (self.width() - text_w) / 2.0
            y = start_y + idx * line_h
            path.addText(x, y, self.custom_font, line)

        outline_pen = QPen(
            QColor("#000000"),
            self.stroke_width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.strokePath(path, outline_pen)
        painter.fillPath(path, QColor("#FFFFFF"))
        painter.end()


class OutlinedActionButton(QPushButton):
    def __init__(self, text: str, font: QFont):
        super().__init__(text)
        self.custom_font = font
        self._hovered = False
        self.setFixedSize(260, 56)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        grad = QLinearGradient(0, 0, self.width(), 0)
        if self._hovered:
            grad.setColorAt(0.0, QColor("#00d2ff"))
            grad.setColorAt(1.0, QColor("#ff2a85"))
        else:
            grad.setColorAt(0.0, QColor("#ff2a85"))
            grad.setColorAt(0.55, QColor("#ff7b00"))
            grad.setColorAt(1.0, QColor("#f59e0b"))

        bg_path = QPainterPath()
        bg_path.addRoundedRect(
            float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()), 26.0, 26.0
        )
        painter.fillPath(bg_path, grad)
        painter.setPen(QPen(QColor("#000000"), 3.0))
        painter.drawPath(bg_path)

        painter.setFont(self.custom_font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(self.text())
        tx = (self.width() - tw) / 2.0
        ty = (self.height() - metrics.height()) / 2.0 + metrics.ascent()

        text_path = QPainterPath()
        text_path.addText(tx, ty, self.custom_font, self.text())
        painter.strokePath(
            text_path,
            QPen(
                QColor("#000000"),
                4.5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            ),
        )
        painter.fillPath(text_path, QColor("#FFFFFF"))
        painter.end()


class MemberCard(QFrame):
    def __init__(self, member: dict, index: int):
        super().__init__()
        self.setFixedSize(230, 305)
        accent_colors = ["#ff2a85", "#00e5ff", "#ff8c00", "#b84dff"]
        accent = accent_colors[index % len(accent_colors)]

        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: rgba(15, 11, 28, 225);
                border: 2px solid {accent};
                border-radius: 12px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        photo_lbl = QLabel()
        photo_lbl.setFixedSize(198, 192)
        photo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        photo_lbl.setStyleSheet(
            f"background-color: #1b1430; border-radius: 8px; border: 1px solid {accent};"
        )

        if os.path.exists(member["foto"]):
            pix = QPixmap(member["foto"]).scaled(
                198,
                192,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            photo_lbl.setPixmap(pix)
        else:
            photo_lbl.setText(f"MEMBER {index + 1}\n{member['foto']}")
            photo_lbl.setStyleSheet(
                f"color: {accent}; font-weight: 700; font-size: 11px; "
                f"background-color: #1b1430; border-radius: 8px;"
            )

        name_lbl = QLabel(member["nama"])
        name_lbl.setWordWrap(True)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: 700;")

        nim_lbl = QLabel(f"NIM: {member['nim']}")
        nim_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nim_lbl.setStyleSheet(f"color: {accent}; font-size: 12px; font-weight: 700;")

        layout.addWidget(photo_lbl)
        layout.addWidget(name_lbl)
        layout.addWidget(nim_lbl)


class WelcomeScreen(QWidget):
    start_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Sego Pecel Image Editor - Studi Eksplorasi Grayscale dan Optik untuk Pengolahan Citra dan Elemen piksel"
        )
        self.resize(1160, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 36, 36, 36)
        root.setSpacing(10)

        title_banner = OutlinedTextBanner(
            "SEGO PECEL IMAGE EDITOR",
            font=_get_gta_font(56),
            stroke_width=6.5,
            fixed_height=76,
        )

        sub_font = QFont("Segoe UI", 13)
        sub_font.setBold(True)
        subtitle_banner = OutlinedTextBanner(
            "Studi Eksplorasi Grayscale dan Optik untuk Pengolahan Citra dan Elemen piksel",
            font=sub_font,
            stroke_width=3.2,
            fixed_height=34,
        )

        cards_row = QHBoxLayout()
        cards_row.setSpacing(20)
        cards_row.addStretch()
        for idx, m in enumerate(TEAM_MEMBERS):
            cards_row.addWidget(MemberCard(m, idx))
        cards_row.addStretch()

        start_btn = OutlinedActionButton("MULAI", font=_get_gta_font(28))
        start_btn.clicked.connect(self.start_clicked.emit)

        btn_wrap = QHBoxLayout()
        btn_wrap.addStretch()
        btn_wrap.addWidget(start_btn)
        btn_wrap.addStretch()

        root.addStretch(1)
        root.addWidget(title_banner)
        root.addWidget(subtitle_banner)
        root.addSpacing(18)
        root.addLayout(cards_row)
        root.addSpacing(22)
        root.addLayout(btn_wrap)
        root.addStretch(1)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QColor("#090514"))
        grad.setColorAt(0.45, QColor("#270d42"))
        grad.setColorAt(0.80, QColor("#66154c"))
        grad.setColorAt(1.0, QColor("#b8432e"))
        painter.fillRect(self.rect(), grad)

        pen = QPen(QColor(0, 240, 255, 38), 1)
        painter.setPen(pen)
        for y in range(self.height() - 130, self.height(), 22):
            painter.drawLine(0, y, self.width(), y)
        painter.end()