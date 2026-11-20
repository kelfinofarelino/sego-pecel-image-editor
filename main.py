import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.welcome_screen import WelcomeScreen


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(
        "Sego Pecel Image Editor - Studi Eksplorasi Grayscale dan Optik untuk Pengolahan Citra dan Elemen piksel"
    )
    welcome = WelcomeScreen()
    editor = MainWindow()

    def launch_editor() -> None:
        welcome.close()
        editor.show()

    welcome.start_clicked.connect(launch_editor)
    welcome.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()