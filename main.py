"""Audio Studio AI - Main Application Entry Point.
Next-Generation AI-Powered Music Production & Audio Workstation.
Supports Windows 11 Desktop Standalone Application packaging,
multiprocessing freeze support, and error boundary handling.
"""
import sys
import os
import multiprocessing
import traceback
from pathlib import Path

# Multiprocessing freeze support must be called immediately for Windows PyInstaller
if __name__ == "__main__":
    multiprocessing.freeze_support()

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont

from config import APP_NAME, APP_VERSION, APP_ORGANIZATION, ASSETS_DIR
from ui.main_window import MainWindow

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to capture unhandled crashes."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"[FATAL ERROR] {err_msg}", file=sys.stderr)

    # If QApplication is running, show warning dialog
    if QApplication.instance() and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        QMessageBox.critical(
            None,
            f"{APP_NAME} - Unexpected Error",
            f"An unexpected error occurred:\n\n{exc_value}\n\nSee console log for full stack trace."
        )

def main():
    sys.excepthook = handle_exception

    # Qt6 enables High-DPI scaling automatically by default
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORGANIZATION)

    # Application Icon
    ico_path = ASSETS_DIR / "icon.ico"
    png_path = ASSETS_DIR / "icon.png"
    if ico_path.exists():
        app.setWindowIcon(QIcon(str(ico_path)))
    elif png_path.exists():
        app.setWindowIcon(QIcon(str(png_path)))

    # Set default clean typography with explicit valid point size
    font = QFont("Segoe UI", 10)
    font.setPointSize(10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # Launch Main Window
    window = MainWindow()
    if ico_path.exists():
        window.setWindowIcon(QIcon(str(ico_path)))
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
