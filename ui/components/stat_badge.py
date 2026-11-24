"""Stat badge / Metric counter component for dashboard and library.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

from ui.themes.theme_manager import ThemeManager

class StatBadge(QWidget):
    def __init__(self, value: str, label: str, icon_text: str = "", parent=None):
        super().__init__(parent)
        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        self.label_title = QLabel(f"{icon_text}  {label}".strip() if icon_text else label)
        self.label_title.setObjectName("StatLabel")

        self.label_value = QLabel(value)
        self.label_value.setObjectName("StatValue")

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_value)
        self._update_style()

    def set_value(self, val: str):
        self.label_value.setText(val)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = c.get("background_card", "#1B1D21")
        border = c.get("border_default", "#30333A")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")

        self.setStyleSheet(f"""
            StatBadge {{
                background-color: {bg_card};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QLabel#StatLabel {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            QLabel#StatValue {{
                color: {text_primary};
                font-size: 24px;
                font-weight: 700;
            }}
        """)
