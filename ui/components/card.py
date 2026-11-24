"""Premium Card component with Obsidian Silver styling, rounded corners,
subtle elevation and hover interactions.
"""
from typing import Optional
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent

from ui.themes.theme_manager import ThemeManager

class PremiumCard(QFrame):
    clicked = Signal()

    def __init__(
        self,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        elevated: bool = False,
        clickable: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self._elevated = elevated
        self._clickable = clickable
        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())

        if self._clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(14)

        # Header area if title given
        if title:
            header_widget = QWidget()
            header_layout = QVBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)

            self.title_label = QLabel(title)
            self.title_label.setObjectName("CardTitle")
            header_layout.addWidget(self.title_label)

            if subtitle:
                self.subtitle_label = QLabel(subtitle)
                self.subtitle_label.setObjectName("CardSubtitle")
                header_layout.addWidget(self.subtitle_label)

            self.main_layout.addWidget(header_widget)

        self._update_style()

    def add_widget(self, widget: QWidget):
        self.main_layout.addWidget(widget)

    def add_layout(self, layout):
        self.main_layout.addLayout(layout)

    def mousePressEvent(self, event: QMouseEvent):
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = c.get("background_elevated", "#22252A") if self._elevated else c.get("background_card", "#1B1D21")
        border_col = c.get("border_default", "#30333A")
        border_hover = c.get("accent_silver", "#C5C9D0")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        radius = 12

        hover_style = f"""
            QFrame:hover {{
                border: 1px solid {border_hover};
            }}
        """ if self._clickable else ""

        self.setStyleSheet(f"""
            PremiumCard {{
                background-color: {bg_card};
                border: 1px solid {border_col};
                border-radius: {radius}px;
            }}
            {hover_style}
            QLabel#CardTitle {{
                color: {text_primary};
                font-size: 16px;
                font-weight: 600;
            }}
            QLabel#CardSubtitle {{
                color: {text_secondary};
                font-size: 13px;
                font-weight: 400;
            }}
        """)
