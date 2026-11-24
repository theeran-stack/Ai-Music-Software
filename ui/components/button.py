"""Premium Button Components for Audio Studio AI.
Supports Primary (Silver), Secondary, Outline, Ghost, and Danger variants with smooth hover and active feedback.
"""
from typing import Optional
from PySide6.QtWidgets import QPushButton, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QCursor, QColor

from ui.themes.theme_manager import ThemeManager

class PremiumButton(QPushButton):
    def __init__(
        self,
        text: str = "",
        variant: str = "primary",  # primary, secondary, outline, ghost, danger
        icon=None,
        parent=None
    ):
        super().__init__(text, parent)
        self._variant = variant
        self._is_loading = False
        self._original_text = text
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(38)

        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(18, 18))

        # Connect theme changes
        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def set_loading(self, loading: bool, message: str = "Processing..."):
        self._is_loading = loading
        self.setEnabled(not loading)
        if loading:
            self.setText(message)
        else:
            self.setText(self._original_text)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = c.get("background_card", "#1B1D21")
        bg_elevated = c.get("background_elevated", "#22252A")
        bg_hover = c.get("background_hover", "#26292F")
        border_default = c.get("border_default", "#30333A")
        border_subtle = c.get("border_subtle", "#25282F")
        accent_silver = c.get("accent_silver", "#C5C9D0")
        accent_hover = c.get("accent_hover", "#D8DCE3")
        accent_active = c.get("accent_active", "#B0B4BC")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        danger = c.get("error", "#E16D75")
        main_bg = c.get("background_main", "#101114")

        if self._variant == "primary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {accent_silver};
                    color: {main_bg};
                    font-weight: 600;
                    font-size: 13px;
                    border: 1px solid {accent_silver};
                    border-radius: 8px;
                    padding: 8px 18px;
                }}
                QPushButton:hover {{
                    background-color: {accent_hover};
                    border-color: {accent_hover};
                }}
                QPushButton:pressed {{
                    background-color: {accent_active};
                    border-color: {accent_active};
                }}
                QPushButton:disabled {{
                    background-color: {border_default};
                    color: {text_secondary};
                    border-color: {border_default};
                }}
            """)
        elif self._variant == "secondary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_elevated};
                    color: {text_primary};
                    font-weight: 500;
                    font-size: 13px;
                    border: 1px solid {border_default};
                    border-radius: 8px;
                    padding: 8px 18px;
                }}
                QPushButton:hover {{
                    background-color: {bg_hover};
                    border-color: {accent_silver};
                }}
                QPushButton:pressed {{
                    background-color: {bg_card};
                }}
                QPushButton:disabled {{
                    background-color: {bg_card};
                    color: {border_default};
                    border-color: {border_subtle};
                }}
            """)
        elif self._variant == "outline":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {accent_silver};
                    font-weight: 500;
                    font-size: 13px;
                    border: 1px solid {border_default};
                    border-radius: 8px;
                    padding: 8px 18px;
                }}
                QPushButton:hover {{
                    border-color: {accent_silver};
                    background-color: rgba(197, 201, 208, 0.08);
                    color: {accent_hover};
                }}
                QPushButton:pressed {{
                    background-color: rgba(197, 201, 208, 0.15);
                }}
                QPushButton:disabled {{
                    color: {border_default};
                    border-color: {border_subtle};
                }}
            """)
        elif self._variant == "ghost":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {text_secondary};
                    font-weight: 500;
                    font-size: 13px;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 14px;
                }}
                QPushButton:hover {{
                    background-color: {bg_hover};
                    color: {text_primary};
                }}
                QPushButton:pressed {{
                    background-color: {bg_elevated};
                }}
            """)
        elif self._variant == "danger":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(225, 109, 117, 0.15);
                    color: {danger};
                    font-weight: 600;
                    font-size: 13px;
                    border: 1px solid rgba(225, 109, 117, 0.3);
                    border-radius: 8px;
                    padding: 8px 18px;
                }}
                QPushButton:hover {{
                    background-color: {danger};
                    color: #FFFFFF;
                    border-color: {danger};
                }}
                QPushButton:pressed {{
                    background-color: #C04D55;
                }}
            """)
