"""Persistent Left Navigation Sidebar for Audio Studio AI.
Supports smooth collapsing animation, silver active navigation indicator,
and exact 11 standalone workspaces.
"""
from typing import Dict, List, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QCursor, QFont, QPainter, QColor, QPen

from config import SIDEBAR_EXPANDED_WIDTH, SIDEBAR_COLLAPSED_WIDTH
from ui.themes.theme_manager import ThemeManager
from ui.animations.animation_manager import AnimationManager

# Exact 11 standalone workspaces
WORKSPACES: List[Tuple[str, str, str]] = [
    ("dashboard", "Dashboard", "🎛️"),
    ("library", "Music Library", "📚"),
    ("vocal_remover", "Vocal Remover", "✂️"),      # Standalone top-level
    ("vocal_studio", "Vocal Studio", "🎙️"),       # Separate workspace
    ("voice_lab", "Voice Lab", "🧪"),
    ("audio_editor", "Audio Editor", "🎚️"),
    ("smart_mix", "Smart Mix Studio", "🎧"),
    ("ai_analyzer", "AI Analyzer", "🧠"),
    ("visualizer", "Audio Visualizer", "📊"),
    ("plugin_manager", "Plugin Manager", "🧩"),
    ("settings", "Settings", "⚙️")
]

class NavItemWidget(QPushButton):
    def __init__(self, key: str, title: str, icon_str: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.title = title
        self.icon_str = icon_str
        self._is_active = False
        self._is_collapsed = False

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(42)
        self.setToolTip(title)

        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_text()
        self._update_style()

    def set_active(self, active: bool):
        self._is_active = active
        self._update_style()

    def set_collapsed(self, collapsed: bool):
        self._is_collapsed = collapsed
        self._update_text()
        self._update_style()

    def _update_text(self):
        if self._is_collapsed:
            self.setText(self.icon_str)
        else:
            self.setText(f"  {self.icon_str}   {self.title}")

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        accent_silver = c.get("accent_silver", "#C5C9D0")
        accent_hover = c.get("accent_hover", "#D8DCE3")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        bg_hover = c.get("background_hover", "#26292F")
        bg_elevated = c.get("background_elevated", "#22252A")
        align = "center" if self._is_collapsed else "left"

        if self._is_active:
            border_left = "none" if self._is_collapsed else f"3px solid {accent_silver}"
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_elevated};
                    color: {text_primary};
                    font-weight: 600;
                    font-size: 13px;
                    border: none;
                    border-left: {border_left};
                    border-radius: {'8px' if self._is_collapsed else '0px 8px 8px 0px'};
                    text-align: {align};
                    padding-left: {'0px' if self._is_collapsed else '14px'};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {text_secondary};
                    font-weight: 500;
                    font-size: 13px;
                    border: none;
                    border-radius: 8px;
                    text-align: {align};
                    padding-left: {'0px' if self._is_collapsed else '14px'};
                }}
                QPushButton:hover {{
                    background-color: {bg_hover};
                    color: {text_primary};
                }}
            """)


class Sidebar(QWidget):
    workspace_selected = Signal(str)  # workspace_key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self._is_collapsed = False
        self._active_key = "dashboard"
        self._nav_items: Dict[str, NavItemWidget] = {}

        self.setFixedWidth(SIDEBAR_EXPANDED_WIDTH)
        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())

        self._build_ui()
        self._update_style()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 14)
        layout.setSpacing(6)

        # Header / Logo
        self.header_widget = QWidget()
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(6, 4, 6, 12)
        header_layout.setSpacing(10)

        self.logo_icon = QLabel("◆")
        self.logo_icon.setObjectName("SidebarLogoIcon")
        self.logo_icon.setFixedSize(28, 28)
        self.logo_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.logo_text = QLabel("AUDIO STUDIO AI")
        self.logo_text.setObjectName("SidebarLogoText")

        header_layout.addWidget(self.logo_icon)
        header_layout.addWidget(self.logo_text, 1)
        layout.addWidget(self.header_widget)

        # Navigation Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent; border: none;")

        nav_container = QWidget()
        self.nav_layout = QVBoxLayout(nav_container)
        self.nav_layout.setContentsMargins(0, 4, 0, 4)
        self.nav_layout.setSpacing(4)

        for key, title, icon in WORKSPACES:
            item = NavItemWidget(key, title, icon, self)
            item.clicked.connect(lambda checked=False, k=key: self.select_workspace(k))
            self.nav_layout.addWidget(item)
            self._nav_items[key] = item

        self.nav_layout.addStretch()
        scroll_area.setWidget(nav_container)
        layout.addWidget(scroll_area, 1)

        # System Status Pill (Local / Offline / GPU)
        self.status_pill = QFrame()
        self.status_pill.setObjectName("StatusPill")
        pill_layout = QHBoxLayout(self.status_pill)
        pill_layout.setContentsMargins(10, 8, 10, 8)
        pill_layout.setSpacing(8)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("StatusDot")
        self.status_dot.setFixedSize(14, 14)

        self.status_label = QLabel("Offline Ready")
        self.status_label.setObjectName("StatusLabel")

        pill_layout.addWidget(self.status_dot)
        pill_layout.addWidget(self.status_label, 1)
        layout.addWidget(self.status_pill)

        # Collapse / Expand Button
        self.btn_collapse = QPushButton("◀  Collapse")
        self.btn_collapse.setObjectName("CollapseBtn")
        self.btn_collapse.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_collapse.setFixedHeight(34)
        self.btn_collapse.clicked.connect(self.toggle_collapse)
        layout.addWidget(self.btn_collapse)

        # Set initial active item
        self.select_workspace("dashboard", emit_signal=False)

    def select_workspace(self, key: str, emit_signal: bool = True):
        if key not in self._nav_items:
            return
        self._active_key = key
        for k, item in self._nav_items.items():
            item.set_active(k == key)
        if emit_signal:
            self.workspace_selected.emit(key)

    def toggle_collapse(self):
        self._is_collapsed = not self._is_collapsed
        target_w = SIDEBAR_COLLAPSED_WIDTH if self._is_collapsed else SIDEBAR_EXPANDED_WIDTH

        # Update items collapsed state
        self.logo_text.setVisible(not self._is_collapsed)
        self.status_label.setVisible(not self._is_collapsed)
        self.btn_collapse.setText("▶" if self._is_collapsed else "◀  Collapse")

        for item in self._nav_items.values():
            item.set_collapsed(self._is_collapsed)

        # Animate width smoothly
        anim_mgr = AnimationManager.get_instance()
        anim_mgr.animate_width(self, self.width(), target_w, duration=200)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_sidebar = c.get("background_sidebar", "#151619")
        border_subtle = c.get("border_subtle", "#25282F")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent_silver = c.get("accent_silver", "#C5C9D0")
        success = c.get("success", "#48B58A")
        bg_elevated = c.get("background_elevated", "#22252A")
        bg_hover = c.get("background_hover", "#26292F")

        self.setStyleSheet(f"""
            QWidget#Sidebar {{
                background-color: {bg_sidebar};
                border-right: 1px solid {border_subtle};
            }}
            QLabel#SidebarLogoIcon {{
                color: {accent_silver};
                font-size: 18px;
                font-weight: bold;
            }}
            QLabel#SidebarLogoText {{
                color: {text_primary};
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 1.5px;
            }}
            QFrame#StatusPill {{
                background-color: {bg_elevated};
                border: 1px solid {border_subtle};
                border-radius: 8px;
            }}
            QLabel#StatusDot {{
                color: {success};
                font-size: 12px;
            }}
            QLabel#StatusLabel {{
                color: {text_secondary};
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton#CollapseBtn {{
                background-color: transparent;
                color: {text_secondary};
                border: 1px solid {border_subtle};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton#CollapseBtn:hover {{
                background-color: {bg_hover};
                color: {text_primary};
            }}
        """)
