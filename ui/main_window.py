"""Main Window for Audio Studio AI.
Orchestrates Obsidian Silver UI layout:
- Top bar with workspace breadcrumb & global search
- Collapsible sidebar with exact 11 standalone workspaces
- Central workspace stacked widget with smooth fade/slide transitions
- Persistent bottom audio player (82px)
- Elegant startup splash animation
"""
from typing import Dict
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QLineEdit, QPushButton, QFrame,
    QGraphicsOpacityEffect, QApplication, QComboBox, QStyledItemDelegate
)
from PySide6.QtCore import (
    Qt, QTimer, QPoint, QEasingCurve, QPropertyAnimation,
    QParallelAnimationGroup, Signal
)
from PySide6.QtGui import QIcon, QFont, QColor

from config import (
    APP_NAME, APP_VERSION, TOP_BAR_HEIGHT, BOTTOM_PLAYER_HEIGHT,
    CUDA_AVAILABLE, CUDA_DEVICE_NAME
)
from ui.themes.theme_manager import ThemeManager
from ui.animations.animation_manager import AnimationManager
from ui.sidebar import Sidebar
from ui.components.player_bar import PlayerBar

# Workspaces
from ui.workspaces.dashboard import DashboardWorkspace
from ui.workspaces.library import LibraryWorkspace
from ui.workspaces.vocal_remover import VocalRemoverWorkspace
from ui.workspaces.vocal_studio import VocalStudioWorkspace
from ui.workspaces.voice_lab import VoiceLabWorkspace
from ui.workspaces.audio_editor import AudioEditorWorkspace
from ui.workspaces.smart_mix import SmartMixWorkspace
from ui.workspaces.ai_analyzer import AIAnalyzerWorkspace
from ui.workspaces.visualizer import AudioVisualizerWorkspace
from ui.workspaces.plugin_manager import PluginManagerWorkspace
from ui.workspaces.settings import SettingsWorkspace
from database.database import DatabaseManager

class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(TOP_BAR_HEIGHT)

        self._theme_mgr = ThemeManager.get_instance()
        self._build_ui()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(16)

        # Breadcrumb / Active Workspace Title
        self.title_label = QLabel("Dashboard")
        self.title_label.setObjectName("TopBarTitle")
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Global Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍  Quick search (Ctrl+K)...")
        self.search_input.setFixedWidth(240)
        layout.addWidget(self.search_input)

        # Hardware & Status Tag
        hw_label = "CUDA Active" if CUDA_AVAILABLE else "CPU Ready"
        self.badge_hw = QLabel(f"● {hw_label}")
        self.badge_hw.setObjectName("TopBarHwBadge")
        layout.addWidget(self.badge_hw)

    def set_title(self, title: str):
        self.title_label.setText(f"{APP_NAME}  /  {title}")

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_main = c.get("background_main", "#101114")
        border_subtle = c.get("border_subtle", "#25282F")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")
        success = c.get("success", "#48B58A")

        self.setStyleSheet(f"""
            QWidget#TopBar {{
                background-color: {bg_main};
                border-bottom: 1px solid {border_subtle};
            }}
            QLabel#TopBarTitle {{
                color: {text_primary};
                font-size: 15px;
                font-weight: 600;
            }}
            QLabel#TopBarHwBadge {{
                color: {success if CUDA_AVAILABLE else accent};
                font-size: 12px;
                font-weight: 600;
                background-color: rgba(255, 255, 255, 0.05);
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid {border_subtle};
            }}
        """)


class SplashScreenOverlay(QWidget):
    """Subtle luxury Obsidian Silver startup screen."""
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)

        self.logo_icon = QLabel("◆")
        self.logo_icon.setStyleSheet("color: #C5C9D0; font-size: 54px;")
        self.logo_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.app_title = QLabel("AUDIO STUDIO AI")
        self.app_title.setStyleSheet("color: #F5F5F5; font-size: 20px; font-weight: 700; letter-spacing: 4px;")
        self.app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.tagline = QLabel("Initializing Neural Audio Engine & Local Library...")
        self.tagline.setStyleSheet("color: #969BA3; font-size: 12px;")
        self.tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.logo_icon)
        layout.addWidget(self.app_title)
        layout.addWidget(self.tagline)

        self.setStyleSheet("background-color: #101114;")

    def start_animation(self):
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)

        # Fade in
        fade_in = QPropertyAnimation(effect, b"opacity", self)
        fade_in.setDuration(400)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.start()

        # Trigger fade out after 1.2s
        QTimer.singleShot(1200, self._fade_out)

    def _fade_out(self):
        effect = self.graphicsEffect()
        if isinstance(effect, QGraphicsOpacityEffect):
            fade_out = QPropertyAnimation(effect, b"opacity", self)
            fade_out.setDuration(350)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.finished.connect(self._on_finished)
            fade_out.start()
        else:
            self._on_finished()

    def _on_finished(self):
        self.setVisible(False)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - Next-Gen Audio Workstation")
        self.setMinimumSize(1120, 740)
        self.resize(1380, 880)

        self._theme_mgr = ThemeManager.get_instance()
        self._anim_mgr = AnimationManager.get_instance()
        self._db = DatabaseManager()

        self._workspaces: Dict[str, QWidget] = {}
        self._current_workspace_key = "dashboard"

        self._build_ui()
        self._apply_theme()
        self._theme_mgr.theme_changed.connect(lambda _: self._apply_theme())

        # Startup Splash Animation
        show_startup = self._db.get_setting("startup_animation", True)
        if show_startup and self._anim_mgr.is_enabled and not self._anim_mgr.reduced_motion:
            self.splash = SplashScreenOverlay(self)
            self.splash.resize(self.size())
            self.splash.show()
            self.splash.start_animation()
            self.splash.finished.connect(lambda: self._set_workspace("dashboard"))
        else:
            self._set_workspace("dashboard")

    def resizeEvent(self, event):
        if hasattr(self, "splash") and self.splash.isVisible():
            self.splash.resize(self.size())
        super().resizeEvent(event)

    def _build_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top Bar
        self.top_bar = TopBar(self)
        root_layout.addWidget(self.top_bar)

        # Middle Area: Sidebar + Stacked Workspaces
        middle_widget = QWidget()
        middle_layout = QHBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar(self)
        self.sidebar.workspace_selected.connect(self._set_workspace)
        middle_layout.addWidget(self.sidebar)

        # Stacked Widget for 11 Workspaces
        self.stack = QStackedWidget()
        self._init_workspaces()
        middle_layout.addWidget(self.stack, 1)

        root_layout.addWidget(middle_widget, 1)

        # Bottom Player (82px)
        self.player_bar = PlayerBar(self)
        root_layout.addWidget(self.player_bar)

    def _init_workspaces(self):
        # 1. Dashboard
        dashboard = DashboardWorkspace(self)
        dashboard.navigate_requested.connect(self._on_dashboard_navigate)
        self._add_workspace("dashboard", "Dashboard", dashboard)

        # 2. Music Library
        library = LibraryWorkspace(self)
        self._add_workspace("library", "Music Library", library)

        # 3. Vocal Remover (Standalone top-level workspace)
        vocal_remover = VocalRemoverWorkspace(self)
        self._add_workspace("vocal_remover", "AI Vocal Remover", vocal_remover)

        # 4. Vocal Studio (Dedicated vocal editing)
        vocal_studio = VocalStudioWorkspace(self)
        self._add_workspace("vocal_studio", "Vocal Studio", vocal_studio)

        # 5. Voice Lab
        voice_lab = VoiceLabWorkspace(self)
        self._add_workspace("voice_lab", "Voice Lab", voice_lab)

        # 6. Audio Editor
        audio_editor = AudioEditorWorkspace(self)
        self._add_workspace("audio_editor", "Audio Editor", audio_editor)

        # 7. Smart Mix Studio
        smart_mix = SmartMixWorkspace(self)
        self._add_workspace("smart_mix", "Smart Mix Studio", smart_mix)

        # 8. AI Analyzer
        ai_analyzer = AIAnalyzerWorkspace(self)
        self._add_workspace("ai_analyzer", "AI Analyzer", ai_analyzer)

        # 9. Audio Visualizer
        visualizer = AudioVisualizerWorkspace(self)
        self._add_workspace("visualizer", "Audio Visualizer", visualizer)

        # 10. Plugin Manager
        plugin_manager = PluginManagerWorkspace(self)
        self._add_workspace("plugin_manager", "Plugin Manager", plugin_manager)

        # 11. Settings
        settings = SettingsWorkspace(self)
        self._add_workspace("settings", "Settings", settings)

    def _add_workspace(self, key: str, title: str, widget: QWidget):
        self._workspaces[key] = widget
        widget.setProperty("workspace_title", title)
        # Apply QStyledItemDelegate directly to all QComboBox widgets within this workspace
        for cb in widget.findChildren(QComboBox):
            cb.setItemDelegate(QStyledItemDelegate(cb))
        self.stack.addWidget(widget)

    def _set_workspace(self, key: str):
        if key not in self._workspaces:
            return

        target_widget = self._workspaces[key]
        title = target_widget.property("workspace_title") or key.replace("_", " ").title()
        self.top_bar.set_title(title)
        self.sidebar.select_workspace(key, emit_signal=False)

        # If already active, do nothing
        if self.stack.currentWidget() == target_widget:
            return

        # Smooth transition
        if self._anim_mgr.is_enabled and not self._anim_mgr.reduced_motion:
            self._anim_mgr.slide_fade_in(target_widget, duration=220)

        self.stack.setCurrentWidget(target_widget)
        self._current_workspace_key = key

        # If dashboard, refresh stats
        if key == "dashboard" and isinstance(target_widget, DashboardWorkspace):
            target_widget.refresh_data()

    def _on_dashboard_navigate(self, key: str):
        self._set_workspace(key)

    def _apply_theme(self):
        qss = self._theme_mgr.generate_qss()
        self.setStyleSheet(qss)
