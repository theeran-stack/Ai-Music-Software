"""Dashboard Workspace for Audio Studio AI.
Displays quick actions, library statistics, recent tracks, and AI task queue.
"""
from typing import Callable, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.stat_badge import StatBadge
from database.database import DatabaseManager
from core.playback_engine import PlaybackEngine
from config import CUDA_AVAILABLE, CUDA_DEVICE_NAME

class DashboardWorkspace(QWidget):
    navigate_requested = Signal(str)  # workspace_key

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = DatabaseManager()
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())

        self._build_ui()
        self.refresh_data()
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(24)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        # 1. Welcome Section
        welcome_card = PremiumCard(elevated=True)
        w_layout = QHBoxLayout()
        w_layout.setContentsMargins(0, 0, 0, 0)

        w_text_widget = QWidget()
        w_text_layout = QVBoxLayout(w_text_widget)
        w_text_layout.setContentsMargins(0, 0, 0, 0)
        w_text_layout.setSpacing(6)

        title = QLabel("Welcome to Audio Studio AI")
        title.setObjectName("DashboardHeroTitle")
        subtitle = QLabel("Professional AI-powered vocal separation, voice conversion, audio editing & smart DJ mixing.")
        subtitle.setObjectName("DashboardHeroSubtitle")

        hw_tag = QLabel(f"⚡ Device: {CUDA_DEVICE_NAME if CUDA_AVAILABLE else 'CPU Acceleration'} • Offline Mode Active")
        hw_tag.setObjectName("DashboardHeroHw")

        w_text_layout.addWidget(title)
        w_text_layout.addWidget(subtitle)
        w_text_layout.addWidget(hw_tag)

        w_layout.addWidget(w_text_widget, 1)

        # Quick action buttons on banner
        btn_box = QVBoxLayout()
        btn_box.setSpacing(8)
        btn_quick_remove = PremiumButton("✂️  Remove Vocals", variant="primary")
        btn_quick_remove.clicked.connect(lambda: self.navigate_requested.emit("vocal_remover"))
        btn_quick_mix = PremiumButton("🎧  Create Smart Mix", variant="secondary")
        btn_quick_mix.clicked.connect(lambda: self.navigate_requested.emit("smart_mix"))

        btn_box.addWidget(btn_quick_remove)
        btn_box.addWidget(btn_quick_mix)
        w_layout.addLayout(btn_box)

        welcome_card.add_layout(w_layout)
        layout.addWidget(welcome_card)

        # 2. Statistics Grid
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.stat_tracks = StatBadge("0", "Tracks in Library", "📚")
        self.stat_duration = StatBadge("0h 0m", "Total Audio Duration", "⏱️")
        self.stat_separated = StatBadge("Ready", "AI Separation Engine", "✂️")
        self.stat_models = StatBadge("Local", "AI Storage Mode", "🔒")

        stats_layout.addWidget(self.stat_tracks)
        stats_layout.addWidget(self.stat_duration)
        stats_layout.addWidget(self.stat_separated)
        stats_layout.addWidget(self.stat_models)
        layout.addLayout(stats_layout)

        # 3. Quick Action Cards (Grid)
        actions_label = QLabel("Studio Workspaces")
        actions_label.setObjectName("SectionHeader")
        layout.addWidget(actions_label)

        actions_grid = QGridLayout()
        actions_grid.setSpacing(16)

        actions = [
            ("✂️ AI Vocal Remover", "Separate vocals and instrumentals locally using Demucs & MDX-Net.", "vocal_remover"),
            ("🎙️ Vocal Studio", "Denoising, pitch shifting, time stretching & vocal enhancement.", "vocal_studio"),
            ("🧪 Voice Lab", "RVC-style neural voice conversion and model management.", "voice_lab"),
            ("🎚️ Audio Editor", "Multitrack DAW timeline, clip trimming, merging & crossfades.", "audio_editor"),
            ("🎧 Smart Mix Studio", "Automatic AI DJ mixing with key & tempo alignment.", "smart_mix"),
            ("🧠 AI Analyzer", "BPM, musical key, song structure and vocal activity timeline.", "ai_analyzer"),
        ]

        for i, (act_title, act_desc, act_key) in enumerate(actions):
            card = PremiumCard(title=act_title, subtitle=act_desc, clickable=True)
            card.clicked.connect(lambda k=act_key: self.navigate_requested.emit(k))
            row = i // 3
            col = i % 3
            actions_grid.addWidget(card, row, col)

        layout.addLayout(actions_grid)

        # 4. Recently Played & AI Queue (Two Columns)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(20)

        # Recent Tracks Card
        self.recent_card = PremiumCard(title="Recently Played", subtitle="Quick access to your listening history")
        self.recent_layout = QVBoxLayout()
        self.recent_layout.setSpacing(8)
        self.recent_empty = QLabel("No recently played tracks yet. Add music in Music Library.")
        self.recent_empty.setObjectName("EmptyNotice")
        self.recent_layout.addWidget(self.recent_empty)
        self.recent_card.add_layout(self.recent_layout)
        bottom_row.addWidget(self.recent_card, 1)

        # AI Processing Queue Card
        self.queue_card = PremiumCard(title="AI Processing Queue", subtitle="Background tasks and active operations")
        self.queue_layout = QVBoxLayout()
        self.queue_layout.setSpacing(8)
        self.queue_empty = QLabel("Queue is idle. Start an AI Vocal Remover or Smart Mix task.")
        self.queue_empty.setObjectName("EmptyNotice")
        self.queue_layout.addWidget(self.queue_empty)
        self.queue_card.add_layout(self.queue_layout)
        bottom_row.addWidget(self.queue_card, 1)

        layout.addLayout(bottom_row)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def refresh_data(self):
        # Update statistics from Database
        stats = self._db.get_library_statistics()
        self.stat_tracks.set_value(str(stats["total_tracks"]))

        total_min = int(stats["total_duration"] / 60)
        hours = total_min // 60
        mins = total_min % 60
        self.stat_duration.set_value(f"{hours}h {mins}m")

        # Update recently played
        recents = self._db.get_recently_played_tracks(limit=5)
        # Clear previous items
        for i in reversed(range(self.recent_layout.count())):
            w = self.recent_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        if not recents:
            self.recent_layout.addWidget(self.recent_empty)
        else:
            for track in recents:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(8, 6, 8, 6)
                t_lbl = QLabel(f"🎵 {track.title}  —  {track.artist}")
                t_lbl.setStyleSheet("color: #F5F5F5; font-size: 13px;")
                btn_play = PremiumButton("Play", variant="ghost")
                btn_play.setFixedHeight(28)
                btn_play.clicked.connect(lambda checked=False, t=track: self._playback.play_track(t))
                row_layout.addWidget(t_lbl, 1)
                row_layout.addWidget(btn_play)
                self.recent_layout.addWidget(row)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")

        self.setStyleSheet(f"""
            QLabel#DashboardHeroTitle {{
                color: {text_primary};
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#DashboardHeroSubtitle {{
                color: {text_secondary};
                font-size: 13px;
                line-height: 1.4;
            }}
            QLabel#DashboardHeroHw {{
                color: {accent};
                font-size: 12px;
                font-weight: 600;
                margin-top: 4px;
            }}
            QLabel#SectionHeader {{
                color: {text_primary};
                font-size: 17px;
                font-weight: 600;
                margin-top: 6px;
            }}
            QLabel#EmptyNotice {{
                color: {text_secondary};
                font-size: 13px;
                font-style: italic;
                padding: 12px 0;
            }}
        """)
