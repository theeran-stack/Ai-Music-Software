"""Global Bottom Audio Player Bar for Audio Studio AI.
Persistent 82px player with artwork, track metadata, seek bar with tabular timing,
transport controls, volume slider, and playback rate options.
"""
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton,
    QFrame, QMenu
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QAction, QPainter, QColor, QFont

from ui.themes.theme_manager import ThemeManager
from core.playback_engine import PlaybackEngine
from database.models import Track

def format_time_ms(ms: int) -> str:
    total_seconds = max(0, ms // 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

class PlayerBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BottomPlayer")
        self.setFixedHeight(82)

        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()
        self._is_seeking = False

        self._build_ui()
        self._connect_signals()
        self._update_style()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(20)

        # --- LEFT SECTION: Track Info & Artwork (Width ~260px) ---
        left_container = QWidget()
        left_container.setFixedWidth(260)
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Artwork Square
        self.art_label = QLabel()
        self.art_label.setFixedSize(50, 50)
        self.art_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.art_label.setText("♫")
        self.art_label.setObjectName("PlayerArtwork")

        # Titles
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 4, 0, 4)
        info_layout.setSpacing(2)

        self.title_label = QLabel("Ready to Play")
        self.title_label.setObjectName("PlayerTitle")
        self.artist_label = QLabel("Select a track to begin")
        self.artist_label.setObjectName("PlayerArtist")

        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.artist_label)

        left_layout.addWidget(self.art_label)
        left_layout.addWidget(info_widget, 1)

        # --- CENTER SECTION: Transport Controls & Seek Bar ---
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Transport Controls Row
        controls_row = QWidget()
        controls_layout = QHBoxLayout(controls_row)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(16)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_shuffle = QPushButton("🔀")
        self.btn_shuffle.setFixedSize(30, 30)
        self.btn_shuffle.setObjectName("PlayerIconBtn")
        self.btn_shuffle.setToolTip("Shuffle Queue")

        self.btn_prev = QPushButton("⏮")
        self.btn_prev.setFixedSize(32, 32)
        self.btn_prev.setObjectName("PlayerIconBtn")
        self.btn_prev.setToolTip("Previous Track")

        self.btn_play_pause = QPushButton("▶")
        self.btn_play_pause.setFixedSize(40, 40)
        self.btn_play_pause.setObjectName("PlayerPlayBtn")
        self.btn_play_pause.setToolTip("Play / Pause")

        self.btn_next = QPushButton("⏭")
        self.btn_next.setFixedSize(32, 32)
        self.btn_next.setObjectName("PlayerIconBtn")
        self.btn_next.setToolTip("Next Track")

        self.btn_repeat = QPushButton("🔁")
        self.btn_repeat.setFixedSize(30, 30)
        self.btn_repeat.setObjectName("PlayerIconBtn")
        self.btn_repeat.setToolTip("Repeat (Off / All / One)")

        controls_layout.addWidget(self.btn_shuffle)
        controls_layout.addWidget(self.btn_prev)
        controls_layout.addWidget(self.btn_play_pause)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addWidget(self.btn_repeat)

        # Seek Bar Row
        seek_row = QWidget()
        seek_layout = QHBoxLayout(seek_row)
        seek_layout.setContentsMargins(0, 0, 0, 0)
        seek_layout.setSpacing(10)

        self.time_current = QLabel("00:00")
        self.time_current.setObjectName("PlayerTiming")
        self.time_current.setFixedWidth(40)
        self.time_current.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setCursor(Qt.CursorShape.PointingHandCursor)

        self.time_total = QLabel("00:00")
        self.time_total.setObjectName("PlayerTiming")
        self.time_total.setFixedWidth(40)
        self.time_total.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        seek_layout.addWidget(self.time_current)
        seek_layout.addWidget(self.seek_slider, 1)
        seek_layout.addWidget(self.time_total)

        center_layout.addWidget(controls_row)
        center_layout.addWidget(seek_row)

        # --- RIGHT SECTION: Speed, Volume & Extra Tools (Width ~240px) ---
        right_container = QWidget()
        right_container.setFixedWidth(240)
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Speed Button Pill
        self.btn_speed = QPushButton("1.0x")
        self.btn_speed.setObjectName("SpeedPill")
        self.btn_speed.setFixedSize(48, 28)
        self.btn_speed.setCursor(Qt.CursorShape.PointingHandCursor)
        speed_font = self.btn_speed.font()
        speed_font.setPointSize(8)
        speed_font.setWeight(QFont.Weight.DemiBold)
        self.btn_speed.setFont(speed_font)
        self._build_speed_menu()

        # Volume Controls
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedSize(30, 30)
        self.btn_mute.setObjectName("PlayerIconBtn")
        self.btn_mute.setToolTip("Mute / Unmute")

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self._playback.get_volume() * 100))
        self.volume_slider.setFixedWidth(90)
        self.volume_slider.setCursor(Qt.CursorShape.PointingHandCursor)

        right_layout.addWidget(self.btn_speed)
        right_layout.addWidget(self.btn_mute)
        right_layout.addWidget(self.volume_slider)

        # Assemble Main Layout
        main_layout.addWidget(left_container)
        main_layout.addWidget(center_container, 1)
        main_layout.addWidget(right_container)

    def _build_speed_menu(self):
        menu = QMenu(self)
        speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for spd in speeds:
            action = QAction(f"{spd}x", self)
            action.triggered.connect(lambda checked, s=spd: self._set_speed(s))
            menu.addAction(action)
        self.btn_speed.setMenu(menu)

    def _set_speed(self, rate: float):
        self.btn_speed.setText(f"{rate}x")
        self._playback.set_playback_rate(rate)

    def _connect_signals(self):
        # Playback events
        self._playback.track_loaded.connect(self._on_track_loaded)
        self._playback.state_changed.connect(self._on_state_changed)
        self._playback.position_changed.connect(self._on_position_changed)
        self._playback.volume_changed.connect(self._on_volume_changed)

        # UI Actions
        self.btn_play_pause.clicked.connect(self._playback.toggle_play_pause)
        self.btn_prev.clicked.connect(self._playback.previous_track)
        self.btn_next.clicked.connect(self._playback.next_track)
        self.btn_shuffle.clicked.connect(self._on_toggle_shuffle)
        self.btn_repeat.clicked.connect(self._on_toggle_repeat)

        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)

        self.volume_slider.valueChanged.connect(lambda val: self._playback.set_volume(val / 100.0))
        self.btn_mute.clicked.connect(self._toggle_mute)

        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())

    def _on_track_loaded(self, track: Track):
        self.title_label.setText(track.title or "Unknown Track")
        self.artist_label.setText(track.artist or "Unknown Artist")

    def _on_state_changed(self, is_playing: bool):
        self.btn_play_pause.setText("⏸" if is_playing else "▶")

    def _on_position_changed(self, current_ms: int, duration_ms: int):
        if not self._is_seeking:
            if duration_ms > 0:
                pos_ratio = current_ms / duration_ms
                self.seek_slider.setValue(int(pos_ratio * 1000))
            else:
                self.seek_slider.setValue(0)
            self.time_current.setText(format_time_ms(current_ms))
            self.time_total.setText(format_time_ms(duration_ms))

    def _on_volume_changed(self, vol: float):
        self.volume_slider.setValue(int(vol * 100))
        if vol <= 0.01:
            self.btn_mute.setText("🔇")
        elif vol < 0.5:
            self.btn_mute.setText("🔉")
        else:
            self.btn_mute.setText("🔊")

    def _toggle_mute(self):
        current_vol = self._playback.get_volume()
        if current_vol > 0.01:
            self._playback.set_volume(0.0)
        else:
            self._playback.set_volume(0.7)

    def _on_seek_pressed(self):
        self._is_seeking = True

    def _on_seek_moved(self, val: int):
        pass

    def _on_seek_released(self):
        val = self.seek_slider.value()
        ratio = val / 1000.0
        self._playback.seek_ratio(ratio)
        self._is_seeking = False

    def _on_toggle_shuffle(self):
        is_shuf = self._playback.toggle_shuffle()
        c = self._theme_mgr.get_current_theme().get("colors", {})
        accent = c.get("accent_silver", "#C5C9D0")
        self.btn_shuffle.setStyleSheet(f"color: {accent}; font-weight: bold;" if is_shuf else "")

    def _on_toggle_repeat(self):
        rep = self._playback.cycle_repeat()
        if rep == 0:
            self.btn_repeat.setText("🔁")
            self.btn_repeat.setStyleSheet("")
        elif rep == 1:
            c = self._theme_mgr.get_current_theme().get("colors", {})
            accent = c.get("accent_silver", "#C5C9D0")
            self.btn_repeat.setText("🔁")
            self.btn_repeat.setStyleSheet(f"color: {accent}; font-weight: bold;")
        elif rep == 2:
            c = self._theme_mgr.get_current_theme().get("colors", {})
            accent = c.get("accent_silver", "#C5C9D0")
            self.btn_repeat.setText("🔂")
            self.btn_repeat.setStyleSheet(f"color: {accent}; font-weight: bold;")

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_player = c.get("player_bg", "#141518")
        bg_card = c.get("background_card", "#1B1D21")
        border_col = c.get("border_default", "#30333A")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent_silver = c.get("accent_silver", "#C5C9D0")
        accent_hover = c.get("accent_hover", "#D8DCE3")
        main_bg = c.get("background_main", "#101114")

        self.setStyleSheet(f"""
            QWidget#BottomPlayer {{
                background-color: {bg_player};
                border-top: 1px solid {border_col};
            }}
            QLabel#PlayerArtwork {{
                background-color: {bg_card};
                border: 1px solid {border_col};
                border-radius: 8px;
                color: {accent_silver};
                font-size: 22px;
            }}
            QLabel#PlayerTitle {{
                color: {text_primary};
                font-size: 14px;
                font-weight: 600;
            }}
            QLabel#PlayerArtist {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 400;
            }}
            QLabel#PlayerTiming {{
                color: {text_secondary};
                font-size: 11px;
                font-family: 'Consolas', 'Segoe UI', monospace;
            }}
            QPushButton#PlayerPlayBtn {{
                background-color: {accent_silver};
                color: {main_bg};
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
                border: none;
            }}
            QPushButton#PlayerPlayBtn:hover {{
                background-color: {accent_hover};
            }}
            QPushButton#PlayerIconBtn {{
                background: transparent;
                color: {text_secondary};
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }}
            QPushButton#PlayerIconBtn:hover {{
                color: {text_primary};
                background: rgba(255, 255, 255, 0.05);
            }}
            QPushButton#SpeedPill {{
                background-color: {bg_card};
                color: {text_secondary};
                border: 1px solid {border_col};
                border-radius: 6px;
                font-size: 8.5pt;
                font-weight: 600;
            }}
            QPushButton#SpeedPill:hover {{
                color: {accent_silver};
                border-color: {accent_silver};
            }}
            QPushButton#SpeedPill::menu-indicator {{
                image: none;
                width: 0px;
            }}
        """)
