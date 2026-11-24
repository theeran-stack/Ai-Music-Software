"""Audio Editor Workspace for Audio Studio AI.
Interactive multitrack audio workstation featuring:
- Non-destructive clip selection, drag-move, and start/end trimming.
- Split clip at playhead position.
- Delete, merge/consolidate, and equal-power crossfading.
- Independent track lanes with volume faders, mute/solo buses, and waveform previews.
- Synchronized playhead timeline tracking.
- 32-bit floating-point master summing with ITU-R BS.1770 -14.0 LUFS broadcast normalisation.
- Master MP3 export at 128k, 192k, 256k, and 320k bitrates.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QFileDialog, QFrame, QScrollArea, QPushButton, QComboBox,
    QMessageBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QMouseEvent

from ui.themes.theme_manager import ThemeManager
from ui.components.button import PremiumButton
from core.playback_engine import PlaybackEngine
from core.clip_manager import TimelineClip, TimelineTrack
from config import AUDIO_FILE_FILTER, AVAILABLE_MP3_BITRATES, DEFAULT_EXPORT_BITRATE, CACHE_DIR
from core.audio_format import validate_mp3_file, encode_array_to_mp3, decode_mp3_to_numpy
from ai.separation.audio_normalizer import AudioNormalizer


class TrackLaneCanvas(QWidget):
    """Interactive timeline canvas for an individual track lane displaying audio clips."""
    clip_selected = Signal(str)      # clip_id
    timeline_changed = Signal()
    seek_requested = Signal(float)   # seconds

    def __init__(self, track: TimelineTrack, parent=None):
        super().__init__(parent)
        self.track = track
        self.setMinimumHeight(86)
        self.setMouseTracking(True)
        self.pixels_per_second: float = 40.0  # Zoom level
        self._playhead_sec: float = 0.0

        # Interaction state
        self._dragging_mode: Optional[str] = None  # "move", "trim_left", "trim_right"
        self._drag_clip: Optional[TimelineClip] = None
        self._drag_start_x: float = 0.0
        self._drag_start_val: float = 0.0

        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self.update())

    def set_playhead(self, sec: float):
        if abs(self._playhead_sec - sec) > 0.02:
            self._playhead_sec = max(0.0, sec)
            self.update()

    def _get_clip_rect(self, clip: TimelineClip) -> QRectF:
        x = clip.timeline_start * self.pixels_per_second
        w = max(24.0, clip.duration * self.pixels_per_second)
        return QRectF(x, 4, w, self.height() - 8)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            x = pos.x()
            clicked_time = max(0.0, x / self.pixels_per_second)

            # Check if clicked inside any clip
            hit_clip = None
            for clip in reversed(self.track.clips):
                rect = self._get_clip_rect(clip)
                if rect.contains(pos):
                    hit_clip = clip
                    break

            if hit_clip:
                self.track.deselect_all()
                hit_clip.is_selected = True
                self.clip_selected.emit(hit_clip.clip_id)

                rect = self._get_clip_rect(hit_clip)
                # Check handle zones (left 8px, right 8px)
                if x <= rect.left() + 8:
                    self._dragging_mode = "trim_left"
                    self._drag_start_val = hit_clip.timeline_start
                elif x >= rect.right() - 8:
                    self._dragging_mode = "trim_right"
                    self._drag_start_val = hit_clip.duration
                else:
                    self._dragging_mode = "move"
                    self._drag_start_val = hit_clip.timeline_start

                self._drag_clip = hit_clip
                self._drag_start_x = x
            else:
                self.track.deselect_all()
                self._dragging_mode = None
                self._drag_clip = None
                self.seek_requested.emit(clicked_time)

            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        x = pos.x()

        if self._dragging_mode and self._drag_clip:
            delta_px = x - self._drag_start_x
            delta_sec = delta_px / self.pixels_per_second

            if self._dragging_mode == "move":
                new_start = max(0.0, self._drag_start_val + delta_sec)
                self._drag_clip.timeline_start = round(new_start, 2)
            elif self._dragging_mode == "trim_left":
                self._drag_clip.trim_start(delta_sec)
                self._drag_start_x = x
            elif self._dragging_mode == "trim_right":
                self._drag_clip.trim_end(delta_sec)
                self._drag_start_x = x

            self.track.sort_clips()
            self.timeline_changed.emit()
            self.update()
        else:
            # Update cursor on hover over trim handles
            cursor_set = False
            for clip in self.track.clips:
                rect = self._get_clip_rect(clip)
                if rect.contains(pos):
                    if pos.x() <= rect.left() + 8 or pos.x() >= rect.right() - 8:
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                    else:
                        self.setCursor(Qt.CursorShape.SizeAllCursor)
                    cursor_set = True
                    break
            if not cursor_set:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_mode:
            self._dragging_mode = None
            self._drag_clip = None
            self.track.sort_clips()
            self.timeline_changed.emit()
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_color = QColor(c.get("background_base", "#141619"))
        card_bg = QColor(c.get("background_card", "#202328"))
        accent_color = QColor(c.get("accent_silver", "#E5AA57"))
        accent_border = QColor(c.get("accent_silver", "#E5AA57"))
        normal_border = QColor(c.get("border_default", "#3A3E47"))
        wave_color = QColor(c.get("waveform_wave", "#5B616D"))
        wave_sel = QColor(c.get("waveform_progress", "#F0F2F5"))
        text_primary = QColor(c.get("text_primary", "#FFFFFF"))
        text_secondary = QColor(c.get("text_secondary", "#969BA3"))

        w = self.width()
        h = self.height()

        # Canvas Background
        painter.fillRect(0, 0, w, h, bg_color)

        # Draw Clips
        font = QFont("Segoe UI", 9)
        font.setPointSize(9)
        painter.setFont(font)

        for clip in self.track.clips:
            rect = self._get_clip_rect(clip)

            # Clip Background
            clip_brush = QBrush(card_bg)
            pen_color = accent_border if clip.is_selected else normal_border
            pen_width = 2 if clip.is_selected else 1
            painter.setPen(QPen(pen_color, pen_width))
            painter.setBrush(clip_brush)
            painter.drawRoundedRect(rect, 6, 6)

            # Clip Header Title
            header_rect = QRectF(rect.x() + 8, rect.y() + 4, rect.width() - 16, 16)
            painter.setPen(text_primary if clip.is_selected else text_secondary)
            title_text = f"{clip.name} ({clip.duration:.1f}s)"
            painter.drawText(header_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title_text)

            # Draw Waveform Peaks inside Clip
            wave_y = rect.y() + 24
            wave_h = max(10, rect.height() - 28)
            mid_y = wave_y + (wave_h / 2.0)

            peaks = clip._peaks_cache
            if peaks is not None and len(peaks) > 0:
                clip_w = rect.width() - 8
                num_bars = min(int(clip_w // 3), len(peaks))
                if num_bars > 0:
                    step = len(peaks) / float(num_bars)
                    bar_w = max(1.5, (clip_w / num_bars) - 1.0)
                    p_color = wave_sel if clip.is_selected else wave_color

                    for i in range(num_bars):
                        peak_idx = int(i * step)
                        val = float(peaks[peak_idx])
                        bh = max(2.0, val * (wave_h * 0.85))
                        bx = rect.x() + 4 + (i * (clip_w / num_bars))
                        by = mid_y - (bh / 2.0)
                        painter.fillRect(QRectF(bx, by, bar_w, bh), p_color)

            # Draw Crossfade indicators
            if clip.fade_in > 0:
                fin_w = clip.fade_in * self.pixels_per_second
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(229, 170, 87, 40))
                painter.drawPolygon([
                    QPointF(rect.x(), rect.bottom()),
                    QPointF(rect.x() + fin_w, rect.top()),
                    QPointF(rect.x(), rect.top())
                ])

            if clip.fade_out > 0:
                fout_w = clip.fade_out * self.pixels_per_second
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(229, 170, 87, 40))
                painter.drawPolygon([
                    QPointF(rect.right() - fout_w, rect.top()),
                    QPointF(rect.right(), rect.bottom()),
                    QPointF(rect.right(), rect.top())
                ])

            # Draw Trim Handles indicators
            if clip.is_selected:
                painter.setBrush(QColor(229, 170, 87, 180))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(rect.left(), rect.top() + 6, 3, rect.height() - 12))
                painter.drawRect(QRectF(rect.right() - 3, rect.top() + 6, 3, rect.height() - 12))

        # Draw Playhead Cursor
        playhead_x = self._playhead_sec * self.pixels_per_second
        if 0 <= playhead_x <= w:
            painter.setPen(QPen(QColor("#E5AA57"), 2))
            painter.drawLine(playhead_x, 0, playhead_x, h)


class TrackLaneWidget(QFrame):
    """Composite track lane control panel + interactive timeline canvas."""
    clip_loaded = Signal(int, str)
    state_changed = Signal()

    def __init__(self, track: TimelineTrack, parent=None):
        super().__init__(parent)
        self.track = track
        self.setMinimumHeight(110)

        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(14)

        # Track Controls Left Panel
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(190)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)

        self.title_lbl = QLabel(f"T{track.track_num}: {track.name}")
        self.title_lbl.setObjectName("TrackLaneTitle")

        # Load MP3 button
        self.btn_load = QPushButton("📂 Load Clip (MP3)")
        self.btn_load.setObjectName("TrackLoadBtn")
        self.btn_load.clicked.connect(self._browse_audio)
        ctrl_layout.addWidget(self.btn_load)

        # Mute / Solo buttons
        ms_row = QHBoxLayout()
        ms_row.setSpacing(6)
        self.btn_mute = QPushButton("M")
        self.btn_mute.setFixedSize(28, 24)
        self.btn_mute.setCheckable(True)
        self.btn_mute.setObjectName("MuteBtn")
        self.btn_mute.toggled.connect(self._on_mute_toggled)

        self.btn_solo = QPushButton("S")
        self.btn_solo.setFixedSize(28, 24)
        self.btn_solo.setCheckable(True)
        self.btn_solo.setObjectName("SoloBtn")
        self.btn_solo.toggled.connect(self._on_solo_toggled)

        ms_row.addWidget(self.btn_mute)
        ms_row.addWidget(self.btn_solo)
        ms_row.addStretch()
        ctrl_layout.addLayout(ms_row)

        # Volume Fader
        vol_row = QHBoxLayout()
        self.vol_lbl = QLabel(f"Vol: {int(track.volume * 100)}%")
        self.vol_lbl.setStyleSheet("color: #969BA3; font-size: 11px;")
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(track.volume * 100))
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.vol_lbl)
        vol_row.addWidget(self.vol_slider)
        ctrl_layout.addLayout(vol_row)
        ctrl_layout.addWidget(self.title_lbl)
        layout.addWidget(ctrl_panel)

        # Interactive Timeline Canvas
        self.canvas = TrackLaneCanvas(track, self)
        self.canvas.timeline_changed.connect(self.state_changed)
        layout.addWidget(self.canvas, 1)

        self._update_style()

    def _browse_audio(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            f"Select MP3 Clip for Track {self.track.track_num}",
            "",
            AUDIO_FILE_FILTER
        )
        if fpath:
            if not fpath.lower().endswith(".mp3"):
                QMessageBox.warning(self, "Invalid Format", "Only MP3 audio files are supported.")
                return
            valid, err = validate_mp3_file(fpath)
            if not valid:
                QMessageBox.warning(self, "Invalid Format", f"Only MP3 audio files are supported.\n\n{err}")
                return

            # Append clip to end of current track clips
            start_t = 0.0
            if self.track.clips:
                start_t = max(c.timeline_end for c in self.track.clips)

            clip = self.track.add_clip(fpath, timeline_start=start_t)
            self.track.deselect_all()
            clip.is_selected = True

            self.canvas.update()
            self.clip_loaded.emit(self.track.track_num, fpath)
            self.state_changed.emit()

    def _on_mute_toggled(self, checked: bool):
        self.track.is_muted = checked
        self.state_changed.emit()

    def _on_solo_toggled(self, checked: bool):
        self.track.is_solo = checked
        self.state_changed.emit()

    def _on_volume_changed(self, val: int):
        self.track.volume = val / 100.0
        self.vol_lbl.setText(f"Vol: {val}%")
        self.state_changed.emit()

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = c.get("background_card", "#1B1D21")
        border = c.get("border_default", "#30333A")
        text_primary = c.get("text_primary", "#F5F5F5")

        self.setStyleSheet(f"""
            TrackLaneWidget {{
                background-color: {bg_card};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QLabel#TrackLaneTitle {{
                color: {text_primary};
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton#TrackLoadBtn {{
                background-color: #22252A;
                color: #C5C9D0;
                border: 1px solid {border};
                border-radius: 4px;
                font-size: 11px;
                padding: 4px 8px;
            }}
            QPushButton#TrackLoadBtn:hover {{
                background-color: #2D3037;
            }}
            QPushButton#MuteBtn, QPushButton#SoloBtn {{
                background-color: #22252A;
                color: #969BA3;
                border: 1px solid {border};
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton#MuteBtn:checked {{
                background-color: #E16D75;
                color: white;
            }}
            QPushButton#SoloBtn:checked {{
                background-color: #E5AA57;
                color: black;
            }}
        """)


class AudioEditorWorkspace(QWidget):
    """Complete 4-Lane Interactive Multitrack Audio Workstation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()

        self._tracks: List[TimelineTrack] = []
        self._lanes: List[TrackLaneWidget] = []
        self._rendered_master_path: Optional[str] = None
        self._playhead_sec: float = 0.0

        self._build_ui()
        self._connect_signals()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(14)

        # Header Title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Audio Editor")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("Interactive multitrack timeline: select, move, trim, split, delete, merge, crossfade, and master MP3 export.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        self.playhead_lbl = QLabel("Playhead: 00:00.00")
        self.playhead_lbl.setStyleSheet("color: #E5AA57; font-size: 13px; font-weight: bold; font-family: Consolas, monospace;")
        header_layout.addWidget(self.playhead_lbl)
        main_layout.addLayout(header_layout)

        # Interactive Editing Tools Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.btn_play_session = PremiumButton("▶ Play Mix", variant="secondary")
        self.btn_play_session.clicked.connect(self._toggle_play_session)

        self.btn_split = QPushButton("✂ Split Clip")
        self.btn_split.setObjectName("ToolBtn")
        self.btn_split.clicked.connect(self._split_active_clip)

        self.btn_merge = QPushButton("🔗 Merge Selected")
        self.btn_merge.setObjectName("ToolBtn")
        self.btn_merge.clicked.connect(self._merge_active_clips)

        self.btn_crossfade = QPushButton("⨯ Add Crossfade")
        self.btn_crossfade.setObjectName("ToolBtn")
        self.btn_crossfade.clicked.connect(self._crossfade_active_clips)

        self.btn_delete = QPushButton("🗑 Delete Clip")
        self.btn_delete.setObjectName("ToolBtn")
        self.btn_delete.clicked.connect(self._delete_active_clip)

        self.btn_add_track = PremiumButton("+ Add Track", variant="primary")
        self.btn_add_track.clicked.connect(lambda: self._add_track())

        toolbar.addWidget(self.btn_play_session)
        toolbar.addWidget(self.btn_split)
        toolbar.addWidget(self.btn_merge)
        toolbar.addWidget(self.btn_crossfade)
        toolbar.addWidget(self.btn_delete)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_add_track)
        main_layout.addLayout(toolbar)

        # Timeline Ruler Bar
        ruler = QFrame()
        ruler.setFixedHeight(24)
        ruler_layout = QHBoxLayout(ruler)
        ruler_layout.setContentsMargins(210, 0, 12, 0)
        ruler_lbl = QLabel("00:00        00:10        00:20        00:30        00:40        00:50        01:00        01:10        01:20        01:30")
        ruler_lbl.setStyleSheet("color: #969BA3; font-size: 11px; font-family: Consolas, monospace;")
        ruler_layout.addWidget(ruler_lbl)
        main_layout.addWidget(ruler)

        # Lanes Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.lanes_container = QWidget()
        self.lanes_layout = QVBoxLayout(self.lanes_container)
        self.lanes_layout.setContentsMargins(0, 0, 0, 0)
        self.lanes_layout.setSpacing(10)

        # Initial 4 default tracks
        self._add_track("Vocals")
        self._add_track("Drums & Percussion")
        self._add_track("Bassline")
        self._add_track("Harmonies & Synth")

        self.lanes_layout.addStretch()
        scroll.setWidget(self.lanes_container)
        main_layout.addWidget(scroll, 1)

        # Bottom Bar: Export Master Mix
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        info_label = QLabel("Non-destructive 32-bit float summing engine • ITU-R BS.1770 -14 LUFS standard")
        info_label.setObjectName("StatusNotice")

        bitrate_lbl = QLabel("MP3 Bitrate:")
        bitrate_lbl.setObjectName("FieldLabel")

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setItemDelegate(QStyledItemDelegate(self.bitrate_combo))
        self.bitrate_combo.addItem("320 kbps (Studio Master)", "320k")
        self.bitrate_combo.addItem("256 kbps (High Quality)", "256k")
        self.bitrate_combo.addItem("192 kbps (Standard)", "192k")
        self.bitrate_combo.addItem("128 kbps (Compact)", "128k")
        self.bitrate_combo.setCurrentIndex(0)

        self.btn_export = PremiumButton("💾  Export Master Mix (MP3)", variant="primary")
        self.btn_export.clicked.connect(self._export_master_mix)

        bottom_bar.addWidget(info_label, 1)
        bottom_bar.addWidget(bitrate_lbl)
        bottom_bar.addWidget(self.bitrate_combo)
        bottom_bar.addWidget(self.btn_export)
        main_layout.addLayout(bottom_bar)

    def _connect_signals(self):
        self._playback.state_changed.connect(self._on_playback_state_changed)
        self._playback.position_changed.connect(self._on_playback_position_changed)

    def _add_track(self, name: Optional[str] = None):
        track_num = len(self._tracks) + 1
        tname = name or f"Audio Track {track_num}"
        track = TimelineTrack(track_num=track_num, name=tname)
        self._tracks.append(track)

        lane = TrackLaneWidget(track, self)
        lane.canvas.seek_requested.connect(self._on_timeline_seek)
        lane.state_changed.connect(self._update_all_canvases)
        self._lanes.append(lane)

        idx = max(0, self.lanes_layout.count() - 1)
        self.lanes_layout.insertWidget(idx, lane)

    def _update_all_canvases(self):
        for lane in self._lanes:
            lane.canvas.update()

    def _on_timeline_seek(self, sec: float):
        self._playhead_sec = sec
        mins = int(sec // 60)
        secs = sec % 60
        self.playhead_lbl.setText(f"Playhead: {mins:02d}:{secs:05.2f}")
        for lane in self._lanes:
            lane.canvas.set_playhead(sec)

    def _get_active_selected_track_and_clip(self) -> Tuple[Optional[TimelineTrack], Optional[TimelineClip]]:
        for t in self._tracks:
            c = t.get_selected_clip()
            if c:
                return t, c
        return None, None

    def _split_active_clip(self):
        track, clip = self._get_active_selected_track_and_clip()
        if not track or not clip:
            QMessageBox.information(self, "No Clip Selected", "Please click on an audio clip to select it for splitting.")
            return

        ok = track.split_selected_clip(self._playhead_sec)
        if ok:
            self._update_all_canvases()
        else:
            QMessageBox.warning(
                self,
                "Cannot Split",
                f"The playhead ({self._playhead_sec:.2f}s) is outside the bounds of the selected clip [{clip.timeline_start:.2f}s - {clip.timeline_end:.2f}s]."
            )

    def _merge_active_clips(self):
        track, _ = self._get_active_selected_track_and_clip()
        if not track:
            QMessageBox.information(self, "No Track Selected", "Please select clips on a track to merge them.")
            return

        ok = track.merge_selected_clips()
        if ok:
            self._update_all_canvases()
        else:
            QMessageBox.information(self, "Cannot Merge", "At least two clips on the same track must be selected to merge.")

    def _crossfade_active_clips(self):
        track, _ = self._get_active_selected_track_and_clip()
        if not track:
            QMessageBox.information(self, "No Track Selected", "Please select a track with adjacent clips to apply crossfading.")
            return

        ok = track.apply_crossfade(crossfade_sec=1.0)
        if ok:
            self._update_all_canvases()
            QMessageBox.information(self, "Crossfade Applied", "Applied 1.0s equal-power crossfade between adjacent clips on this track.")
        else:
            QMessageBox.information(self, "No Adjacent Clips", "Track must have at least two adjacent or touching clips to crossfade.")

    def _delete_active_clip(self):
        track, clip = self._get_active_selected_track_and_clip()
        if not track or not clip:
            return

        track.remove_clip(clip.clip_id)
        self._update_all_canvases()

    def _get_session_max_duration(self) -> float:
        max_t = 0.0
        for t in self._tracks:
            for c in t.clips:
                max_t = max(max_t, c.timeline_end)
        return max_t

    def _render_session(self, max_duration: Optional[float] = None) -> np.ndarray:
        dur = max_duration or self._get_session_max_duration()
        if dur <= 0.1:
            return np.zeros((2, 44100), dtype=np.float32)

        sr = 44100
        has_solo = any(t.is_solo for t in self._tracks)
        active_tracks = [t for t in self._tracks if (t.is_solo if has_solo else not t.is_muted)]

        master_sum = np.zeros((2, int(dur * sr)), dtype=np.float32)
        for t in active_tracks:
            t_audio = t.render_track(dur, sr=sr)
            master_sum += t_audio

        norm_master, _ = AudioNormalizer.normalize_stem(
            master_sum,
            sr=sr,
            enable_norm=True,
            mode="lufs",
            target_lufs=-14.0,
            target_peak_db=-1.0,
            peak_protection=True
        )
        return norm_master

    def _toggle_play_session(self):
        if self._playback.is_playing:
            self._playback.pause()
            return

        # Render current session to a temporary playback file
        dur = self._get_session_max_duration()
        if dur <= 0.1:
            QMessageBox.information(self, "Session Empty", "Please load audio clips into tracks to play the session.")
            return

        rendered = self._render_session(dur)
        preview_p = Path(CACHE_DIR) / "session_preview.mp3"
        encode_array_to_mp3(rendered, str(preview_p), sample_rate=44100, bitrate="320k")
        self._playback.play_file(str(preview_p), title="Multitrack Session", artist="Audio Studio AI")

    def _export_master_mix(self):
        dur = self._get_session_max_duration()
        if dur <= 0.1:
            QMessageBox.warning(self, "No Audio Loaded", "Please load at least one audio clip into a track lane.")
            return

        bitrate = self.bitrate_combo.currentData() or "320k"
        default_file = str(Path(CACHE_DIR) / "master_mix.mp3")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Master Mix (MP3 @ {bitrate})",
            default_file,
            AUDIO_FILE_FILTER
        )
        if not save_path:
            return

        dest = Path(save_path).with_suffix(".mp3")
        rendered = self._render_session(dur)
        encode_array_to_mp3(rendered, str(dest), sample_rate=44100, bitrate=bitrate)
        self._rendered_master_path = str(dest.resolve())

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(
                self,
                "Export Complete",
                f"Master multitrack mix exported successfully!\n\n"
                f"Tracks: {len(self._tracks)}\n"
                f"Duration: {dur:.2f}s\n"
                f"Format: MP3 ({bitrate})\n"
                f"File: {dest.name}"
            )

    def _on_playback_state_changed(self, is_playing: bool):
        self.btn_play_session.setText("⏸ Pause Mix" if is_playing else "▶ Play Mix")

    def _on_playback_position_changed(self, curr_ms: int, dur_ms: int):
        if dur_ms > 0:
            sec = curr_ms / 1000.0
            self._on_timeline_seek(sec)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        border = c.get("border_default", "#30333A")

        self.setStyleSheet(f"""
            QPushButton#ToolBtn {{
                background-color: #22252A;
                color: #C5C9D0;
                border: 1px solid {border};
                border-radius: 6px;
                font-size: 11px;
                font-weight: 500;
                padding: 6px 12px;
            }}
            QPushButton#ToolBtn:hover {{
                background-color: #2D3037;
                color: #FFFFFF;
            }}
            QPushButton#ToolBtn:pressed {{
                background-color: #1B1D21;
            }}
        """)
