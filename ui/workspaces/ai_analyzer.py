"""AI Analyzer Workspace for Audio Studio AI.
Comprehensive song analysis workspace displaying real BPM, musical key, Camelot notation,
energy curves, structural breakdown (intro/verse/chorus/outro), and vocal activity heatmap.
"""
from pathlib import Path
from typing import Optional, List, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QFrame, QScrollArea, QGridLayout, QProgressBar,
    QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.stat_badge import StatBadge
from ui.components.waveform_view import WaveformView
from core.playback_engine import PlaybackEngine
from core.waveform_engine import WaveformEngine
from config import AUDIO_FILE_FILTER
from core.audio_format import validate_mp3_file
from ai.mixing.mix_plan import TrackAnalysis
from workers.analyzer_worker import AnalyzerWorker

class StructureTimelineBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self._segments: List[Tuple[str, float, float, QColor]] = [
            ("Intro", 0.0, 0.15, QColor("#495057")),
            ("Verse 1", 0.15, 0.35, QColor("#3D5A80")),
            ("Chorus 1", 0.35, 0.55, QColor("#C5C9D0")),
            ("Verse 2", 0.55, 0.70, QColor("#3D5A80")),
            ("Chorus 2", 0.70, 0.88, QColor("#C5C9D0")),
            ("Outro", 0.88, 1.0, QColor("#495057"))
        ]

    def set_boundaries(self, boundaries_sec: List[float], total_dur_sec: float):
        if not boundaries_sec or total_dur_sec <= 0:
            return

        colors = [QColor("#495057"), QColor("#3D5A80"), QColor("#C5C9D0"), QColor("#3D5A80"), QColor("#C5C9D0"), QColor("#98C1D9"), QColor("#495057")]
        labels = ["Intro", "Verse 1", "Chorus 1", "Verse 2", "Chorus 2", "Bridge", "Outro"]

        points = [0.0] + [min(1.0, max(0.0, b / total_dur_sec)) for b in boundaries_sec] + [1.0]
        points = sorted(list(set(points)))

        new_segs = []
        for i in range(len(points) - 1):
            s = points[i]
            e = points[i + 1]
            if e - s > 0.03:
                lbl = labels[i % len(labels)]
                col = colors[i % len(colors)]
                new_segs.append((lbl, s, e, col))

        if new_segs:
            self._segments = new_segs
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        for name, start_pct, end_pct, color in self._segments:
            x = int(start_pct * w)
            seg_w = max(4, int((end_pct - start_pct) * w))
            painter.fillRect(x, 0, seg_w - 2, h, color)

            # Draw Label
            painter.setPen(QColor("#101114") if color.lightness() > 140 else QColor("#F5F5F5"))
            painter.drawText(x, 0, seg_w - 2, h, Qt.AlignmentFlag.AlignCenter, name)


class AIAnalyzerWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()
        self._loaded_path: Optional[str] = None
        self._worker: Optional[AnalyzerWorker] = None

        self._build_ui()
        self._connect_signals()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(20)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Header Title
        header_box = QHBoxLayout()
        header_box.setSpacing(16)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("AI Song Analyzer")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("Deep acoustic intelligence: Tempo, Key, Phrase segmentation, Energy, and Vocal presence.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_box.addLayout(title_box, 1)

        self.btn_load_song = PremiumButton("📂  Analyze Audio File (MP3)", variant="primary")
        self.btn_load_song.clicked.connect(self._select_file)
        header_box.addWidget(self.btn_load_song)
        layout.addLayout(header_box)

        # Key Metrics Row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)

        self.badge_bpm = StatBadge("-- BPM", "Detected Tempo", "⏱️")
        self.badge_key = StatBadge("--", "Musical Key (Camelot)", "🎹")
        self.badge_energy = StatBadge("-- %", "Dynamic Energy", "⚡")
        self.badge_vocal = StatBadge("-- %", "Vocal Presence", "🎤")
        self.badge_genre = StatBadge("--", "Estimated Genre", "🎧")

        metrics_layout.addWidget(self.badge_bpm)
        metrics_layout.addWidget(self.badge_key)
        metrics_layout.addWidget(self.badge_energy)
        metrics_layout.addWidget(self.badge_vocal)
        metrics_layout.addWidget(self.badge_genre)
        layout.addLayout(metrics_layout)

        # Waveform & Playback Card
        self.wave_card = PremiumCard(title="Audio Waveform & Beat Grid", subtitle="Synchronized beat markers and playhead")
        self.wave_view = WaveformView()
        self.wave_view.setFixedHeight(85)
        self.wave_view.seek_requested.connect(self._playback.seek_ratio)

        self.play_row = QHBoxLayout()
        self.btn_play = PremiumButton("▶ Play Track", variant="secondary")
        self.btn_play.clicked.connect(self._play_track)
        self.track_name_label = QLabel("No MP3 track loaded for analysis")
        self.track_name_label.setObjectName("LoadedTrackLabel")

        self.play_row.addWidget(self.btn_play)
        self.play_row.addWidget(self.track_name_label, 1)

        self.wave_card.add_layout(self.play_row)
        self.wave_card.add_widget(self.wave_view)
        layout.addWidget(self.wave_card)

        # Song Structure Timeline Card
        struct_card = PremiumCard(title="AI Song Structural Segmentation", subtitle="Automatic classification of Intro, Verse, Chorus, Bridge, and Outro")
        struct_layout = QVBoxLayout()
        struct_layout.setSpacing(10)

        self.structure_bar = StructureTimelineBar()
        struct_layout.addWidget(self.structure_bar)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(14)
        for name, col_hex in [("Intro / Outro", "#495057"), ("Verse", "#3D5A80"), ("Chorus (Hook)", "#C5C9D0"), ("Bridge", "#98C1D9")]:
            dot = QLabel(f"■ {name}")
            dot.setStyleSheet(f"color: {col_hex}; font-size: 12px; font-weight: 600;")
            legend_row.addWidget(dot)
        legend_row.addStretch()
        struct_layout.addLayout(legend_row)

        struct_card.add_layout(struct_layout)
        layout.addWidget(struct_card)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self._playback.state_changed.connect(self._on_playback_state_changed)
        self._playback.position_changed.connect(self._on_playback_position_changed)

    def _select_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select MP3 Audio File to Analyze",
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

            self._loaded_path = str(Path(fpath).resolve())
            p = Path(fpath)
            self.track_name_label.setText(f"Analyzing: {p.name}...")
            self.btn_load_song.setEnabled(False)

            peaks = WaveformEngine.load_peaks(self._loaded_path)
            if peaks is not None:
                self.wave_view.set_peaks(peaks)

            self._worker = AnalyzerWorker(self._loaded_path, self)
            self._worker.completed.connect(self._on_analysis_completed)
            self._worker.failed.connect(self._on_analysis_failed)
            self._worker.start()

    def _on_analysis_completed(self, ta: TrackAnalysis):
        self.btn_load_song.setEnabled(True)
        p = Path(self._loaded_path)
        self.track_name_label.setText(f"Analyzed: {p.name}")

        self.badge_bpm.set_value(f"{ta.bpm:.1f} BPM")
        self.badge_key.set_value(f"{ta.camelot_key} ({ta.musical_key})")
        self.badge_energy.set_value(f"{int(ta.energy_level * 100)} %")
        self.badge_vocal.set_value(f"{int(ta.vocal_presence_ratio * 100)} %")

        # Intelligent genre estimation
        bpm = ta.bpm
        if bpm > 165:
            genre = "Drum & Bass / Fast"
        elif bpm >= 135:
            genre = "Trance / Techno / EDM"
        elif bpm >= 120:
            genre = "House / Dance Pop"
        elif bpm >= 95:
            genre = "Hip-Hop / R&B / Pop"
        elif bpm >= 75:
            genre = "Trap / Reggaeton"
        else:
            genre = "Downtempo / Ambient"
        self.badge_genre.set_value(genre)

        # Update structural segmentation
        self.structure_bar.set_boundaries(ta.phrase_boundaries_sec, ta.duration_seconds)

    def _on_analysis_failed(self, err: str):
        self.btn_load_song.setEnabled(True)
        self.track_name_label.setText("Analysis failed.")
        QMessageBox.critical(self, "Analysis Error", f"Acoustic analysis failed:\n\n{err}")

    def _play_track(self):
        if self._loaded_path:
            self._playback.play_file(self._loaded_path, title=Path(self._loaded_path).stem, artist="AI Analyzer")

    def _on_playback_state_changed(self, is_playing: bool):
        self.btn_play.setText("⏸ Pause Track" if is_playing else "▶ Play Track")

    def _on_playback_position_changed(self, curr_ms: int, dur_ms: int):
        if dur_ms > 0:
            self.wave_view.set_progress(curr_ms / float(dur_ms))

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")

        self.setStyleSheet(f"""
            QLabel#WorkspaceTitle {{
                color: {text_primary};
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#WorkspaceSubtitle {{
                color: {text_secondary};
                font-size: 13px;
            }}
            QLabel#LoadedTrackLabel {{
                color: {accent};
                font-size: 13px;
                font-weight: 500;
            }}
        """)
