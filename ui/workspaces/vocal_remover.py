"""Standalone AI Vocal Remover Workspace for Audio Studio AI.
Fully functional neural source separation powered by HTDemucs v4.
Provides drag-and-drop audio import, real-time waveform display,
multi-pass AI separation, intelligent loudness normalization & stem enhancement,
independent A/B stem playback, and separate/batch export in WAV, MP3, and FLAC.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QFileDialog, QProgressBar, QFrame, QScrollArea, QSlider,
    QMessageBox, QCheckBox, QGridLayout, QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from ui.themes.theme_manager import ThemeManager
from ui.animations.animation_manager import AnimationManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.waveform_view import WaveformView
from ai.model_manager import ModelManager
from ai.separation.demucs_adapter import SeparationEngine, SEPARATED_OUTPUT_DIR
from ai.separation.audio_normalizer import AudioNormalizer
from workers.separation_worker import SeparationWorker
from core.playback_engine import PlaybackEngine
from core.waveform_engine import WaveformEngine
from config import (
    SUPPORTED_AUDIO_EXTENSIONS, CUDA_AVAILABLE, CUDA_DEVICE_NAME,
    AUDIO_FILE_FILTER, AVAILABLE_MP3_BITRATES, DEFAULT_EXPORT_BITRATE
)
from core.audio_format import validate_mp3_file, encode_array_to_mp3, decode_mp3_to_numpy

def format_duration_sec(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

class DropZoneWidget(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(130)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)

        self.icon_label = QLabel("📥")
        self.icon_label.setStyleSheet("font-size: 28px;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel("Drag & Drop your MP3 track here, or click to browse")
        self.text_label.setObjectName("DropText")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.subtext_label = QLabel("Supported format: MP3 (*.mp3) only (128k - 320k)")
        self.subtext_label.setObjectName("DropSubtext")
        self.subtext_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.subtext_label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(Path(u.toLocalFile()).suffix.lower() == ".mp3" for u in urls):
                event.acceptProposedAction()
                self.setStyleSheet("border: 2px dashed #C5C9D0; background-color: rgba(197, 201, 208, 0.08);")

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        for u in event.mimeData().urls():
            fpath = u.toLocalFile()
            if not fpath.lower().endswith(".mp3"):
                QMessageBox.warning(self, "Invalid Format", "Only MP3 audio files are supported.")
                return
            is_valid, err = validate_mp3_file(fpath)
            if not is_valid:
                QMessageBox.warning(self, "Invalid Format", f"Only MP3 audio files are supported.\n\n{err}")
                return
            self.file_dropped.emit(fpath)
            break

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            fpath, _ = QFileDialog.getOpenFileName(
                self,
                "Select MP3 Audio File for AI Vocal Separation",
                "",
                AUDIO_FILE_FILTER
            )
            if fpath:
                if not fpath.lower().endswith(".mp3"):
                    QMessageBox.warning(self, "Invalid Format", "Only MP3 audio files are supported.")
                    return
                is_valid, err = validate_mp3_file(fpath)
                if not is_valid:
                    QMessageBox.warning(self, "Invalid Format", f"Only MP3 audio files are supported.\n\n{err}")
                    return
                self.file_dropped.emit(fpath)
        super().mousePressEvent(event)


class VocalRemoverWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._model_mgr = ModelManager.get_instance()
        self._theme_mgr = ThemeManager.get_instance()
        self._anim_mgr = AnimationManager.get_instance()

        self._loaded_file_path: Optional[str] = None
        self._loaded_duration: float = 0.0
        self._loaded_sr: int = 44100
        self._stem_base_name: str = ""

        # Separation output paths
        self._vocals_path: Optional[str] = None
        self._instrumental_path: Optional[str] = None
        self._vocals_raw_path: Optional[str] = None
        self._instrumental_raw_path: Optional[str] = None
        self._currently_playing_stem: Optional[str] = None
        self._preview_normalized_mode: bool = True
        self._latest_metrics: Dict[str, Any] = {}

        # Worker
        self._worker: Optional[SeparationWorker] = None

        self._build_ui()
        self._connect_signals()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 28, 28, 28)
        main_layout.setSpacing(20)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background: transparent; border: none;")

        self.container = QWidget()
        self.content_layout = QVBoxLayout(self.container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(20)

        # 1. Header Title & Device Status
        header_box = QHBoxLayout()
        header_box.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("AI Vocal Remover")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("Extract studio-grade isolated Vocals and Instrumental accompaniment locally using neural HTDemucs.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_box.addLayout(title_box, 1)

        dev_text = f"⚡ {CUDA_DEVICE_NAME}" if CUDA_AVAILABLE else "⚙️ CPU Multi-threading"
        self.badge_device = QLabel(dev_text)
        self.badge_device.setObjectName("DeviceTag")
        header_box.addWidget(self.badge_device)

        self.content_layout.addLayout(header_box)

        # 2. Input Audio Section (Drop Zone + Loaded Track Card)
        self.drop_card = PremiumCard()
        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_dropped.connect(self._load_audio_file)
        self.drop_card.add_widget(self.drop_zone)
        self.content_layout.addWidget(self.drop_card)

        # Loaded Track Info Card (Hidden initially until file dropped)
        self.track_card = PremiumCard(title="Input Song Waveform & Details", subtitle="Original unseparated mix")
        self.track_card.setVisible(False)

        info_header_row = QHBoxLayout()
        self.track_title_label = QLabel("No file loaded")
        self.track_title_label.setObjectName("TrackTitleHighlight")
        self.track_meta_label = QLabel("")
        self.track_meta_label.setObjectName("TrackMetaDetails")

        self.btn_replace = PremiumButton("↺  Replace Song", variant="ghost")
        self.btn_replace.clicked.connect(self._prompt_replace_file)

        self.btn_play_original = PremiumButton("▶  Play Original", variant="secondary")
        self.btn_play_original.clicked.connect(self._toggle_play_original)

        info_header_row.addWidget(self.track_title_label)
        info_header_row.addWidget(self.track_meta_label, 1)
        info_header_row.addWidget(self.btn_replace)
        info_header_row.addWidget(self.btn_play_original)
        self.track_card.add_layout(info_header_row)

        self.original_waveform = WaveformView()
        self.original_waveform.setFixedHeight(75)
        self.original_waveform.seek_requested.connect(self._playback.seek_ratio)
        self.track_card.add_widget(self.original_waveform)

        self.content_layout.addWidget(self.track_card)

        # 3. Separation Configuration Card
        self.config_card = PremiumCard(title="Neural Model & Processing Engine", subtitle="Select separation model, precision quality, and compute acceleration")
        cfg_layout = QHBoxLayout()
        cfg_layout.setSpacing(20)

        # Model Selector
        m_box = QVBoxLayout()
        m_lbl = QLabel("AI Model Architecture:")
        m_lbl.setObjectName("FieldLabel")
        self.model_combo = QComboBox()
        self.model_combo.setItemDelegate(QStyledItemDelegate(self.model_combo))
        self.model_combo.addItem("HTDemucs v4 (Vocals & Instrumental) [Ready]", "htdemucs")
        m_box.addWidget(m_lbl)
        m_box.addWidget(self.model_combo)
        cfg_layout.addLayout(m_box, 2)

        # Quality Mode Selector
        q_box = QVBoxLayout()
        q_lbl = QLabel("Quality & Speed Profile:")
        q_lbl.setObjectName("FieldLabel")
        self.quality_combo = QComboBox()
        self.quality_combo.setItemDelegate(QStyledItemDelegate(self.quality_combo))
        self.quality_combo.addItem("Balanced (1 Shift, 25% Overlap)", "balanced")
        self.quality_combo.addItem("Fast (0 Shifts, Low Latency)", "fast")
        self.quality_combo.addItem("High Quality (2 Shifts, 50% Overlap)", "quality")
        q_box.addWidget(q_lbl)
        q_box.addWidget(self.quality_combo)
        cfg_layout.addLayout(q_box, 1)

        # Compute Device Selector
        d_box = QVBoxLayout()
        d_lbl = QLabel("Compute Device:")
        d_lbl.setObjectName("FieldLabel")
        self.device_combo = QComboBox()
        self.device_combo.setItemDelegate(QStyledItemDelegate(self.device_combo))
        self.device_combo.addItem("Auto-Detect (Recommended)", "auto")
        if CUDA_AVAILABLE:
            self.device_combo.addItem("NVIDIA CUDA (GPU)", "cuda")
        self.device_combo.addItem("CPU Acceleration", "cpu")
        d_box.addWidget(d_lbl)
        d_box.addWidget(self.device_combo)
        cfg_layout.addLayout(d_box, 1)

        self.config_card.add_layout(cfg_layout)

        # Destination Folder Selector Row
        dest_row = QHBoxLayout()
        dest_row.setSpacing(10)
        dest_lbl = QLabel("Output Folder:")
        dest_lbl.setObjectName("FieldLabel")
        self.dest_folder_label = QLabel(str(SEPARATED_OUTPUT_DIR))
        self.dest_folder_label.setObjectName("DestFolderLabel")
        self.btn_browse_dest = PremiumButton("Browse...", variant="outline")
        self.btn_browse_dest.setFixedHeight(30)
        self.btn_browse_dest.clicked.connect(self._select_dest_folder)

        dest_row.addWidget(dest_lbl)
        dest_row.addWidget(self.dest_folder_label, 1)
        dest_row.addWidget(self.btn_browse_dest)
        self.config_card.add_layout(dest_row)

        # Action Button: Remove Vocals
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 10, 0, 0)
        self.status_tip = QLabel("Select an audio track above to begin separation.")
        self.status_tip.setObjectName("StatusNotice")

        self.btn_remove_vocals = PremiumButton("✂️  REMOVE VOCALS", variant="primary")
        self.btn_remove_vocals.setMinimumWidth(220)
        self.btn_remove_vocals.setFixedHeight(44)
        self.btn_remove_vocals.setEnabled(False)
        self.btn_remove_vocals.clicked.connect(self._start_separation)

        action_row.addWidget(self.status_tip, 1)
        action_row.addWidget(self.btn_remove_vocals)
        self.config_card.add_layout(action_row)

        self.content_layout.addWidget(self.config_card)

        # 4. Processing Progress Card (Hidden initially, shown only during processing)
        self.progress_card = PremiumCard(elevated=True)
        self.progress_card.setVisible(False)
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(12)

        prog_header_row = QHBoxLayout()
        self.prog_title = QLabel("AI Neural Separation in Progress...")
        self.prog_title.setObjectName("ProcessingTitle")
        self.prog_pct_label = QLabel("0%")
        self.prog_pct_label.setObjectName("ProcessingPercent")

        prog_header_row.addWidget(self.prog_title)
        prog_header_row.addStretch()
        prog_header_row.addWidget(self.prog_pct_label)
        prog_layout.addLayout(prog_header_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setObjectName("SeparationProgressBar")
        prog_layout.addWidget(self.progress_bar)

        prog_footer_row = QHBoxLayout()
        self.prog_stage_label = QLabel("Initializing neural model...")
        self.prog_stage_label.setObjectName("ProgressStage")
        self.btn_cancel = PremiumButton("✕ Cancel Separation", variant="danger")
        self.btn_cancel.setFixedHeight(30)
        self.btn_cancel.clicked.connect(self._cancel_separation)

        prog_footer_row.addWidget(self.prog_stage_label, 1)
        prog_footer_row.addWidget(self.btn_cancel)
        prog_layout.addLayout(prog_footer_row)

        self.progress_card.add_layout(prog_layout)
        self.content_layout.addWidget(self.progress_card)

        # 5. Results Section (Hidden until separation succeeds!)
        self.results_container = QWidget()
        self.results_container.setObjectName("ResultsContainer")
        self.results_container.setVisible(False)
        res_layout = QVBoxLayout(self.results_container)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(18)

        # --- A. PREVIEW & DOWNLOAD BOTH TOOLBAR ---
        res_toolbar = PremiumCard()
        tb_layout = QHBoxLayout()
        tb_layout.setContentsMargins(0, 0, 0, 0)
        tb_layout.setSpacing(14)

        ab_title = QLabel("Preview Mode:")
        ab_title.setObjectName("FieldLabel")
        tb_layout.addWidget(ab_title)

        self.btn_solo_original = PremiumButton("🎵 Original", variant="secondary")
        self.btn_solo_original.clicked.connect(lambda: self._switch_preview("original"))

        self.btn_solo_vocals = PremiumButton("🎤 Vocals Solo", variant="secondary")
        self.btn_solo_vocals.clicked.connect(lambda: self._switch_preview("vocals"))

        self.btn_solo_inst = PremiumButton("🎸 Instrumental Solo", variant="secondary")
        self.btn_solo_inst.clicked.connect(lambda: self._switch_preview("instrumental"))

        tb_layout.addWidget(self.btn_solo_original)
        tb_layout.addWidget(self.btn_solo_vocals)
        tb_layout.addWidget(self.btn_solo_inst)
        tb_layout.addStretch()

        # DOWNLOAD BOTH: Bitrate selector & button directly in UI
        both_fmt_lbl = QLabel("MP3 Bitrate:")
        both_fmt_lbl.setObjectName("FieldLabel")
        tb_layout.addWidget(both_fmt_lbl)

        self.both_format_combo = QComboBox()
        self.both_format_combo.setObjectName("FormatSelector")
        self.both_format_combo.setItemDelegate(QStyledItemDelegate(self.both_format_combo))
        self.both_format_combo.addItem("320 kbps (Studio Master)", "320k")
        self.both_format_combo.addItem("256 kbps (High Quality)", "256k")
        self.both_format_combo.addItem("192 kbps (Standard)", "192k")
        self.both_format_combo.addItem("128 kbps (Compact)", "128k")
        self.both_format_combo.setCurrentIndex(0)  # Default 320k
        tb_layout.addWidget(self.both_format_combo)

        self.btn_download_both = PremiumButton("📦  Export Both Stems (MP3)", variant="primary")
        self.btn_download_both.clicked.connect(self._download_both_action)
        tb_layout.addWidget(self.btn_download_both)

        res_toolbar.add_layout(tb_layout)
        res_layout.addWidget(res_toolbar)

        # --- B. TWO STEM CARDS (VOCALS & INSTRUMENTAL) ---
        stems_grid = QHBoxLayout()
        stems_grid.setSpacing(20)

        # CARD 1: VOCALS STEM
        self.vocals_card = PremiumCard(title="🎤 Vocals Stem", subtitle="Isolated acapella vocal track without accompaniment")
        v_layout = QVBoxLayout()
        v_layout.setSpacing(12)

        self.vocals_waveform = WaveformView()
        self.vocals_waveform.setFixedHeight(85)
        self.vocals_waveform.seek_requested.connect(self._playback.seek_ratio)
        v_layout.addWidget(self.vocals_waveform)

        v_ctrl_row = QHBoxLayout()
        self.btn_play_vocal_card = PremiumButton("▶ Play Vocals", variant="primary")
        self.btn_play_vocal_card.clicked.connect(lambda: self._switch_preview("vocals"))

        self.vocal_duration_label = QLabel("00:00")
        self.vocal_duration_label.setObjectName("DurationBadge")

        v_ctrl_row.addWidget(self.btn_play_vocal_card)
        v_ctrl_row.addWidget(self.vocal_duration_label)
        v_ctrl_row.addStretch()
        v_layout.addLayout(v_ctrl_row)

        # Export Row: Bitrate Selector & Export Vocals Button
        v_exp_row = QHBoxLayout()
        v_exp_row.setSpacing(10)
        v_fmt_lbl = QLabel("MP3 Bitrate:")
        v_fmt_lbl.setObjectName("FieldLabel")

        self.vocal_format_combo = QComboBox()
        self.vocal_format_combo.setObjectName("FormatSelector")
        self.vocal_format_combo.setItemDelegate(QStyledItemDelegate(self.vocal_format_combo))
        self.vocal_format_combo.addItem("320 kbps (Studio Master)", "320k")
        self.vocal_format_combo.addItem("256 kbps (High Quality)", "256k")
        self.vocal_format_combo.addItem("192 kbps (Standard)", "192k")
        self.vocal_format_combo.addItem("128 kbps (Compact)", "128k")
        self.vocal_format_combo.setCurrentIndex(0)

        self.btn_download_vocals = PremiumButton("💾  Export Vocals (MP3)", variant="primary")
        self.btn_download_vocals.clicked.connect(self._download_vocals_dialog)

        v_exp_row.addWidget(v_fmt_lbl)
        v_exp_row.addWidget(self.vocal_format_combo)
        v_exp_row.addStretch()
        v_exp_row.addWidget(self.btn_download_vocals)
        v_layout.addLayout(v_exp_row)

        self.vocals_card.add_layout(v_layout)
        stems_grid.addWidget(self.vocals_card, 1)

        # CARD 2: INSTRUMENTAL STEM
        self.inst_card = PremiumCard(title="🎸 Instrumental Stem", subtitle="Full karaoke backing track with drums, bass, and instruments")
        i_layout = QVBoxLayout()
        i_layout.setSpacing(12)

        self.inst_waveform = WaveformView()
        self.inst_waveform.setFixedHeight(85)
        self.inst_waveform.seek_requested.connect(self._playback.seek_ratio)
        i_layout.addWidget(self.inst_waveform)

        i_ctrl_row = QHBoxLayout()
        self.btn_play_inst_card = PremiumButton("▶ Play Instrumental", variant="primary")
        self.btn_play_inst_card.clicked.connect(lambda: self._switch_preview("instrumental"))

        self.inst_duration_label = QLabel("00:00")
        self.inst_duration_label.setObjectName("DurationBadge")

        i_ctrl_row.addWidget(self.btn_play_inst_card)
        i_ctrl_row.addWidget(self.inst_duration_label)
        i_ctrl_row.addStretch()
        i_layout.addLayout(i_ctrl_row)

        # Export Row: Bitrate Selector & Export Instrumental Button
        i_exp_row = QHBoxLayout()
        i_exp_row.setSpacing(10)
        i_fmt_lbl = QLabel("MP3 Bitrate:")
        i_fmt_lbl.setObjectName("FieldLabel")

        self.inst_format_combo = QComboBox()
        self.inst_format_combo.setObjectName("FormatSelector")
        self.inst_format_combo.setItemDelegate(QStyledItemDelegate(self.inst_format_combo))
        self.inst_format_combo.addItem("320 kbps (Studio Master)", "320k")
        self.inst_format_combo.addItem("256 kbps (High Quality)", "256k")
        self.inst_format_combo.addItem("192 kbps (Standard)", "192k")
        self.inst_format_combo.addItem("128 kbps (Compact)", "128k")
        self.inst_format_combo.setCurrentIndex(0)

        self.btn_download_inst = PremiumButton("💾  Export Instrumental (MP3)", variant="primary")
        self.btn_download_inst.clicked.connect(self._download_inst_dialog)

        i_exp_row.addWidget(i_fmt_lbl)
        i_exp_row.addWidget(self.inst_format_combo)
        i_exp_row.addStretch()
        i_exp_row.addWidget(self.btn_download_inst)
        i_layout.addLayout(i_exp_row)

        self.inst_card.add_layout(i_layout)
        stems_grid.addWidget(self.inst_card, 1)

        res_layout.addLayout(stems_grid)

        # --- C. STEM AUDIO ENHANCEMENT SECTION ---
        self.enhancement_card = PremiumCard(
            title="STEM AUDIO ENHANCEMENT",
            subtitle="Intelligent loudness normalization, true-peak clipping protection, and independent stem leveling"
        )
        enh_layout = QVBoxLayout()
        enh_layout.setSpacing(14)

        # Toggles Row
        toggles_row = QHBoxLayout()
        self.cb_enable_norm = QCheckBox("Enable Intelligent Loudness Normalisation")
        self.cb_enable_norm.setChecked(True)
        self.cb_enable_norm.setObjectName("EnhanceCheck")

        self.cb_peak_protect = QCheckBox("Output Peak Protection (-1.0 dBTP Ceiling)")
        self.cb_peak_protect.setChecked(True)
        self.cb_peak_protect.setObjectName("EnhanceCheck")

        toggles_row.addWidget(self.cb_enable_norm)
        toggles_row.addWidget(self.cb_peak_protect)
        toggles_row.addStretch()
        enh_layout.addLayout(toggles_row)

        # Controls Grid: Target Loudness & Sliders
        sliders_grid = QGridLayout()
        sliders_grid.setSpacing(16)

        # Target Loudness selector
        tl_box = QVBoxLayout()
        tl_lbl = QLabel("Target Broadcast Loudness:")
        tl_lbl.setObjectName("FieldLabel")
        self.target_loudness_combo = QComboBox()
        self.target_loudness_combo.setItemDelegate(QStyledItemDelegate(self.target_loudness_combo))
        self.target_loudness_combo.addItem("-14 LUFS (Streaming Standard / Full Presence)", -14.0)
        self.target_loudness_combo.addItem("-16 LUFS (Balanced Studio Reference)", -16.0)
        self.target_loudness_combo.addItem("-18 LUFS (Natural Dynamic Mix)", -18.0)
        self.target_loudness_combo.addItem("-1.0 dB Peak Normalisation", "peak")
        tl_box.addWidget(tl_lbl)
        tl_box.addWidget(self.target_loudness_combo)
        sliders_grid.addLayout(tl_box, 0, 0)

        # Vocal Gain Slider
        vg_box = QVBoxLayout()
        self.vocal_gain_label = QLabel("Vocal Level Gain: +0.0 dB")
        self.vocal_gain_label.setObjectName("FieldLabel")
        self.slider_vocal_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_vocal_gain.setRange(-12, 12)
        self.slider_vocal_gain.setValue(0)
        self.slider_vocal_gain.valueChanged.connect(self._on_vocal_gain_slider_changed)
        vg_box.addWidget(self.vocal_gain_label)
        vg_box.addWidget(self.slider_vocal_gain)
        sliders_grid.addLayout(vg_box, 0, 1)

        # Instrumental Gain Slider
        ig_box = QVBoxLayout()
        self.inst_gain_label = QLabel("Instrumental Level Gain: +0.0 dB")
        self.inst_gain_label.setObjectName("FieldLabel")
        self.slider_inst_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_inst_gain.setRange(-12, 12)
        self.slider_inst_gain.setValue(0)
        self.slider_inst_gain.valueChanged.connect(self._on_inst_gain_slider_changed)
        ig_box.addWidget(self.inst_gain_label)
        ig_box.addWidget(self.slider_inst_gain)
        sliders_grid.addLayout(ig_box, 0, 2)

        enh_layout.addLayout(sliders_grid)

        # Level Measurements Grid (Before-and-After)
        self.metrics_frame = QFrame()
        self.metrics_frame.setObjectName("MetricsFrame")
        metrics_layout = QGridLayout(self.metrics_frame)
        metrics_layout.setContentsMargins(14, 12, 14, 12)
        metrics_layout.setSpacing(10)

        m_title = QLabel("Acoustic Loudness & Dynamic Measurements")
        m_title.setObjectName("MetricsHeader")
        metrics_layout.addWidget(m_title, 0, 0, 1, 4)

        self.lbl_orig_meas = QLabel("Original Mix: Peak = -- dBFS | RMS = -- dBFS | Loudness = -- LUFS")
        self.lbl_orig_meas.setObjectName("MetricRow")
        self.lbl_voc_meas = QLabel("Vocals Stem: Raw = -- dBFS (-- LUFS) ➔ Enhanced = -- dBFS (-- LUFS)")
        self.lbl_voc_meas.setObjectName("MetricRow")
        self.lbl_inst_meas = QLabel("Instrumental: Raw = -- dBFS (-- LUFS) ➔ Enhanced = -- dBFS (-- LUFS)")
        self.lbl_inst_meas.setObjectName("MetricRow")

        metrics_layout.addWidget(self.lbl_orig_meas, 1, 0, 1, 4)
        metrics_layout.addWidget(self.lbl_voc_meas, 2, 0, 1, 4)
        metrics_layout.addWidget(self.lbl_inst_meas, 3, 0, 1, 4)

        enh_layout.addWidget(self.metrics_frame)

        # Action bar on Enhancement Card: Apply/Re-level & Preview Mode
        enh_action_row = QHBoxLayout()
        self.cb_preview_mode = QCheckBox("Audition Enhanced Loudness (Uncheck to hear raw unnormalized stems)")
        self.cb_preview_mode.setChecked(True)
        self.cb_preview_mode.toggled.connect(self._toggle_preview_mode)

        self.btn_apply_enhancement = PremiumButton("⚡  Apply Enhancement & Re-level Stems", variant="primary")
        self.btn_apply_enhancement.clicked.connect(self._reapply_enhancement)

        enh_action_row.addWidget(self.cb_preview_mode)
        enh_action_row.addStretch()
        enh_action_row.addWidget(self.btn_apply_enhancement)
        enh_layout.addLayout(enh_action_row)

        self.enhancement_card.add_layout(enh_layout)
        res_layout.addWidget(self.enhancement_card)

        self.content_layout.addWidget(self.results_container)

        self.scroll.setWidget(self.container)
        main_layout.addWidget(self.scroll)

    def _connect_signals(self):
        self._playback.state_changed.connect(self._on_playback_state_changed)
        self._playback.position_changed.connect(self._on_playback_position_changed)

    def _load_audio_file(self, fpath: str):
        path = Path(fpath)
        if not path.exists():
            return

        if path.suffix.lower() != ".mp3":
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.warning(self, "Invalid Format", "Only MP3 audio files are supported.")
            return

        is_valid, err = validate_mp3_file(str(path))
        if not is_valid:
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.warning(self, "Invalid Format", f"Only MP3 audio files are supported.\n\n{err}")
            return

        self._loaded_file_path = str(path.resolve())
        self._stem_base_name = path.stem

        stat = path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        fmt = path.suffix.upper().replace(".", "")

        dur_sec = 0.0
        try:
            import soundfile as sf
            info = sf.info(self._loaded_file_path)
            dur_sec = info.duration
            self._loaded_sr = info.samplerate
        except Exception:
            dur_sec = 180.0

        self._loaded_duration = dur_sec

        self.track_title_label.setText(path.name)
        self.track_meta_label.setText(f"•  {fmt}  •  {format_duration_sec(dur_sec)}  •  {size_mb:.2f} MB  •  {self._loaded_sr} Hz")

        peaks = WaveformEngine.load_peaks(self._loaded_file_path)
        if peaks is not None:
            self.original_waveform.set_peaks(peaks)

        self.drop_card.setVisible(False)
        self.track_card.setVisible(True)
        self.btn_remove_vocals.setEnabled(True)
        self.status_tip.setText("Ready to separate. Click REMOVE VOCALS to initiate neural extraction.")

        self.results_container.setVisible(False)
        self._vocals_path = None
        self._instrumental_path = None
        self._vocals_raw_path = None
        self._instrumental_raw_path = None

    def _prompt_replace_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select New MP3 Audio File",
            "",
            AUDIO_FILE_FILTER
        )
        if fpath:
            self._load_audio_file(fpath)

    def _select_dest_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder for Stems", self.dest_folder_label.text())
        if folder:
            self.dest_folder_label.setText(folder)

    def _on_vocal_gain_slider_changed(self, val: int):
        sign = "+" if val > 0 else ""
        self.vocal_gain_label.setText(f"Vocal Level Gain: {sign}{val}.0 dB")

    def _on_inst_gain_slider_changed(self, val: int):
        sign = "+" if val > 0 else ""
        self.inst_gain_label.setText(f"Instrumental Level Gain: {sign}{val}.0 dB")

    def _start_separation(self):
        if not self._loaded_file_path:
            return

        model_name = self.model_combo.currentData() or "htdemucs"
        quality_mode = self.quality_combo.currentData() or "balanced"
        device_mode = self.device_combo.currentData() or "auto"
        output_dir = self.dest_folder_label.text()

        enable_norm = self.cb_enable_norm.isChecked()
        norm_val = self.target_loudness_combo.currentData()
        norm_mode = "peak" if norm_val == "peak" else "lufs"
        target_lufs = -14.0 if norm_val == "peak" else float(norm_val)
        peak_protect = self.cb_peak_protect.isChecked()
        voc_gain = float(self.slider_vocal_gain.value())
        inst_gain = float(self.slider_inst_gain.value())

        self.btn_remove_vocals.setEnabled(False)
        self.btn_replace.setEnabled(False)
        self.progress_card.setVisible(True)
        self.progress_bar.setValue(0)
        self.prog_pct_label.setText("0%")
        self.prog_stage_label.setText("Starting background separation worker...")
        self.results_container.setVisible(False)

        self._playback.stop()

        self._worker = SeparationWorker(
            file_path=self._loaded_file_path,
            output_dir=output_dir,
            model_name=model_name,
            quality_mode=quality_mode,
            device=device_mode,
            enable_normalization=enable_norm,
            norm_mode=norm_mode,
            target_lufs=target_lufs,
            target_peak_db=-1.0,
            vocal_gain_db=voc_gain,
            inst_gain_db=inst_gain,
            peak_protection=peak_protect,
            parent=self
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _cancel_separation(self):
        if self._worker and self._worker.isRunning():
            self.prog_stage_label.setText("Cancelling separation...")
            self.btn_cancel.setEnabled(False)
            self._worker.cancel()

    def _on_worker_progress(self, pct: float, stage: str):
        val = int(pct * 100)
        self.progress_bar.setValue(val)
        self.prog_pct_label.setText(f"{val}%")
        self.prog_stage_label.setText(stage)

    def _on_worker_completed(self, result: Dict[str, Any]):
        self.progress_card.setVisible(False)
        self.btn_remove_vocals.setEnabled(True)
        self.btn_replace.setEnabled(True)
        self.btn_cancel.setEnabled(True)

        self._vocals_path = result["vocals_path"]
        self._instrumental_path = result["instrumental_path"]
        self._vocals_raw_path = result.get("vocals_raw_path", self._vocals_path)
        self._instrumental_raw_path = result.get("instrumental_raw_path", self._instrumental_path)
        self._latest_metrics = result.get("metrics", {})

        dur = result.get("duration", self._loaded_duration)

        self._refresh_waveforms()
        self.vocal_duration_label.setText(format_duration_sec(dur))
        self.inst_duration_label.setText(format_duration_sec(dur))

        self._update_metrics_display()

        self.results_container.setVisible(True)
        self._anim_mgr.slide_fade_in(self.results_container, duration=240)
        self.status_tip.setText("Separation complete! Preview and export isolated stems below.")

        # Automatically scroll down so the results and format selectors are front and center
        QTimer.singleShot(100, lambda: self.scroll.verticalScrollBar().setValue(self.results_container.y()))

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(
                self,
                "Separation Complete",
                f"Successfully separated into 2 independent stems using {result.get('model_used', 'HTDemucs v4')} "
                f"on {result.get('device_used', 'CPU')}!\n\n"
                f"• Vocals: {Path(self._vocals_path).name}\n"
                f"• Instrumental: {Path(self._instrumental_path).name}"
            )

    def _on_worker_failed(self, error_msg: str):
        self.progress_card.setVisible(False)
        self.btn_remove_vocals.setEnabled(True)
        self.btn_replace.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.status_tip.setText("Separation failed. Check error details.")

        QMessageBox.critical(
            self,
            "Separation Error",
            f"An error occurred during AI stem separation:\n\n{error_msg}\n\nPlease verify available system memory and audio format."
        )

    def _on_worker_cancelled(self):
        self.progress_card.setVisible(False)
        self.btn_remove_vocals.setEnabled(True)
        self.btn_replace.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.status_tip.setText("Separation was cancelled.")

    def _refresh_waveforms(self):
        active_voc = self._vocals_path if self._preview_normalized_mode else self._vocals_raw_path
        active_inst = self._instrumental_path if self._preview_normalized_mode else self._instrumental_raw_path

        if active_voc:
            v_peaks = WaveformEngine.load_peaks(active_voc)
            if v_peaks is not None:
                self.vocals_waveform.set_peaks(v_peaks)

        if active_inst:
            i_peaks = WaveformEngine.load_peaks(active_inst)
            if i_peaks is not None:
                self.inst_waveform.set_peaks(i_peaks)

    def _update_metrics_display(self):
        m = self._latest_metrics
        if not m:
            return

        orig = m.get("original", {})
        vr = m.get("vocals_raw", {})
        vn = m.get("vocals_norm", {})
        vg = m.get("vocals_gain_applied_db", 0.0)

        ir = m.get("instrumental_raw", {})
        inorm = m.get("instrumental_norm", {})
        ig = m.get("instrumental_gain_applied_db", 0.0)

        self.lbl_orig_meas.setText(
            f"Original Mix: Peak = {orig.get('peak_db', '--')} dBFS  |  "
            f"RMS = {orig.get('rms_db', '--')} dBFS  |  "
            f"Loudness = {orig.get('lufs', '--')} LUFS"
        )
        self.lbl_voc_meas.setText(
            f"Vocals Stem: Raw Peak = {vr.get('peak_db', '--')} dBFS ({vr.get('lufs', '--')} LUFS)  ➔  "
            f"Enhanced Peak = {vn.get('peak_db', '--')} dBFS ({vn.get('lufs', '--')} LUFS)  [Gain: {vg:+.1f} dB]"
        )
        self.lbl_inst_meas.setText(
            f"Instrumental: Raw Peak = {ir.get('peak_db', '--')} dBFS ({ir.get('lufs', '--')} LUFS)  ➔  "
            f"Enhanced Peak = {inorm.get('peak_db', '--')} dBFS ({inorm.get('lufs', '--')} LUFS)  [Gain: {ig:+.1f} dB]"
        )

    def _toggle_preview_mode(self, checked: bool):
        self._preview_normalized_mode = checked
        self._refresh_waveforms()

        if self._currently_playing_stem in ("vocals", "instrumental"):
            curr_pos = self._playback._player.position()
            self._switch_preview(self._currently_playing_stem)
            self._playback.seek_ms(curr_pos)

    def _reapply_enhancement(self):
        """Re-runs instant normalization on raw separated stems without re-running Demucs inference."""
        if not self._vocals_raw_path or not self._instrumental_raw_path:
            return

        enable_norm = self.cb_enable_norm.isChecked()
        norm_val = self.target_loudness_combo.currentData()
        norm_mode = "peak" if norm_val == "peak" else "lufs"
        target_lufs = -14.0 if norm_val == "peak" else float(norm_val)
        peak_protect = self.cb_peak_protect.isChecked()
        voc_gain = float(self.slider_vocal_gain.value())
        inst_gain = float(self.slider_inst_gain.value())

        voc_tensor, sr, _ = SeparationEngine.load_audio(self._vocals_raw_path)
        inst_tensor, _, _ = SeparationEngine.load_audio(self._instrumental_raw_path)

        norm_voc, voc_meta = AudioNormalizer.normalize_stem(
            voc_tensor.numpy().T,
            sr=sr,
            enable_norm=enable_norm,
            mode=norm_mode,
            target_lufs=target_lufs,
            target_peak_db=-1.0,
            user_gain_db=voc_gain,
            peak_protection=peak_protect
        )

        norm_inst, inst_meta = AudioNormalizer.normalize_stem(
            inst_tensor.numpy().T,
            sr=sr,
            enable_norm=enable_norm,
            mode=norm_mode,
            target_lufs=target_lufs,
            target_peak_db=-1.0,
            user_gain_db=inst_gain,
            peak_protection=peak_protect
        )

        SeparationEngine.save_audio(norm_voc, self._vocals_path, sr=sr, format_type="wav")
        SeparationEngine.save_audio(norm_inst, self._instrumental_path, sr=sr, format_type="wav")

        self._latest_metrics["vocals_norm"] = voc_meta["norm_metrics"]
        self._latest_metrics["vocals_gain_applied_db"] = voc_meta["gain_applied_db"]
        self._latest_metrics["instrumental_norm"] = inst_meta["norm_metrics"]
        self._latest_metrics["instrumental_gain_applied_db"] = inst_meta["gain_applied_db"]

        self._update_metrics_display()
        self._refresh_waveforms()

        if self._currently_playing_stem in ("vocals", "instrumental"):
            pos = self._playback._player.position()
            self._switch_preview(self._currently_playing_stem)
            self._playback.seek_ms(pos)

        self.status_tip.setText("Stems re-normalized and leveled successfully!")

    # --- Preview & Comparison Management ---
    def _switch_preview(self, stem: str):
        self._currently_playing_stem = stem

        self.btn_solo_original._variant = "primary" if stem == "original" else "secondary"
        self.btn_solo_vocals._variant = "primary" if stem == "vocals" else "secondary"
        self.btn_solo_inst._variant = "primary" if stem == "instrumental" else "secondary"
        self.btn_solo_original._update_style()
        self.btn_solo_vocals._update_style()
        self.btn_solo_inst._update_style()

        active_voc = self._vocals_path if self._preview_normalized_mode else self._vocals_raw_path
        active_inst = self._instrumental_path if self._preview_normalized_mode else self._instrumental_raw_path

        curr_ms = self._playback._player.position()

        if stem == "original" and self._loaded_file_path:
            self._playback.play_file(self._loaded_file_path, title=f"{self._stem_base_name} (Original)", artist="Original Mix")
        elif stem == "vocals" and active_voc:
            label = "Enhanced Vocals" if self._preview_normalized_mode else "Raw Vocals"
            self._playback.play_file(active_voc, title=f"{self._stem_base_name} (Vocals)", artist=label)
        elif stem == "instrumental" and active_inst:
            label = "Enhanced Instrumental" if self._preview_normalized_mode else "Raw Instrumental"
            self._playback.play_file(active_inst, title=f"{self._stem_base_name} (Instrumental)", artist=label)

        if curr_ms > 0:
            self._playback.seek_ms(curr_ms)

    def _toggle_play_original(self):
        if self._loaded_file_path:
            if self._currently_playing_stem == "original" and self._playback.is_playing:
                self._playback.pause()
            else:
                self._switch_preview("original")

    def _on_playback_state_changed(self, is_playing: bool):
        play_icon = "⏸" if is_playing else "▶"
        if self._currently_playing_stem == "original":
            self.btn_play_original.setText(f"{play_icon} Play Original")
            self.btn_play_vocal_card.setText("▶ Play Vocals")
            self.btn_play_inst_card.setText("▶ Play Instrumental")
        elif self._currently_playing_stem == "vocals":
            self.btn_play_vocal_card.setText(f"{play_icon} Play Vocals")
            self.btn_play_original.setText("▶ Play Original")
            self.btn_play_inst_card.setText("▶ Play Instrumental")
        elif self._currently_playing_stem == "instrumental":
            self.btn_play_inst_card.setText(f"{play_icon} Play Instrumental")
            self.btn_play_original.setText("▶ Play Original")
            self.btn_play_vocal_card.setText("▶ Play Vocals")
        else:
            self.btn_play_original.setText("▶ Play Original")
            self.btn_play_vocal_card.setText("▶ Play Vocals")
            self.btn_play_inst_card.setText("▶ Play Instrumental")

    def _on_playback_position_changed(self, curr_ms: int, dur_ms: int):
        if dur_ms > 0:
            ratio = curr_ms / float(dur_ms)
            if self._currently_playing_stem == "original":
                self.original_waveform.set_progress(ratio)
            elif self._currently_playing_stem == "vocals":
                self.vocals_waveform.set_progress(ratio)
            elif self._currently_playing_stem == "instrumental":
                self.inst_waveform.set_progress(ratio)

    # --- Export & Download Handlers (Strictly MP3) ---
    def _download_vocals_dialog(self):
        if not self._vocals_path:
            return
        bitrate = self.vocal_format_combo.currentData() or "320k"
        dest_folder = self.dest_folder_label.text()
        default_file = str(Path(dest_folder) / f"{self._stem_base_name}_Vocals.mp3")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Vocals Stem (MP3 @ {bitrate})",
            default_file,
            AUDIO_FILE_FILTER
        )
        if save_path:
            active_src = self._vocals_path if self._preview_normalized_mode else self._vocals_raw_path
            final_path = self._export_stem_file(active_src, save_path, bitrate=bitrate)
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Vocals stem exported successfully!\n\nFormat: MP3 ({bitrate})\nFile: {final_path}"
                )

    def _download_inst_dialog(self):
        if not self._instrumental_path:
            return
        bitrate = self.inst_format_combo.currentData() or "320k"
        dest_folder = self.dest_folder_label.text()
        default_file = str(Path(dest_folder) / f"{self._stem_base_name}_Instrumental.mp3")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Instrumental Stem (MP3 @ {bitrate})",
            default_file,
            AUDIO_FILE_FILTER
        )
        if save_path:
            active_src = self._instrumental_path if self._preview_normalized_mode else self._instrumental_raw_path
            final_path = self._export_stem_file(active_src, save_path, bitrate=bitrate)
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Instrumental stem exported successfully!\n\nFormat: MP3 ({bitrate})\nFile: {final_path}"
                )

    def _download_both_action(self):
        if not self._vocals_path or not self._instrumental_path:
            return

        bitrate = self.both_format_combo.currentData() or "320k"
        folder = QFileDialog.getExistingDirectory(
            self,
            f"Select Destination Folder for Both Stems (MP3 @ {bitrate})",
            self.dest_folder_label.text()
        )
        if folder:
            dest_dir = Path(folder)
            v_target = str(dest_dir / f"{self._stem_base_name}_Vocals.mp3")
            i_target = str(dest_dir / f"{self._stem_base_name}_Instrumental.mp3")

            active_voc = self._vocals_path if self._preview_normalized_mode else self._vocals_raw_path
            active_inst = self._instrumental_path if self._preview_normalized_mode else self._instrumental_raw_path

            v_final = self._export_stem_file(active_voc, v_target, bitrate=bitrate)
            i_final = self._export_stem_file(active_inst, i_target, bitrate=bitrate)

            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Both stems successfully exported in MP3 ({bitrate}) format!\n\n"
                    f"Destination Folder: {folder}\n\n"
                    f"• {Path(v_final).name}\n"
                    f"• {Path(i_final).name}"
                )

    def _export_stem_file(self, src_wav_path: str, dest_path: str, bitrate: str = "320k") -> str:
        dest = Path(dest_path)
        if dest.suffix.lower() != ".mp3":
            dest = dest.with_suffix(".mp3")

        dest.parent.mkdir(parents=True, exist_ok=True)
        data, sr = decode_mp3_to_numpy(src_wav_path)
        final_saved = encode_array_to_mp3(data, str(dest), sample_rate=sr, bitrate=bitrate)
        return final_saved

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        border_col = c.get("border_default", "#30333A")
        border_subtle = c.get("border_subtle", "#25282F")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")
        success = c.get("success", "#48B58A")
        bg_card = c.get("background_card", "#1B1D21")
        bg_elevated = c.get("background_elevated", "#22252A")
        bg_input = c.get("background_input", "#181A1E")

        self.drop_zone.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_card};
                border: 2px dashed {border_col};
                border-radius: 12px;
            }}
            QFrame:hover {{
                border-color: {accent};
            }}
            QLabel#DropText {{
                color: {text_primary};
                font-size: 14px;
                font-weight: 600;
            }}
            QLabel#DropSubtext {{
                color: {text_secondary};
                font-size: 12px;
            }}
        """)

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
            QLabel#DeviceTag {{
                color: {success if CUDA_AVAILABLE else accent};
                font-size: 12px;
                font-weight: 600;
                background-color: rgba(255, 255, 255, 0.05);
                padding: 6px 12px;
                border-radius: 6px;
                border: 1px solid {border_subtle};
            }}
            QLabel#FieldLabel {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#TrackTitleHighlight {{
                color: {text_primary};
                font-size: 14px;
                font-weight: 600;
            }}
            QLabel#TrackMetaDetails {{
                color: {text_secondary};
                font-size: 12px;
            }}
            QLabel#DestFolderLabel {{
                color: {accent};
                font-size: 12px;
                font-family: Consolas, monospace;
            }}
            QLabel#StatusNotice {{
                color: {text_secondary};
                font-size: 13px;
            }}
            QLabel#ProcessingTitle {{
                color: {text_primary};
                font-size: 15px;
                font-weight: 600;
            }}
            QLabel#ProcessingPercent {{
                color: {accent};
                font-size: 16px;
                font-weight: 700;
                font-family: Consolas, monospace;
            }}
            QLabel#ProgressStage {{
                color: {text_secondary};
                font-size: 12px;
                font-style: italic;
            }}
            QLabel#DurationBadge {{
                color: {text_secondary};
                font-size: 12px;
                font-family: Consolas, monospace;
                padding-left: 6px;
            }}
            QProgressBar#SeparationProgressBar {{
                background-color: {bg_elevated};
                border: 1px solid {border_subtle};
                border-radius: 4px;
            }}
            QProgressBar#SeparationProgressBar::chunk {{
                background-color: {accent};
                border-radius: 3px;
            }}
            QFrame#MetricsFrame {{
                background-color: #15171B;
                border: 1px solid {border_col};
                border-radius: 8px;
            }}
            QLabel#MetricsHeader {{
                color: {accent};
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            QLabel#MetricRow {{
                color: {text_primary};
                font-size: 12px;
                font-family: 'Consolas', monospace;
            }}
            QCheckBox#EnhanceCheck {{
                color: {text_primary};
                font-size: 13px;
                font-weight: 500;
            }}
            QComboBox#FormatSelector {{
                background-color: {bg_input};
                color: {text_primary};
                border: 1px solid {border_col};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 9.5pt;
                font-weight: 600;
                min-width: 110px;
            }}
            QComboBox#FormatSelector:hover {{
                border-color: {accent};
            }}
        """)
