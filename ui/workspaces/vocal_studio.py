"""Vocal Studio Workspace for Audio Studio AI.
Advanced vocal editing workspace featuring waveform inspection, AI spectral noise reduction,
independent pitch shifting, independent time stretching, clarity/formant enhancement,
broadcast loudness normalisation (-14 LUFS, -1.0 dBTP ceiling),
seamless A/B auditioning, undo/redo history, and MP3 export at 128k-320k.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QFileDialog, QFrame, QScrollArea, QGridLayout, QComboBox,
    QProgressBar, QMessageBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.waveform_view import WaveformView
from core.playback_engine import PlaybackEngine
from core.waveform_engine import WaveformEngine
from config import AUDIO_FILE_FILTER, AVAILABLE_MP3_BITRATES, DEFAULT_EXPORT_BITRATE, CACHE_DIR
from core.audio_format import validate_mp3_file, encode_array_to_mp3, decode_mp3_to_numpy
from workers.enhancer_worker import EnhancerWorker

class VocalStudioWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()

        self._original_path: Optional[str] = None
        self._processed_path: Optional[str] = None
        self._is_ab_b_mode: bool = False  # False = Original (A), True = Processed (B)

        # Undo / Redo History
        self._undo_stack: List[str] = []
        self._redo_stack: List[str] = []

        # Background worker
        self._worker: Optional[EnhancerWorker] = None

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

        # Header Bar
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Vocal Studio")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("AI vocal enhancement, background noise suppression, pitch shifting, and time stretching.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        self.engine_mode_combo = QComboBox()
        self.engine_mode_combo.setItemDelegate(QStyledItemDelegate(self.engine_mode_combo))
        self.engine_mode_combo.addItem("Neural AI Speech Enhancement (PyTorch Deep Spectral Masking)", "neural")
        self.engine_mode_combo.addItem("Studio DSP Enhancement (FFmpeg Spectral afftdn + Rubberband)", "dsp")
        self.engine_mode_combo.setToolTip("Select whether to use the Deep Neural AI enhancement model or traditional studio DSP")
        header_layout.addWidget(self.engine_mode_combo)

        self.btn_master_enhance = PremiumButton("✨  One-Click AI Audio Enhance", variant="primary")
        self.btn_master_enhance.setEnabled(False)
        self.btn_master_enhance.clicked.connect(self._run_one_click_enhance)
        header_layout.addWidget(self.btn_master_enhance)

        self.btn_load_vocal = PremiumButton("📂  Load Vocal File (MP3)", variant="secondary")
        self.btn_load_vocal.clicked.connect(self._browse_vocal_file)
        header_layout.addWidget(self.btn_load_vocal)

        layout.addLayout(header_layout)

        # Waveform Editor Card
        self.waveform_card = PremiumCard(title="Vocal Waveform Editor", subtitle="Interactive timeline and comparison")
        self.waveform_view = WaveformView()
        self.waveform_view.setFixedHeight(90)
        self.waveform_view.seek_requested.connect(self._playback.seek_ratio)

        # Waveform toolbar (Play/Pause, A/B toggle, Undo, Redo)
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(12)

        self.btn_play = PremiumButton("▶ Play Track", variant="secondary")
        self.btn_play.clicked.connect(self._toggle_playback)

        self.btn_ab_toggle = PremiumButton("A/B: Original (A)", variant="outline")
        self.btn_ab_toggle.setEnabled(False)
        self.btn_ab_toggle.clicked.connect(self._toggle_ab)

        self.btn_undo = PremiumButton("↺ Undo", variant="ghost")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self._undo)

        self.btn_redo = PremiumButton("↻ Redo", variant="ghost")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self._redo)

        self.loaded_label = QLabel("No vocal track loaded (MP3 only)")
        self.loaded_label.setObjectName("VocalStatusText")

        toolbar_row.addWidget(self.btn_play)
        toolbar_row.addWidget(self.btn_ab_toggle)
        toolbar_row.addWidget(self.btn_undo)
        toolbar_row.addWidget(self.btn_redo)
        toolbar_row.addStretch()
        toolbar_row.addWidget(self.loaded_label)

        self.waveform_card.add_layout(toolbar_row)
        self.waveform_card.add_widget(self.waveform_view)
        layout.addWidget(self.waveform_card)

        # Progress Card (Hidden initially, shown only during processing)
        self.progress_card = PremiumCard(elevated=True)
        self.progress_card.setVisible(False)
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(10)

        prog_hdr = QHBoxLayout()
        self.prog_title = QLabel("AI Audio Processing...")
        self.prog_title.setObjectName("FieldLabel")
        self.prog_pct = QLabel("0%")
        self.prog_pct.setObjectName("FieldLabel")
        prog_hdr.addWidget(self.prog_title)
        prog_hdr.addStretch()
        prog_hdr.addWidget(self.prog_pct)
        prog_layout.addLayout(prog_hdr)

        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(8)
        self.prog_bar.setTextVisible(False)
        prog_layout.addWidget(self.prog_bar)

        prog_ftr = QHBoxLayout()
        self.prog_stage_lbl = QLabel("Initializing...")
        self.prog_stage_lbl.setObjectName("VocalStatusText")
        self.btn_cancel = PremiumButton("✕ Cancel", variant="danger")
        self.btn_cancel.setFixedHeight(28)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        prog_ftr.addWidget(self.prog_stage_lbl, 1)
        prog_ftr.addWidget(self.btn_cancel)
        prog_layout.addLayout(prog_ftr)

        self.progress_card.add_layout(prog_layout)
        layout.addWidget(self.progress_card)

        # Processing Panels (2x2 Grid)
        panels_grid = QGridLayout()
        panels_grid.setSpacing(16)

        # 1. Background Noise Reduction
        denoise_card = PremiumCard(title="🛡️ AI Noise Reduction", subtitle="Adaptive spectral ambient suppression")
        dn_layout = QVBoxLayout()
        dn_layout.setSpacing(10)

        self.dn_slider_label = QLabel("Suppression Intensity: 65%")
        self.dn_slider = QSlider(Qt.Orientation.Horizontal)
        self.dn_slider.setRange(0, 100)
        self.dn_slider.setValue(65)
        self.dn_slider.valueChanged.connect(lambda v: self.dn_slider_label.setText(f"Suppression Intensity: {v}%"))

        self.btn_apply_denoise = PremiumButton("Apply Noise Reduction", variant="secondary")
        self.btn_apply_denoise.setEnabled(False)
        self.btn_apply_denoise.clicked.connect(self._apply_denoise)

        dn_layout.addWidget(self.dn_slider_label)
        dn_layout.addWidget(self.dn_slider)
        dn_layout.addWidget(self.btn_apply_denoise)
        denoise_card.add_layout(dn_layout)
        panels_grid.addWidget(denoise_card, 0, 0)

        # 2. Independent Pitch Shifting (Without changing speed)
        pitch_card = PremiumCard(title="🎵 Pitch Shifting", subtitle="Shift pitch without modifying tempo")
        p_layout = QVBoxLayout()
        p_layout.setSpacing(10)

        self.pitch_label = QLabel("Semitones: 0 (Original Key)")
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(self._on_pitch_changed)

        self.btn_apply_pitch = PremiumButton("Apply Pitch Shift", variant="secondary")
        self.btn_apply_pitch.setEnabled(False)
        self.btn_apply_pitch.clicked.connect(self._apply_pitch_shift)

        p_layout.addWidget(self.pitch_label)
        p_layout.addWidget(self.pitch_slider)
        p_layout.addWidget(self.btn_apply_pitch)
        pitch_card.add_layout(p_layout)
        panels_grid.addWidget(pitch_card, 0, 1)

        # 3. Independent Time Stretching (Without changing pitch)
        speed_card = PremiumCard(title="⏱️ Time Stretching", subtitle="Change tempo without altering pitch")
        s_layout = QVBoxLayout()
        s_layout.setSpacing(10)

        self.stretch_label = QLabel("Tempo: 100% (1.00x)")
        self.stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.stretch_slider.setRange(50, 150)
        self.stretch_slider.setValue(100)
        self.stretch_slider.valueChanged.connect(lambda v: self.stretch_label.setText(f"Tempo: {v}% ({v/100:.2f}x)"))

        self.btn_apply_stretch = PremiumButton("Apply Time Stretch", variant="secondary")
        self.btn_apply_stretch.setEnabled(False)
        self.btn_apply_stretch.clicked.connect(self._apply_time_stretch)

        s_layout.addWidget(self.stretch_label)
        s_layout.addWidget(self.stretch_slider)
        s_layout.addWidget(self.btn_apply_stretch)
        speed_card.add_layout(s_layout)
        panels_grid.addWidget(speed_card, 1, 0)

        # 4. Vocal Enhancement & Envelopes
        enh_card = PremiumCard(title="✨ Vocal Enhancer & Dynamics", subtitle="Clarity booster & presence shaping")
        e_layout = QVBoxLayout()
        e_layout.setSpacing(10)

        self.gain_label = QLabel("Clarity / Presence Boost: +3.0 dB")
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-12, 12)
        self.gain_slider.setValue(3)
        self.gain_slider.valueChanged.connect(lambda v: self.gain_label.setText(f"Clarity / Presence Boost: {'+' if v>0 else ''}{v}.0 dB"))

        self.btn_apply_enh = PremiumButton("Apply Enhancement", variant="secondary")
        self.btn_apply_enh.setEnabled(False)
        self.btn_apply_enh.clicked.connect(self._apply_vocal_enhancement)

        e_layout.addWidget(self.gain_label)
        e_layout.addWidget(self.gain_slider)
        e_layout.addWidget(self.btn_apply_enh)
        enh_card.add_layout(e_layout)
        panels_grid.addWidget(enh_card, 1, 1)

        layout.addLayout(panels_grid)

        # Export Row (Strictly MP3)
        export_card = PremiumCard()
        export_row = QHBoxLayout()
        export_row.setSpacing(14)

        bitrate_lbl = QLabel("MP3 Bitrate:")
        bitrate_lbl.setObjectName("FieldLabel")
        export_row.addWidget(bitrate_lbl)

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setItemDelegate(QStyledItemDelegate(self.bitrate_combo))
        self.bitrate_combo.addItem("320 kbps (Studio Master)", "320k")
        self.bitrate_combo.addItem("256 kbps (High Quality)", "256k")
        self.bitrate_combo.addItem("192 kbps (Standard)", "192k")
        self.bitrate_combo.addItem("128 kbps (Compact)", "128k")
        self.bitrate_combo.setCurrentIndex(0)
        export_row.addWidget(self.bitrate_combo)

        export_row.addStretch()

        self.btn_export = PremiumButton("💾  Export Mastered Vocals (MP3)", variant="primary")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_audio)
        export_row.addWidget(self.btn_export)

        export_card.add_layout(export_row)
        layout.addWidget(export_card)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self._playback.state_changed.connect(self._on_playback_state_changed)
        self._playback.position_changed.connect(self._on_playback_position_changed)

    def _browse_vocal_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vocal Audio File (MP3)",
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

            self._original_path = str(Path(fpath).resolve())
            self._processed_path = self._original_path
            self._is_ab_b_mode = False
            self._undo_stack.clear()
            self._redo_stack.clear()

            p = Path(fpath)
            self.loaded_label.setText(f"Loaded: {p.name} (Original Mix)")
            peaks = WaveformEngine.load_peaks(fpath)
            if peaks is not None:
                self.waveform_view.set_peaks(peaks)

            # Enable action buttons
            self.btn_master_enhance.setEnabled(True)
            self.btn_apply_denoise.setEnabled(True)
            self.btn_apply_pitch.setEnabled(True)
            self.btn_apply_stretch.setEnabled(True)
            self.btn_apply_enh.setEnabled(True)
            self.btn_ab_toggle.setEnabled(True)
            self.btn_ab_toggle.setText("A/B: Original (A)")
            self.btn_export.setEnabled(True)
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)

    def _toggle_playback(self):
        active_file = self._processed_path if self._is_ab_b_mode else self._original_path
        if active_file:
            if self._playback.is_playing:
                self._playback.pause()
            else:
                label = "Processed (B)" if self._is_ab_b_mode else "Original (A)"
                self._playback.play_file(active_file, title=Path(active_file).stem, artist=f"Vocal Studio ({label})")

    def _toggle_ab(self):
        if not self._processed_path or not self._original_path:
            return
        self._is_ab_b_mode = not self._is_ab_b_mode
        curr_ms = self._playback._player.position()
        was_playing = self._playback.is_playing

        if self._is_ab_b_mode:
            self.btn_ab_toggle.setText("A/B: Processed (B)")
            self.btn_ab_toggle._variant = "primary"
            self.loaded_label.setText(f"Auditioning: Processed (B)")
            active_file = self._processed_path
        else:
            self.btn_ab_toggle.setText("A/B: Original (A)")
            self.btn_ab_toggle._variant = "outline"
            self.loaded_label.setText(f"Auditioning: Original (A)")
            active_file = self._original_path

        self.btn_ab_toggle._update_style()

        peaks = WaveformEngine.load_peaks(active_file)
        if peaks is not None:
            self.waveform_view.set_peaks(peaks)

        if was_playing:
            label = "Processed (B)" if self._is_ab_b_mode else "Original (A)"
            self._playback.play_file(active_file, title=Path(active_file).stem, artist=f"Vocal Studio ({label})")
            if curr_ms > 0:
                self._playback.seek_ms(curr_ms)

    def _on_playback_state_changed(self, is_playing: bool):
        self.btn_play.setText("⏸ Pause" if is_playing else "▶ Play Track")

    def _on_playback_position_changed(self, curr_ms: int, dur_ms: int):
        if dur_ms > 0:
            self.waveform_view.set_progress(curr_ms / float(dur_ms))

    def _on_pitch_changed(self, val: int):
        sign = "+" if val > 0 else ""
        self.pitch_label.setText(f"Semitones: {sign}{val} {'(Original Key)' if val == 0 else ''}")

    # --- Processing Actions ---
    def _run_one_click_enhance(self):
        if not self._original_path:
            return
        out_p = Path(CACHE_DIR) / f"vocal_master_enhanced.mp3"
        self._start_worker(
            input_path=self._processed_path or self._original_path,
            output_path=str(out_p),
            is_one_click=True
        )

    def _apply_denoise(self):
        if not self._original_path:
            return
        intensity = self.dn_slider.value() / 100.0
        out_p = Path(CACHE_DIR) / f"vocal_denoised.mp3"
        self._start_worker(
            input_path=self._processed_path or self._original_path,
            output_path=str(out_p),
            noise_reduction=intensity
        )

    def _apply_pitch_shift(self):
        if not self._original_path:
            return
        semi = self.pitch_slider.value()
        out_p = Path(CACHE_DIR) / f"vocal_pitch_{semi}.mp3"
        self._start_worker(
            input_path=self._processed_path or self._original_path,
            output_path=str(out_p),
            pitch_semitones=semi
        )

    def _apply_time_stretch(self):
        if not self._original_path:
            return
        tempo = self.stretch_slider.value() / 100.0
        out_p = Path(CACHE_DIR) / f"vocal_tempo_{int(tempo*100)}.mp3"
        self._start_worker(
            input_path=self._processed_path or self._original_path,
            output_path=str(out_p),
            tempo_ratio=tempo
        )

    def _apply_vocal_enhancement(self):
        if not self._original_path:
            return
        clarity = float(self.gain_slider.value())
        out_p = Path(CACHE_DIR) / f"vocal_clarity_{int(clarity)}.mp3"
        self._start_worker(
            input_path=self._processed_path or self._original_path,
            output_path=str(out_p),
            clarity_boost_db=clarity
        )

    def _start_worker(
        self,
        input_path: str,
        output_path: str,
        noise_reduction: float = 0.0,
        pitch_semitones: int = 0,
        tempo_ratio: float = 1.0,
        clarity_boost_db: float = 0.0,
        is_one_click: bool = False
    ):
        bitrate = self.bitrate_combo.currentData() or "320k"
        mode = self.engine_mode_combo.currentData() or "neural"
        self._playback.stop()

        self.progress_card.setVisible(True)
        self.prog_bar.setValue(0)
        self.prog_pct.setText("0%")
        self.prog_stage_lbl.setText("Starting enhancement worker...")
        self._set_buttons_enabled(False)

        self._worker = EnhancerWorker(
            input_path=input_path,
            output_path=output_path,
            mode=mode,
            noise_reduction=noise_reduction,
            pitch_semitones=pitch_semitones,
            tempo_ratio=tempo_ratio,
            clarity_boost_db=clarity_boost_db,
            export_bitrate=bitrate,
            is_one_click_master=is_one_click,
            parent=self
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _cancel_processing(self):
        if self._worker and self._worker.isRunning():
            self.prog_stage_lbl.setText("Cancelling...")
            self.btn_cancel.setEnabled(False)
            self._worker.cancel()

    def _on_worker_progress(self, pct: float, stage: str):
        val = int(pct * 100)
        self.prog_bar.setValue(val)
        self.prog_pct.setText(f"{val}%")
        self.prog_stage_lbl.setText(stage)

    def _on_worker_completed(self, result: Dict[str, Any]):
        self.progress_card.setVisible(False)
        self._set_buttons_enabled(True)

        new_path = result["output_path"]
        if self._processed_path and self._processed_path != self._original_path:
            self._undo_stack.append(self._processed_path)
            self.btn_undo.setEnabled(True)
            self._redo_stack.clear()
            self.btn_redo.setEnabled(False)
        elif self._processed_path == self._original_path:
            self._undo_stack.append(self._original_path)
            self.btn_undo.setEnabled(True)

        self._processed_path = new_path
        self._is_ab_b_mode = True
        self.btn_ab_toggle.setText("A/B: Processed (B)")
        self.btn_ab_toggle._variant = "primary"
        self.btn_ab_toggle._update_style()
        self.loaded_label.setText("Enhancement applied! Auditioning Processed (B).")

        peaks = WaveformEngine.load_peaks(new_path)
        if peaks is not None:
            self.waveform_view.set_peaks(peaks)

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(
                self,
                "Enhancement Complete",
                f"Audio enhancement applied successfully!\n\n"
                f"Target Loudness: {result.get('metrics', {}).get('lufs', '-14.0')} LUFS\n"
                f"Peak: {result.get('metrics', {}).get('peak_db', '-1.0')} dBFS\n"
                f"Format: MP3 ({result.get('bitrate', '320k')})"
            )

    def _on_worker_failed(self, error_msg: str):
        self.progress_card.setVisible(False)
        self._set_buttons_enabled(True)
        self.loaded_label.setText("Processing failed.")
        QMessageBox.critical(self, "Processing Error", f"Audio enhancement failed:\n\n{error_msg}")

    def _on_worker_cancelled(self):
        self.progress_card.setVisible(False)
        self._set_buttons_enabled(True)
        self.loaded_label.setText("Processing cancelled.")

    def _undo(self):
        if self._undo_stack:
            prev = self._undo_stack.pop()
            if self._processed_path:
                self._redo_stack.append(self._processed_path)
                self.btn_redo.setEnabled(True)
            self._processed_path = prev
            self.btn_undo.setEnabled(len(self._undo_stack) > 0)

            peaks = WaveformEngine.load_peaks(prev)
            if peaks is not None:
                self.waveform_view.set_peaks(peaks)
            self.loaded_label.setText("Undone to previous state.")

    def _redo(self):
        if self._redo_stack:
            nxt = self._redo_stack.pop()
            if self._processed_path:
                self._undo_stack.append(self._processed_path)
                self.btn_undo.setEnabled(True)
            self._processed_path = nxt
            self.btn_redo.setEnabled(len(self._redo_stack) > 0)

            peaks = WaveformEngine.load_peaks(nxt)
            if peaks is not None:
                self.waveform_view.set_peaks(peaks)
            self.loaded_label.setText("Redone to next state.")

    def _export_audio(self):
        active_path = self._processed_path or self._original_path
        if not active_path:
            return

        bitrate = self.bitrate_combo.currentData() or "320k"
        base_name = Path(active_path).stem.replace("_processed", "").replace("_master_enhanced", "")
        default_file = str(Path(active_path).parent / f"{base_name}_Mastered.mp3")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Mastered Vocals (MP3 @ {bitrate})",
            default_file,
            AUDIO_FILE_FILTER
        )
        if save_path:
            dest = Path(save_path).with_suffix(".mp3")
            data, sr = decode_mp3_to_numpy(active_path)
            encode_array_to_mp3(data, str(dest), sample_rate=sr, bitrate=bitrate)

            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Mastered vocals exported successfully!\n\nFile: {dest.name}\nBitrate: {bitrate}"
                )

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_load_vocal.setEnabled(enabled)
        self.btn_master_enhance.setEnabled(enabled)
        self.btn_apply_denoise.setEnabled(enabled)
        self.btn_apply_pitch.setEnabled(enabled)
        self.btn_apply_stretch.setEnabled(enabled)
        self.btn_apply_enh.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)

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
            QLabel#VocalStatusText {{
                color: {accent};
                font-size: 12px;
                font-weight: 500;
            }}
            QLabel#FieldLabel {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 600;
            }}
        """)
