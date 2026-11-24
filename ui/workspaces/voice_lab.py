"""Voice Lab Workspace for Audio Studio AI.
Dedicated voice conversion workspace supporting local offline voice morphing,
reference voice formant transfer, RVC-compatible local model inference,
pitch transposition (-12 to +12 semitones), conversion strength blend,
explicit ethical consent validation, and MP3 export at 128k-320k.
"""
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QFileDialog, QFrame, QScrollArea, QGridLayout,
    QStyledItemDelegate, QCheckBox, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.waveform_view import WaveformView
from core.playback_engine import PlaybackEngine
from core.waveform_engine import WaveformEngine
from config import MODELS_DIR, CACHE_DIR, AUDIO_FILE_FILTER, AVAILABLE_MP3_BITRATES, DEFAULT_EXPORT_BITRATE
from core.audio_format import validate_mp3_file, encode_array_to_mp3, decode_mp3_to_numpy
from ai.voice.voice_converter import VoiceConverter
from ai.voice.rvc_engine import RVCModelValidator
from workers.voice_conversion_worker import VoiceConversionWorker

class VoiceLabWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()

        self._input_vocal_path: Optional[str] = None
        self._reference_vocal_path: Optional[str] = None
        self._model_path: Optional[str] = None
        self._output_converted_path: Optional[str] = None

        self._worker: Optional[VoiceConversionWorker] = None

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

        # Header
        header_box = QVBoxLayout()
        header_box.setSpacing(2)
        title = QLabel("Voice Lab")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("AI neural voice conversion with formant morphing, reference audio transfer, and pitch matching.")
        subtitle.setObjectName("WorkspaceSubtitle")
        header_box.addWidget(title)
        header_box.addWidget(subtitle)
        layout.addLayout(header_box)

        # Step 1: Source Vocal Audio
        input_card = PremiumCard(title="1. Source Vocal Audio", subtitle="Select dry or separated vocal audio to convert (MP3 only)")
        in_row = QHBoxLayout()
        self.input_lbl = QLabel("No vocal track selected")
        self.input_lbl.setObjectName("FileDetail")
        self.btn_select_vocal = PremiumButton("📂 Select Source Vocal (MP3)", variant="primary")
        self.btn_select_vocal.clicked.connect(self._select_vocal)

        in_row.addWidget(self.input_lbl, 1)
        in_row.addWidget(self.btn_select_vocal)
        input_card.add_layout(in_row)

        self.input_waveform = WaveformView()
        self.input_waveform.setFixedHeight(70)
        self.input_waveform.seek_requested.connect(self._playback.seek_ratio)
        input_card.add_widget(self.input_waveform)
        layout.addWidget(input_card)

        # Step 2: Target Voice (Reference Audio OR Model Checkpoint)
        target_card = PremiumCard(title="2. Target Voice Identity", subtitle="Provide a reference vocal MP3 or select a local voice checkpoint (.pth)")
        tgt_grid = QGridLayout()
        tgt_grid.setSpacing(14)

        # Reference Vocal Audio Option
        ref_box = QVBoxLayout()
        ref_lbl = QLabel("Target Reference Voice (MP3 Audio):")
        ref_lbl.setObjectName("FieldLabel")
        ref_subrow = QHBoxLayout()
        self.ref_audio_lbl = QLabel("No reference voice loaded (Using internal timbre morphing)")
        self.ref_audio_lbl.setObjectName("FileDetail")
        self.btn_load_ref = PremiumButton("📂 Load Reference MP3", variant="secondary")
        self.btn_load_ref.clicked.connect(self._select_reference_audio)
        ref_subrow.addWidget(self.ref_audio_lbl, 1)
        ref_subrow.addWidget(self.btn_load_ref)
        ref_box.addWidget(ref_lbl)
        ref_box.addLayout(ref_subrow)
        tgt_grid.addLayout(ref_box, 0, 0, 1, 2)

        # Voice Model Dropdown (for .pth checkpoints)
        m_box = QVBoxLayout()
        m_lbl = QLabel("Or Select Pretrained Voice Model (.pth):")
        m_lbl.setObjectName("FieldLabel")
        self.model_combo = QComboBox()
        self.model_combo.setItemDelegate(QStyledItemDelegate(self.model_combo))
        self._populate_models_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        m_box.addWidget(m_lbl)
        m_box.addWidget(self.model_combo)
        tgt_grid.addLayout(m_box, 1, 0, 1, 2)

        target_card.add_layout(tgt_grid)
        layout.addWidget(target_card)

        # Step 3: Synthesis Settings & Pitch Matching
        params_card = PremiumCard(title="3. Voice Morphing & Synthesis Settings", subtitle="Pitch transposition, formant blend strength, and extraction algorithm")
        grid = QGridLayout()
        grid.setSpacing(16)

        # F0 Pitch Transposition
        p_box = QVBoxLayout()
        self.pitch_lbl = QLabel("Pitch Transposition (Key Match): 0 Semitones")
        self.pitch_lbl.setObjectName("FieldLabel")
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(lambda v: self.pitch_lbl.setText(f"Pitch Transposition: {'+' if v>0 else ''}{v} Semitones"))
        p_box.addWidget(self.pitch_lbl)
        p_box.addWidget(self.pitch_slider)
        grid.addLayout(p_box, 0, 0)

        # Conversion Strength (Index Rate)
        s_box = QVBoxLayout()
        self.strength_lbl = QLabel("Voice Conversion Strength: 80%")
        self.strength_lbl.setObjectName("FieldLabel")
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(80)
        self.strength_slider.valueChanged.connect(lambda v: self.strength_lbl.setText(f"Voice Conversion Strength: {v}%"))
        s_box.addWidget(self.strength_lbl)
        s_box.addWidget(self.strength_slider)
        grid.addLayout(s_box, 0, 1)

        # F0 Extraction Method
        f0_box = QVBoxLayout()
        f0_lbl = QLabel("Pitch Extraction Algorithm:")
        f0_lbl.setObjectName("FieldLabel")
        self.f0_combo = QComboBox()
        self.f0_combo.setItemDelegate(QStyledItemDelegate(self.f0_combo))
        self.f0_combo.addItems(["RMVPE (High Quality)", "Harvest (Clean Vocals)", "Crepe (Complex Audio)"])
        f0_box.addWidget(f0_lbl)
        f0_box.addWidget(self.f0_combo)
        grid.addLayout(f0_box, 1, 0)

        # Conversion Mode Indicator
        mode_box = QVBoxLayout()
        mode_title = QLabel("Active Processing Mode:")
        mode_title.setObjectName("FieldLabel")
        self.mode_badge = QLabel("Formant & Timbre Transfer (Offline Spectral Morphing)")
        self.mode_badge.setObjectName("FileDetail")
        mode_box.addWidget(mode_title)
        mode_box.addWidget(self.mode_badge)
        grid.addLayout(mode_box, 1, 1)

        params_card.add_layout(grid)

        # Ethical Usage Consent Checkbox
        consent_row = QHBoxLayout()
        consent_row.setContentsMargins(0, 8, 0, 0)
        self.cb_consent = QCheckBox("⚖️  I confirm I have explicit rights, consent, or authorization to use this voice.")
        self.cb_consent.setObjectName("EnhanceCheck")
        self.cb_consent.toggled.connect(self._on_consent_toggled)
        consent_row.addWidget(self.cb_consent)
        params_card.add_layout(consent_row)

        # Convert Action Button Row
        act_row = QHBoxLayout()
        act_row.setContentsMargins(0, 10, 0, 0)
        self.status_lbl = QLabel("Load a source vocal track and verify consent to begin.")
        self.status_lbl.setObjectName("StatusNotice")

        self.btn_convert = PremiumButton("⚡  Convert Voice", variant="primary")
        self.btn_convert.setEnabled(False)
        self.btn_convert.clicked.connect(self._run_conversion)

        act_row.addWidget(self.status_lbl, 1)
        act_row.addWidget(self.btn_convert)
        params_card.add_layout(act_row)

        layout.addWidget(params_card)

        # Progress Card (Hidden initially, shown only during processing)
        self.progress_card = PremiumCard(elevated=True)
        self.progress_card.setVisible(False)
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(10)

        prog_hdr = QHBoxLayout()
        self.prog_title = QLabel("AI Voice Conversion in Progress...")
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
        self.prog_stage_lbl = QLabel("Initializing voice synthesis...")
        self.prog_stage_lbl.setObjectName("VocalStatusText")
        self.btn_cancel = PremiumButton("✕ Cancel", variant="danger")
        self.btn_cancel.setFixedHeight(28)
        self.btn_cancel.clicked.connect(self._cancel_conversion)
        prog_ftr.addWidget(self.prog_stage_lbl, 1)
        prog_ftr.addWidget(self.btn_cancel)
        prog_layout.addLayout(prog_ftr)

        self.progress_card.add_layout(prog_layout)
        layout.addWidget(self.progress_card)

        # Step 4: Converted Output & A/B Comparison
        out_card = PremiumCard(title="4. Converted Voice Output", subtitle="Listen, compare before-and-after, and export in MP3")
        out_row = QHBoxLayout()
        out_row.setSpacing(14)

        self.btn_play_original = PremiumButton("▶ Play Original", variant="outline")
        self.btn_play_original.clicked.connect(self._play_original)

        self.btn_play_converted = PremiumButton("▶ Play Converted", variant="secondary")
        self.btn_play_converted.setEnabled(False)
        self.btn_play_converted.clicked.connect(self._play_converted)

        bitrate_lbl = QLabel("MP3 Bitrate:")
        bitrate_lbl.setObjectName("FieldLabel")

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setItemDelegate(QStyledItemDelegate(self.bitrate_combo))
        self.bitrate_combo.addItem("320 kbps (Studio Master)", "320k")
        self.bitrate_combo.addItem("256 kbps (High Quality)", "256k")
        self.bitrate_combo.addItem("192 kbps (Standard)", "192k")
        self.bitrate_combo.addItem("128 kbps (Compact)", "128k")
        self.bitrate_combo.setCurrentIndex(0)

        self.btn_export = PremiumButton("💾 Export Converted Voice (MP3)", variant="primary")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_converted)

        out_row.addWidget(self.btn_play_original)
        out_row.addWidget(self.btn_play_converted)
        out_row.addStretch()
        out_row.addWidget(bitrate_lbl)
        out_row.addWidget(self.bitrate_combo)
        out_row.addWidget(self.btn_export)
        out_card.add_layout(out_row)

        self.output_waveform = WaveformView()
        self.output_waveform.setFixedHeight(70)
        self.output_waveform.seek_requested.connect(self._playback.seek_ratio)
        out_card.add_widget(self.output_waveform)
        layout.addWidget(out_card)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self._playback.state_changed.connect(self._on_playback_state_changed)
        self._playback.position_changed.connect(self._on_playback_position_changed)

    def _populate_models_combo(self):
        self.model_combo.clear()
        self.model_combo.addItem("Internal Formant Morphing (No checkpoint required)", None)
        models = VoiceConverter.list_available_models()
        for m in models:
            self.model_combo.addItem(f"Neural Checkpoint: {m['name']}", m["path"])
        self.model_combo.addItem("Browse Custom Model (*.pth)...", "browse")

    def _on_model_changed(self, idx: int):
        val = self.model_combo.currentData()
        if val == "browse":
            fpath, _ = QFileDialog.getOpenFileName(
                self,
                "Select PyTorch Voice Model Checkpoint",
                "",
                "PyTorch Models (*.pth *.pt)"
            )
            if fpath:
                meta = RVCModelValidator.inspect_checkpoint(fpath)
                if not meta["valid"]:
                    QMessageBox.warning(
                        self,
                        "Unsupported Model Format",
                        f"The selected model file is not compatible with RVC neural inference:\n\n{meta.get('error')}\n\n"
                        "Please provide an authorized RVC v1 or v2 PyTorch checkpoint (.pth)."
                    )
                    self.model_combo.setCurrentIndex(0)
                    return
                self._model_path = fpath
                self.model_combo.insertItem(1, f"RVC Neural: {meta['name']} ({meta['version']})", fpath)
                self.model_combo.setCurrentIndex(1)
                self.mode_badge.setText(f"RVC Neural Conversion: {meta['name']} ({meta['version']})")
            else:
                self.model_combo.setCurrentIndex(0)
        elif val:
            meta = RVCModelValidator.inspect_checkpoint(val)
            if meta["valid"]:
                self._model_path = val
                self.mode_badge.setText(f"RVC Neural Conversion: {meta['name']} ({meta['version']})")
            else:
                self._model_path = None
                self.mode_badge.setText("DSP Voice Morphing (Formant & Timbre Transfer)")
        else:
            self._model_path = None
            if self._reference_vocal_path:
                self.mode_badge.setText("DSP Voice Morphing (Reference Timbre Transfer)")
            else:
                self.mode_badge.setText("DSP Voice Morphing (Internal Timbre Synthesis)")

    def _select_vocal(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dry Vocal File (MP3)",
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

            self._input_vocal_path = str(Path(fpath).resolve())
            p = Path(fpath)
            self.input_lbl.setText(f"Loaded: {p.name}")
            peaks = WaveformEngine.load_peaks(fpath)
            if peaks is not None:
                self.input_waveform.set_peaks(peaks)

            self._update_convert_button_state()

    def _select_reference_audio(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Voice Audio File (MP3)",
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

            self._reference_vocal_path = str(Path(fpath).resolve())
            self.ref_audio_lbl.setText(f"Reference Audio: {Path(fpath).name}")
            if not self._model_path:
                self.mode_badge.setText(f"Reference Audio Formant Morphing ({Path(fpath).name})")

    def _on_consent_toggled(self, checked: bool):
        self._update_convert_button_state()

    def _update_convert_button_state(self):
        has_input = bool(self._input_vocal_path)
        has_consent = self.cb_consent.isChecked()
        self.btn_convert.setEnabled(has_input and has_consent)
        if not has_input:
            self.status_lbl.setText("Select a source vocal track (MP3) to proceed.")
        elif not has_consent:
            self.status_lbl.setText("Please confirm ethical rights & consent checkbox above.")
        else:
            self.status_lbl.setText("Ready to convert voice.")

    def _run_conversion(self):
        if not self._input_vocal_path or not self.cb_consent.isChecked():
            return

        out_p = Path(CACHE_DIR) / f"converted_voice.mp3"
        bitrate = self.bitrate_combo.currentData() or "320k"
        pitch = self.pitch_slider.value()
        strength = self.strength_slider.value() / 100.0
        f0_choice = self.f0_combo.currentText().split()[0].lower()

        self._playback.stop()
        self.progress_card.setVisible(True)
        self.prog_bar.setValue(0)
        self.prog_pct.setText("0%")
        self.prog_stage_lbl.setText("Starting voice conversion worker...")
        self.btn_convert.setEnabled(False)

        self._worker = VoiceConversionWorker(
            source_audio_path=self._input_vocal_path,
            output_audio_path=str(out_p),
            reference_audio_path=self._reference_vocal_path,
            model_checkpoint_path=self._model_path,
            pitch_semitones=pitch,
            conversion_strength=strength,
            f0_method=f0_choice,
            export_bitrate=bitrate,
            parent=self
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _cancel_conversion(self):
        if self._worker and self._worker.isRunning():
            self.prog_stage_lbl.setText("Cancelling voice conversion...")
            self.btn_cancel.setEnabled(False)
            self._worker.cancel()

    def _on_worker_progress(self, pct: float, stage: str):
        val = int(pct * 100)
        self.prog_bar.setValue(val)
        self.prog_pct.setText(f"{val}%")
        self.prog_stage_lbl.setText(stage)

    def _on_worker_completed(self, result: Dict[str, Any]):
        self.progress_card.setVisible(False)
        self._update_convert_button_state()
        self.btn_cancel.setEnabled(True)

        self._output_converted_path = result["output_path"]
        self.btn_play_converted.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.status_lbl.setText(f"Voice converted successfully via {result.get('mode', 'AI Morphing')}!")

        peaks = WaveformEngine.load_peaks(self._output_converted_path)
        if peaks is not None:
            self.output_waveform.set_peaks(peaks)

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(
                self,
                "Voice Conversion Complete",
                f"Voice conversion completed successfully!\n\n"
                f"Mode: {result.get('mode')}\n"
                f"Format: MP3 ({result.get('bitrate')})\n"
                f"Loudness: {result.get('metrics', {}).get('lufs', '-14.0')} LUFS"
            )

    def _on_worker_failed(self, err: str):
        self.progress_card.setVisible(False)
        self._update_convert_button_state()
        self.btn_cancel.setEnabled(True)
        self.status_lbl.setText("Voice conversion failed.")
        QMessageBox.critical(self, "Conversion Error", f"Voice conversion failed:\n\n{err}")

    def _on_worker_cancelled(self):
        self.progress_card.setVisible(False)
        self._update_convert_button_state()
        self.btn_cancel.setEnabled(True)
        self.status_lbl.setText("Voice conversion was cancelled.")

    def _play_original(self):
        if self._input_vocal_path:
            self._playback.play_file(self._input_vocal_path, title="Source Vocal", artist="Original Voice")

    def _play_converted(self):
        if self._output_converted_path:
            self._playback.play_file(self._output_converted_path, title="Converted Vocal", artist="Morphed Voice")

    def _export_converted(self):
        if not self._output_converted_path:
            return

        bitrate = self.bitrate_combo.currentData() or "320k"
        base_name = Path(self._input_vocal_path).stem if self._input_vocal_path else "Vocal"
        default_file = str(Path(self._output_converted_path).parent / f"{base_name}_Converted.mp3")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Converted Voice (MP3 @ {bitrate})",
            default_file,
            AUDIO_FILE_FILTER
        )
        if save_path:
            dest = Path(save_path).with_suffix(".mp3")
            data, sr = decode_mp3_to_numpy(self._output_converted_path)
            encode_array_to_mp3(data, str(dest), sample_rate=sr, bitrate=bitrate)

            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Converted voice exported successfully!\n\nFile: {dest.name}\nBitrate: {bitrate}"
                )

    def _on_playback_state_changed(self, is_playing: bool):
        pass

    def _on_playback_position_changed(self, curr_ms: int, dur_ms: int):
        if dur_ms > 0:
            ratio = curr_ms / float(dur_ms)
            if self._playback.current_track and "Original" in self._playback.current_track.artist:
                self.input_waveform.set_progress(ratio)
            elif self._playback.current_track and "Morphed" in self._playback.current_track.artist:
                self.output_waveform.set_progress(ratio)

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
            QLabel#FieldLabel {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#FileDetail {{
                color: {accent};
                font-size: 13px;
                font-weight: 500;
            }}
            QLabel#StatusNotice {{
                color: {text_secondary};
                font-size: 13px;
            }}
            QLabel#VocalStatusText {{
                color: {accent};
                font-size: 12px;
                font-weight: 500;
            }}
            QCheckBox#EnhanceCheck {{
                color: {text_primary};
                font-size: 13px;
                font-weight: 500;
            }}
        """)
