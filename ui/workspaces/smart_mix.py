"""Smart Mix Studio Workspace for Audio Studio AI.
AI-powered automatic DJ music mixing workstation:
- Multi-track sequence queue with Move Up, Move Down, and Remove.
- Automatic acoustic analysis (BPM, Krumhansl-Schmuckler harmonic key, Camelot wheel).
- AI transition planning with vocal clash collision avoidance.
- Interactive multi-track continuous mix timeline with phrase boundaries and crossfade markers.
- Continuous mix preview with playhead synchronization and seeking.
- Mix project saving (.json).
- Master continuous DJ mix rendering strictly in MP3 (128k-320k).
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QFileDialog, QFrame, QScrollArea, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal

from ui.themes.theme_manager import ThemeManager
from ui.components.card import PremiumCard
from ui.components.button import PremiumButton
from ui.components.mix_timeline import MixTimelineWidget
from core.playback_engine import PlaybackEngine
from core.waveform_engine import WaveformEngine
from config import AUDIO_FILE_FILTER, AVAILABLE_MP3_BITRATES, DEFAULT_EXPORT_BITRATE, CACHE_DIR
from core.audio_format import validate_mp3_file, encode_array_to_mp3, decode_mp3_to_numpy
from ai.mixing.mix_plan import MixPlan
from ai.mixing.audio_analyzer import AudioAnalyzer
from workers.mix_worker import SmartMixWorker


class SmartMixWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback = PlaybackEngine.get_instance()
        self._theme_mgr = ThemeManager.get_instance()

        self._track_queue: List[Dict[str, Any]] = []
        self._current_plan: Optional[MixPlan] = None
        self._preview_mix_path: Optional[str] = None
        self._rendered_mix_path: Optional[str] = None
        self._worker: Optional[SmartMixWorker] = None

        self._build_ui()
        self._connect_signals()
        self._theme_mgr.theme_changed.connect(lambda _: self._update_style())
        self._update_style()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Header Title Bar
        header_box = QHBoxLayout()
        header_box.setSpacing(16)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Smart Mix Studio")
        title.setObjectName("WorkspaceTitle")
        subtitle = QLabel("AI automatic DJ mixing: Harmonic Camelot matching, phrase alignment, vocal avoidance, and seamless continuous set rendering.")
        subtitle.setObjectName("WorkspaceSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_box.addLayout(title_box, 1)

        self.btn_import_songs = PremiumButton("➕  Import Songs (MP3)", variant="primary")
        self.btn_import_songs.clicked.connect(self._import_songs)
        header_box.addWidget(self.btn_import_songs)
        layout.addLayout(header_box)

        # Content Row: Queue on Left, Mix Configuration on Right
        content_row = QHBoxLayout()
        content_row.setSpacing(16)

        # Left Column: Track Arrangement & Sequence List
        left_card = PremiumCard(title="1. Track Sequence & Harmonic Analysis", subtitle="Imported MP3 songs with detected BPM, musical key, and vocal cues")
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)

        self.track_list = QListWidget()
        self.track_list.setObjectName("TrackSequenceList")
        self.track_list.setMinimumHeight(210)
        left_layout.addWidget(self.track_list)

        track_ctrl_row = QHBoxLayout()
        self.btn_move_up = PremiumButton("▲ Move Up", variant="secondary")
        self.btn_move_up.clicked.connect(self._move_up)
        self.btn_move_down = PremiumButton("▼ Move Down", variant="secondary")
        self.btn_move_down.clicked.connect(self._move_down)
        self.btn_remove = PremiumButton("✕ Remove", variant="outline")
        self.btn_remove.clicked.connect(self._remove_selected)

        track_ctrl_row.addWidget(self.btn_move_up)
        track_ctrl_row.addWidget(self.btn_move_down)
        track_ctrl_row.addWidget(self.btn_remove)
        left_layout.addLayout(track_ctrl_row)

        left_card.add_layout(left_layout)
        content_row.addWidget(left_card, 1)

        # Right Column: Mix Style & Transition Rules
        right_card = PremiumCard(title="2. AI Mix Planning Parameters", subtitle="Transition algorithm, phrase alignment, and tempo adaptation")
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # Transition Type
        t_box = QVBoxLayout()
        t_lbl = QLabel("Primary Transition Style:")
        t_lbl.setObjectName("FieldLabel")
        self.transition_combo = QComboBox()
        self.transition_combo.setItemDelegate(QStyledItemDelegate(self.transition_combo))
        self.transition_combo.addItems([
            "AI-Selected (Smart Adaptation)",
            "Beat-Matched Phrase Transition",
            "Harmonic Crossfade (Camelot Wheel)",
            "Echo Out / High-Pass Fade",
            "Reverb Tail Washout"
        ])
        t_box.addWidget(t_lbl)
        t_box.addWidget(self.transition_combo)
        right_layout.addLayout(t_box)

        # Transition Length
        dur_box = QVBoxLayout()
        dur_lbl = QLabel("Transition Duration (Bars):")
        dur_lbl.setObjectName("FieldLabel")
        self.dur_combo = QComboBox()
        self.dur_combo.setItemDelegate(QStyledItemDelegate(self.dur_combo))
        self.dur_combo.addItems(["8 Bars (Standard)", "16 Bars (Long Progressive)", "4 Bars (Quick Drop)", "32 Bars (Extended Club)"])
        dur_box.addWidget(dur_lbl)
        dur_box.addWidget(self.dur_combo)
        right_layout.addLayout(dur_box)

        # Vocal Overlap Prevention
        vocal_box = QVBoxLayout()
        v_lbl = QLabel("Vocal Activity Safety:")
        v_lbl.setObjectName("FieldLabel")
        self.vocal_combo = QComboBox()
        self.vocal_combo.setItemDelegate(QStyledItemDelegate(self.vocal_combo))
        self.vocal_combo.addItems([
            "Strict: Never overlap prominent vocals",
            "Balanced: Overlap vocal intro over instrumental outro",
            "Permissive: Allow crossfading anywhere"
        ])
        vocal_box.addWidget(v_lbl)
        vocal_box.addWidget(self.vocal_combo)
        right_layout.addLayout(vocal_box)

        # Action Button: Generate Proposed AI Mix Plan
        self.btn_generate_plan = PremiumButton("⚡  Generate Proposed AI Mix Plan", variant="primary")
        self.btn_generate_plan.clicked.connect(self._generate_plan)
        right_layout.addWidget(self.btn_generate_plan)

        self.plan_status = QLabel("Import at least 2 MP3 tracks to generate a continuous DJ mix.")
        self.plan_status.setObjectName("StatusNotice")
        right_layout.addWidget(self.plan_status)

        right_card.add_layout(right_layout)
        content_row.addWidget(right_card, 1)

        layout.addLayout(content_row)

        # Progress Card (Hidden initially, shown only during planning/rendering)
        self.progress_card = PremiumCard(elevated=True)
        self.progress_card.setVisible(False)
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(10)

        prog_hdr = QHBoxLayout()
        self.prog_title = QLabel("AI Mix Engine in Progress...")
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
        self.prog_stage_lbl = QLabel("Analyzing tracks...")
        self.prog_stage_lbl.setObjectName("StatusNotice")
        self.btn_cancel = PremiumButton("✕ Cancel", variant="danger")
        self.btn_cancel.setFixedHeight(28)
        self.btn_cancel.clicked.connect(self._cancel_worker)
        prog_ftr.addWidget(self.prog_stage_lbl, 1)
        prog_ftr.addWidget(self.btn_cancel)
        prog_layout.addLayout(prog_ftr)

        self.progress_card.add_layout(prog_layout)
        layout.addWidget(self.progress_card)

        # 3. Continuous Mix Timeline Preview Card
        timeline_card = PremiumCard(
            title="3. Continuous Mix Timeline Preview",
            subtitle="Interactive DJ set arrangement: Track sequence, start/end timestamps, phrase transition zones, and crossfades"
        )
        tl_layout = QVBoxLayout()
        tl_layout.setSpacing(12)

        # Dedicated Interactive Continuous Mix Timeline Widget
        self.mix_timeline = MixTimelineWidget()
        self.mix_timeline.set_empty_message("Mix timeline will appear here after analysis")
        self.mix_timeline.seek_requested.connect(self._on_timeline_seek)
        tl_layout.addWidget(self.mix_timeline)

        # Timeline Controls & Master Export Toolbar
        tl_ctrl_row = QHBoxLayout()
        tl_ctrl_row.setSpacing(12)

        self.btn_play_mix = PremiumButton("▶ Preview Continuous Mix", variant="secondary")
        self.btn_play_mix.setEnabled(False)
        self.btn_play_mix.clicked.connect(self._play_continuous_mix)

        self.btn_save_project = PremiumButton("💾 Save Mix Project", variant="outline")
        self.btn_save_project.setEnabled(False)
        self.btn_save_project.clicked.connect(self._save_mix_project)

        bitrate_lbl = QLabel("Export Bitrate:")
        bitrate_lbl.setObjectName("FieldLabel")

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setItemDelegate(QStyledItemDelegate(self.bitrate_combo))
        self.bitrate_combo.addItem("320 kbps (Studio Master)", "320k")
        self.bitrate_combo.addItem("256 kbps (High Quality)", "256k")
        self.bitrate_combo.addItem("192 kbps (Standard)", "192k")
        self.bitrate_combo.addItem("128 kbps (Compact)", "128k")
        self.bitrate_combo.setCurrentIndex(0)

        self.btn_render_export = PremiumButton("🚀 Render & Export DJ Mix (MP3)", variant="primary")
        self.btn_render_export.setEnabled(False)
        self.btn_render_export.clicked.connect(self._render_and_export_mix)

        tl_ctrl_row.addWidget(self.btn_play_mix)
        tl_ctrl_row.addWidget(self.btn_save_project)
        tl_ctrl_row.addStretch()
        tl_ctrl_row.addWidget(bitrate_lbl)
        tl_ctrl_row.addWidget(self.bitrate_combo)
        tl_ctrl_row.addWidget(self.btn_render_export)
        tl_layout.addLayout(tl_ctrl_row)

        timeline_card.add_layout(tl_layout)
        layout.addWidget(timeline_card)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self._playback.state_changed.connect(self._on_playback_state_changed)
        self._playback.position_changed.connect(self._on_playback_position_changed)

    def _import_songs(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Songs for DJ Mix (MP3)",
            "",
            AUDIO_FILE_FILTER
        )
        if files:
            for fpath in files:
                if not fpath.lower().endswith(".mp3"):
                    QMessageBox.warning(self, "Invalid Format", "Only MP3 audio files are supported.")
                    continue
                valid, err = validate_mp3_file(fpath)
                if not valid:
                    QMessageBox.warning(self, "Invalid Format", f"Only MP3 audio files are supported.\n\n{err}")
                    continue

                p = Path(fpath)
                # Run rapid acoustic pre-analysis for immediate acoustic feedback in the list
                try:
                    analysis = AudioAnalyzer.analyze_track(str(p.resolve()))
                    track_info = {
                        "path": str(p.resolve()),
                        "title": p.stem,
                        "bpm": analysis.bpm,
                        "key": f"{analysis.camelot_key} ({analysis.musical_key})",
                        "duration": analysis.duration_seconds
                    }
                    item = QListWidgetItem(f"🎵 {p.name}   |   BPM: {analysis.bpm:.1f}   |   Key: {analysis.camelot_key} ({analysis.musical_key})")
                except Exception:
                    track_info = {
                        "path": str(p.resolve()),
                        "title": p.stem,
                        "bpm": 0.0,
                        "key": "Pending",
                        "duration": 0.0
                    }
                    item = QListWidgetItem(f"🎵 {p.name}   [Analysis Ready]")

                self._track_queue.append(track_info)
                self.track_list.addItem(item)

            self.plan_status.setText(f"{len(self._track_queue)} MP3 tracks loaded. Click 'Generate Proposed AI Mix Plan' to arrange.")

    def _move_up(self):
        row = self.track_list.currentRow()
        if row > 0:
            item = self.track_list.takeItem(row)
            self.track_list.insertItem(row - 1, item)
            self.track_list.setCurrentRow(row - 1)
            self._track_queue[row], self._track_queue[row - 1] = self._track_queue[row - 1], self._track_queue[row]
            self._on_queue_modified()

    def _move_down(self):
        row = self.track_list.currentRow()
        if 0 <= row < self.track_list.count() - 1:
            item = self.track_list.takeItem(row)
            self.track_list.insertItem(row + 1, item)
            self.track_list.setCurrentRow(row + 1)
            self._track_queue[row], self._track_queue[row + 1] = self._track_queue[row + 1], self._track_queue[row]
            self._on_queue_modified()

    def _remove_selected(self):
        row = self.track_list.currentRow()
        if row >= 0:
            self.track_list.takeItem(row)
            if row < len(self._track_queue):
                self._track_queue.pop(row)
            self._on_queue_modified()

    def _on_queue_modified(self):
        self._current_plan = None
        self._preview_mix_path = None
        self.btn_play_mix.setEnabled(False)
        self.btn_save_project.setEnabled(False)
        self.btn_render_export.setEnabled(False)
        self.mix_timeline.set_plan(None)
        self.mix_timeline.set_peaks(None)
        self.plan_status.setText(f"{len(self._track_queue)} tracks in queue. Mix plan must be regenerated after reordering.")

    def _generate_plan(self):
        if len(self._track_queue) < 2:
            QMessageBox.information(
                self,
                "Need At Least 2 Tracks",
                "Please import at least 2 MP3 tracks into the queue to generate an AI continuous DJ mix."
            )
            return

        bars_text = self.dur_combo.currentText().split()[0]
        try:
            bars_val = int(bars_text)
        except ValueError:
            bars_val = 8

        settings = {
            "transition_style": self.transition_combo.currentText(),
            "transition_duration_bars": bars_val,
            "vocal_safety": self.vocal_combo.currentText()
        }

        self.progress_card.setVisible(True)
        self.prog_title.setText("AI Mix Planning & Acoustic Analysis...")
        self.prog_bar.setValue(0)
        self.prog_pct.setText("0%")
        self.prog_stage_lbl.setText("Analyzing track acoustics and harmonic Camelot wheel...")
        self.btn_generate_plan.setEnabled(False)
        self.btn_play_mix.setEnabled(False)
        self.btn_render_export.setEnabled(False)

        tracks_to_plan = [t["path"] for t in self._track_queue]

        self._worker = SmartMixWorker(
            mode="plan",
            tracks=tracks_to_plan,
            settings=settings,
            parent=self
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.plan_completed.connect(self._on_plan_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _on_plan_completed(self, plan: MixPlan):
        self.progress_card.setVisible(False)
        self.btn_generate_plan.setEnabled(True)
        self._current_plan = plan

        # Update track list items with analyzed acoustic data and transition tags
        self.track_list.clear()
        for idx, t in enumerate(plan.tracks):
            trans_info = ""
            if idx < len(plan.transitions):
                tr = plan.transitions[idx]
                trans_info = f"➔ [{tr.transition_type} | {tr.duration_bars} Bars]"
            item_text = f"🎵 {t.title}   |   BPM: {t.bpm:.1f}   |   Key: {t.camelot_key} ({t.musical_key})   {trans_info}"
            self.track_list.addItem(QListWidgetItem(item_text))

        # Immediately update the Interactive Continuous Mix Timeline
        self.mix_timeline.set_plan(plan)

        mins = plan.total_duration_seconds / 60.0
        self.plan_status.setText(
            f"Mix Plan Ready! {len(plan.tracks)} tracks, {len(plan.transitions)} transitions "
            f"({mins:.1f} mins total). Generating continuous preview..."
        )

        self.btn_save_project.setEnabled(True)
        self.btn_render_export.setEnabled(True)

        # Automatically render continuous preview in background for instant auditioning
        self._start_preview_rendering(plan)

    def _start_preview_rendering(self, plan: MixPlan):
        preview_file = str(Path(CACHE_DIR) / f"smart_mix_preview_{plan.id}.mp3")
        self._worker = SmartMixWorker(
            mode="preview",
            plan=plan,
            output_path=preview_file,
            export_bitrate="192k",
            parent=self
        )
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.failed.connect(lambda err: self.plan_status.setText("Preview render warning: " + err))
        self._worker.start()

    def _on_preview_ready(self, preview_path: str, result: Dict[str, Any]):
        self._preview_mix_path = preview_path
        self.btn_play_mix.setEnabled(True)
        self.plan_status.setText(
            f"Mix Plan Ready! {len(self._current_plan.tracks)} tracks, {len(self._current_plan.transitions)} transitions "
            f"({self._current_plan.total_duration_seconds / 60.0:.1f} mins). Click 'Preview Continuous Mix' to audition."
        )

        # Load waveform peaks into timeline
        peaks = WaveformEngine.load_peaks(preview_path)
        if peaks is not None:
            self.mix_timeline.set_peaks(peaks)

    def _play_continuous_mix(self):
        active_audio = self._rendered_mix_path or self._preview_mix_path
        if not active_audio or not Path(active_audio).exists():
            if self._current_plan:
                # If preview was not yet completed, trigger render and play
                self.progress_card.setVisible(True)
                self.prog_title.setText("Rendering Mix Preview...")
                self.prog_bar.setValue(0)
                self._start_preview_rendering(self._current_plan)
            return

        if self._playback.is_playing:
            self._playback.pause()
        else:
            self._playback.play_file(
                active_audio,
                title="Continuous DJ Mix Preview",
                artist="Smart Mix Studio AI"
            )

    def _on_timeline_seek(self, ratio: float):
        dur_ms = getattr(self._playback, "duration_ms", getattr(self._playback, "_duration_ms", 0))
        if dur_ms > 0 and hasattr(self._playback, "seek_ratio"):
            self._playback.seek_ratio(ratio)
        elif dur_ms > 0 and hasattr(self._playback, "seek_ms"):
            self._playback.seek_ms(int(ratio * dur_ms))
        self.mix_timeline.set_playhead_ratio(ratio)

    def _save_mix_project(self):
        if not self._current_plan:
            return

        default_file = str(Path(CACHE_DIR) / f"{self._current_plan.name}.json")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Smart Mix Project",
            default_file,
            "Smart Mix Project (*.json)"
        )
        if save_path:
            p = Path(save_path).with_suffix(".json")
            p.write_text(json.dumps(self._current_plan.to_dict(), indent=2), encoding="utf-8")
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self,
                    "Project Saved",
                    f"Smart Mix project saved successfully to:\n\n{p.name}"
                )

    def _render_and_export_mix(self):
        if not self._current_plan:
            return

        bitrate = self.bitrate_combo.currentData() or "320k"
        default_file = str(Path(CACHE_DIR) / f"smart_dj_mix.mp3")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Continuous DJ Mix (MP3 @ {bitrate})",
            default_file,
            AUDIO_FILE_FILTER
        )
        if not save_path:
            return

        self.progress_card.setVisible(True)
        self.prog_title.setText("Rendering Continuous DJ Mix...")
        self.prog_bar.setValue(0)
        self.prog_pct.setText("0%")
        self.prog_stage_lbl.setText("Crossfading and stitching tracks to master MP3...")
        self.btn_render_export.setEnabled(False)

        self._worker = SmartMixWorker(
            mode="render",
            plan=self._current_plan,
            output_path=save_path,
            export_bitrate=bitrate,
            parent=self
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.render_completed.connect(self._on_render_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _cancel_worker(self):
        if self._worker and self._worker.isRunning():
            self.prog_stage_lbl.setText("Cancelling...")
            self.btn_cancel.setEnabled(False)
            self._worker.cancel()

    def _on_worker_progress(self, pct: float, stage: str):
        val = int(pct * 100)
        self.prog_bar.setValue(val)
        self.prog_pct.setText(f"{val}%")
        self.prog_stage_lbl.setText(stage)

    def _on_render_completed(self, result: Dict[str, Any]):
        self.progress_card.setVisible(False)
        self.btn_render_export.setEnabled(True)
        self._rendered_mix_path = result["output_path"]

        self.btn_play_mix.setEnabled(True)
        self.plan_status.setText(f"Continuous DJ set rendered! Saved to {Path(self._rendered_mix_path).name}")

        peaks = WaveformEngine.load_peaks(self._rendered_mix_path)
        if peaks is not None:
            self.mix_timeline.set_peaks(peaks)

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(
                self,
                "Mix Render Complete",
                f"Continuous DJ mix rendered and exported successfully!\n\n"
                f"File: {self._rendered_mix_path}\n"
                f"Format: MP3 ({result.get('bitrate', '320k')})\n"
                f"Loudness: {result.get('metrics', {}).get('lufs', '-14.0')} LUFS"
            )

    def _on_worker_failed(self, err: str):
        self.progress_card.setVisible(False)
        self.btn_generate_plan.setEnabled(True)
        self.btn_render_export.setEnabled(self._current_plan is not None)
        self.plan_status.setText("Processing failed.")
        QMessageBox.critical(self, "Mix Error", f"Smart Mix operation failed:\n\n{err}")

    def _on_worker_cancelled(self):
        self.progress_card.setVisible(False)
        self.btn_generate_plan.setEnabled(True)
        self.btn_render_export.setEnabled(self._current_plan is not None)
        self.plan_status.setText("Operation cancelled.")

    def _on_playback_state_changed(self, is_playing: bool):
        self.btn_play_mix.setText("⏸ Pause Continuous Mix" if is_playing else "▶ Preview Continuous Mix")

    def _on_playback_position_changed(self, curr_ms: int, dur_ms: int):
        if dur_ms > 0:
            ratio = curr_ms / float(dur_ms)
            self.mix_timeline.set_playhead_ratio(ratio)

    def _update_style(self):
        c = self._theme_mgr.get_current_theme().get("colors", {})
        border = c.get("border_default", "#30333A")
        text_primary = c.get("text_primary", "#F5F5F5")
        text_secondary = c.get("text_secondary", "#969BA3")
        accent = c.get("accent_silver", "#C5C9D0")

        self.track_list.setStyleSheet(f"""
            QListWidget {{
                background-color: #141619;
                border: 1px solid {border};
                border-radius: 8px;
                color: {text_primary};
                font-size: 13px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-bottom: 1px solid #20232A;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: #26292F;
                color: {accent};
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
            QLabel#FieldLabel {{
                color: {text_secondary};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#StatusNotice {{
                color: {text_secondary};
                font-size: 13px;
                font-style: italic;
            }}
        """)
