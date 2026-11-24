"""Background Worker Thread for AI Audio Separation and Stem Loudness Normalization.
Ensures UI stays completely responsive at 60 FPS while running PyTorch Demucs inference.
Supports clean cancellation tokens, progress reporting, and error handling.
"""
import threading
from typing import Optional, Dict, Any

from PySide6.QtCore import QThread, Signal

from ai.separation.demucs_adapter import SeparationEngine, SeparationCancelledException

class SeparationWorker(QThread):
    progress_changed = Signal(float, str)  # (progress_ratio: 0.0-1.0, stage_description: str)
    completed = Signal(dict)              # results dictionary with stems and audio metrics
    failed = Signal(str)                 # error message
    cancelled = Signal()

    def __init__(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        model_name: str = "htdemucs",
        quality_mode: str = "balanced",
        device: str = "auto",
        enable_normalization: bool = True,
        norm_mode: str = "lufs",
        target_lufs: float = -14.0,
        target_peak_db: float = -1.0,
        vocal_gain_db: float = 0.0,
        inst_gain_db: float = 0.0,
        peak_protection: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.quality_mode = quality_mode
        self.device = device
        self.enable_normalization = enable_normalization
        self.norm_mode = norm_mode
        self.target_lufs = target_lufs
        self.target_peak_db = target_peak_db
        self.vocal_gain_db = vocal_gain_db
        self.inst_gain_db = inst_gain_db
        self.peak_protection = peak_protection

        self._cancel_event = threading.Event()

    def cancel(self):
        """Signals cancellation to running neural inference."""
        self._cancel_event.set()
        self.requestInterruption()

    def run(self):
        try:
            def on_progress(pct: float, msg: str):
                if not self._cancel_event.is_set():
                    self.progress_changed.emit(pct, msg)

            result = SeparationEngine.separate_2stem(
                file_path=self.file_path,
                output_dir=self.output_dir,
                model_name=self.model_name,
                quality_mode=self.quality_mode,
                device=self.device,
                enable_normalization=self.enable_normalization,
                norm_mode=self.norm_mode,
                target_lufs=self.target_lufs,
                target_peak_db=self.target_peak_db,
                vocal_gain_db=self.vocal_gain_db,
                inst_gain_db=self.inst_gain_db,
                peak_protection=self.peak_protection,
                progress_callback=on_progress,
                cancel_event=self._cancel_event
            )

            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.completed.emit(result)

        except SeparationCancelledException:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))
