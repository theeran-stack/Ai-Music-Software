"""Background QThread Worker for AI Voice Lab in Audio Studio AI.
Handles asynchronous voice morphing with progress reporting and cancellation.
"""
import threading
from typing import Optional, Dict, Any

from PySide6.QtCore import QThread, Signal

from ai.voice.voice_converter import VoiceConverter

class VoiceConversionWorker(QThread):
    progress_changed = Signal(float, str)   # (progress_ratio, stage_description)
    completed = Signal(dict)                # result metadata
    failed = Signal(str)                   # error message
    cancelled = Signal()                   # emitted if cancelled

    def __init__(
        self,
        source_audio_path: str,
        output_audio_path: str,
        reference_audio_path: Optional[str] = None,
        model_checkpoint_path: Optional[str] = None,
        pitch_semitones: int = 0,
        conversion_strength: float = 0.8,
        f0_method: str = "rmvpe",
        export_bitrate: str = "320k",
        parent=None
    ):
        super().__init__(parent)
        self._source_path = source_audio_path
        self._output_path = output_audio_path
        self._ref_path = reference_audio_path
        self._model_path = model_checkpoint_path
        self._pitch_semi = pitch_semitones
        self._strength = conversion_strength
        self._f0_method = f0_method
        self._export_bitrate = export_bitrate

        self._cancel_token = threading.Event()

    def cancel(self):
        self._cancel_token.set()

    def run(self):
        try:
            def _progress_cb(pct: float, stage: str):
                self.progress_changed.emit(pct, stage)

            result = VoiceConverter.convert_voice(
                source_audio_path=self._source_path,
                output_audio_path=self._output_path,
                reference_audio_path=self._ref_path,
                model_checkpoint_path=self._model_path,
                pitch_semitones=self._pitch_semi,
                conversion_strength=self._strength,
                f0_method=self._f0_method,
                export_bitrate=self._export_bitrate,
                cancel_token=self._cancel_token,
                progress_cb=_progress_cb
            )

            self.completed.emit(result)

        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))
