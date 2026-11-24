"""Background QThread Worker for AI Audio Enhancer in Audio Studio AI.
Supports dual-mode processing:
1. Studio DSP Enhancement (FFmpeg afftdn + rubberband pitch/time + biquad EQ).
2. Neural AI Speech Enhancement (Deep Conv-Recurrent Spectral Masking in PyTorch).
Provides asynchronous execution, cancellation support, and live progress reporting.
"""
import sys
import threading
from typing import Optional, Dict, Any

from PySide6.QtCore import QThread, Signal

from ai.enhancement.audio_enhancer import AudioEnhancer
from ai.enhancement.neural_enhancer import NeuralAudioEnhancer

class EnhancerWorker(QThread):
    progress_changed = Signal(float, str)   # (progress_ratio 0.0-1.0, stage_description)
    completed = Signal(dict)                # result metadata dict
    failed = Signal(str)                   # error message string
    cancelled = Signal()                   # emitted when task was safely aborted

    def __init__(
        self,
        input_path: str,
        output_path: str,
        mode: str = "dsp",                   # "dsp" or "neural"
        noise_reduction: float = 0.0,
        pitch_semitones: int = 0,
        tempo_ratio: float = 1.0,
        clarity_boost_db: float = 0.0,
        vocal_gain_db: float = 0.0,
        enable_normalisation: bool = True,
        target_lufs: float = -14.0,
        export_bitrate: str = "320k",
        is_one_click_master: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self._input_path = input_path
        self._output_path = output_path
        self._mode = mode.lower()
        self._noise_reduction = noise_reduction
        self._pitch_semitones = pitch_semitones
        self._tempo_ratio = tempo_ratio
        self._clarity_boost_db = clarity_boost_db
        self._vocal_gain_db = vocal_gain_db
        self._enable_norm = enable_normalisation
        self._target_lufs = target_lufs
        self._export_bitrate = export_bitrate
        self._is_one_click_master = is_one_click_master

        self._cancel_token = threading.Event()

    def cancel(self):
        self._cancel_token.set()

    def run(self):
        try:
            def _progress_cb(pct: float, stage: str):
                self.progress_changed.emit(pct, stage)

            if self._mode == "neural":
                result = NeuralAudioEnhancer.enhance_audio(
                    input_path=self._input_path,
                    output_path=self._output_path,
                    noise_suppression_strength=max(0.1, self._noise_reduction),
                    speech_clarity_boost=self._clarity_boost_db,
                    export_bitrate=self._export_bitrate,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb
                )
            elif self._is_one_click_master:
                result = AudioEnhancer.one_click_master_enhance(
                    input_path=self._input_path,
                    output_path=self._output_path,
                    export_bitrate=self._export_bitrate,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb
                )
            else:
                result = AudioEnhancer.enhance_audio(
                    input_path=self._input_path,
                    output_path=self._output_path,
                    noise_reduction=self._noise_reduction,
                    pitch_semitones=self._pitch_semitones,
                    tempo_ratio=self._tempo_ratio,
                    clarity_boost_db=self._clarity_boost_db,
                    vocal_gain_db=self._vocal_gain_db,
                    enable_normalisation=self._enable_norm,
                    target_lufs=self._target_lufs,
                    export_bitrate=self._export_bitrate,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb
                )

            self.completed.emit(result)

        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e))
