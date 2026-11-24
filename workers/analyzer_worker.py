"""Background QThread Worker for AI Song Analyzer in Audio Studio AI.
Extracts acoustic intelligence (BPM, Key, Energy, Vocal Segments) asynchronously.
"""
from typing import Optional

from PySide6.QtCore import QThread, Signal

from ai.mixing.audio_analyzer import AudioAnalyzer
from ai.mixing.mix_plan import TrackAnalysis

class AnalyzerWorker(QThread):
    completed = Signal(object)  # TrackAnalysis
    failed = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self._file_path = file_path

    def run(self):
        try:
            analysis = AudioAnalyzer.analyze_track(self._file_path)
            self.completed.emit(analysis)
        except Exception as e:
            self.failed.emit(str(e))
