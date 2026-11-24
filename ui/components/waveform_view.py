"""Waveform visualization component for Audio Studio AI.
High performance custom-painted QWidget with peak caching, playhead synchronization,
and seek interaction.
"""
from typing import Optional, List
import numpy as np

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QMouseEvent

from ui.themes.theme_manager import ThemeManager

class WaveformView(QWidget):
    seek_requested = Signal(float)  # normalized position 0.0 - 1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        self._peaks: Optional[np.ndarray] = None  # Normalized [-1.0, 1.0] or [0.0, 1.0]
        self._progress: float = 0.0  # 0.0 to 1.0
        self._hover_pos: Optional[float] = None  # 0.0 to 1.0
        self._is_loading: bool = False
        self._empty_message: str = "No audio loaded"

        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self.update())

    def set_peaks(self, peaks: Optional[np.ndarray]):
        self._peaks = peaks
        self._is_loading = False
        self.update()

    def set_progress(self, progress: float):
        new_progress = max(0.0, min(1.0, progress))
        if abs(new_progress - self._progress) > 0.001:
            self._progress = new_progress
            self.update()

    def set_loading(self, loading: bool):
        self._is_loading = loading
        self.update()

    def set_empty_message(self, message: str):
        self._empty_message = message
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.width() > 0:
            pos_ratio = max(0.0, min(1.0, event.position().x() / self.width()))
            self.seek_requested.emit(pos_ratio)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.width() > 0:
            self._hover_pos = max(0.0, min(1.0, event.position().x() / self.width()))
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_col = QColor(c.get("background_card", "#1B1D21"))
        wave_unplayed = QColor(c.get("waveform_wave", "#444952"))
        wave_played = QColor(c.get("waveform_progress", "#C5C9D0"))
        accent_col = QColor(c.get("accent_silver", "#C5C9D0"))
        text_muted = QColor(c.get("text_muted", "#60646D"))

        w = self.width()
        h = self.height()
        mid_y = h / 2.0

        # Background
        painter.fillRect(0, 0, w, h, bg_col)

        if self._peaks is None or len(self._peaks) == 0:
            # Render empty placeholder state
            painter.setPen(text_muted)
            painter.drawText(
                QRectF(0, 0, w, h),
                Qt.AlignmentFlag.AlignCenter,
                "Analyzing waveform..." if self._is_loading else self._empty_message
            )
            return

        # Resample peaks to match pixel width
        num_bars = min(w // 3, len(self._peaks))
        if num_bars <= 0:
            return

        step = len(self._peaks) / float(num_bars)
        indices = (np.arange(num_bars) * step).astype(int)
        sampled_peaks = self._peaks[indices]

        progress_x = int(self._progress * w)
        bar_width = max(1.5, (w / num_bars) - 1.0)

        for i, val in enumerate(sampled_peaks):
            x = i * (w / num_bars)
            bar_height = max(2.0, float(val) * (h * 0.85))
            y = mid_y - (bar_height / 2.0)

            if x <= progress_x:
                painter.fillRect(QRectF(x, y, bar_width, bar_height), wave_played)
            else:
                painter.fillRect(QRectF(x, y, bar_width, bar_height), wave_unplayed)

        # Draw Playhead Cursor
        playhead_pen = QPen(accent_col, 2)
        painter.setPen(playhead_pen)
        painter.drawLine(progress_x, 0, progress_x, h)

        # Draw Hover Line if applicable
        if self._hover_pos is not None:
            hover_x = int(self._hover_pos * w)
            hover_pen = QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DashLine)
            painter.setPen(hover_pen)
            painter.drawLine(hover_x, 0, hover_x, h)
