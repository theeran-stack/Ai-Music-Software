"""Audio Visualizer Component for Audio Studio AI.
Supports Spectrum Bars, Circular Visualizer, Oscilloscope Waveform, and Frequency Analyser modes.
"""
import math
import numpy as np
from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QPainterPath

from ui.themes.theme_manager import ThemeManager

class VisualizerWidget(QWidget):
    def __init__(self, mode: str = "spectrum", parent=None):
        super().__init__(parent)
        self._mode = mode  # spectrum, circular, oscilloscope, frequency
        self._is_active = False
        self._sensitivity = 1.0
        self._num_bands = 48
        self._band_values = np.zeros(self._num_bands)
        self._decay_rate = 0.88

        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self.update())

        # Animation timer (target 60fps when active)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

        self._phase = 0.0

    def set_mode(self, mode: str):
        self._mode = mode
        self.update()

    def set_sensitivity(self, val: float):
        self._sensitivity = max(0.1, min(3.0, val))

    def set_active(self, active: bool):
        self._is_active = active
        if active:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self._band_values.fill(0)
            self.update()

    def push_audio_data(self, samples: np.ndarray):
        """Feed real audio samples to compute FFT magnitudes."""
        if len(samples) == 0:
            return
        try:
            fft_data = np.abs(np.fft.rfft(samples))
            if len(fft_data) > self._num_bands:
                # Bin FFT into bands
                binned = np.array_split(fft_data, self._num_bands)
                magnitudes = np.array([np.mean(b) for b in binned])
                # Normalize
                max_val = np.max(magnitudes) if np.max(magnitudes) > 0 else 1.0
                normalized = (magnitudes / max_val) * self._sensitivity
                # Smooth update
                self._band_values = np.maximum(normalized, self._band_values * self._decay_rate)
            self.update()
        except Exception:
            pass

    def _on_tick(self):
        if not self._is_active:
            return
        self._phase += 0.08
        # If no real data pushed recently, create a gentle ambient rhythm
        ambient = (np.sin(np.linspace(0, math.pi * 3, self._num_bands) + self._phase) + 1.0) * 0.25 * self._sensitivity
        self._band_values = np.maximum(ambient, self._band_values * self._decay_rate)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = QColor(c.get("background_card", "#1B1D21"))
        accent_silver = QColor(c.get("accent_silver", "#C5C9D0"))
        accent_hover = QColor(c.get("accent_hover", "#D8DCE3"))
        border_col = QColor(c.get("border_default", "#30333A"))

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, bg_card)

        if self._mode == "spectrum":
            self._draw_spectrum(painter, w, h, accent_silver, accent_hover)
        elif self._mode == "circular":
            self._draw_circular(painter, w, h, accent_silver, accent_hover)
        elif self._mode == "oscilloscope":
            self._draw_oscilloscope(painter, w, h, accent_silver)
        elif self._mode == "frequency":
            self._draw_frequency(painter, w, h, accent_silver)

    def _draw_spectrum(self, painter: QPainter, w: int, h: int, color1: QColor, color2: QColor):
        num_bars = len(self._band_values)
        padding = 2
        bar_width = max(2.0, (w - (num_bars * padding)) / num_bars)

        for i, val in enumerate(self._band_values):
            x = i * (bar_width + padding)
            bar_height = min(h - 10, val * (h - 20))
            y = h - bar_height - 10

            gradient = QLinearGradient(x, y + bar_height, x, y)
            gradient.setColorAt(0.0, color1)
            gradient.setColorAt(1.0, color2)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_height), 2, 2)

    def _draw_circular(self, painter: QPainter, w: int, h: int, color1: QColor, color2: QColor):
        center_x = w / 2.0
        center_y = h / 2.0
        base_radius = min(w, h) * 0.22
        num_points = len(self._band_values)

        painter.setPen(QPen(color1, 2))
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i - (math.pi / 2)
            val = float(self._band_values[i])
            r_out = base_radius + (val * base_radius * 0.9)

            x1 = center_x + math.cos(angle) * base_radius
            y1 = center_y + math.sin(angle) * base_radius
            x2 = center_x + math.cos(angle) * r_out
            y2 = center_y + math.sin(angle) * r_out

            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_oscilloscope(self, painter: QPainter, w: int, h: int, color: QColor):
        mid_y = h / 2.0
        path = QPainterPath()
        path.moveTo(0, mid_y)
        num_points = len(self._band_values)

        for i in range(num_points):
            x = (w / (num_points - 1)) * i
            val = (float(self._band_values[i]) - 0.5) * (h * 0.6)
            y = mid_y + val
            path.lineTo(x, y)

        painter.setPen(QPen(color, 2))
        painter.drawPath(path)

    def _draw_frequency(self, painter: QPainter, w: int, h: int, color: QColor):
        path = QPainterPath()
        path.moveTo(0, h)
        num_points = len(self._band_values)

        for i in range(num_points):
            x = (w / (num_points - 1)) * i
            y = h - (self._band_values[i] * (h * 0.8))
            path.lineTo(x, y)

        path.lineTo(w, h)
        path.closeSubpath()

        fill_color = QColor(color.red(), color.green(), color.blue(), 40)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(color, 2))
        painter.drawPath(path)
