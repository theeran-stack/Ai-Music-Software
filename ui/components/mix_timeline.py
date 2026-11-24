"""Interactive Continuous Mix Timeline Widget for Smart Mix Studio in Audio Studio AI.
Visualizes the complete AI MixPlan:
- Track sequencing, order, start and end timestamps.
- Detected BPM and musical Camelot key badges.
- Transition overlap markers with equal-power crossfade curves.
- Transition style and vocal safety status.
- Waveform preview display.
- Synchronized playhead cursor with click-to-seek interaction.
"""
from typing import Optional, List, Dict, Any
import numpy as np

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QMouseEvent

from ui.themes.theme_manager import ThemeManager
from ai.mixing.mix_plan import MixPlan, TrackAnalysis, TransitionPlan


class MixTimelineWidget(QWidget):
    seek_requested = Signal(float)  # ratio 0.0 to 1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._plan: Optional[MixPlan] = None
        self._peaks: Optional[np.ndarray] = None
        self._playhead_ratio: float = 0.0
        self._hover_ratio: Optional[float] = None
        self._empty_message: str = "Mix timeline will appear here after analysis"

        self._track_layouts: List[Dict[str, Any]] = []
        self._transition_layouts: List[Dict[str, Any]] = []
        self._total_duration: float = 0.0

        self._theme_mgr = ThemeManager.get_instance()
        self._theme_mgr.theme_changed.connect(lambda _: self.update())

    def set_empty_message(self, msg: str):
        self._empty_message = msg
        self.update()

    def set_plan(self, plan: Optional[MixPlan]):
        self._plan = plan
        self._recalculate_layout()
        self.update()

    def set_peaks(self, peaks: Optional[np.ndarray]):
        self._peaks = peaks
        self.update()

    def set_playhead_ratio(self, ratio: float):
        r = max(0.0, min(1.0, ratio))
        if abs(self._playhead_ratio - r) > 0.002:
            self._playhead_ratio = r
            self.update()

    def _recalculate_layout(self):
        self._track_layouts.clear()
        self._transition_layouts.clear()
        self._total_duration = 0.0

        if not self._plan or not self._plan.tracks:
            return

        tracks = self._plan.tracks
        transitions = self._plan.transitions

        # Calculate exact timeline start and end positions for every track
        curr_offset = 0.0
        for i, t in enumerate(tracks):
            t_start = curr_offset
            t_end = t_start + t.duration_seconds
            self._track_layouts.append({
                "idx": i,
                "track": t,
                "start_sec": t_start,
                "end_sec": t_end,
                "duration_sec": t.duration_seconds
            })

            if i < len(transitions):
                tr = transitions[i]
                # Transition overlap position on the continuous timeline
                trans_start = t_start + tr.from_exit_start_sec
                trans_end = t_start + tr.from_exit_end_sec
                self._transition_layouts.append({
                    "idx": i,
                    "transition": tr,
                    "start_sec": trans_start,
                    "end_sec": trans_end,
                    "duration_sec": tr.duration_seconds
                })
                # Incoming track start aligns with the transition entry
                curr_offset = trans_start - tr.to_entry_start_sec
            else:
                self._total_duration = max(self._total_duration, t_end)

        if self._track_layouts:
            self._total_duration = max(t["end_sec"] for t in self._track_layouts)

    @staticmethod
    def _format_time(sec: float) -> str:
        s = max(0.0, sec)
        m = int(s // 60)
        rem = int(s % 60)
        return f"{m:02d}:{rem:02d}"

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.width() > 0:
            ratio = max(0.0, min(1.0, event.position().x() / self.width()))
            self.seek_requested.emit(ratio)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.width() > 0:
            self._hover_ratio = max(0.0, min(1.0, event.position().x() / self.width()))
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_ratio = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        c = self._theme_mgr.get_current_theme().get("colors", {})
        bg_card = QColor(c.get("background_card", "#1B1D21"))
        border = QColor(c.get("border_default", "#30333A"))
        text_primary = QColor(c.get("text_primary", "#F5F5F5"))
        text_muted = QColor(c.get("text_muted", "#60646D"))
        accent = QColor(c.get("accent_silver", "#E5AA57"))
        accent_alt = QColor("#5B9BD5")

        w = self.width()
        h = self.height()

        # Canvas Background
        painter.fillRect(0, 0, w, h, bg_card)

        # Empty State
        if not self._plan or not self._track_layouts or self._total_duration <= 0.1:
            painter.setPen(text_muted)
            f = QFont("Segoe UI", 11)
            painter.setFont(f)
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self._empty_message)
            return

        # Time Scale
        scale = w / self._total_duration
        ruler_h = 20.0
        lanes_y = ruler_h
        lane_h = (h - ruler_h) / 2.0  # Dual staggered lanes for clear transition visualization

        # 1. Draw Top Time Ruler
        painter.setPen(QColor(255, 255, 255, 30))
        painter.drawLine(0, int(ruler_h), w, int(ruler_h))

        font_sm = QFont("Segoe UI", 8)
        font_sm.setPointSize(8)
        painter.setFont(font_sm)

        step_sec = 30.0 if self._total_duration > 180 else (10.0 if self._total_duration > 30 else 2.0)
        curr_t = 0.0
        while curr_t <= self._total_duration:
            x_pos = int(curr_t * scale)
            painter.setPen(QColor(255, 255, 255, 40))
            painter.drawLine(x_pos, int(ruler_h - 4), x_pos, int(ruler_h))
            painter.setPen(text_muted)
            painter.drawText(QRectF(x_pos - 20, 2, 40, 14), Qt.AlignmentFlag.AlignCenter, self._format_time(curr_t))
            curr_t += step_sec

        # 2. Draw Track Blocks
        for tl in self._track_layouts:
            idx = tl["idx"]
            t = tl["track"]
            t_x = tl["start_sec"] * scale
            t_w = max(40.0, tl["duration_sec"] * scale)
            t_y = lanes_y + (0 if idx % 2 == 0 else lane_h)

            rect = QRectF(t_x, t_y + 2, t_w, lane_h - 4)

            # Block Background gradient
            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            if idx % 2 == 0:
                grad.setColorAt(0.0, QColor("#222730"))
                grad.setColorAt(1.0, QColor("#181B22"))
                stroke_color = QColor("#3D4654")
            else:
                grad.setColorAt(0.0, QColor("#28241F"))
                grad.setColorAt(1.0, QColor("#1B1916"))
                stroke_color = QColor("#544A3C")

            painter.setBrush(QBrush(grad))
            painter.setPen(QPen(stroke_color, 1))
            painter.drawRoundedRect(rect, 4, 4)

            # Track Title & Info Badges
            painter.setPen(text_primary)
            font_title = QFont("Segoe UI", 9)
            font_title.setPointSize(9)
            font_title.setBold(True)
            painter.setFont(font_title)

            title_str = f"Track {idx + 1}: {t.title}"
            painter.drawText(QRectF(rect.x() + 8, rect.y() + 4, rect.width() - 16, 16), Qt.AlignmentFlag.AlignLeft, title_str)

            # Acoustic Pills
            font_pill = QFont("Segoe UI", 8)
            font_pill.setPointSize(8)
            painter.setFont(font_pill)
            painter.setPen(accent)
            pill_str = f"⚡ {t.bpm:.1f} BPM  |  🔑 {t.camelot_key} ({t.musical_key})  |  ⏱ {self._format_time(t.duration_seconds)}"
            painter.drawText(QRectF(rect.x() + 8, rect.y() + 22, rect.width() - 16, 14), Qt.AlignmentFlag.AlignLeft, pill_str)

            # Timestamps
            painter.setPen(text_muted)
            painter.drawText(QRectF(rect.x() + 8, rect.bottom() - 16, 60, 12), Qt.AlignmentFlag.AlignLeft, self._format_time(tl["start_sec"]))
            painter.drawText(QRectF(rect.right() - 68, rect.bottom() - 16, 60, 12), Qt.AlignmentFlag.AlignRight, self._format_time(tl["end_sec"]))

        # 3. Draw Transition Overlap Zones
        for trl in self._transition_layouts:
            tr = trl["transition"]
            tx_start = trl["start_sec"] * scale
            tx_end = trl["end_sec"] * scale
            tx_w = max(24.0, tx_end - tx_start)

            t_rect = QRectF(tx_start, lanes_y, tx_w, h - lanes_y)

            # Overlap Zone Translucent Highlight
            painter.setBrush(QColor(229, 170, 87, 25))
            painter.setPen(QPen(QColor(229, 170, 87, 120), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(t_rect, 4, 4)

            # Draw Equal-Power Crossfade Curves inside transition
            fade_y_mid = lanes_y + ((h - lanes_y) / 2.0)
            painter.setPen(QPen(QColor(229, 170, 87, 200), 1.5))
            # Outgoing Track Fade Out curve
            p1 = QPointF(tx_start, lanes_y + 8)
            p2 = QPointF(tx_end, h - 8)
            painter.drawLine(p1, p2)

            # Incoming Track Fade In curve
            painter.setPen(QPen(QColor(91, 155, 213, 200), 1.5))
            p3 = QPointF(tx_start, h - 8)
            p4 = QPointF(tx_end, lanes_y + 8)
            painter.drawLine(p3, p4)

            # Transition Type Badge
            badge_rect = QRectF(tx_start - 35, fade_y_mid - 12, max(70.0, tx_w + 70.0), 24)
            painter.setBrush(QColor(15, 17, 21, 230))
            painter.setPen(QPen(QColor(229, 170, 87, 220), 1))
            painter.drawRoundedRect(badge_rect, 12, 12)

            painter.setPen(text_primary)
            font_tr = QFont("Segoe UI", 8)
            font_tr.setPointSize(8)
            font_tr.setBold(True)
            painter.setFont(font_tr)
            badge_text = f"⇄ {tr.transition_type} ({tr.duration_seconds:.1f}s)"
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # 4. Optional Waveform Overlay if rendered peaks available
        if self._peaks is not None and len(self._peaks) > 0:
            num_bars = min(w // 3, len(self._peaks))
            if num_bars > 0:
                step = len(self._peaks) / float(num_bars)
                bar_w = max(1.5, (w / num_bars) - 1.0)
                wave_y = h - 30
                wave_h = 24.0
                progress_x = int(self._playhead_ratio * w)

                for i in range(num_bars):
                    idx = int(i * step)
                    val = float(self._peaks[idx])
                    bh = max(2.0, val * wave_h)
                    bx = i * (w / num_bars)
                    by = wave_y + (wave_h - bh) / 2.0

                    if bx <= progress_x:
                        painter.fillRect(QRectF(bx, by, bar_w, bh), QColor("#E5AA57"))
                    else:
                        painter.fillRect(QRectF(bx, by, bar_w, bh), QColor(255, 255, 255, 50))

        # 5. Draw Playhead
        playhead_x = int(self._playhead_ratio * w)
        if 0 <= playhead_x <= w:
            painter.setPen(QPen(QColor("#E5AA57"), 2))
            painter.drawLine(playhead_x, 0, playhead_x, h)

            # Playhead time tooltip
            curr_sec = self._playhead_ratio * self._total_duration
            ptip_text = self._format_time(curr_sec)
            tip_rect = QRectF(max(0, min(w - 50, playhead_x - 25)), 2, 50, 16)
            painter.setBrush(QColor("#E5AA57"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(tip_rect, 3, 3)
            painter.setPen(QColor("#000000"))
            painter.setFont(font_sm)
            painter.drawText(tip_rect, Qt.AlignmentFlag.AlignCenter, ptip_text)

        # 6. Hover indicator
        if self._hover_ratio is not None and self._hover_ratio != self._playhead_ratio:
            hx = int(self._hover_ratio * w)
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DashLine))
            painter.drawLine(hx, 0, hx, h)
