"""Animation Manager for Audio Studio AI.
Coordinates UI transitions, fade effects, hover animations, and respects reduced motion settings.
"""
from typing import Optional, Callable
from PySide6.QtCore import (
    QObject, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, QPoint, QRect, Signal, Property
)
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect

from database.database import DatabaseManager

class AnimationManager(QObject):
    _instance = None

    @classmethod
    def get_instance(cls) -> "AnimationManager":
        if cls._instance is None:
            cls._instance = AnimationManager()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = DatabaseManager()
        self._enabled: bool = self._db.get_setting("animations_enabled", True)
        self._reduced_motion: bool = self._db.get_setting("reduced_motion", False)
        self._duration_multiplier: float = 1.0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @is_enabled.setter
    def is_enabled(self, val: bool):
        self._enabled = bool(val)
        self._db.set_setting("animations_enabled", self._enabled)

    @property
    def reduced_motion(self) -> bool:
        return self._reduced_motion

    @reduced_motion.setter
    def reduced_motion(self, val: bool):
        self._reduced_motion = bool(val)
        self._db.set_setting("reduced_motion", self._reduced_motion)

    def fade_transition(
        self,
        widget: QWidget,
        duration: int = 220,
        start_opacity: float = 0.0,
        end_opacity: float = 1.0,
        on_finished: Optional[Callable] = None
    ) -> Optional[QPropertyAnimation]:
        if not self._enabled or self._reduced_motion:
            effect = widget.graphicsEffect()
            if isinstance(effect, QGraphicsOpacityEffect):
                effect.setOpacity(end_opacity)
            if on_finished:
                on_finished()
            return None

        # Check or create opacity effect
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)

        effect.setOpacity(start_opacity)
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(int(duration * self._duration_multiplier))
        anim.setStartValue(start_opacity)
        anim.setEndValue(end_opacity)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        if on_finished:
            anim.finished.connect(on_finished)

        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        return anim

    def slide_fade_in(
        self,
        widget: QWidget,
        duration: int = 240,
        offset_y: int = 8,
        on_finished: Optional[Callable] = None
    ):
        """Elegant slide up slightly while fading in."""
        if not self._enabled or self._reduced_motion:
            if on_finished:
                on_finished()
            return

        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)

        group = QParallelAnimationGroup(widget)

        # Fade
        fade_anim = QPropertyAnimation(effect, b"opacity")
        fade_anim.setDuration(duration)
        fade_anim.setStartValue(0.1)
        fade_anim.setEndValue(1.0)
        fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(fade_anim)

        # Slide
        pos = widget.pos()
        slide_anim = QPropertyAnimation(widget, b"pos")
        slide_anim.setDuration(duration)
        slide_anim.setStartValue(QPoint(pos.x(), pos.y() + offset_y))
        slide_anim.setEndValue(pos)
        slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(slide_anim)

        if on_finished:
            group.finished.connect(on_finished)

        group.start(QParallelAnimationGroup.DeletionPolicy.DeleteWhenStopped)

    def animate_width(
        self,
        widget: QWidget,
        start_w: int,
        end_w: int,
        duration: int = 220,
        on_finished: Optional[Callable] = None
    ) -> Optional[QPropertyAnimation]:
        if not self._enabled or self._reduced_motion:
            widget.setFixedWidth(end_w)
            if on_finished:
                on_finished()
            return None

        anim = QPropertyAnimation(widget, b"maximumWidth", widget)
        anim.setDuration(int(duration * self._duration_multiplier))
        anim.setStartValue(start_w)
        anim.setEndValue(end_w)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        widget.setMinimumWidth(min(start_w, end_w))

        def finish_handler():
            widget.setFixedWidth(end_w)
            if on_finished:
                on_finished()

        anim.finished.connect(finish_handler)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        return anim
