"""Central Audio Playback Service for Audio Studio AI.
Single source of truth for audio playback using PySide6.QtMultimedia.
"""
from typing import Optional, List
import random
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from database.models import Track
from database.database import DatabaseManager

class PlaybackEngine(QObject):
    _instance = None

    track_loaded = Signal(object)          # Track
    state_changed = Signal(bool)           # is_playing
    position_changed = Signal(int, int)    # current_ms, duration_ms
    volume_changed = Signal(float)         # 0.0 - 1.0
    playback_rate_changed = Signal(float)  # 0.5 - 2.0
    queue_updated = Signal()

    @classmethod
    def get_instance(cls) -> "PlaybackEngine":
        if cls._instance is None:
            cls._instance = PlaybackEngine()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)

        self._current_track: Optional[Track] = None
        self._queue: List[Track] = []
        self._queue_index: int = -1
        self._is_shuffle: bool = False
        self._repeat_mode: int = 0  # 0: off, 1: repeat all, 2: repeat one
        self._duration_ms: int = 0

        # Wire QtMultimedia signals
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

        # Restore volume setting
        db = DatabaseManager()
        saved_vol = db.get_setting("playback_volume", 0.8)
        self._audio_output.setVolume(saved_vol)

    @property
    def current_track(self) -> Optional[Track]:
        return self._current_track

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def is_shuffle(self) -> bool:
        return self._is_shuffle

    @property
    def repeat_mode(self) -> int:
        return self._repeat_mode

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    def set_queue(self, tracks: List[Track], start_index: int = 0):
        self._queue = list(tracks)
        self._queue_index = max(0, min(start_index, len(self._queue) - 1)) if self._queue else -1
        self.queue_updated.emit()
        if 0 <= self._queue_index < len(self._queue):
            self.play_track(self._queue[self._queue_index])

    def play_track(self, track: Track):
        self._current_track = track
        file_path = Path(track.file_path)
        if file_path.exists():
            url = QUrl.fromLocalFile(str(file_path.resolve()))
            self._player.setSource(url)
            self._player.play()
            self.track_loaded.emit(track)

            if track.id:
                DatabaseManager().record_track_played(track.id)
        else:
            # File missing or placeholder track
            self.track_loaded.emit(track)

    def play_file(self, file_path_str: str, title: str = "", artist: str = ""):
        p = Path(file_path_str)
        track = Track(
            file_path=str(p.resolve()),
            title=title if title else p.stem,
            artist=artist if artist else "Local Audio",
            format=p.suffix.lower()
        )
        self.play_track(track)

    def toggle_play_pause(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            if self._player.source().isEmpty() and self._current_track:
                self.play_track(self._current_track)
            else:
                self._player.play()

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()

    def seek_ms(self, pos_ms: int):
        self._player.setPosition(pos_ms)

    def seek_ratio(self, ratio: float):
        if self._duration_ms > 0:
            target_ms = int(ratio * self._duration_ms)
            self.seek_ms(target_ms)

    def set_volume(self, vol: float):
        v = max(0.0, min(1.0, vol))
        self._audio_output.setVolume(v)
        DatabaseManager().set_setting("playback_volume", v)
        self.volume_changed.emit(v)

    def get_volume(self) -> float:
        return self._audio_output.volume()

    def set_playback_rate(self, rate: float):
        r = max(0.25, min(2.5, rate))
        self._player.setPlaybackRate(r)
        self.playback_rate_changed.emit(r)

    def toggle_shuffle(self) -> bool:
        self._is_shuffle = not self._is_shuffle
        return self._is_shuffle

    def cycle_repeat(self) -> int:
        self._repeat_mode = (self._repeat_mode + 1) % 3
        return self._repeat_mode

    def next_track(self):
        if not self._queue:
            return
        if self._is_shuffle:
            self._queue_index = random.randint(0, len(self._queue) - 1)
        else:
            self._queue_index = (self._queue_index + 1) % len(self._queue)
        self.play_track(self._queue[self._queue_index])

    def previous_track(self):
        if not self._queue:
            return
        if self._player.position() > 3000:
            # If playing more than 3 sec, restart track
            self.seek_ms(0)
            return
        self._queue_index = (self._queue_index - 1) % len(self._queue)
        self.play_track(self._queue[self._queue_index])

    def _on_playback_state_changed(self, state):
        is_playing = (state == QMediaPlayer.PlaybackState.PlayingState)
        self.state_changed.emit(is_playing)

    def _on_position_changed(self, pos_ms: int):
        self.position_changed.emit(pos_ms, self._duration_ms)

    def _on_duration_changed(self, dur_ms: int):
        self._duration_ms = dur_ms
        self.position_changed.emit(self._player.position(), self._duration_ms)

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self._repeat_mode == 2:
                # Repeat one
                self.seek_ms(0)
                self.play()
            elif self._repeat_mode == 1 or (self._queue_index + 1 < len(self._queue)):
                self.next_track()
