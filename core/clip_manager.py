"""Interactive Multitrack Clip Management and Rendering Engine for Audio Studio AI.
Supports non-destructive clip editing:
- Selection, drag-move, left-edge start trim, right-edge end trim.
- Split at playhead.
- Delete, merge/consolidate, and equal-power crossfading.
- 32-bit floating-point multitrack summing with track volume faders, mute/solo buses,
  ITU-R BS.1770 -14 LUFS loudness normalisation, and MP3 master export.
"""
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

from core.audio_format import decode_mp3_to_numpy, encode_array_to_mp3
from ai.separation.audio_normalizer import AudioNormalizer


class TimelineClip:
    """Non-destructive audio clip on the multitrack timeline."""

    def __init__(
        self,
        file_path: str,
        name: Optional[str] = None,
        timeline_start: float = 0.0,
        in_point: float = 0.0,
        duration: Optional[float] = None,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
        clip_id: Optional[str] = None
    ):
        self.clip_id = clip_id or str(uuid.uuid4())[:8]
        self.file_path = str(Path(file_path).resolve())
        self.name = name or Path(file_path).stem

        # Decode source audio into memory
        self.source_data, self.sr = decode_mp3_to_numpy(self.file_path, target_sr=44100)
        if self.source_data.ndim == 1:
            self.source_data = np.vstack([self.source_data, self.source_data])

        self.source_duration = self.source_data.shape[1] / float(self.sr)

        self.timeline_start = max(0.0, float(timeline_start))
        self.in_point = max(0.0, min(self.source_duration, float(in_point)))

        if duration is None:
            self.duration = max(0.1, self.source_duration - self.in_point)
        else:
            self.duration = max(0.1, min(self.source_duration - self.in_point, float(duration)))

        self.fade_in = max(0.0, float(fade_in))
        self.fade_out = max(0.0, float(fade_out))
        self.is_selected = False

        # Precompute visual waveform peaks for the clip
        self._peaks_cache: Optional[np.ndarray] = None
        self._update_peaks()

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.duration

    def _update_peaks(self, num_peaks: int = 120):
        """Generates peak envelope for rendering inside the clip bounding box."""
        sub = self.get_rendered_audio()
        if sub.shape[1] == 0:
            self._peaks_cache = np.zeros(num_peaks, dtype=np.float32)
            return

        mono = np.mean(np.abs(sub), axis=0)
        chunk = max(1, len(mono) // num_peaks)
        peaks = []
        for i in range(num_peaks):
            s = i * chunk
            e = min(len(mono), s + chunk)
            if s >= len(mono):
                peaks.append(0.0)
            else:
                peaks.append(float(np.max(mono[s:e])))

        max_val = max(1e-4, max(peaks))
        self._peaks_cache = (np.array(peaks, dtype=np.float32) / max_val)

    def get_rendered_audio(self) -> np.ndarray:
        """Extracts the trimmed slice of audio and applies fade in / fade out."""
        start_samp = int(self.in_point * self.sr)
        end_samp = min(self.source_data.shape[1], int((self.in_point + self.duration) * self.sr))

        if start_samp >= end_samp:
            return np.zeros((2, 0), dtype=np.float32)

        sliced = self.source_data[:, start_samp:end_samp].copy()
        n_samples = sliced.shape[1]

        # Apply fade in
        if self.fade_in > 0:
            fade_in_samples = min(n_samples // 2, int(self.fade_in * self.sr))
            if fade_in_samples > 0:
                t = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
                curve = np.sin(t * (np.pi / 2.0))
                sliced[:, :fade_in_samples] *= curve

        # Apply fade out
        if self.fade_out > 0:
            fade_out_samples = min(n_samples // 2, int(self.fade_out * self.sr))
            if fade_out_samples > 0:
                t = np.linspace(0.0, 1.0, fade_out_samples, dtype=np.float32)
                curve = np.cos(t * (np.pi / 2.0))
                sliced[:, -fade_out_samples:] *= curve

        return sliced

    def trim_start(self, delta_sec: float):
        """Adjusts the in_point (start trim) and shifts timeline_start accordingly."""
        new_in = max(0.0, min(self.in_point + self.duration - 0.1, self.in_point + delta_sec))
        actual_delta = new_in - self.in_point
        self.in_point = new_in
        self.duration = max(0.1, self.duration - actual_delta)
        self.timeline_start = max(0.0, self.timeline_start + actual_delta)
        self._update_peaks()

    def trim_end(self, delta_sec: float):
        """Adjusts the end trim (duration)."""
        max_dur = self.source_duration - self.in_point
        self.duration = max(0.1, min(max_dur, self.duration + delta_sec))
        self._update_peaks()

    def split(self, playhead_sec: float) -> Optional[Tuple['TimelineClip', 'TimelineClip']]:
        """
        Splits clip into two independent clips at the specified timeline playhead position.
        Returns (left_clip, right_clip) or None if playhead is outside the clip.
        """
        if playhead_sec <= self.timeline_start + 0.05 or playhead_sec >= self.timeline_end - 0.05:
            return None

        split_offset = playhead_sec - self.timeline_start
        dur_left = split_offset
        dur_right = self.duration - split_offset

        left_clip = TimelineClip(
            file_path=self.file_path,
            name=f"{self.name} (Part 1)",
            timeline_start=self.timeline_start,
            in_point=self.in_point,
            duration=dur_left,
            fade_in=self.fade_in,
            fade_out=0.01,
            clip_id=str(uuid.uuid4())[:8]
        )

        right_clip = TimelineClip(
            file_path=self.file_path,
            name=f"{self.name} (Part 2)",
            timeline_start=playhead_sec,
            in_point=self.in_point + split_offset,
            duration=dur_right,
            fade_in=0.01,
            fade_out=self.fade_out,
            clip_id=str(uuid.uuid4())[:8]
        )

        return left_clip, right_clip


class TimelineTrack:
    """Manages audio clips, volume fader, and mute/solo controls for a single track lane."""

    def __init__(self, track_num: int, name: str = "Audio Track"):
        self.track_num = track_num
        self.name = name
        self.volume: float = 0.8  # 0.0 to 1.0 (80% default)
        self.is_muted: bool = False
        self.is_solo: bool = False
        self.clips: List[TimelineClip] = []

    def add_clip(self, file_path: str, timeline_start: float = 0.0, name: Optional[str] = None) -> TimelineClip:
        clip = TimelineClip(file_path=file_path, timeline_start=timeline_start, name=name)
        self.clips.append(clip)
        self.sort_clips()
        return clip

    def remove_clip(self, clip_id: str) -> bool:
        initial = len(self.clips)
        self.clips = [c for c in self.clips if c.clip_id != clip_id]
        return len(self.clips) < initial

    def sort_clips(self):
        self.clips.sort(key=lambda c: c.timeline_start)

    def get_selected_clip(self) -> Optional[TimelineClip]:
        for c in self.clips:
            if c.is_selected:
                return c
        return None

    def deselect_all(self):
        for c in self.clips:
            c.is_selected = False

    def split_selected_clip(self, playhead_sec: float) -> bool:
        sel = self.get_selected_clip()
        if not sel:
            return False

        res = sel.split(playhead_sec)
        if not res:
            return False

        left, right = res
        idx = self.clips.index(sel)
        self.clips[idx] = left
        self.clips.insert(idx + 1, right)
        right.is_selected = True
        self.sort_clips()
        return True

    def merge_selected_clips(self) -> bool:
        """Consolidates all selected clips or adjacent clips on this track into a single rendered clip."""
        selected = [c for c in self.clips if c.is_selected]
        if len(selected) < 2:
            return False

        selected.sort(key=lambda c: c.timeline_start)
        min_start = selected[0].timeline_start
        max_end = max(c.timeline_end for c in selected)
        total_duration = max_end - min_start
        sr = 44100
        total_samples = int(total_duration * sr)

        merged_data = np.zeros((2, total_samples), dtype=np.float32)
        for c in selected:
            audio = c.get_rendered_audio()
            offset_samp = int((c.timeline_start - min_start) * sr)
            end_samp = min(total_samples, offset_samp + audio.shape[1])
            audio_slice = audio[:, :end_samp - offset_samp]
            merged_data[:, offset_samp:end_samp] += audio_slice

        # Save consolidated clip to a cache file
        out_p = Path(selected[0].file_path).parent / f"merged_{uuid.uuid4().hex[:6]}.mp3"
        encode_array_to_mp3(merged_data, str(out_p), sample_rate=sr, bitrate="320k")

        # Replace selected clips with new merged clip
        merged_clip = TimelineClip(
            file_path=str(out_p),
            name=f"Merged: {selected[0].name} + {len(selected)-1} clips",
            timeline_start=min_start,
            duration=total_duration
        )
        merged_clip.is_selected = True

        for c in selected:
            self.clips.remove(c)
        self.clips.append(merged_clip)
        self.sort_clips()
        return True

    def apply_crossfade(self, crossfade_sec: float = 1.0) -> bool:
        """Applies an equal-power crossfade between adjacent overlapping or touching clips."""
        if len(self.clips) < 2:
            return False

        self.sort_clips()
        applied = False
        for i in range(len(self.clips) - 1):
            c1 = self.clips[i]
            c2 = self.clips[i + 1]
            gap = c2.timeline_start - c1.timeline_end
            if gap <= 0.1:  # Touching or overlapping
                c1.fade_out = crossfade_sec
                c2.fade_in = crossfade_sec
                c1._update_peaks()
                c2._update_peaks()
                applied = True
        return applied

    def render_track(self, max_duration_sec: float, sr: int = 44100) -> np.ndarray:
        """Renders the entire track into a continuous 2D float32 audio buffer."""
        total_samples = int(max_duration_sec * sr)
        buffer = np.zeros((2, total_samples), dtype=np.float32)

        if self.is_muted:
            return buffer

        gain = self.volume
        for c in self.clips:
            audio = c.get_rendered_audio()
            start_samp = int(c.timeline_start * sr)
            if start_samp >= total_samples:
                continue
            end_samp = min(total_samples, start_samp + audio.shape[1])
            clip_slice = audio[:, :end_samp - start_samp] * gain
            buffer[:, start_samp:end_samp] += clip_slice

        return buffer
