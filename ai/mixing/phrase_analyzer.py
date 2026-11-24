"""Musical Phrasing and Beat Alignment Engine for Smart Mix Studio.
Detects 4/8/16/32-bar phrase boundaries, snaps transitions to downbeats,
and identifies natural exit and entry transition windows.
"""
from typing import List, Tuple, Dict, Any
import numpy as np

class PhraseAnalyzer:
    @staticmethod
    def calculate_beat_grid(duration_sec: float, bpm: float, first_downbeat_sec: float = 0.0) -> List[float]:
        """Generates exact timestamps for every quarter-note beat across the track duration."""
        if bpm <= 0 or duration_sec <= 0:
            return []
        beat_interval = 60.0 / bpm
        start = max(0.0, first_downbeat_sec)
        num_beats = int((duration_sec - start) / beat_interval)
        return [round(start + i * beat_interval, 3) for i in range(max(0, num_beats))]

    @staticmethod
    def calculate_phrase_boundaries(duration_sec: float, bpm: float, first_downbeat_sec: float = 0.0, bars_per_phrase: int = 8) -> List[float]:
        """
        Calculates timestamps for musical phrase boundaries (e.g. 8-bar / 32-beat boundaries in 4/4 time).
        """
        if bpm <= 0 or duration_sec <= 0:
            return [0.0, duration_sec]

        beat_sec = 60.0 / bpm
        phrase_sec = bars_per_phrase * 4.0 * beat_sec
        if phrase_sec <= 0:
            return [0.0, duration_sec]

        boundaries = []
        curr = max(0.0, first_downbeat_sec)
        while curr < duration_sec:
            boundaries.append(round(curr, 2))
            curr += phrase_sec

        if not boundaries or boundaries[-1] < duration_sec:
            boundaries.append(round(duration_sec, 2))
        return boundaries

    @classmethod
    def find_optimal_transition_points(
        cls,
        track_a_duration: float,
        track_a_bpm: float,
        track_b_duration: float,
        track_b_bpm: float,
        transition_duration_bars: int = 8,
        track_a_downbeat: float = 0.0,
        track_b_downbeat: float = 0.0
    ) -> Dict[str, Any]:
        """
        Finds the exact beat-aligned timestamps for the transition zone between Track A and Track B.
        - Track A: Exit window starts at the beginning of the final phrase or outro section.
        - Track B: Entry window starts at Track B's first downbeat (Intro).
        """
        beat_sec_a = 60.0 / max(30.0, track_a_bpm)
        beat_sec_b = 60.0 / max(30.0, track_b_bpm)

        # Average mix tempo for transition duration calculation
        mix_bpm = (track_a_bpm + track_b_bpm) / 2.0
        mix_beat_sec = 60.0 / max(30.0, mix_bpm)
        transition_duration_sec = round(transition_duration_bars * 4.0 * mix_beat_sec, 2)

        # Cap transition duration so it doesn't exceed track lengths
        max_possible = max(0.5, min(track_a_duration * 0.45, track_b_duration * 0.45))
        if transition_duration_sec > max_possible:
            transition_duration_sec = round(max_possible, 2)

        # Track A: Place transition start at (duration - transition_duration_sec) snapped to nearest bar boundary
        bar_sec_a = 4.0 * beat_sec_a
        target_exit_start = max(0.0, track_a_duration - transition_duration_sec)

        # Snap to nearest downbeat/bar
        bar_idx = int(round((target_exit_start - track_a_downbeat) / max(0.1, bar_sec_a)))
        from_exit_start = max(0.0, track_a_downbeat + bar_idx * bar_sec_a)
        if from_exit_start + transition_duration_sec > track_a_duration:
            from_exit_start = max(0.0, track_a_duration - transition_duration_sec)
        from_exit_end = min(track_a_duration, from_exit_start + transition_duration_sec)

        # Track B: Place transition start at track_b_downbeat (or 0.0)
        to_entry_start = max(0.0, track_b_downbeat)
        if to_entry_start + transition_duration_sec > track_b_duration:
            to_entry_start = max(0.0, track_b_duration - transition_duration_sec)
        to_entry_end = min(track_b_duration, to_entry_start + transition_duration_sec)

        return {
            "duration_bars": transition_duration_bars,
            "duration_seconds": transition_duration_sec,
            "from_exit_start_sec": round(from_exit_start, 2),
            "from_exit_end_sec": round(from_exit_end, 2),
            "to_entry_start_sec": round(to_entry_start, 2),
            "to_entry_end_sec": round(to_entry_end, 2),
            "mix_bpm": round(mix_bpm, 1)
        }

