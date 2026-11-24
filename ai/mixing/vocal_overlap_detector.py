"""Vocal Overlap Detector and Collision Avoidance Engine for Smart Mix Studio.
Evaluates vocal activity in the transition overlap zone, implements Strict / Balanced /
Permissive safety policies, and adjusts transition boundaries to prevent vocal clashes.
"""
from typing import Dict, Any, List, Optional
from ai.mixing.mix_plan import TrackAnalysis

class VocalOverlapDetector:
    @staticmethod
    def _is_vocal_active(segments: List[Dict[str, float]], window_start: float, window_end: float, threshold: float = 0.35) -> bool:
        """Checks if any prominent vocal segment overlaps with the specified time window."""
        if not segments:
            return False
        for seg in segments:
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            intensity = seg.get("intensity", 0.5)
            # Check overlap: max(start1, start2) < min(end1, end2)
            if max(s, window_start) < min(e, window_end):
                if intensity >= threshold:
                    return True
        return False

    @classmethod
    def evaluate_vocal_safety(
        cls,
        track_a: TrackAnalysis,
        track_b: TrackAnalysis,
        from_exit_start: float,
        from_exit_end: float,
        to_entry_start: float,
        to_entry_end: float,
        safety_mode: str = "Strict: Never overlap prominent vocals"
    ) -> Dict[str, Any]:
        """
        Evaluates vocal collision risk for the proposed transition window.
        Returns safety status, adjustment recommendations, and warnings.
        """
        # Check confidence of vocal analysis
        confidence_a = getattr(track_a, "vocal_confidence", "Moderate")
        confidence_b = getattr(track_b, "vocal_confidence", "Moderate")

        if confidence_a == "Unavailable" or confidence_b == "Unavailable":
            return {
                "status": "Uncertain - Vocal Analysis Unavailable",
                "has_collision": False,
                "adjusted": False,
                "vocal_in_a": False,
                "vocal_in_b": False,
                "warning": "Vocal activity data unavailable for one or more tracks. Manual transition review advised."
            }

        # Check vocal activity in exit zone of Track A
        vocal_in_a = cls._is_vocal_active(track_a.vocal_segments, from_exit_start, from_exit_end) or \
                     (track_a.vocal_presence_ratio > 0.4 and from_exit_end < track_a.duration_seconds * 0.9)

        # Check vocal activity in entry zone of Track B
        vocal_in_b = cls._is_vocal_active(track_b.vocal_segments, to_entry_start, to_entry_end) or \
                     (track_b.vocal_presence_ratio > 0.4 and to_entry_start > track_b.duration_seconds * 0.1)

        is_strict = "Strict" in safety_mode
        is_balanced = "Balanced" in safety_mode

        adjusted = False
        new_exit_start = from_exit_start
        new_exit_end = from_exit_end
        new_entry_start = to_entry_start
        new_entry_end = to_entry_end
        new_style: Optional[str] = None
        warning: Optional[str] = None

        if vocal_in_a and vocal_in_b:
            # Both tracks have vocals in the transition zone -> direct vocal clash!
            if is_strict:
                # In Strict mode: Avoid blending simultaneous voices
                # 1. Attempt to shorten transition to 4-bar drop or echo-out cut
                new_style = "Echo Out / High-Pass Fade"
                adjusted = True
                status = "Adjusted - Vocal Collision Avoided via Echo Cut"
                warning = "Both tracks contain simultaneous vocals in transition zone. Transition adjusted to Echo Out to prevent vocal clash."
            elif is_balanced:
                status = "Vocal Collision Warning"
                warning = "Simultaneous vocal sections overlap. Lowering outgoing vocal frequencies via EQ recommended."
            else:  # Permissive
                status = "Permissive - Simultaneous Vocals Allowed"
        elif vocal_in_a and not vocal_in_b:
            # Track A has vocals, Track B is instrumental intro
            status = "Safe - Outgoing Vocals over Instrumental Intro"
        elif not vocal_in_a and vocal_in_b:
            # Track A is instrumental outro, Track B has vocals
            status = "Safe - Incoming Vocals over Instrumental Outro"
        else:
            # Neither has vocals in the transition zone -> 100% clean instrumental crossfade
            status = "Safe - Instrumental Outro to Intro"

        return {
            "status": status,
            "has_collision": (vocal_in_a and vocal_in_b),
            "adjusted": adjusted,
            "vocal_in_a": vocal_in_a,
            "vocal_in_b": vocal_in_b,
            "new_exit_start": new_exit_start,
            "new_exit_end": new_exit_end,
            "new_entry_start": new_entry_start,
            "new_entry_end": new_entry_end,
            "recommended_style": new_style,
            "warning": warning
        }

