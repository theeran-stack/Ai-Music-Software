"""Master AI Mix Planning Engine for Smart Mix Studio.
Coordinates track acoustic analysis, harmonic compatibility, phrase boundary alignment,
vocal collision avoidance, and transition plan generation.
"""
from typing import List, Dict, Any, Optional, Callable, Union
import time
from pathlib import Path

from ai.mixing.mix_plan import TrackAnalysis, TransitionPlan, MixPlan
from ai.mixing.audio_analyzer import AudioAnalyzer
from ai.mixing.transition_analyzer import TransitionAnalyzer
from ai.mixing.phrase_analyzer import PhraseAnalyzer
from ai.mixing.vocal_overlap_detector import VocalOverlapDetector

class MixPlanner:
    def __init__(self):
        pass

    @classmethod
    def plan_mix(
        cls,
        tracks: List[Union[str, TrackAnalysis]],
        settings: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> MixPlan:
        """
        Generates a comprehensive, musically coherent continuous DJ mix plan.
        - Validates tracks (minimum 2).
        - Analyzes acoustic metadata for any unanalyzed tracks.
        - Plans beat-matched, harmonically aligned transitions between consecutive tracks.
        - Enforces vocal safety policies.
        """
        if len(tracks) < 2:
            raise ValueError("At least 2 tracks are required to generate an AI mix plan.")

        cfg = {
            "transition_style": "AI-Selected (Smart Adaptation)",
            "transition_duration_bars": 8,
            "vocal_safety": "Strict: Never overlap prominent vocals"
        }
        if settings:
            cfg.update(settings)

        num_tracks = len(tracks)
        analyzed_tracks: List[TrackAnalysis] = []

        # 1. Acoustic Analysis Phase
        for idx, item in enumerate(tracks):
            if isinstance(item, TrackAnalysis):
                analyzed_tracks.append(item)
            elif isinstance(item, str):
                if progress_callback:
                    p = Path(item)
                    progress_callback(
                        0.05 + 0.45 * (idx / num_tracks),
                        f"Analyzing acoustic characteristics ({idx + 1}/{num_tracks}): {p.name}..."
                    )
                ta = AudioAnalyzer.analyze_track(item)
                analyzed_tracks.append(ta)
            elif isinstance(item, dict):
                # Dict from UI queue
                fpath = item.get("path")
                if not fpath or not Path(fpath).exists():
                    raise FileNotFoundError(f"Track file not found: {fpath}")
                if progress_callback:
                    p = Path(fpath)
                    progress_callback(
                        0.05 + 0.45 * (idx / num_tracks),
                        f"Analyzing acoustic characteristics ({idx + 1}/{num_tracks}): {p.name}..."
                    )
                ta = AudioAnalyzer.analyze_track(fpath)
                analyzed_tracks.append(ta)
            else:
                raise TypeError(f"Unsupported track input type: {type(item)}")

        # 2. Transition Planning Phase
        transitions: List[TransitionPlan] = []
        num_transitions = num_tracks - 1

        for idx in range(num_transitions):
            track_a = analyzed_tracks[idx]
            track_b = analyzed_tracks[idx + 1]

            if progress_callback:
                progress_callback(
                    0.55 + 0.35 * (idx / max(1, num_transitions)),
                    f"Planning transition {idx + 1}/{num_transitions}: '{track_a.title}' ➔ '{track_b.title}'..."
                )

            # Analyze Transition Style & Compatibility
            rec = TransitionAnalyzer.recommend_transition(
                bpm1=track_a.bpm,
                bpm2=track_b.bpm,
                key1=track_a.camelot_key,
                key2=track_b.camelot_key,
                requested_style=cfg.get("transition_style", "AI-Selected (Smart Adaptation)"),
                requested_duration_bars=int(cfg.get("transition_duration_bars", 8))
            )

            # Determine Phrase Boundaries and Placement
            placement = PhraseAnalyzer.find_optimal_transition_points(
                track_a_duration=track_a.duration_seconds,
                track_a_bpm=track_a.bpm,
                track_b_duration=track_b.duration_seconds,
                track_b_bpm=track_b.bpm,
                transition_duration_bars=rec["duration_bars"],
                track_a_downbeat=track_a.first_downbeat_sec,
                track_b_downbeat=track_b.first_downbeat_sec
            )

            # Evaluate Vocal Overlap & Safety
            vocal_safety = VocalOverlapDetector.evaluate_vocal_safety(
                track_a=track_a,
                track_b=track_b,
                from_exit_start=placement["from_exit_start_sec"],
                from_exit_end=placement["from_exit_end_sec"],
                to_entry_start=placement["to_entry_start_sec"],
                to_entry_end=placement["to_entry_end_sec"],
                safety_mode=cfg.get("vocal_safety", "Strict: Never overlap prominent vocals")
            )

            final_style = vocal_safety.get("recommended_style") or rec["style"]
            final_exit_start = vocal_safety["new_exit_start"]
            final_exit_end = vocal_safety["new_exit_end"]
            final_entry_start = vocal_safety["new_entry_start"]
            final_entry_end = vocal_safety["new_entry_end"]

            warnings = list(rec["warnings"])
            if vocal_safety.get("warning"):
                warnings.append(vocal_safety["warning"])

            trans_plan = TransitionPlan(
                from_track_idx=idx,
                to_track_idx=idx + 1,
                from_track_title=track_a.title,
                to_track_title=track_b.title,
                transition_type=final_style,
                duration_bars=placement["duration_bars"],
                duration_seconds=placement["duration_seconds"],
                from_exit_start_sec=final_exit_start,
                from_exit_end_sec=final_exit_end,
                to_entry_start_sec=final_entry_start,
                to_entry_end_sec=final_entry_end,
                bpm_source=track_a.bpm,
                bpm_target=track_b.bpm,
                bpm_mix=placement["mix_bpm"],
                bpm_diff_pct=rec["tempo"]["diff_pct"],
                bpm_compatibility_score=rec["tempo"]["score"],
                key_source=track_a.musical_key,
                key_target=track_b.musical_key,
                camelot_source=track_a.camelot_key,
                camelot_target=track_b.camelot_key,
                camelot_relation=rec["harmonic"]["relation"],
                key_compatibility_score=rec["harmonic"]["score"],
                vocal_safety_status=vocal_safety["status"],
                warnings=warnings,
                eq_strategy=rec["eq_strategy"]
            )
            transitions.append(trans_plan)

        # 3. Calculate Total Continuous Mix Duration
        total_track_time = sum(t.duration_seconds for t in analyzed_tracks)
        total_overlap_time = sum(tr.duration_seconds for tr in transitions)
        mix_duration = max(0.0, round(total_track_time - total_overlap_time, 2))

        if progress_callback:
            progress_callback(1.0, f"AI Mix Plan generated successfully ({len(transitions)} transitions, {round(mix_duration/60.0, 1)} mins).")

        return MixPlan(
            name=f"Smart Mix - {analyzed_tracks[0].title} to {analyzed_tracks[-1].title}",
            tracks=analyzed_tracks,
            transitions=transitions,
            total_duration_seconds=mix_duration,
            settings=cfg
        )

