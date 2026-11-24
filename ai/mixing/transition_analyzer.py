"""Transition & Harmonic Compatibility Analyzer for Smart Mix Studio.
Implements Camelot Wheel harmonic distance, tempo difference evaluation,
energy compatibility, and smart transition style recommendation.
"""
from typing import Tuple, Dict, Any, List

CAMELOT_MAP = {
    # Minor Keys (A)
    "Ab Minor": "1A", "G# Minor": "1A",
    "Eb Minor": "2A", "D# Minor": "2A",
    "Bb Minor": "3A", "A# Minor": "3A",
    "F Minor": "4A",
    "C Minor": "5A",
    "G Minor": "6A",
    "D Minor": "7A",
    "A Minor": "8A",
    "E Minor": "9A",
    "B Minor": "10A",
    "F# Minor": "11A", "Gb Minor": "11A",
    "Db Minor": "12A", "C# Minor": "12A",

    # Major Keys (B)
    "B Major": "1B", "Cb Major": "1B",
    "F# Major": "2B", "Gb Major": "2B",
    "Db Major": "3B", "C# Major": "3B",
    "Ab Major": "4B", "G# Major": "4B",
    "Eb Major": "5B", "D# Major": "5B",
    "Bb Major": "6B", "A# Major": "6B",
    "F Major": "7B",
    "C Major": "8B",
    "G Major": "9B",
    "D Major": "10B",
    "A Major": "11B",
    "E Major": "12B"
}

def normalize_key_to_camelot(key_str: str) -> Tuple[str, str]:
    """Converts key string (e.g. 'A Minor', '8A', 'A Minor (8A)') to (standard_key, camelot)."""
    clean = key_str.strip()
    # Check if already Camelot code like '8A' or '11B'
    for k, v in CAMELOT_MAP.items():
        if clean.upper() == v.upper() or f"({v})" in clean or f"[{v}]" in clean:
            return k, v
        if k.lower() in clean.lower():
            return k, v

    # Fallback to C Major / 8B
    return "C Major", "8B"


class TransitionAnalyzer:
    @staticmethod
    def analyze_harmonic_compatibility(key1_str: str, key2_str: str) -> Dict[str, Any]:
        """
        Calculates Camelot Wheel harmonic distance and relationship between two musical keys.
        Returns compatibility score (0.0 to 1.0), relationship title, and descriptive advice.
        """
        name1, cam1 = normalize_key_to_camelot(key1_str)
        name2, cam2 = normalize_key_to_camelot(key2_str)

        num1 = int("".join(filter(str.isdigit, cam1)) or "8")
        let1 = "".join(filter(str.isalpha, cam1)).upper() or "B"

        num2 = int("".join(filter(str.isdigit, cam2)) or "8")
        let2 = "".join(filter(str.isalpha, cam2)).upper() or "B"

        # Calculate circular distance (1 to 12)
        diff_num = abs(num1 - num2)
        if diff_num > 6:
            diff_num = 12 - diff_num

        same_letter = (let1 == let2)

        if num1 == num2 and same_letter:
            score = 1.0
            relation = "Exact Harmonic Match"
            desc = "Identical scale and tonal center. Flawless harmonic overlay."
            safe_harmonic = True
        elif num1 == num2 and not same_letter:
            score = 0.95
            relation = "Relative Major/Minor"
            desc = f"Shares same key signature ({let1} ➔ {let2}). Dramatic mood shift."
            safe_harmonic = True
        elif diff_num == 1 and same_letter:
            # Check direction (+1 = Energy Boost, -1 = Energy Drop)
            step = (num2 - num1) if abs(num2 - num1) <= 6 else (num2 - num1 - 12 if num2 > num1 else num2 - num1 + 12)
            direction = "+1 Energy Boost" if step > 0 else "-1 Energy Drop"
            score = 0.90
            relation = direction
            desc = f"Adjacent fifths step ({direction}). Smooth natural progression."
            safe_harmonic = True
        elif diff_num == 2 and same_letter:
            score = 0.70
            relation = "+2 Energy Flash"
            desc = "Two-step key jump. High energy transition with noticeable tension."
            safe_harmonic = True
        elif diff_num == 1 and not same_letter:
            score = 0.75
            relation = "Diagonal Harmonic Shift"
            desc = "Subtle mode and scale modulation. Recommended for progressive crossfades."
            safe_harmonic = True
        else:
            score = max(0.2, 0.6 - diff_num * 0.08)
            relation = "Harmonic Key Clash"
            desc = f"Distinct keys with {diff_num} steps of separation. Filter fade, echo out, or drop transition recommended."
            safe_harmonic = False

        return {
            "key_source": name1,
            "key_target": name2,
            "camelot_source": cam1,
            "camelot_target": cam2,
            "camelot_distance": diff_num,
            "score": round(score, 2),
            "relation": relation,
            "description": desc,
            "safe_harmonic": safe_harmonic
        }

    @staticmethod
    def analyze_tempo_compatibility(bpm1: float, bpm2: float) -> Dict[str, Any]:
        """
        Analyzes tempo alignment, difference percentage, and half/double-time relationships.
        """
        if bpm1 <= 0 or bpm2 <= 0:
            return {
                "bpm_source": bpm1,
                "bpm_target": bpm2,
                "diff_bpm": 0.0,
                "diff_pct": 0.0,
                "score": 1.0,
                "is_half_double": False,
                "target_bpm": max(bpm1, bpm2, 120.0),
                "rating": "Default Tempo"
            }

        diff_raw = abs(bpm1 - bpm2)
        base = min(bpm1, bpm2)
        diff_pct = (diff_raw / base) * 100.0

        # Check for half/double-time (e.g. 70 vs 140, 85 vs 170)
        ratio = max(bpm1, bpm2) / min(bpm1, bpm2)
        is_half_double = abs(ratio - 2.0) <= 0.06

        if is_half_double:
            score = 0.95
            rating = "Half/Double-Time Match"
            target_bpm = bpm1  # Lock to incoming tempo or double
        elif diff_pct <= 2.5:
            score = 1.0
            rating = "Flawless Tempo Match"
            target_bpm = round((bpm1 + bpm2) / 2.0, 1)
        elif diff_pct <= 5.0:
            score = 0.90
            rating = "Smooth Tempo Blend"
            target_bpm = round((bpm1 + bpm2) / 2.0, 1)
        elif diff_pct <= 8.5:
            score = 0.75
            rating = "Moderate Tempo Stretch"
            target_bpm = round((bpm1 + bpm2) / 2.0, 1)
        elif diff_pct <= 14.0:
            score = 0.50
            rating = "Wide Tempo Difference (Drop Recommended)"
            target_bpm = bpm2
        else:
            score = 0.25
            rating = "Severe Tempo Clash"
            target_bpm = bpm2

        return {
            "bpm_source": round(bpm1, 1),
            "bpm_target": round(bpm2, 1),
            "diff_bpm": round(diff_raw, 1),
            "diff_pct": round(diff_pct, 1),
            "score": round(score, 2),
            "is_half_double": is_half_double,
            "target_bpm": target_bpm,
            "rating": rating
        }

    @classmethod
    def recommend_transition(
        cls,
        bpm1: float,
        bpm2: float,
        key1: str,
        key2: str,
        requested_style: str = "AI-Selected (Smart Adaptation)",
        requested_duration_bars: int = 8
    ) -> Dict[str, Any]:
        """
        Determines the optimal transition style, duration, and EQ automation strategy.
        """
        harmonic = cls.analyze_harmonic_compatibility(key1, key2)
        tempo = cls.analyze_tempo_compatibility(bpm1, bpm2)

        # Overall compatibility composite score
        overall_score = round(0.55 * harmonic["score"] + 0.45 * tempo["score"], 2)

        style = requested_style
        bars = requested_duration_bars
        eq = {
            "bass_swap": True,
            "highpass_intro": True,
            "curve": "exponential"
        }

        # If user selected AI-Selected, choose the musically best transition:
        if "AI-Selected" in requested_style:
            if not harmonic["safe_harmonic"] or tempo["diff_pct"] > 9.0:
                # Key clash or wide tempo difference -> cut or echo out so clashing tones don't blend
                style = "Echo Out / High-Pass Fade"
                bars = min(bars, 8)
                eq["filter"] = "highpass_sweep"
            elif harmonic["score"] >= 0.90 and tempo["score"] >= 0.85:
                style = "Harmonic Crossfade (Camelot Wheel)"
                bars = max(bars, 8)
                eq["bass_swap"] = True
            else:
                style = "Beat-Matched Phrase Transition"
                bars = requested_duration_bars
        elif "Echo Out" in requested_style:
            style = "Echo Out / High-Pass Fade"
            eq["filter"] = "echo_washout"
        elif "Reverb Tail" in requested_style:
            style = "Reverb Tail Washout"
            eq["filter"] = "reverb_washout"
        elif "Harmonic Crossfade" in requested_style:
            style = "Harmonic Crossfade (Camelot Wheel)"
            eq["bass_swap"] = True
        else:
            style = "Beat-Matched Phrase Transition"

        # Calculate transition duration in seconds based on target mix BPM
        # 1 bar = 4 beats = 4 * (60.0 / target_bpm) seconds
        sec_per_beat = 60.0 / max(30.0, tempo["target_bpm"])
        duration_sec = round(bars * 4.0 * sec_per_beat, 2)

        warnings = []
        if not harmonic["safe_harmonic"]:
            warnings.append(f"Harmonic Clash: {harmonic['key_source']} ({harmonic['camelot_source']}) vs {harmonic['key_target']} ({harmonic['camelot_target']})")
        if tempo["diff_pct"] > 8.0:
            warnings.append(f"Large Tempo Difference: {tempo['diff_bpm']} BPM ({tempo['diff_pct']}%)")

        return {
            "style": style,
            "duration_bars": bars,
            "duration_seconds": duration_sec,
            "harmonic": harmonic,
            "tempo": tempo,
            "overall_score": overall_score,
            "eq_strategy": eq,
            "warnings": warnings
        }

