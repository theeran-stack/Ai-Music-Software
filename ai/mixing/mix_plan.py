"""Data models and serialization for Smart Mix Studio AI Mix Planning.
Represents analyzed track metadata, transition specifications, and full mix plans.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import time
import uuid
from pathlib import Path

@dataclass
class TrackAnalysis:
    path: str
    title: str
    duration_seconds: float
    sample_rate: int = 44100
    bpm: float = 120.0
    musical_key: str = "C Major"
    camelot_key: str = "8B"
    energy_level: float = 0.7
    beat_interval: float = 0.5
    first_downbeat_sec: float = 0.0
    phrase_boundaries_sec: List[float] = field(default_factory=list)
    vocal_presence_ratio: float = 0.0
    vocal_segments: List[Dict[str, float]] = field(default_factory=list)  # [{"start": 0.0, "end": 10.0, "intensity": 0.8}]
    vocal_confidence: str = "High"  # High, Moderate, Low, Unavailable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackAnalysis":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TransitionPlan:
    from_track_idx: int
    to_track_idx: int
    from_track_title: str
    to_track_title: str
    transition_type: str = "Beat-Matched Phrase Transition"
    duration_bars: int = 8
    duration_seconds: float = 16.0
    from_exit_start_sec: float = 0.0
    from_exit_end_sec: float = 0.0
    to_entry_start_sec: float = 0.0
    to_entry_end_sec: float = 0.0
    bpm_source: float = 120.0
    bpm_target: float = 120.0
    bpm_mix: float = 120.0
    bpm_diff_pct: float = 0.0
    bpm_compatibility_score: float = 1.0  # 0.0 to 1.0
    key_source: str = "C Major"
    key_target: str = "C Major"
    camelot_source: str = "8B"
    camelot_target: str = "8B"
    camelot_relation: str = "Exact Match"
    key_compatibility_score: float = 1.0  # 0.0 to 1.0
    vocal_safety_status: str = "Safe - No Vocal Collision"
    warnings: List[str] = field(default_factory=list)
    eq_strategy: Dict[str, Any] = field(default_factory=lambda: {
        "bass_swap": True,
        "highpass_intro": True,
        "curve": "exponential"
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionPlan":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MixPlan:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Smart Mix Project"
    tracks: List[TrackAnalysis] = field(default_factory=list)
    transitions: List[TransitionPlan] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "transition_style": "AI-Selected (Smart Adaptation)",
        "transition_duration_bars": 8,
        "vocal_safety": "Strict: Never overlap prominent vocals"
    })
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tracks": [t.to_dict() for t in self.tracks],
            "transitions": [tr.to_dict() for tr in self.transitions],
            "total_duration_seconds": self.total_duration_seconds,
            "settings": self.settings,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixPlan":
        plan = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Smart Mix Project"),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            settings=data.get("settings", {}),
            created_at=data.get("created_at", time.time())
        )
        plan.tracks = [TrackAnalysis.from_dict(t) for t in data.get("tracks", [])]
        plan.transitions = [TransitionPlan.from_dict(tr) for tr in data.get("transitions", [])]
        return plan

    def save_json(self, file_path: str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "MixPlan":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Mix plan file not found: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

