"""Data models for Audio Studio AI SQLite database.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import time

@dataclass
class Track:
    id: Optional[int] = None
    file_path: str = ""
    title: str = ""
    artist: str = "Unknown Artist"
    album: str = "Unknown Album"
    genre: str = ""
    duration_seconds: float = 0.0
    bpm: Optional[float] = None
    musical_key: Optional[str] = None
    sample_rate: int = 44100
    channels: int = 2
    bitrate: int = 0
    format: str = ""
    waveform_cached: bool = False
    is_favorite: bool = False
    play_count: int = 0
    last_played: Optional[float] = None
    date_added: float = field(default_factory=time.time)
    file_size_bytes: int = 0
    file_modified_time: float = 0.0

@dataclass
class Playlist:
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    track_count: int = 0

@dataclass
class ProcessingTask:
    id: Optional[int] = None
    task_type: str = ""  # separation, noise_reduction, conversion, mix, analysis
    title: str = ""
    input_file: str = ""
    output_files: str = ""  # JSON string or comma-separated
    status: str = "queued"  # queued, processing, completed, failed, cancelled
    progress: float = 0.0
    status_message: str = ""
    error_message: Optional[str] = None
    model_name: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
