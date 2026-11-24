"""Waveform extraction and caching engine for Audio Studio AI.
Extracts min/max peak data using SoundFile with downsampling and disk caching.
"""
import os
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

from config import CACHE_DIR

WAVEFORM_CACHE_DIR = CACHE_DIR / "waveforms"
WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class WaveformEngine:
    @staticmethod
    def get_cache_path(file_path: str) -> Path:
        p = Path(file_path)
        stat = p.stat() if p.exists() else None
        mtime = stat.st_mtime if stat else 0
        size = stat.st_size if stat else 0
        hash_input = f"{p.resolve()}_{mtime}_{size}".encode("utf-8")
        hash_str = hashlib.sha256(hash_input).hexdigest()[:16]
        return WAVEFORM_CACHE_DIR / f"{hash_str}.npy"

    @classmethod
    def load_peaks(cls, file_path: str, num_bins: int = 1200) -> Optional[np.ndarray]:
        cache_file = cls.get_cache_path(file_path)
        if cache_file.exists():
            try:
                return np.load(str(cache_file))
            except Exception:
                pass

        # Compute peaks from audio file
        try:
            if not Path(file_path).exists():
                return None

            info = sf.info(file_path)
            total_frames = info.frames
            if total_frames <= 0:
                return None

            # Read audio data in blocks or downsample
            # For efficiency on large files, block-read
            block_size = max(1, total_frames // num_bins)
            peaks = []

            with sf.SoundFile(file_path) as f:
                while len(peaks) < num_bins:
                    data = f.read(block_size, dtype='float32', always_2d=True)
                    if len(data) == 0:
                        break
                    # Peak magnitude of channels
                    peak = np.max(np.abs(data))
                    peaks.append(peak)

            peaks_arr = np.array(peaks, dtype=np.float32)
            # Normalize to 0.0 - 1.0
            max_val = np.max(peaks_arr) if len(peaks_arr) > 0 and np.max(peaks_arr) > 0 else 1.0
            peaks_arr = peaks_arr / max_val

            # Save to cache
            np.save(str(cache_file), peaks_arr)
            return peaks_arr
        except Exception as e:
            print(f"Error extracting waveform for {file_path}: {e}")
            return None
