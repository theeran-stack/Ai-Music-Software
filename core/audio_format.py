"""Central Audio Format Validation & Conversion Module for Audio Studio AI.
Enforces strict MP3-only import and export policy across all 11 application workspaces.
Provides robust MP3 magic-byte validation, chunk-safe PCM decoding, and 
broadcast-standard libmp3lame encoding with configurable bitrates (128k - 320k).
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import soundfile as sf

from config import BUNDLED_BIN, BASE_DIR

def get_ffmpeg_binary() -> str:
    """Locates the bundled or system ffmpeg binary."""
    # 1. Bundled bin folder
    bundled = BUNDLED_BIN / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)

    # 2. Project root bin
    root_bin = BASE_DIR / "bin" / "ffmpeg.exe"
    if root_bin.exists():
        return str(root_bin)

    # 3. System PATH
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path

    return "ffmpeg"


def validate_mp3_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validates that a file is strictly an MP3 file by checking both its
    extension and binary header markers (ID3 or MPEG audio frame sync).
    Returns (is_valid, error_message).
    """
    if not file_path:
        return False, "No file selected."

    path = Path(file_path)
    if path.suffix.lower() != ".mp3":
        return False, "Only MP3 audio files are supported."

    if not path.exists():
        return False, f"File does not exist: {path.name}"

    if not path.is_file():
        return False, f"Not a valid file: {path.name}"

    # Validate file size
    try:
        size = path.stat().st_size
        if size < 128:
            return False, "File is too small to be a valid audio recording."
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Validate magic bytes / MPEG headers
    try:
        with open(path, "rb") as f:
            header = f.read(1024)

        # Check for ID3v2 header ('ID3')
        if header.startswith(b"ID3"):
            return True, ""

        # Check for MPEG audio frame sync (11 bits set: 0xFF followed by 0xEx or 0xFx)
        for i in range(len(header) - 1):
            if header[i] == 0xFF and (header[i + 1] & 0xE0) == 0xE0:
                return True, ""

        return False, "Only MP3 audio files are supported."
    except Exception as e:
        return False, f"Failed to inspect audio file: {e}"


def decode_mp3_to_numpy(file_path: Union[str, Path], target_sr: int = 44100, mono: bool = False) -> Tuple[np.ndarray, int]:
    """
    Decodes an MP3 file into a float32 NumPy array [-1.0, 1.0].
    Returns (audio_data, sample_rate).
    If stereo: audio_data has shape (channels, samples).
    If mono: audio_data has shape (samples,).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Try soundfile
    try:
        data, sr = sf.read(str(path), dtype="float32")
        # soundfile returns (samples, channels)
        if data.ndim == 2:
            data = data.T  # Transpose to (channels, samples)
            if mono:
                data = np.mean(data, axis=0)
        elif mono:
            pass  # already 1D
        else:
            # Duplicate mono to 2 channels
            data = np.stack([data, data], axis=0)
        return data, sr
    except Exception:
        pass

    # 2. Robust FFmpeg raw PCM fallback
    ffmpeg_exe = get_ffmpeg_binary()
    channels = 1 if mono else 2
    cmd = [
        ffmpeg_exe,
        "-y",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(target_sr),
        "-ac", str(channels),
        "-"
    ]
    creation_flags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg decoding failed: {err.decode('utf-8', errors='ignore')}")

    audio = np.frombuffer(out, dtype=np.float32)
    if channels == 2:
        audio = audio.reshape(-1, 2).T
    return audio, target_sr


def encode_array_to_mp3(
    audio_data: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = 44100,
    bitrate: str = "320k",
    sr: Optional[int] = None
) -> bool:
    """
    Encodes a float32 NumPy array to a broadcast-quality MP3 file using FFmpeg libmp3lame.
    Supported bitrates: '128k', '192k', '256k', '320k'. Default: '320k'.
    """
    if sr is not None:
        sample_rate = sr
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Normalize bitrate format
    clean_bitrate = str(bitrate).lower().replace("kbps", "").replace("k", "").strip()
    if clean_bitrate not in ["128", "192", "256", "320"]:
        clean_bitrate = "320"
    b_flag = f"{clean_bitrate}k"

    # Shape preparation: (samples, channels)
    if audio_data.ndim == 1:
        channels = 1
        raw_pcm = audio_data.astype(np.float32).tobytes()
    elif audio_data.ndim == 2:
        if audio_data.shape[0] in (1, 2) and audio_data.shape[1] > audio_data.shape[0]:
            channels = audio_data.shape[0]
            raw_pcm = audio_data.T.astype(np.float32).tobytes()
        else:
            channels = audio_data.shape[1]
            raw_pcm = audio_data.astype(np.float32).tobytes()
    else:
        raise ValueError(f"Unsupported audio array dimension: {audio_data.ndim}")

    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-v", "error",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-i", "-",
        "-c:a", "libmp3lame",
        "-b:a", b_flag,
        "-id3v2_version", "3",
        str(out_p)
    ]

    creation_flags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
    _, err = proc.communicate(input=raw_pcm)

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg MP3 encoding failed: {err.decode('utf-8', errors='ignore')}")

    return out_p.exists() and out_p.stat().st_size > 0


def convert_file_to_mp3(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    bitrate: str = "320k"
) -> bool:
    """Converts any audio file to MP3 via FFmpeg."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    clean_bitrate = str(bitrate).lower().replace("kbps", "").replace("k", "").strip()
    if clean_bitrate not in ["128", "192", "256", "320"]:
        clean_bitrate = "320"

    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-v", "error",
        "-i", str(in_p),
        "-c:a", "libmp3lame",
        "-b:a", f"{clean_bitrate}k",
        "-id3v2_version", "3",
        str(out_p)
    ]

    creation_flags = 0x08000000 if os.name == "nt" else 0
    res = subprocess.run(cmd, capture_output=True, creationflags=creation_flags)
    return res.returncode == 0 and out_p.exists() and out_p.stat().st_size > 0
