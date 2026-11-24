"""AI Audio Enhancer Engine for Audio Studio AI.
Provides studio-grade audio enhancement:
1. Adaptive spectral noise reduction (via FFmpeg afftdn or spectral gating).
2. Rumble elimination (< 80 Hz high-pass).
3. Formant presence & clarity booster (parametric EQ at 3.5 kHz & 10 kHz air).
4. Independent pitch shifting (rubberband pitch shift without tempo alteration).
5. Independent time stretching (rubberband / atempo without pitch alteration).
6. ITU-R BS.1770 broadcast loudness normalisation (-14 LUFS standard, -1.0 dBTP ceiling).
7. One-Click AI Master Polish preset.
Strictly exports in MP3 at configurable bitrates (128k, 192k, 256k, 320k).
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import threading

import numpy as np
import soundfile as sf

from config import BASE_DIR, BUNDLED_BIN, DEFAULT_EXPORT_BITRATE, CACHE_DIR
from core.audio_format import encode_array_to_mp3, decode_mp3_to_numpy, validate_mp3_file, get_ffmpeg_binary
from ai.separation.audio_normalizer import AudioNormalizer

class AudioEnhancer:
    """Core DSP and AI Audio Enhancement Engine."""

    @staticmethod
    def _get_ffmpeg() -> str:
        return get_ffmpeg_binary()

    @classmethod
    def enhance_audio(
        cls,
        input_path: str,
        output_path: str,
        noise_reduction: float = 0.0,       # 0.0 to 1.0 (intensity)
        pitch_semitones: int = 0,            # -12 to +12
        tempo_ratio: float = 1.0,           # 0.5 to 1.5
        clarity_boost_db: float = 0.0,      # -12 to +12 dB
        vocal_gain_db: float = 0.0,         # -12 to +12 dB
        enable_normalisation: bool = True,  # Broadcast loudness normalisation
        target_lufs: float = -14.0,         # Streaming standard
        export_bitrate: str = "320k",       # 128k, 192k, 256k, 320k
        cancel_token: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Enhances an audio file using specified DSP modules and exports strictly to MP3.
        """
        in_p = Path(input_path)
        out_p = Path(output_path).with_suffix(".mp3")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            raise FileNotFoundError(f"Source audio file not found: {input_path}")

        if progress_cb:
            progress_cb(0.1, "Validating and preparing input...")

        if cancel_token and cancel_token.is_set():
            raise InterruptedError("Enhancement cancelled by user.")

        # Build FFmpeg audio filter chain
        filters = []

        # 1. Highpass filter to eliminate sub-bass stage rumble / microphone handling noise
        filters.append("highpass=f=75:width_type=q:width=0.707")

        # 2. Spectral noise suppression
        if noise_reduction > 0.01:
            nr_db = min(35.0, max(6.0, noise_reduction * 35.0))
            # afftdn: adaptive FFT denoiser with noise reduction amount in dB
            filters.append(f"afftdn=nr={nr_db:.1f}:nf=-50:tn=1")

        # 3. Independent pitch shifting (rubberband pitch shift factor = 2^(semitones/12))
        if pitch_semitones != 0:
            pitch_scale = 2.0 ** (pitch_semitones / 12.0)
            filters.append(f"rubberband=pitch={pitch_scale:.6f}")

        # 4. Independent time stretching
        if abs(tempo_ratio - 1.0) > 0.01:
            tempo = max(0.5, min(2.0, tempo_ratio))
            filters.append(f"atempo={tempo:.4f}")

        # 5. Formant Clarity & Presence Enhancement (3.5 kHz bell boost + 10 kHz air shelf)
        if abs(clarity_boost_db) > 0.1:
            filters.append(f"equalizer=f=3500:t=q:w=1.2:g={clarity_boost_db:.2f}")
            air_gain = clarity_boost_db * 0.6
            filters.append(f"highshelf=f=9500:width_type=q:w=0.707:g={air_gain:.2f}")

        # 6. Vocal gain boost
        if abs(vocal_gain_db) > 0.1:
            filters.append(f"volume={vocal_gain_db:+.2f}dB")

        filter_str = ",".join(filters) if filters else "anull"

        if progress_cb:
            progress_cb(0.3, "Executing DSP audio enhancement pipeline...")

        if cancel_token and cancel_token.is_set():
            raise InterruptedError("Enhancement cancelled by user.")

        # Intermediate WAV processing
        temp_wav = out_p.with_suffix(".temp_proc.wav")
        try:
            cmd = [
                cls._get_ffmpeg(),
                "-y",
                "-loglevel", "error",
                "-i", str(in_p),
                "-af", filter_str,
                "-ac", "2",
                "-ar", "44100",
                str(temp_wav)
            ]
            subprocess.run(cmd, check=True)

            if cancel_token and cancel_token.is_set():
                raise InterruptedError("Enhancement cancelled by user.")

            if progress_cb:
                progress_cb(0.65, "Applying intelligent loudness normalisation...")

            # Load audio for normalization & measurement
            data, sr = sf.read(str(temp_wav), dtype="float32")
            if data.ndim == 1:
                data = np.column_stack([data, data])

            norm_meta = {}
            if enable_normalisation:
                norm_audio, norm_meta = AudioNormalizer.normalize_stem(
                    data,
                    sr=sr,
                    enable_norm=True,
                    mode="lufs",
                    target_lufs=target_lufs,
                    target_peak_db=-1.0,
                    peak_protection=True
                )
            else:
                norm_audio = np.clip(data, -0.9999, 0.9999)

            if progress_cb:
                progress_cb(0.85, f"Encoding MP3 at {export_bitrate}...")

            if cancel_token and cancel_token.is_set():
                raise InterruptedError("Enhancement cancelled by user.")

            # Encode strictly to MP3
            encode_array_to_mp3(
                norm_audio,
                str(out_p),
                sample_rate=sr,
                bitrate=export_bitrate
            )

            # Measure final audio
            final_meas = AudioNormalizer.measure_audio(norm_audio, sr)

            if progress_cb:
                progress_cb(1.0, "Enhancement completed successfully!")

            return {
                "status": "success",
                "output_path": str(out_p.resolve()),
                "bitrate": export_bitrate,
                "metrics": final_meas,
                "norm_meta": norm_meta
            }

        finally:
            if temp_wav.exists():
                try:
                    temp_wav.unlink()
                except OSError:
                    pass

    @classmethod
    def one_click_master_enhance(
        cls,
        input_path: str,
        output_path: str,
        export_bitrate: str = "320k",
        cancel_token: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        One-Click Master Vocal Polish preset:
        - 60% adaptive noise reduction
        - +3.0 dB clarity presence boost at 3.5 kHz & air at 10 kHz
        - 75 Hz stage rumble cut
        - ITU-R BS.1770 broadcast normalization (-14 LUFS, -1.0 dBTP ceiling)
        - Master 320 kbps MP3 export
        """
        return cls.enhance_audio(
            input_path=input_path,
            output_path=output_path,
            noise_reduction=0.60,
            pitch_semitones=0,
            tempo_ratio=1.0,
            clarity_boost_db=3.0,
            vocal_gain_db=1.0,
            enable_normalisation=True,
            target_lufs=-14.0,
            export_bitrate=export_bitrate,
            cancel_token=cancel_token,
            progress_cb=progress_cb
        )
