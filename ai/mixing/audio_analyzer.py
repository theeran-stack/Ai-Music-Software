"""Real Audio Analysis Engine for Smart Mix Studio.
Performs acoustic extraction of BPM, beat grid, musical key, Camelot notation,
energy curves, and vocal activity heatmap using scipy, numpy, and soundfile.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf

from ai.mixing.mix_plan import TrackAnalysis

# Krumhansl-Schmuckler Key Profiles for Major and Minor Keys
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KS_MAJOR = KS_MAJOR / np.linalg.norm(KS_MAJOR)
KS_MINOR = KS_MINOR / np.linalg.norm(KS_MINOR)

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CAMELOT_MAP = {
    ("C", "major"): "8B", ("C", "minor"): "5A",
    ("C#", "major"): "3B", ("C#", "minor"): "12A",
    ("D", "major"): "10B", ("D", "minor"): "7A",
    ("D#", "major"): "5B", ("D#", "minor"): "2A",
    ("E", "major"): "12B", ("E", "minor"): "9A",
    ("F", "major"): "7B", ("F", "minor"): "4A",
    ("F#", "major"): "2B", ("F#", "minor"): "11A",
    ("G", "major"): "9B", ("G", "minor"): "6A",
    ("G#", "major"): "4B", ("G#", "minor"): "1A",
    ("A", "major"): "11B", ("A", "minor"): "8A",
    ("A#", "major"): "6B", ("A#", "minor"): "3A",
    ("B", "major"): "1B", ("B", "minor"): "10A"
}

class AudioAnalyzer:
    @staticmethod
    def load_mono_audio(file_path: str, target_sr: int = 22050, max_duration: Optional[float] = None) -> Tuple[np.ndarray, int, float]:
        """
        Loads audio file as a mono float32 array at target sample rate.
        Supports WAV, MP3, FLAC, M4A, OGG via soundfile or FFmpeg fallback.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            duration_sec = len(mono) / float(orig_sr)

            if max_duration and duration_sec > max_duration:
                mono = mono[:int(max_duration * orig_sr)]
                duration_sec = max_duration

            if orig_sr != target_sr:
                num_target = int(len(mono) * target_sr / orig_sr)
                mono = signal.resample(mono, num_target)
                sr = target_sr
            else:
                sr = orig_sr

            return mono.astype(np.float32), sr, duration_sec

        except Exception:
            # Fallback to FFmpeg decoding for complex containers or corrupt headers
            cmd = [
                "ffmpeg", "-y", "-loglevel", "panic",
                "-i", str(path)
            ]
            if max_duration:
                cmd.extend(["-t", str(max_duration)])
            cmd.extend([
                "-ar", str(target_sr),
                "-ac", "1",
                "-f", "f32le",
                "-"
            ])
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_bytes, _ = proc.communicate()
            if proc.returncode != 0 or not raw_bytes:
                raise RuntimeError(f"FFmpeg failed to decode audio file: {file_path}")

            mono = np.frombuffer(raw_bytes, dtype=np.float32)
            duration_sec = len(mono) / float(target_sr)
            return mono, target_sr, duration_sec

    @staticmethod
    def detect_bpm_and_downbeat(mono: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Detects primary tempo (BPM) and first downbeat position using spectral flux autocorrelation.
        """
        if len(mono) < sr * 1.0:
            return 120.0, 0.0

        hop_length = 512
        n_fft = 2048
        f, t, Zxx = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)

        # Spectral flux onset envelope
        flux = np.diff(mag, axis=1)
        flux = np.maximum(0, flux)
        onset_env = np.sum(flux, axis=0)

        if np.max(onset_env) > 0:
            onset_env = onset_env / np.max(onset_env)

        # Autocorrelation
        corr = np.correlate(onset_env, onset_env, mode="full")
        corr = corr[len(corr)//2:]

        fps = sr / hop_length
        # Search range: 68 to 175 BPM
        min_lag = int(fps * 60.0 / 175.0)
        max_lag = int(fps * 60.0 / 68.0)

        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            return 120.0, 0.0

        valid_corr = corr[min_lag:max_lag]
        best_lag = min_lag + np.argmax(valid_corr)
        bpm = (fps * 60.0) / float(best_lag)

        # Find first prominent downbeat peak in onset envelope
        beat_interval_frames = int(best_lag)
        first_peaks, _ = signal.find_peaks(onset_env[:beat_interval_frames * 2], height=0.25, distance=max(1, beat_interval_frames // 2))
        first_downbeat_sec = float(t[first_peaks[0]]) if len(first_peaks) > 0 else 0.0

        return round(float(bpm), 1), round(first_downbeat_sec, 3)

    @staticmethod
    def detect_musical_key(mono: np.ndarray, sr: int) -> Tuple[str, str, float]:
        """
        Detects musical key and Camelot notation via 12-bin chromagram and Krumhansl-Schmuckler profiles.
        Returns (Key Name, Camelot Code, Correlation Score).
        """
        if len(mono) < sr * 1.0:
            return "C Major", "8B", 0.5

        hop_length = 1024
        n_fft = 4096
        f, t, Zxx = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)

        # Map frequency bins into 12 chroma pitch classes (A4 = 440Hz -> MIDI 69)
        chroma = np.zeros(12)
        for idx, freq in enumerate(f):
            if 65.0 <= freq <= 2200.0:
                midi = 69.0 + 12.0 * np.log2(freq / 440.0)
                pitch_class = int(np.round(midi)) % 12
                chroma[pitch_class] += np.sum(mag[idx, :])

        norm = np.linalg.norm(chroma)
        if norm > 0:
            chroma = chroma / norm

        best_score = -999.0
        best_key = ("C", "major")

        for shift in range(12):
            rotated = np.roll(chroma, -shift)
            c_maj = np.corrcoef(rotated, KS_MAJOR)[0, 1]
            if c_maj > best_score:
                best_score = c_maj
                best_key = (PITCH_NAMES[shift], "major")

            c_min = np.corrcoef(rotated, KS_MINOR)[0, 1]
            if c_min > best_score:
                best_score = c_min
                best_key = (PITCH_NAMES[shift], "minor")

        key_name = f"{best_key[0]} {best_key[1].title()}"
        camelot = CAMELOT_MAP.get(best_key, "8B")
        return key_name, camelot, round(float(best_score), 3)

    @staticmethod
    def analyze_vocal_presence(mono: np.ndarray, sr: int) -> Tuple[float, List[Dict[str, float]], str]:
        """
        Analyzes vocal frequency band (300 Hz - 3400 Hz) spectral energy and harmonicity.
        Returns (overall_vocal_ratio, vocal_segments, confidence_level).
        """
        if len(mono) < sr * 1.0:
            return 0.0, [], "Low"

        chunk_size = sr * 2  # 2-second windows
        num_chunks = len(mono) // chunk_size
        if num_chunks == 0:
            return 0.0, [], "Low"

        vocal_segments = []
        vocal_chunks_count = 0

        # Frequency filter for vocal speech/singing formant band (300Hz - 3400Hz)
        sos = signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
        vocal_filtered = signal.sosfilt(sos, mono)

        for i in range(num_chunks):
            start_sec = i * 2.0
            end_sec = (i + 1) * 2.0
            chunk_raw = mono[i * chunk_size : (i + 1) * chunk_size]
            chunk_vocal = vocal_filtered[i * chunk_size : (i + 1) * chunk_size]

            rms_total = np.sqrt(np.mean(chunk_raw ** 2)) + 1e-7
            rms_vocal = np.sqrt(np.mean(chunk_vocal ** 2))

            ratio = float(rms_vocal / rms_total)
            if ratio > 0.42 and rms_vocal > 0.02:
                vocal_chunks_count += 1
                intensity = min(1.0, round((ratio - 0.40) * 3.5, 2))
                vocal_segments.append({
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "intensity": intensity
                })

        overall_ratio = round(vocal_chunks_count / max(1, num_chunks), 2)
        confidence = "High" if len(mono) > sr * 10 else "Moderate"
        return overall_ratio, vocal_segments, confidence

    @classmethod
    def analyze_track(cls, file_path: str) -> TrackAnalysis:
        """
        Performs complete acoustic analysis on an audio file.
        Returns a structured TrackAnalysis object.
        """
        path = Path(file_path)
        mono, sr, duration_sec = cls.load_mono_audio(file_path)

        # 1. BPM & Downbeat
        bpm, downbeat = cls.detect_bpm_and_downbeat(mono, sr)

        # 2. Key & Camelot
        key_name, camelot, key_score = cls.detect_musical_key(mono, sr)

        # 3. Dynamic Energy
        rms = float(np.sqrt(np.mean(mono ** 2)))
        energy_level = round(min(1.0, rms * 4.0), 2)

        # 4. Vocal Activity
        vocal_ratio, vocal_segs, vocal_conf = cls.analyze_vocal_presence(mono, sr)

        # 5. Phrase Boundaries (8-bar phrases)
        beat_interval = 60.0 / bpm
        phrase_sec = 8 * 4 * beat_interval
        boundaries = []
        curr = downbeat
        while curr < duration_sec:
            boundaries.append(round(curr, 2))
            curr += phrase_sec
        boundaries.append(round(duration_sec, 2))

        return TrackAnalysis(
            path=str(path.resolve()),
            title=path.stem,
            duration_seconds=round(duration_sec, 2),
            sample_rate=sr,
            bpm=bpm,
            musical_key=key_name,
            camelot_key=camelot,
            energy_level=energy_level,
            beat_interval=round(beat_interval, 3),
            first_downbeat_sec=downbeat,
            phrase_boundaries_sec=boundaries,
            vocal_presence_ratio=vocal_ratio,
            vocal_segments=vocal_segs,
            vocal_confidence=vocal_conf
        )

