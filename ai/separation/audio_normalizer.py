"""Intelligent Audio Normalization and Loudness Enhancement Engine for Audio Studio AI.
Provides ITU-R BS.1770 / EBU R128 loudness measurement, true-peak calculation,
headroom protection, and safe gain adjustments without clipping or distortion.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
import scipy.signal

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except Exception:
    PYLOUDNORM_AVAILABLE = False

class AudioNormalizer:
    @staticmethod
    def measure_audio(audio: np.ndarray, sr: int = 44100) -> Dict[str, Any]:
        """
        Calculates acoustic metrics:
        - Peak linear and dBFS
        - True-Peak linear and dBTP (using 4x oversampling)
        - RMS linear and dBFS
        - Integrated Loudness (LUFS / LKFS)
        - Clipping and Silence detection
        """
        # Ensure 2D (samples, channels)
        if audio.ndim == 1:
            data = np.column_stack([audio, audio])
        elif audio.shape[0] < audio.shape[1] and audio.shape[0] in (1, 2):
            data = audio.T
        else:
            data = audio

        samples, channels = data.shape

        # 1. Peak linear & dBFS
        peak_lin = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
        peak_db = 20.0 * np.log10(max(1e-6, peak_lin))

        # 2. RMS linear & dBFS
        rms_lin = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
        rms_db = 20.0 * np.log10(max(1e-6, rms_lin))

        # 3. True-Peak estimation (4x oversampling)
        true_peak_lin = peak_lin
        try:
            if samples > 64:
                # Downsample chunk for efficiency on long files, or use full
                test_chunk = data[:min(sr * 30, samples)]
                oversampled = scipy.signal.resample_poly(test_chunk, 4, 1)
                true_peak_lin = float(np.max(np.abs(oversampled)))
        except Exception:
            true_peak_lin = peak_lin
        true_peak_db = 20.0 * np.log10(max(1e-6, true_peak_lin))

        # 4. Integrated Loudness (LUFS)
        lufs = -70.0
        if PYLOUDNORM_AVAILABLE and samples >= sr * 0.4 and peak_lin > 1e-4:
            try:
                meter = pyln.Meter(sr)
                measured_lufs = meter.integrated_loudness(data)
                if not np.isinf(measured_lufs) and not np.isnan(measured_lufs):
                    lufs = float(measured_lufs)
                else:
                    lufs = rms_db - 3.0  # RMS approximation
            except Exception:
                lufs = rms_db - 3.0
        else:
            lufs = rms_db - 3.0 if rms_db > -70.0 else -70.0

        is_silent = (peak_db < -55.0) or (rms_db < -60.0)
        has_clipping = peak_lin >= 0.999

        return {
            "peak_linear": peak_lin,
            "peak_db": round(peak_db, 2),
            "true_peak_linear": true_peak_lin,
            "true_peak_db": round(true_peak_db, 2),
            "rms_linear": rms_lin,
            "rms_db": round(rms_db, 2),
            "lufs": round(lufs, 2),
            "is_silent": is_silent,
            "has_clipping": has_clipping
        }

    @classmethod
    def normalize_stem(
        cls,
        audio: np.ndarray,
        sr: int = 44100,
        enable_norm: bool = True,
        mode: str = "lufs",             # "lufs", "peak", "custom"
        target_lufs: float = -14.0,     # e.g. -14 LUFS, -16 LUFS, -18 LUFS
        target_peak_db: float = -1.0,   # true-peak ceiling (-1.0 dBFS)
        user_gain_db: float = 0.0,      # manual user slider offset
        peak_protection: bool = True,
        max_boost_db: float = 14.0      # safety cap against noise amplification
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies safe, intelligent gain adjustment to isolated stems.
        Preserves signal fidelity, avoids pumping, and guarantees headroom ceiling.
        """
        # Ensure 2D (samples, channels)
        if audio.ndim == 1:
            data = np.column_stack([audio, audio]).astype(np.float32)
        elif audio.shape[0] < audio.shape[1] and audio.shape[0] in (1, 2):
            data = audio.T.astype(np.float32)
        else:
            data = audio.copy().astype(np.float32)

        raw_metrics = cls.measure_audio(data, sr)

        if not enable_norm and abs(user_gain_db) < 0.01:
            # Bypass normalization
            return data, {
                "gain_applied_db": 0.0,
                "raw_metrics": raw_metrics,
                "norm_metrics": raw_metrics
            }

        # Handle nearly silent stems: do NOT blow up noise floor
        if raw_metrics["is_silent"]:
            # If silent, only apply safe manual gain or keep untouched
            gain_db = min(user_gain_db, 3.0)
            scale = 10.0 ** (gain_db / 20.0)
            out_data = data * scale
            norm_metrics = cls.measure_audio(out_data, sr)
            return out_data, {
                "gain_applied_db": round(gain_db, 2),
                "raw_metrics": raw_metrics,
                "norm_metrics": norm_metrics
            }

        calc_gain_db = 0.0
        if enable_norm:
            if mode == "lufs":
                current_lufs = raw_metrics["lufs"]
                if current_lufs > -65.0:
                    delta = target_lufs - current_lufs
                    calc_gain_db = max(-18.0, min(max_boost_db, delta))
            elif mode == "peak":
                current_peak = raw_metrics["peak_db"]
                delta = target_peak_db - current_peak
                calc_gain_db = max(-18.0, min(max_boost_db, delta))

        # Add manual user gain offset
        total_gain_db = calc_gain_db + user_gain_db

        # Output Peak Protection (True-Peak ceiling)
        ceiling_lin = 10.0 ** (target_peak_db / 20.0)  # e.g. -1.0 dBFS = 0.891
        potential_scale = 10.0 ** (total_gain_db / 20.0)
        potential_peak = raw_metrics["peak_linear"] * potential_scale

        if peak_protection and potential_peak > ceiling_lin:
            # Pull back gain to respect ceiling exactly without clipping
            allowed_scale = ceiling_lin / max(1e-6, raw_metrics["peak_linear"])
            total_gain_db = 20.0 * np.log10(max(1e-6, allowed_scale))

        # Apply final gain
        final_scale = 10.0 ** (total_gain_db / 20.0)
        processed = data * final_scale

        # Hard safety clamp to prevent DAC digital clipping
        processed = np.clip(processed, -0.9999, 0.9999)

        norm_metrics = cls.measure_audio(processed, sr)

        return processed, {
            "gain_applied_db": round(total_gain_db, 2),
            "raw_metrics": raw_metrics,
            "norm_metrics": norm_metrics
        }

