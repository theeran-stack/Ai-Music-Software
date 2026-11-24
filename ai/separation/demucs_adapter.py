"""Demucs Neural Separation Adapter for Audio Studio AI.
Provides real 2-stem (Vocals & Instrumental) neural source separation using HTDemucs v4.
Supports WAV, MP3, FLAC, M4A, AAC, OGG decoding, intelligent audio level normalization,
cancellation, and multi-format export.
"""
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, Union

import torch
import numpy as np
import soundfile as sf
import julius

import demucs.hf
import huggingface_hub

# Patch demucs HuggingFace downloader to prefer local cache without network warnings
_orig_hf_download = huggingface_hub.hf_hub_download

def _safe_hf_hub_download(repo_id, filename, *args, **kwargs):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and "token" not in kwargs:
        kwargs["token"] = hf_token
    # Prefer local cached file without sending unauthenticated requests
    try:
        return _orig_hf_download(repo_id, filename, *args, local_files_only=True, **kwargs)
    except Exception:
        # Fallback to remote fetch if not present in local cache
        return _orig_hf_download(repo_id, filename, *args, **kwargs)

demucs.hf.hf_hub_download = _safe_hf_hub_download

from demucs.pretrained import get_model
from demucs.apply import apply_model
from config import MODELS_DIR, DATA_DIR, CUDA_AVAILABLE, HF_CACHE_DIR
from ai.separation.audio_normalizer import AudioNormalizer

SEPARATED_OUTPUT_DIR = DATA_DIR / "separated"
SEPARATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SeparationCancelledException(Exception):
    """Raised when separation is cancelled by user."""
    pass

class SeparationEngine:
    @staticmethod
    def is_model_installed(model_name: str = "htdemucs") -> bool:
        """Check if model checkpoint is available locally without network requests."""
        # Check AppData Models HuggingFace cache
        hub_path = HF_CACHE_DIR / "hub" / "models--adefossez--HTDemucs"
        if hub_path.exists() and any(hub_path.rglob("*.safetensors")):
            return True

        # Check default user home cache
        user_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--adefossez--HTDemucs"
        if user_cache.exists() and any(user_cache.rglob("*.safetensors")):
            return True

        # Check models directory
        for ext in [".pth", ".safetensors", ".onnx"]:
            if (MODELS_DIR / f"{model_name}{ext}").exists():
                return True

        return False

    @staticmethod
    def load_audio(file_path: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int, float]:
        """
        Loads audio from any supported format (WAV, MP3, FLAC, M4A, AAC, OGG).
        Returns stereo torch.Tensor (2, samples), sample_rate, duration_seconds.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        wav_tensor = None
        sr = target_sr

        # 1. Try reading with soundfile first
        try:
            data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
            channels = data.shape[1]
            if channels == 1:
                stereo_data = np.column_stack([data[:, 0], data[:, 0]])
            elif channels >= 2:
                stereo_data = data[:, :2]
            else:
                stereo_data = np.column_stack([data, data])

            wav_tensor = torch.from_numpy(stereo_data.T)
            sr = orig_sr
        except Exception:
            # 2. Fallback to FFmpeg decoding for complex containers (m4a, aac, etc.)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "panic",
                "-i", str(path.resolve()),
                "-f", "f32le",
                "-ac", "2",
                "-ar", str(target_sr),
                "-"
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            raw = np.frombuffer(res.stdout, dtype=np.float32)
            stereo = raw.reshape(-1, 2)
            wav_tensor = torch.from_numpy(stereo.T)
            sr = target_sr

        # Resample to target sample rate if needed
        if sr != target_sr:
            wav_tensor = julius.resample_frac(wav_tensor, sr, target_sr)
            sr = target_sr

        duration = wav_tensor.shape[-1] / float(sr)
        return wav_tensor, sr, duration

    @staticmethod
    def save_audio(
        tensor_or_array: Union[torch.Tensor, np.ndarray],
        output_path: str,
        sr: int = 44100,
        format_type: str = "wav"
    ) -> str:
        """
        Exports audio to WAV (16-bit PCM), MP3 (320kbps), or FLAC (lossless).
        Guarantees correct file extension and valid stream encoding.
        Returns final resolved path.
        """
        fmt = format_type.lower().replace(".", "").strip()
        if fmt not in ("wav", "mp3", "flac"):
            fmt = "wav"

        out_p = Path(output_path)
        # Ensure correct suffix
        expected_suffix = f".{fmt}"
        if out_p.suffix.lower() != expected_suffix:
            out_p = out_p.with_suffix(expected_suffix)

        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy array with shape (samples, channels)
        if isinstance(tensor_or_array, torch.Tensor):
            data = tensor_or_array.detach().cpu().numpy()
        else:
            data = np.asarray(tensor_or_array)

        if data.ndim == 1:
            data = np.column_stack([data, data])
        elif data.shape[0] < data.shape[1] and data.shape[0] in (1, 2):
            data = data.T

        # Ensure stereo contiguous float32 clamped to [-0.9999, 0.9999]
        audio_np = np.ascontiguousarray(data, dtype=np.float32)
        audio_np = np.clip(audio_np, -0.9999, 0.9999)

        if fmt == "wav":
            sf.write(str(out_p), audio_np, sr, subtype="PCM_16")
        elif fmt == "flac":
            sf.write(str(out_p), audio_np, sr, format="FLAC", subtype="PCM_16")
        elif fmt == "mp3":
            # Encode via lameenc for 320 kbps high-fidelity MP3
            encoded_success = False
            try:
                import lameenc
                encoder = lameenc.Encoder()
                encoder.set_bit_rate(320)
                encoder.set_in_sample_rate(sr)
                encoder.set_channels(audio_np.shape[1])
                encoder.set_quality(2)
                int16_data = (audio_np * 32767.0).astype(np.int16)
                int16_contiguous = np.ascontiguousarray(int16_data)
                mp3_bytes = encoder.encode(int16_contiguous.tobytes())
                mp3_bytes += encoder.flush()
                with open(out_p, "wb") as f:
                    f.write(mp3_bytes)
                encoded_success = True
            except Exception:
                encoded_success = False

            if not encoded_success or out_p.stat().st_size == 0:
                # Robust fallback to FFmpeg for MP3 encoding
                temp_wav = out_p.with_suffix(".temp_enc.wav")
                sf.write(str(temp_wav), audio_np, sr, subtype="PCM_16")
                cmd = ["ffmpeg", "-y", "-loglevel", "panic", "-i", str(temp_wav), "-b:a", "320k", str(out_p)]
                subprocess.run(cmd, check=True)
                if temp_wav.exists():
                    temp_wav.unlink()

        return str(out_p.resolve())

    @classmethod
    def separate_2stem(
        cls,
        file_path: str,
        output_dir: Optional[str] = None,
        model_name: str = "htdemucs",
        quality_mode: str = "balanced",  # fast, balanced, quality
        device: str = "auto",            # auto, cuda, cpu
        enable_normalization: bool = True,
        norm_mode: str = "lufs",         # "lufs", "peak"
        target_lufs: float = -14.0,
        target_peak_db: float = -1.0,
        vocal_gain_db: float = 0.0,
        inst_gain_db: float = 0.0,
        peak_protection: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Dict[str, Any]:
        """
        Runs real neural 2-stem separation with intelligent loudness enhancement.
        Preserves original separated stems as raw intermediates while exporting
        optimally leveled, broadcast-standard stems.
        """
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {file_path} not found.")

        # Determine target compute device
        if device == "auto":
            target_device = "cuda" if CUDA_AVAILABLE else "cpu"
        elif device == "cuda":
            target_device = "cuda" if CUDA_AVAILABLE else "cpu"
        else:
            target_device = "cpu"

        device_obj = torch.device(target_device)

        # Quality settings mapping to Demucs parameters
        if quality_mode == "fast":
            shifts = 0
            overlap = 0.1
        elif quality_mode == "quality":
            shifts = 2
            overlap = 0.5
        else:  # balanced
            shifts = 1
            overlap = 0.25

        if progress_callback:
            progress_callback(0.04, f"Loading neural model ({model_name}) on {target_device.upper()}...")

        if cancel_event and cancel_event.is_set():
            raise SeparationCancelledException("Cancelled by user")

        # Load Demucs Model
        model = get_model(model_name)
        model.to(device_obj)
        model.eval()

        if progress_callback:
            progress_callback(0.10, "Decoding and analyzing input audio...")

        if cancel_event and cancel_event.is_set():
            raise SeparationCancelledException("Cancelled by user")

        # Load and resample audio
        wav_tensor, sr, duration = cls.load_audio(str(input_path), target_sr=model.samplerate)
        total_samples = wav_tensor.shape[-1]

        # Analyze original input audio metrics
        orig_np = wav_tensor.numpy().T
        orig_metrics = AudioNormalizer.measure_audio(orig_np, sr)

        # Prepare Demucs batch tensor (1, channels, samples)
        mix = wav_tensor.unsqueeze(0).to(device_obj)

        if progress_callback:
            progress_callback(0.18, f"Running AI source separation ({quality_mode.title()} mode)...")

        # Determine segment length in seconds
        segment_sec = getattr(model, "segment", None)
        if segment_sec is None and hasattr(model, "models") and len(model.models) > 0:
            segment_sec = getattr(model.models[0], "segment", 7.8)
        if segment_sec is None:
            segment_sec = 7.8
        segment_samples = float(segment_sec) * model.samplerate

        def demucs_callback(info: Dict[str, Any]):
            if cancel_event and cancel_event.is_set():
                raise SeparationCancelledException("Separation cancelled by user")

            offset = info.get("segment_offset", 0)
            model_idx = info.get("model_idx_in_bag", 0)
            num_models = info.get("models", 1)

            chunk_ratio = min(1.0, float(offset + segment_samples) / max(1.0, float(total_samples)))
            model_ratio = (model_idx + chunk_ratio) / max(1.0, float(num_models))
            pct = 0.18 + (model_ratio * 0.65)  # 18% to 83%

            if progress_callback:
                progress_callback(pct, f"Extracting stems: {int(pct * 100)}% complete...")

        # Run Neural Inference
        with torch.no_grad():
            try:
                sources = apply_model(
                    model,
                    mix,
                    shifts=shifts,
                    split=True,
                    overlap=overlap,
                    device=device_obj,
                    progress=False,
                    callback=demucs_callback
                )
            except SeparationCancelledException:
                raise
            except Exception as e:
                if cancel_event and cancel_event.is_set():
                    raise SeparationCancelledException("Separation cancelled by user")
                raise RuntimeError(f"Error during AI neural inference: {e}")

        if cancel_event and cancel_event.is_set():
            raise SeparationCancelledException("Cancelled by user")

        if progress_callback:
            progress_callback(0.85, "Reconstructing Vocals and Instrumental stems...")

        vocals_idx = model.sources.index("vocals") if "vocals" in model.sources else 3
        vocals_tensor = sources[0, vocals_idx]  # (2, samples)

        other_indices = [i for i in range(len(model.sources)) if i != vocals_idx]
        instrumental_tensor = torch.zeros_like(vocals_tensor)
        for idx in other_indices:
            instrumental_tensor += sources[0, idx]

        # Destination directory
        dest_dir = Path(output_dir) if output_dir else SEPARATED_OUTPUT_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem_base_name = input_path.stem

        # Save Untouched Raw Intermediates
        if progress_callback:
            progress_callback(0.88, "Saving raw intermediate stems...")

        raw_vocals_file = dest_dir / f"{stem_base_name}_Vocals_raw.wav"
        raw_inst_file = dest_dir / f"{stem_base_name}_Instrumental_raw.wav"

        cls.save_audio(vocals_tensor, str(raw_vocals_file), sr=sr, format_type="wav")
        cls.save_audio(instrumental_tensor, str(raw_inst_file), sr=sr, format_type="wav")

        # Intelligent Normalization & Loudness Enhancement
        if progress_callback:
            progress_callback(0.92, "Analyzing loudness and applying peak-protected normalization...")

        voc_np = vocals_tensor.cpu().numpy().T
        inst_np = instrumental_tensor.cpu().numpy().T

        norm_voc_np, voc_norm_meta = AudioNormalizer.normalize_stem(
            voc_np,
            sr=sr,
            enable_norm=enable_normalization,
            mode=norm_mode,
            target_lufs=target_lufs,
            target_peak_db=target_peak_db,
            user_gain_db=vocal_gain_db,
            peak_protection=peak_protection
        )

        norm_inst_np, inst_norm_meta = AudioNormalizer.normalize_stem(
            inst_np,
            sr=sr,
            enable_norm=enable_normalization,
            mode=norm_mode,
            target_lufs=target_lufs,
            target_peak_db=target_peak_db,
            user_gain_db=inst_gain_db,
            peak_protection=peak_protection
        )

        if progress_callback:
            progress_callback(0.96, "Saving enhanced master stems...")

        vocals_file = dest_dir / f"{stem_base_name}_Vocals.wav"
        instrumental_file = dest_dir / f"{stem_base_name}_Instrumental.wav"

        cls.save_audio(norm_voc_np, str(vocals_file), sr=sr, format_type="wav")
        cls.save_audio(norm_inst_np, str(instrumental_file), sr=sr, format_type="wav")

        if progress_callback:
            progress_callback(1.0, "Separation & audio level enhancement complete!")

        return {
            "status": "success",
            "vocals_path": str(vocals_file.resolve()),
            "instrumental_path": str(instrumental_file.resolve()),
            "vocals_raw_path": str(raw_vocals_file.resolve()),
            "instrumental_raw_path": str(raw_inst_file.resolve()),
            "duration": duration,
            "sample_rate": sr,
            "device_used": target_device.upper(),
            "model_used": model_name,
            "quality_mode": quality_mode,
            "metrics": {
                "original": orig_metrics,
                "vocals_raw": voc_norm_meta["raw_metrics"],
                "vocals_norm": voc_norm_meta["norm_metrics"],
                "vocals_gain_applied_db": voc_norm_meta["gain_applied_db"],
                "instrumental_raw": inst_norm_meta["raw_metrics"],
                "instrumental_norm": inst_norm_meta["norm_metrics"],
                "instrumental_gain_applied_db": inst_norm_meta["gain_applied_db"]
            }
        }
