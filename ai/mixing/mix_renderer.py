"""Continuous DJ Mix Audio Renderer for Smart Mix Studio.
Executes seamless beat-aligned transitions and harmonic crossfading
between sequenced audio tracks according to an AI MixPlan,
normalises to broadcast loudness standards (-14 LUFS, -1.0 dBTP),
and exports strictly to MP3 (128k, 192k, 256k, 320k).
"""
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import numpy as np

from config import DEFAULT_EXPORT_BITRATE, CACHE_DIR
from core.audio_format import encode_array_to_mp3, decode_mp3_to_numpy
from ai.separation.audio_normalizer import AudioNormalizer
from ai.mixing.mix_plan import MixPlan

class MixRenderer:
    """Renders a continuous DJ set from a validated MixPlan."""

    @classmethod
    def render_mix(
        cls,
        plan: MixPlan,
        output_path: str,
        export_bitrate: str = "320k",
        cancel_token: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        out_p = Path(output_path).with_suffix(".mp3")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if len(plan.tracks) < 2:
            raise ValueError("MixPlan must have at least 2 tracks.")

        sr = 44100
        num_tracks = len(plan.tracks)

        if progress_cb:
            progress_cb(0.05, f"Loading audio for Track 1: {plan.tracks[0].title}...")

        # Load first track audio
        first_track_path = getattr(plan.tracks[0], "path", getattr(plan.tracks[0], "file_path", ""))
        current_audio, cur_sr = decode_mp3_to_numpy(first_track_path, target_sr=sr)
        if current_audio.ndim == 1:
            current_audio = np.vstack([current_audio, current_audio])

        # Sequentially stitch and crossfade subsequent tracks
        for idx in range(len(plan.transitions)):
            if cancel_token and cancel_token.is_set():
                raise InterruptedError("Mix rendering cancelled.")

            trans = plan.transitions[idx]
            next_track_info = plan.tracks[idx + 1]

            if progress_cb:
                pct = 0.1 + 0.65 * (idx / max(1, len(plan.transitions)))
                progress_cb(pct, f"Crossfading '{trans.from_track_title}' ➔ '{trans.to_track_title}'...")

            next_track_path = getattr(next_track_info, "path", getattr(next_track_info, "file_path", ""))
            next_audio, _ = decode_mp3_to_numpy(next_track_path, target_sr=sr)
            if next_audio.ndim == 1:
                next_audio = np.vstack([next_audio, next_audio])

            # Transition sample calculations
            exit_start_samp = int(trans.from_exit_start_sec * sr)
            exit_end_samp = min(current_audio.shape[1], int(trans.from_exit_end_sec * sr))
            trans_len = max(sr // 2, exit_end_samp - exit_start_samp)  # at least 0.5s

            entry_start_samp = int(trans.to_entry_start_sec * sr)
            entry_end_samp = entry_start_samp + trans_len

            # Clamp track A
            part_a_pre = current_audio[:, :exit_start_samp]
            part_a_xfade = current_audio[:, exit_start_samp:exit_start_samp + trans_len]
            if part_a_xfade.shape[1] < trans_len:
                part_a_xfade = np.pad(part_a_xfade, ((0, 0), (0, trans_len - part_a_xfade.shape[1])))

            # Clamp track B
            if entry_end_samp > next_audio.shape[1]:
                entry_end_samp = next_audio.shape[1]
                trans_len = entry_end_samp - entry_start_samp
                part_a_xfade = part_a_xfade[:, :trans_len]

            part_b_xfade = next_audio[:, entry_start_samp:entry_end_samp]
            if part_b_xfade.shape[1] < trans_len:
                part_b_xfade = np.pad(part_b_xfade, ((0, 0), (0, trans_len - part_b_xfade.shape[1])))

            part_b_post = next_audio[:, entry_end_samp:]

            # Equal-power crossfade curves
            t = np.linspace(0.0, 1.0, trans_len, endpoint=False, dtype=np.float32)
            fade_out = np.cos(t * (np.pi / 2.0))
            fade_in = np.sin(t * (np.pi / 2.0))

            blended_trans = (part_a_xfade * fade_out) + (part_b_xfade * fade_in)

            # Concatenate to form new continuous mix
            current_audio = np.hstack([part_a_pre, blended_trans, part_b_post])

        if cancel_token and cancel_token.is_set():
            raise InterruptedError("Mix rendering cancelled.")

        if progress_cb:
            progress_cb(0.85, "Normalising continuous set to broadcast standards (-14 LUFS)...")

        norm_audio, norm_meta = AudioNormalizer.normalize_stem(
            current_audio,
            sr=sr,
            enable_norm=True,
            mode="lufs",
            target_lufs=-14.0,
            target_peak_db=-1.0,
            peak_protection=True
        )

        if progress_cb:
            progress_cb(0.92, f"Encoding full continuous set to MP3 ({export_bitrate})...")

        encode_array_to_mp3(
            norm_audio,
            str(out_p),
            sample_rate=sr,
            bitrate=export_bitrate
        )

        metrics = AudioNormalizer.measure_audio(norm_audio, sr)

        if progress_cb:
            progress_cb(1.0, "Continuous DJ mix rendered successfully!")

        n_samples = norm_audio.shape[0] if norm_audio.shape[0] > norm_audio.shape[1] else norm_audio.shape[1]

        return {
            "status": "success",
            "output_path": str(out_p.resolve()),
            "bitrate": export_bitrate,
            "duration_sec": n_samples / float(sr),
            "metrics": metrics
        }
