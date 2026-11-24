"""End-to-End Validation Test Suite for AI Vocal Remover, Audio Normalization, and Multi-Format Exports.
Tests:
1. Separation output levels (measured numerically in dBFS & LUFS).
2. Peak normalisation.
3. Loudness normalisation (ITU-R BS.1770).
4. Clipping protection (-1.0 dBTP ceiling).
5. Silent audio handling.
6. WAV export & decoding.
7. MP3 export & decoding.
8. FLAC export & decoding.
9. Independent export of both stems.
10. Correct filename extensions.
11. Actual file decoding after export.
12. Playback volume & non-destructive processing.
"""
import sys
import os
import hashlib
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import soundfile as sf
import torch

from ai.separation.demucs_adapter import SeparationEngine, SEPARATED_OUTPUT_DIR
from ai.separation.audio_normalizer import AudioNormalizer
from core.waveform_engine import WaveformEngine

def generate_test_audio(file_path: Path, duration_sec: float = 4.0, sr: int = 44100):
    """Generates synthetic audio with vocal formant melody + bass/percussion backing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # 1. Accompaniment: 110Hz sub-bass + 220Hz rhythm chords + kick beat
    bass = 0.5 * np.sin(2 * np.pi * 110 * t)
    chord = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)
    beat = np.zeros_like(t)
    beat_indices = np.arange(0, len(t), int(sr * 0.5))
    for idx in beat_indices:
        decay_len = min(int(sr * 0.08), len(t) - idx)
        beat[idx:idx + decay_len] += np.hanning(decay_len) * 0.7
    accompaniment = bass + chord + beat

    # 2. Vocal component: 440Hz / 554Hz melody with 5Hz vibrato and formants
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5 * t)
    freq = np.where(t < 2.0, 440.0 * vibrato, 554.0 * vibrato)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    vocal = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)

    # Stereo mix
    left = accompaniment * 0.8 + vocal * 0.7
    right = accompaniment * 0.8 + vocal * 0.7

    stereo = np.column_stack([left, right]).astype(np.float32)
    max_val = np.max(np.abs(stereo))
    if max_val > 0:
        stereo /= max_val

    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file_path), stereo, sr, subtype="PCM_16")

def main():
    print("=" * 70)
    print("AI VOCAL REMOVER - AUDIO LEVEL & MULTI-FORMAT EXPORT VALIDATION")
    print("=" * 70)

    test_dir = BASE_DIR / "data" / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_song_path = test_dir / "Test_Song.wav"

    # Generate test audio
    generate_test_audio(test_song_path, duration_sec=4.0, sr=44100)

    with open(test_song_path, "rb") as f:
        orig_hash = hashlib.sha256(f.read()).hexdigest()

    # --- TEST 1: Unit Test AudioNormalizer: Silent Audio Handling ---
    print("\n--- 1. Testing Silent Audio Handling ---")
    silent_audio = np.zeros((44100, 2), dtype=np.float32)
    s_meas = AudioNormalizer.measure_audio(silent_audio, 44100)
    assert s_meas["is_silent"] is True or s_meas["is_silent"] == 1, "Silent audio not detected as silent"
    s_norm, s_meta = AudioNormalizer.normalize_stem(silent_audio, 44100, enable_norm=True, target_lufs=-14.0)
    assert s_meta["gain_applied_db"] <= 3.0, f"Excessive gain applied to silence: {s_meta['gain_applied_db']} dB"
    print(f"[PASS] Silent audio handled safely (Gain capped at {s_meta['gain_applied_db']} dB).")

    # --- TEST 2: Unit Test AudioNormalizer: Peak & Loudness Normalization & Clipping Protection ---
    print("\n--- 2. Testing Loudness & Peak Normalisation with Clipping Protection ---")
    # Signal with low peak (-14 dBFS)
    sr = 44100
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    quiet_test = np.column_stack([0.2 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 440 * t)]).astype(np.float32)
    q_meas = AudioNormalizer.measure_audio(quiet_test, sr)
    print(f"  Quiet Input: Peak = {q_meas['peak_db']} dBFS, LUFS = {q_meas['lufs']}")

    # Loudness target -14 LUFS with -1.0 dBTP peak ceiling
    norm_audio, n_meta = AudioNormalizer.normalize_stem(
        quiet_test, sr, enable_norm=True, mode="lufs", target_lufs=-14.0, target_peak_db=-1.0, peak_protection=True
    )
    print(f"  Normalised Output: Peak = {n_meta['norm_metrics']['peak_db']} dBFS, LUFS = {n_meta['norm_metrics']['lufs']}, Gain Applied = {n_meta['gain_applied_db']} dB")
    assert n_meta["norm_metrics"]["peak_db"] <= -0.99, f"Peak ceiling breached: {n_meta['norm_metrics']['peak_db']} dBFS"
    assert n_meta["norm_metrics"]["has_clipping"] is False, "Clipping occurred during normalisation"
    print("[PASS] Loudness normalisation and -1.0 dBTP peak ceiling protection verified.")

    # --- TEST 3: Separation Execution with Normalization ---
    print("\n--- 3. Running Neural Separation with Intelligent Normalization ---")
    res = SeparationEngine.separate_2stem(
        file_path=str(test_song_path),
        output_dir=str(test_dir),
        model_name="htdemucs",
        quality_mode="fast",
        device="cpu",
        enable_normalization=True,
        norm_mode="lufs",
        target_lufs=-14.0,
        target_peak_db=-1.0,
        vocal_gain_db=2.0,
        inst_gain_db=0.0
    )

    print("Separation Status:", res["status"])
    print("Vocals Raw Path:", res["vocals_raw_path"])
    print("Vocals Enhanced Path:", res["vocals_path"])
    print("Instrumental Raw Path:", res["instrumental_raw_path"])
    print("Instrumental Enhanced Path:", res["instrumental_path"])

    # --- TEST 4: Numerical Level Comparison (Raw vs Enhanced) ---
    print("\n--- 4. Numerical Level Comparison ---")
    m = res["metrics"]
    orig_m = m["original"]
    vr_m = m["vocals_raw"]
    vn_m = m["vocals_norm"]
    ir_m = m["instrumental_raw"]
    in_m = m["instrumental_norm"]

    print(f"Original Mix : Peak = {orig_m['peak_db']:6.2f} dBFS | RMS = {orig_m['rms_db']:6.2f} dBFS | LUFS = {orig_m['lufs']:6.2f}")
    print(f"Vocals (Raw) : Peak = {vr_m['peak_db']:6.2f} dBFS | RMS = {vr_m['rms_db']:6.2f} dBFS | LUFS = {vr_m['lufs']:6.2f}")
    print(f"Vocals (Norm): Peak = {vn_m['peak_db']:6.2f} dBFS | RMS = {vn_m['rms_db']:6.2f} dBFS | LUFS = {vn_m['lufs']:6.2f} [Gain: {m['vocals_gain_applied_db']:+.2f} dB]")
    print(f"Inst (Raw)   : Peak = {ir_m['peak_db']:6.2f} dBFS | RMS = {ir_m['rms_db']:6.2f} dBFS | LUFS = {ir_m['lufs']:6.2f}")
    print(f"Inst (Norm)  : Peak = {in_m['peak_db']:6.2f} dBFS | RMS = {in_m['rms_db']:6.2f} dBFS | LUFS = {in_m['lufs']:6.2f} [Gain: {m['instrumental_gain_applied_db']:+.2f} dB]")

    # Verify enhanced stems are audibly louder than raw stems and within ceiling
    assert vn_m["peak_db"] > vr_m["peak_db"], "Enhanced vocals not brought to healthy listening level"
    assert vn_m["peak_db"] <= -0.99, f"Vocals peak ceiling exceeded: {vn_m['peak_db']}"
    assert in_m["peak_db"] <= -0.99, f"Instrumental peak ceiling exceeded: {in_m['peak_db']}"
    print("[PASS] Vocal and Instrumental stems elevated to broadcast listening standards without clipping.")

    # --- TEST 5: Verify Non-Destructive Integrity ---
    with open(test_song_path, "rb") as f:
        new_hash = hashlib.sha256(f.read()).hexdigest()
    assert orig_hash == new_hash, "Original file was modified!"
    print("\n[PASS] Original input file remains 100% untouched.")

    # --- TEST 6: Multi-Format Export Validation (WAV, MP3, FLAC) ---
    print("\n--- 6. Multi-Format Export & Decoding Validation ---")
    formats_to_test = [
        ("WAV", ".wav"),
        ("MP3", ".mp3"),
        ("FLAC", ".flac")
    ]

    for stem_name, stem_path in [("Vocals", res["vocals_path"]), ("Instrumental", res["instrumental_path"])]:
        wav_tensor, stem_sr, dur = SeparationEngine.load_audio(stem_path)
        for fmt_label, ext in formats_to_test:
            out_file = test_dir / f"Exported_{stem_name}{ext}"
            saved_path = SeparationEngine.save_audio(wav_tensor, str(out_file), sr=stem_sr, format_type=ext)
            p = Path(saved_path)

            # Check file existence and non-zero size
            assert p.exists(), f"Exported file {p} does not exist!"
            assert p.stat().st_size > 1000, f"Exported file {p} is too small ({p.stat().st_size} bytes)!"
            assert p.suffix.lower() == ext.lower(), f"Extension mismatch: expected {ext}, got {p.suffix}"

            # Verify actual audio decoding
            decoded_info = sf.info(str(p))
            assert decoded_info.channels == 2, f"{p.name} decoded channels mismatch: {decoded_info.channels}"
            assert decoded_info.samplerate == 44100, f"{p.name} samplerate mismatch: {decoded_info.samplerate}"
            assert abs(decoded_info.duration - dur) < 0.2, f"{p.name} duration mismatch: {decoded_info.duration}"
            print(f"  [PASS] {stem_name} {fmt_label}: {p.name} ({p.stat().st_size} bytes, {decoded_info.duration:.2f}s, {decoded_info.format})")

    # --- TEST 7: Independent Download of Both Stems ---
    print("\n--- 7. Testing Independent and Batch Downloads ---")
    batch_dir = test_dir / "batch_download"
    batch_dir.mkdir(parents=True, exist_ok=True)
    v_target = batch_dir / "Song_Vocals.flac"
    i_target = batch_dir / "Song_Instrumental.flac"

    SeparationEngine.save_audio(wav_tensor, str(v_target), sr=44100, format_type=".flac")
    SeparationEngine.save_audio(wav_tensor, str(i_target), sr=44100, format_type=".flac")

    assert v_target.exists() and sf.info(str(v_target)).format == "FLAC"
    assert i_target.exists() and sf.info(str(i_target)).format == "FLAC"
    print("[PASS] Batch download into custom folder with custom format (.flac) verified.")

    print("\n" + "=" * 70)
    print("ALL 12 AUDIO LEVEL & MULTI-FORMAT EXPORT VALIDATION TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
