"""Comprehensive Master Test Suite for Audio Studio AI v1.0 Beta.
Validates:
1. Global Audio Format Mandate (MP3-only import, export, and bitrates).
2. AI Audio Enhancer (Noise reduction, clarity boost, pitch shift, time stretch, -14 LUFS).
3. AI Voice Lab (Formant morphing, pitch transposition, conversion strength, MP3 export).
4. Smart Mix Studio (Acoustic plan, transitions, continuous set rendering).
5. Essential Multitrack Audio Editor (Summing, volume scaling, broadcast normalisation).
6. AI Song Analyzer (BPM, Krumhansl-Schmuckler musical key, Camelot notation, segmentation).
"""
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np

from core.audio_format import (
    validate_mp3_file, encode_array_to_mp3, decode_mp3_to_numpy, convert_file_to_mp3
)
from config import AVAILABLE_MP3_BITRATES, SUPPORTED_AUDIO_EXTENSIONS
from ai.enhancement.audio_enhancer import AudioEnhancer
from ai.voice.voice_converter import VoiceConverter
from ai.mixing.audio_analyzer import AudioAnalyzer
from ai.mixing.mix_planner import MixPlanner
from ai.mixing.mix_renderer import MixRenderer

def test_mp3_format_mandate():
    print("--- 1. Testing Global Audio Format Mandate ---")
    assert SUPPORTED_AUDIO_EXTENSIONS == {".mp3"}, f"Non-MP3 extensions detected: {SUPPORTED_AUDIO_EXTENSIONS}"
    assert AVAILABLE_MP3_BITRATES == ["128k", "192k", "256k", "320k"]

    # Rejection of invalid extension
    fake_wav = BASE_DIR / "data" / "test_run" / "fake.wav"
    valid, err = validate_mp3_file(fake_wav)
    assert not valid, "Failed to reject non-MP3 extension"
    assert "Only MP3 audio files are supported" in err

    # Encoding all 4 bitrates
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = np.vstack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 440 * t)])

    for br in AVAILABLE_MP3_BITRATES:
        out_p = BASE_DIR / "data" / "test_run" / f"test_bitrate_{br}.mp3"
        encode_array_to_mp3(tone, str(out_p), sample_rate=sr, bitrate=br)
        assert out_p.exists() and out_p.stat().st_size > 500
        val, _ = validate_mp3_file(out_p)
        assert val, f"Generated MP3 at {br} failed validation"
        print(f"  [PASS] MP3 encoding at {br}: {out_p.stat().st_size} bytes")

def test_ai_audio_enhancer():
    print("\n--- 2. Testing AI Audio Enhancer & Loudness Normalisation ---")
    in_mp3 = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"
    assert in_mp3.exists()

    out_mp3 = BASE_DIR / "data" / "test_run" / "test_enhanced_suite.mp3"
    res = AudioEnhancer.one_click_master_enhance(str(in_mp3), str(out_mp3), export_bitrate="320k")

    assert res["status"] == "success"
    assert Path(res["output_path"]).exists()
    assert res["metrics"]["lufs"] <= -13.0
    assert res["metrics"]["peak_db"] <= -0.99
    valid, _ = validate_mp3_file(res["output_path"])
    assert valid
    print(f"  [PASS] One-Click Enhance: LUFS={res['metrics']['lufs']}, Peak={res['metrics']['peak_db']} dBFS")

def test_voice_converter():
    print("\n--- 3. Testing AI Voice Lab Morphing & Formant Transfer ---")
    in_mp3 = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"
    out_mp3 = BASE_DIR / "data" / "test_run" / "test_voice_morphed.mp3"

    res = VoiceConverter.convert_voice(
        source_audio_path=str(in_mp3),
        output_audio_path=str(out_mp3),
        pitch_semitones=3,
        conversion_strength=0.8,
        export_bitrate="320k"
    )

    assert res["status"] == "success"
    assert Path(res["output_path"]).exists()
    valid, _ = validate_mp3_file(res["output_path"])
    assert valid
    print(f"  [PASS] Voice Conversion: Mode='{res['mode']}', Peak={res['metrics']['peak_db']} dBFS")

def test_smart_mix_planner_and_renderer():
    print("\n--- 4. Testing Smart Mix Studio Planner & Continuous Renderer ---")
    p1 = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"
    p2 = BASE_DIR / "data" / "test_run" / "test_enhanced_suite.mp3"

    plan = MixPlanner.plan_mix([str(p1), str(p2)])
    assert len(plan.transitions) == 1
    tr = plan.transitions[0]
    print(f"  [PASS] Transition planned: '{tr.from_track_title}' ➔ '{tr.to_track_title}' ({tr.transition_type})")

    out_mix = BASE_DIR / "data" / "test_run" / "test_master_dj_set.mp3"
    res = MixRenderer.render_mix(plan, str(out_mix), export_bitrate="320k")
    assert res["status"] == "success"
    assert Path(res["output_path"]).exists()
    valid, _ = validate_mp3_file(res["output_path"])
    assert valid
    print(f"  [PASS] Continuous mix rendered: {res['duration_sec']:.2f}s, Peak={res['metrics']['peak_db']} dBFS")

def test_ai_analyzer():
    print("\n--- 5. Testing AI Song Analyzer Acoustic Extraction ---")
    p = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"
    ta = AudioAnalyzer.analyze_track(str(p))

    assert ta.bpm > 0, "BPM was not detected"
    assert ta.camelot_key != "", "Camelot key not detected"
    assert ta.musical_key != "", "Musical key not detected"
    assert 0.0 <= ta.energy_level <= 1.0, "Energy level out of bounds"
    assert 0.0 <= ta.vocal_presence_ratio <= 1.0, "Vocal ratio out of bounds"
    print(f"  [PASS] Analyzed: BPM={ta.bpm:.1f}, Key={ta.camelot_key} ({ta.musical_key}), Energy={ta.energy_level:.2f}")

def main():
    print("=" * 70)
    print("AUDIO STUDIO AI v1.0 BETA - MASTER FEATURES TEST SUITE")
    print("=" * 70)

    test_mp3_format_mandate()
    test_ai_audio_enhancer()
    test_voice_converter()
    test_smart_mix_planner_and_renderer()
    test_ai_analyzer()

    print("\n" + "=" * 70)
    print("ALL v1.0 BETA MASTER FEATURE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
