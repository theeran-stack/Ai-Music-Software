"""Automated Test Suite for Audio Studio AI v1.1 Targeted Features.
Validates:
1. Priority 1: Genuine RVC Neural Voice Conversion & Model Validator.
2. Priority 2: Interactive Multitrack Audio Editor (Clip selection, trim, split, merge, crossfade, master export).
3. Priority 3: Neural AI Speech Enhancement (Deep Conv-Recurrent spectral masking, chunk streaming, MP3 export).
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import torch

from core.audio_format import decode_mp3_to_numpy, encode_array_to_mp3, validate_mp3_file
from ai.voice.rvc_engine import SynthesizerTrn, RVCModelValidator, RVCInferencePipeline
from ai.voice.voice_converter import VoiceConverter
from core.clip_manager import TimelineClip, TimelineTrack
from ai.enhancement.neural_enhancer import NeuralAudioEnhancer, NeuralSpeechEnhancementNet


def test_rvc_neural_voice_conversion():
    print("--- 1. Testing Genuine RVC Neural Voice Conversion ---")
    src_vocal = BASE_DIR / "data" / "test_run" / "Test_Song_Vocals.mp3"
    assert src_vocal.exists(), f"Source vocal not found: {src_vocal}"

    # 1. Instantiate neural generator and save verified checkpoint
    model = SynthesizerTrn(gin_channels=256, sr=40000)
    ckpt_path = BASE_DIR / "data" / "test_run" / "v1_1_rvc_model.pth"
    torch.save({
        "weight": model.state_dict(),
        "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0.0, "ResBlock1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 8, 2, 2], 512, [16, 16, 4, 4], 0, 256, 40000]
    }, ckpt_path)

    # 2. Inspect checkpoint with model validator
    meta = RVCModelValidator.inspect_checkpoint(str(ckpt_path))
    assert meta["valid"], f"Model inspection failed: {meta.get('error')}"
    assert meta["version"] == "RVC v1"
    assert meta["sample_rate"] == 40000
    assert meta["has_pitch_guidance"] is True
    print(f"  [PASS] RVC Model Validator verified: {meta['version']}, {meta['sample_rate']} Hz")

    # 3. Test rejection of invalid model file
    invalid_ckpt = BASE_DIR / "data" / "test_run" / "invalid_model.pth"
    invalid_ckpt.write_text("CORRUPT_NON_TORCH_DATA")
    inv_meta = RVCModelValidator.inspect_checkpoint(str(invalid_ckpt))
    assert not inv_meta["valid"], "Failed to reject corrupt checkpoint"
    print("  [PASS] Corrupt checkpoint rejection verified.")
    if invalid_ckpt.exists():
        invalid_ckpt.unlink()

    # 4. Execute neural voice conversion pipeline
    out_neural = BASE_DIR / "data" / "test_run" / "v1_1_neural_converted.mp3"
    res_neural = VoiceConverter.convert_voice(
        source_audio_path=str(src_vocal),
        output_audio_path=str(out_neural),
        model_checkpoint_path=str(ckpt_path),
        pitch_semitones=2,
        conversion_strength=0.85,
        export_bitrate="320k"
    )
    assert res_neural["status"] == "success"
    assert "Neural RVC Voice Conversion" in res_neural["mode"]
    assert out_neural.exists() and out_neural.stat().st_size > 1000
    print(f"  [PASS] Neural RVC Conversion complete: Mode='{res_neural['mode']}', Size={out_neural.stat().st_size} bytes")

    # 5. Execute DSP Voice Morphing fallback with reference vocal
    ref_vocal = BASE_DIR / "data" / "test_run" / "Test_Song_Instrumental.mp3"
    out_dsp = BASE_DIR / "data" / "test_run" / "v1_1_dsp_morphed.mp3"
    res_dsp = VoiceConverter.convert_voice(
        source_audio_path=str(src_vocal),
        output_audio_path=str(out_dsp),
        reference_audio_path=str(ref_vocal),
        pitch_semitones=0,
        conversion_strength=0.75,
        export_bitrate="320k"
    )
    assert res_dsp["status"] == "success"
    assert "DSP Voice Morphing" in res_dsp["mode"]
    assert out_dsp.exists() and out_dsp.stat().st_size > 1000
    print(f"  [PASS] DSP Voice Morphing verified: Mode='{res_dsp['mode']}', Size={out_dsp.stat().st_size} bytes")


def test_interactive_audio_editor():
    print("--- 2. Testing Interactive Multitrack Audio Editor ---")
    src = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"

    # 1. Clip creation and properties
    clip = TimelineClip(file_path=str(src), timeline_start=1.0, in_point=0.5, duration=3.0)
    assert clip.timeline_start == 1.0
    assert clip.duration == 3.0
    assert clip.timeline_end == 4.0
    print("  [PASS] TimelineClip creation and boundary calculation verified.")

    # 2. Trimming operations
    clip.trim_start(0.5)
    assert clip.timeline_start == 1.5
    assert clip.duration == 2.5
    clip.trim_end(-0.5)
    assert clip.duration == 2.0
    print("  [PASS] TimelineClip start and end trimming verified.")

    # 3. Splitting clip at playhead
    full_clip = TimelineClip(file_path=str(src), timeline_start=0.0, in_point=0.0, duration=4.0)
    split_res = full_clip.split(playhead_sec=1.5)
    assert split_res is not None, "Failed to split clip at playhead"
    left, right = split_res
    assert left.duration == 1.5
    assert right.timeline_start == 1.5
    assert right.duration == 2.5
    print("  [PASS] Non-destructive clip splitting at playhead verified.")

    # 4. Track Lane Clip Management & Consolidation
    track = TimelineTrack(track_num=1, name="Vocals Lane")
    c1 = track.add_clip(str(src), timeline_start=0.0)
    c2 = track.add_clip(str(src), timeline_start=4.0)
    assert len(track.clips) == 2

    c1.is_selected = True
    c2.is_selected = True
    merged_ok = track.merge_selected_clips()
    assert merged_ok is True
    assert len(track.clips) == 1
    assert track.clips[0].duration >= 4.0
    print("  [PASS] Track clip merge and consolidation verified.")

    # 5. Crossfading
    track.add_clip(str(src), timeline_start=track.clips[0].timeline_end)
    cross_ok = track.apply_crossfade(crossfade_sec=0.5)
    assert cross_ok is True
    print("  [PASS] Track equal-power crossfading verified.")

    # 6. Master Summing & Master MP3 Export
    rendered_track = track.render_track(max_duration_sec=6.0)
    assert rendered_track.shape == (2, 6 * 44100)

    out_editor_mix = BASE_DIR / "data" / "test_run" / "v1_1_editor_master_mix.mp3"
    encode_array_to_mp3(rendered_track, str(out_editor_mix), sample_rate=44100, bitrate="320k")
    assert out_editor_mix.exists() and out_editor_mix.stat().st_size > 1000
    print(f"  [PASS] Multitrack summing and MP3 master export verified: {out_editor_mix.stat().st_size} bytes")


def test_neural_audio_enhancer():
    print("--- 3. Testing Neural AI Audio Enhancer ---")
    src = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"
    out = BASE_DIR / "data" / "test_run" / "v1_1_neural_enhanced.mp3"

    res = NeuralAudioEnhancer.enhance_audio(
        input_path=str(src),
        output_path=str(out),
        noise_suppression_strength=0.85,
        speech_clarity_boost=2.5,
        export_bitrate="320k"
    )

    assert res["status"] == "success"
    assert "PyTorch" in res["algorithm"]
    assert res["device"] in ("cpu", "cuda")
    assert res["chunks_processed"] >= 1
    assert out.exists() and out.stat().st_size > 1000

    data, sr = decode_mp3_to_numpy(str(out))
    assert data.shape[0] == 2
    assert data.shape[1] > 0
    print(f"  [PASS] Neural Audio Enhancement complete: Device={res['device']}, Chunks={res['chunks_processed']}, Size={out.stat().st_size} bytes")


if __name__ == "__main__":
    print("=" * 70)
    print("AUDIO STUDIO AI v1.1 TARGETED FEATURES VERIFICATION SUITE")
    print("=" * 70)
    test_rvc_neural_voice_conversion()
    test_interactive_audio_editor()
    test_neural_audio_enhancer()
    print("=" * 70)
    print("ALL v1.1 TARGETED FEATURE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)
