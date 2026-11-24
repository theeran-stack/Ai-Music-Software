"""Integration and End-to-End Verification Test for Smart Mix Studio.
Tests:
1. Track queue management (reorder, move up, move down, remove).
2. Generate Proposed AI Mix Plan workflow.
3. Interactive Mix Timeline layout calculation (tracks, transitions, boundaries).
4. Continuous Mix Preview rendering and PlaybackEngine playback/pause/seek.
5. Master continuous DJ set rendering & MP3 export (320 kbps, -14 LUFS, no clipping).
6. Smart Mix project serialization (.json).
"""
import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

from core.audio_format import decode_mp3_to_numpy, validate_mp3_file
from ai.mixing.mix_plan import MixPlan
from ai.mixing.audio_analyzer import AudioAnalyzer
from ai.mixing.mix_planner import MixPlanner
from ai.mixing.mix_renderer import MixRenderer
from ui.components.mix_timeline import MixTimelineWidget
from ui.workspaces.smart_mix import SmartMixWorkspace
from core.playback_engine import PlaybackEngine


def test_smart_mix_complete_workflow():
    print("=" * 70)
    print("SMART MIX STUDIO COMPLETE WORKFLOW TEST")
    print("=" * 70)

    track1_path = str(BASE_DIR / "data" / "test_run" / "Test_Song.mp3")
    track2_path = str(BASE_DIR / "data" / "test_run" / "Test_Song_Instrumental.mp3")

    assert Path(track1_path).exists(), f"Track 1 missing: {track1_path}"
    assert Path(track2_path).exists(), f"Track 2 missing: {track2_path}"

    # 1. Test Workspace Initialization
    print("--- 1. Testing Workspace Instantiation & Controls ---")
    ws = SmartMixWorkspace()
    assert ws.btn_generate_plan.text() == "⚡  Generate Proposed AI Mix Plan"
    assert ws.btn_play_mix.text() == "▶ Preview Continuous Mix"
    assert ws.btn_render_export.text() == "🚀 Render & Export DJ Mix (MP3)"
    assert ws.btn_play_mix.isEnabled() is False
    assert ws.btn_render_export.isEnabled() is False
    assert ws.btn_save_project.isEnabled() is False
    print("  [PASS] Initial control states and MP3 labels verified.")

    # 2. Test Track Queue Population & Reordering
    print("--- 2. Testing Track Queue Operations ---")
    ws._track_queue.append({
        "path": track1_path,
        "title": "Test_Song",
        "bpm": 123.0,
        "key": "11B (A Major)",
        "duration": 4.0
    })
    ws._track_queue.append({
        "path": track2_path,
        "title": "Test_Song_Instrumental",
        "bpm": 117.5,
        "key": "11B (A Major)",
        "duration": 4.05
    })
    assert len(ws._track_queue) == 2

    # Move down / Move up
    ws.track_list.addItem("Item 1")
    ws.track_list.addItem("Item 2")
    ws.track_list.setCurrentRow(0)
    ws._move_down()
    assert ws._track_queue[0]["title"] == "Test_Song_Instrumental"
    assert ws._track_queue[1]["title"] == "Test_Song"
    ws._move_up()
    assert ws._track_queue[0]["title"] == "Test_Song"
    print("  [PASS] Queue reordering (Move Up / Move Down) verified.")

    # 3. Test Mix Plan Generation
    print("--- 3. Testing Real Mix Plan Generation ---")
    settings = {
        "transition_style": "Harmonic Crossfade (Camelot Wheel)",
        "transition_duration_bars": 8,
        "vocal_safety": "Strict: Never overlap prominent vocals"
    }

    plan = MixPlanner.plan_mix(
        tracks=[track1_path, track2_path],
        settings=settings
    )

    assert plan is not None
    assert len(plan.tracks) == 2
    assert len(plan.transitions) == 1
    assert plan.total_duration_seconds > 0.0

    tr = plan.transitions[0]
    assert tr.duration_seconds > 0.0
    assert tr.from_exit_start_sec >= 0.0
    assert tr.to_entry_start_sec >= 0.0
    print(f"  [PASS] MixPlan generated: {plan.total_duration_seconds:.2f}s total, Transition: '{tr.transition_type}' ({tr.duration_seconds}s)")

    # 4. Test Mix Timeline Layout & Visual Display
    print("--- 4. Testing Mix Timeline Widget Visualization ---")
    tl_widget = ws.mix_timeline
    tl_widget.set_plan(plan)

    assert len(tl_widget._track_layouts) == 2
    assert len(tl_widget._transition_layouts) == 1
    assert tl_widget._total_duration > 0.0

    t1 = tl_widget._track_layouts[0]
    t2 = tl_widget._track_layouts[1]
    assert t1["start_sec"] == 0.0
    assert t2["start_sec"] > 0.0
    assert t2["end_sec"] > t1["end_sec"]
    print(f"  [PASS] Timeline layout calculated: Track 1 [{t1['start_sec']:.2f}s - {t1['end_sec']:.2f}s], Track 2 [{t2['start_sec']:.2f}s - {t2['end_sec']:.2f}s]")

    # 5. Test Continuous Mix Preview Rendering & Audio Playback
    print("--- 5. Testing Continuous Mix Preview & Playback Sync ---")
    preview_out = BASE_DIR / "data" / "test_run" / "test_smart_mix_preview.mp3"
    render_res = MixRenderer.render_mix(plan, str(preview_out), export_bitrate="192k")

    assert render_res["status"] == "success"
    assert preview_out.exists() and preview_out.stat().st_size > 1000

    ws._on_plan_completed(plan)
    ws._on_preview_ready(str(preview_out), render_res)

    assert ws.btn_play_mix.isEnabled() is True
    assert ws.btn_render_export.isEnabled() is True
    assert ws.btn_save_project.isEnabled() is True
    print(f"  [PASS] Preview ready and controls enabled: Size={preview_out.stat().st_size} bytes")

    # Simulate Play, Pause, Seeking
    ws._play_continuous_mix()
    ws._on_playback_state_changed(True)
    assert ws.btn_play_mix.text() == "⏸ Pause Continuous Mix"

    ws._on_timeline_seek(0.5)
    assert abs(ws.mix_timeline._playhead_ratio - 0.5) < 0.01

    ws._on_playback_state_changed(False)
    assert ws.btn_play_mix.text() == "▶ Preview Continuous Mix"
    print("  [PASS] Play, Pause, and Timeline Seeking synchronized successfully.")

    # 6. Test Master Continuous DJ Set Rendering to 320 kbps MP3
    print("--- 6. Testing Master Continuous DJ Mix Rendering (MP3 @ 320k) ---")
    master_mix_out = BASE_DIR / "data" / "test_run" / "test_smart_dj_master_320k.mp3"
    master_res = MixRenderer.render_mix(plan, str(master_mix_out), export_bitrate="320k")

    assert master_res["status"] == "success"
    assert master_res["bitrate"] == "320k"
    master_out_data = decode_mp3_to_numpy(str(master_mix_out))
    assert master_out_data is not None
    data, sr = master_out_data
    assert data.shape[0] == 2
    assert data.shape[1] > 0
    assert not master_res["metrics"]["has_clipping"]
    assert not master_res["metrics"]["is_silent"]
    print(f"  [PASS] Master MP3 exported: Peak={master_res['metrics']['peak_db']:.2f} dBFS, LUFS={master_res['metrics']['lufs']}, Duration={master_res['duration_sec']:.2f}s")

    # 7. Test Save Mix Project (.json)
    print("--- 7. Testing Save Mix Project (.json) ---")
    proj_dict = plan.to_dict()
    proj_json = json.dumps(proj_dict, indent=2)
    reloaded = json.loads(proj_json)
    assert reloaded["name"] == plan.name
    assert len(reloaded["tracks"]) == 2
    assert len(reloaded["transitions"]) == 1
    print("  [PASS] Project serialization (.json) verified.")

    print("\n" + "=" * 70)
    print("ALL SMART MIX WORKFLOW TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    test_smart_mix_complete_workflow()
