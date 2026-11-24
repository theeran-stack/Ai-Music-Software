"""UI Integration Test for VocalRemoverWorkspace.
Tests audio loading, UI state transitions, worker execution, and results interface.
"""
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QEventLoop

from ui.workspaces.vocal_remover import VocalRemoverWorkspace
from core.playback_engine import PlaybackEngine

def test_vocal_remover_ui():
    app = QApplication.instance() or QApplication(sys.argv)

    workspace = VocalRemoverWorkspace()
    workspace.show()

    test_file = BASE_DIR / "data" / "test_run" / "Test_Song.mp3"
    if not test_file.exists():
        from core.audio_format import convert_file_to_mp3
        wav_file = BASE_DIR / "data" / "test_run" / "Test_Song.wav"
        if wav_file.exists():
            convert_file_to_mp3(wav_file, test_file, bitrate="320k")
    assert test_file.exists(), f"Test file {test_file} does not exist"

    print("--- 1. Testing Audio File Load in UI ---")
    workspace._load_audio_file(str(test_file))

    assert workspace.track_card.isVisible() is True
    assert workspace.drop_card.isVisible() is False
    assert workspace.btn_remove_vocals.isEnabled() is True
    assert workspace.results_container.isVisible() is False
    print("UI state transitioned correctly on audio load.")

    print("--- 2. Testing Separation Execution via UI ---")
    loop = QEventLoop()

    def on_worker_done():
        loop.quit()

    # Fast mode for quick test
    idx = workspace.quality_combo.findData("fast")
    if idx >= 0:
        workspace.quality_combo.setCurrentIndex(idx)

    workspace._start_separation()
    assert workspace._worker is not None
    workspace._worker.completed.connect(lambda res: on_worker_done())
    workspace._worker.failed.connect(lambda err: loop.quit())

    # Wait for completion in Qt event loop
    loop.exec()

    # Verify results interface
    assert workspace.results_container.isVisible() is True
    assert workspace._vocals_path is not None
    assert workspace._instrumental_path is not None
    assert workspace.vocals_waveform._peaks is not None
    assert workspace.inst_waveform._peaks is not None
    print("Results interface displayed successfully with valid waveform peaks.")

    print("--- 3. Testing Audio Preview Switching ---")
    workspace._switch_preview("vocals")
    assert workspace._currently_playing_stem == "vocals"

    workspace._switch_preview("instrumental")
    assert workspace._currently_playing_stem == "instrumental"

    workspace._switch_preview("original")
    assert workspace._currently_playing_stem == "original"

    print("Audio preview switches cleanly without multi-stream conflicts.")

    print("\n==============================================")
    print("ALL VOCAL REMOVER UI INTEGRATION TESTS PASSED!")
    print("==============================================")
    workspace.close()

if __name__ == "__main__":
    test_vocal_remover_ui()

