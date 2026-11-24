"""Automated test suite for Milestone 1 - Audio Studio AI.
Tests initialization of all 11 workspaces, theme manager, animations, database, and bottom player.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Set offscreen platform for headless verification
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from ui.main_window import MainWindow
from ui.themes.theme_manager import ThemeManager
from ui.animations.animation_manager import AnimationManager
from database.database import DatabaseManager

def test_full_application():
    print("--- 1. Initializing QApplication ---")
    app = QApplication.instance() or QApplication(sys.argv)

    print("--- 2. Initializing Database ---")
    db = DatabaseManager()
    stats = db.get_library_statistics()
    print("Database OK. Stats:", stats)

    print("--- 3. Testing Theme Manager & 4 Presets ---")
    tm = ThemeManager.get_instance()
    themes = tm.get_available_themes()
    print("Available themes:", themes)
    assert len(themes) >= 4, f"Expected at least 4 themes, got {len(themes)}"

    for tid in ["obsidian_silver", "graphite", "platinum_light", "midnight"]:
        success = tm.set_theme(tid)
        assert success, f"Failed to activate theme {tid}"
        qss = tm.generate_qss()
        assert len(qss) > 100, f"QSS generation failed for theme {tid}"
        print(f"Theme '{tid}' verified successfully.")

    # Reset back to obsidian_silver
    tm.set_theme("obsidian_silver")

    print("--- 4. Initializing MainWindow & All 11 Workspaces ---")
    window = MainWindow()
    window.show()

    workspaces = [
        "dashboard",
        "library",
        "vocal_remover",
        "vocal_studio",
        "voice_lab",
        "audio_editor",
        "smart_mix",
        "ai_analyzer",
        "visualizer",
        "plugin_manager",
        "settings"
    ]

    for wk in workspaces:
        assert wk in window._workspaces, f"Workspace {wk} missing from window!"
        window._set_workspace(wk)
        curr = window.stack.currentWidget()
        assert curr == window._workspaces[wk], f"Workspace {wk} failed to set as current widget!"
        print(f"Workspace '{wk}' initialized and switched successfully.")

    print("--- 5. Testing Animation Manager ---")
    anim = AnimationManager.get_instance()
    assert anim.is_enabled is True
    print("Animation Manager OK.")

    print("--- 6. Testing Resize Events ---")
    window.resize(1200, 800)
    window.resize(1600, 1000)
    print("Resize OK.")

    print("--- 7. Testing Bottom Player Bar ---")
    assert window.player_bar is not None
    assert window.player_bar.height() == 82
    print("Player Bar OK.")

    print("\n==========================================")
    print("ALL MILESTONE 1 VERIFICATION TESTS PASSED!")
    print("==========================================")
    window.close()

if __name__ == "__main__":
    test_full_application()
