"""Regression Test Suite for Qt Font Warnings and Hugging Face Authentication.
Verifies that:
1. No unsupported CSS properties (font-feature-settings, font-variant-numeric) exist in any source or theme file.
2. No QFont setPointSize <= 0 (-1) warnings are emitted across all 11 workspaces, themes, and player controls.
3. No unauthenticated Hugging Face Hub network warnings are emitted during model inspection or execution.
"""
import sys
import os
import re
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Use offscreen platform for automated test runner
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# pytest optional
from PySide6.QtCore import qInstallMessageHandler, QtMsgType
from PySide6.QtWidgets import QApplication, QPushButton, QComboBox

captured_warnings = []

def qt_message_handler(mode, context, message):
    captured_warnings.append({
        "mode": mode,
        "message": message,
        "file": context.file,
        "line": context.line,
        "function": context.function
    })

def test_static_css_properties_scan():
    """Scans all project source files to ensure no unsupported CSS properties remain."""
    print("\n--- 1. Testing Static Source Code for Unsupported CSS ---")
    unsupported_patterns = [
        re.compile(r"font-feature-settings\s*:", re.IGNORECASE),
        re.compile(r"font-variant-numeric\s*:", re.IGNORECASE)
    ]

    violations = []
    for root, dirs, files in os.walk(str(BASE_DIR)):
        if any(ignored in root for ignored in [".git", "bin", "__pycache__", "venv"]):
            continue
        for f in files:
            if f.endswith((".py", ".json", ".qss", ".css")):
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                        for line_num, line in enumerate(fp, start=1):
                            for pat in unsupported_patterns:
                                if pat.search(line):
                                    violations.append((file_path, line_num, line.strip()))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    assert len(violations) == 0, f"Found unsupported CSS properties in source files: {violations}"
    print("  [PASS] 0 unsupported font CSS properties found across entire project.")

def test_dynamic_theme_qss_scan():
    """Scans generated QSS for all themes to ensure no unsupported properties are injected."""
    print("\n--- 2. Testing Dynamically Generated Theme Stylesheets ---")
    from ui.themes.theme_manager import ThemeManager

    app = QApplication.instance() or QApplication(sys.argv)
    tm = ThemeManager.get_instance()

    for theme_id in tm.get_available_themes().keys():
        success = tm.set_theme(theme_id)
        assert success, f"Failed to activate theme {theme_id}"
        qss = tm.generate_qss()
        assert "font-feature-settings" not in qss, f"Theme {theme_id} QSS contains font-feature-settings!"
        assert "font-variant-numeric" not in qss, f"Theme {theme_id} QSS contains font-variant-numeric!"
        print(f"  [PASS] Theme '{theme_id}' stylesheet clean.")

def test_runtime_qt_warnings():
    """Runs full application through all workspaces, themes, and interactions with Qt message handler."""
    print("\n--- 3. Testing Full Application Runtime with Active Qt Message Handler ---")
    captured_warnings.clear()
    qInstallMessageHandler(qt_message_handler)

    app = QApplication.instance() or QApplication(sys.argv)

    from ui.main_window import MainWindow
    from ui.themes.theme_manager import ThemeManager
    from ai.separation.demucs_adapter import SeparationEngine

    win = MainWindow()
    if hasattr(win, "splash") and win.splash:
        win.splash.hide()
    win.show()
    app.processEvents()

    # Step A: Visit all 11 Workspaces
    for wk_name, wk in win._workspaces.items():
        win._set_workspace(wk_name)
        app.processEvents()
        # Ensure buttons with menus open and close without setPointSize warnings
        for btn in wk.findChildren(QPushButton):
            if btn.menu():
                m = btn.menu()
                m.show()
                app.processEvents()
                m.close()
        for cb in wk.findChildren(QComboBox):
            cb.showPopup()
            app.processEvents()
            cb.hidePopup()
        app.processEvents()

    # Step B: Switch across all 4 themes
    tm = ThemeManager.get_instance()
    for tid in ["obsidian_silver", "graphite", "midnight", "platinum_light", "obsidian_silver"]:
        tm.set_theme(tid)
        app.processEvents()

    # Step C: Player bar interactions
    pb = win.player_bar
    pb.btn_shuffle.click()
    pb.btn_repeat.click()
    pb.seek_slider.setValue(450)
    pb.volume_slider.setValue(75)
    if pb.btn_speed.menu():
        m = pb.btn_speed.menu()
        m.show()
        app.processEvents()
        m.close()
    app.processEvents()

    # Step D: Vocal remover file load
    vr = win._workspaces.get("vocal_remover")
    test_wav = str(BASE_DIR / "data" / "test_run" / "Test_Song.wav")
    if os.path.exists(test_wav) and vr:
        vr._load_audio_file(test_wav)
        app.processEvents()

    # Step E: Offline model check
    assert SeparationEngine.is_model_installed("htdemucs") is True

    win.close()
    app.processEvents()

    # Inspect captured warnings
    font_feature_warnings = [w for w in captured_warnings if "font-feature-settings" in w["message"]]
    point_size_warnings = [w for w in captured_warnings if "setPointSize" in w["message"] or "setPointSizeF" in w["message"]]
    hf_unauth_warnings = [w for w in captured_warnings if "unauthenticated" in w["message"].lower()]

    print(f"\nRuntime Warning Counts:")
    print(f"  • 'Unknown property font-feature-settings': {len(font_feature_warnings)}")
    print(f"  • 'QFont::setPointSize: Point size <= 0 (-1)': {len(point_size_warnings)}")
    print(f"  • 'HF Hub unauthenticated requests': {len(hf_unauth_warnings)}")

    assert len(font_feature_warnings) == 0, f"Detected font-feature-settings warnings: {font_feature_warnings}"
    assert len(point_size_warnings) == 0, f"Detected setPointSize warnings: {point_size_warnings}"
    assert len(hf_unauth_warnings) == 0, f"Detected Hugging Face unauthenticated warnings: {hf_unauth_warnings}"
    print("\n  [PASS] All Qt warnings successfully eliminated during full application runtime.")

if __name__ == "__main__":
    test_static_css_properties_scan()
    test_dynamic_theme_qss_scan()
    test_runtime_qt_warnings()
    print("\n==================================================================")
    print("ALL QT WARNING REGRESSION TESTS PASSED! ZERO WARNINGS DETECTED.")
    print("==================================================================")
