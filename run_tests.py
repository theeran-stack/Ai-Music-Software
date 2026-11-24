"""Convenience runner for Audio Studio AI test suites.
Discovers and executes all test suites in tests/ without requiring third-party runners.
"""
import sys
import subprocess
from pathlib import Path

TESTS = [
    "tests/test_milestone1.py",
    "tests/test_qt_warnings_regression.py",
    "tests/test_vocal_remover.py",
    "tests/test_vocal_remover_ui.py",
    "tests/test_hf_download_scenarios.py",
    "tests/test_v1_beta_features.py",
    "tests/test_v1_1_features.py",
    "tests/test_smart_mix_integration.py",
]

def main():
    root = Path(__file__).resolve().parent
    passed = 0
    failed = 0

    print("==================================================================")
    print("AUDIO STUDIO AI — COMPLETE TEST SUITE RUNNER")
    print("==================================================================\n")

    for test_path in TESTS:
        full_path = root / test_path
        if not full_path.exists():
            print(f"[SKIP] {test_path} not found.")
            continue

        print(f"Running: {test_path} ...")
        res = subprocess.run([sys.executable, str(full_path)], cwd=str(root))
        if res.returncode == 0:
            print(f"  [PASS] {test_path}\n")
            passed += 1
        else:
            print(f"  [FAIL] {test_path} (Exit Code: {res.returncode})\n")
            failed += 1

    print("==================================================================")
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED across {len(TESTS)} test suites.")
    print("==================================================================")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
