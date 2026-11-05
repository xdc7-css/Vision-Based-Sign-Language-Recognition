#!/usr/bin/env python3
"""Root-level application launcher for Vision-Based Sign Language Recognition.

Enables simple one-command launching:
    python run.py
    python run
    .\\.venv\\Scripts\\python.exe run.py

Preserves all existing application architecture, dependency injection,
MediaPipe patches, pretrained models, and PySide6 UI components.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _verify_environment() -> bool:
    """Perform pre-flight checks on required runtime dependencies."""
    missing_deps: list[str] = []

    deps = [
        ("PySide6", "PySide6 (Qt 6 Framework)"),
        ("cv2", "OpenCV (opencv-python)"),
        ("mediapipe", "Google MediaPipe"),
        ("onnxruntime", "ONNX Runtime"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
    ]

    for module_name, display_name in deps:
        try:
            __import__(module_name)
        except ImportError:
            missing_deps.append(f"  • {display_name} (import name: '{module_name}')")

    if missing_deps:
        print("\n" + "=" * 76)
        print(" [!] CRITICAL STARTUP ERROR: Missing Required Dependencies")
        print("=" * 76)
        print("The following essential libraries were not found in the current Python environment:")
        for dep in missing_deps:
            print(dep)
        print("\nTroubleshooting Instructions:")
        print("  1. Verify your Python virtual environment is activated:")
        print("       PowerShell:  .\\.venv\\Scripts\\Activate.ps1")
        print("       CMD:         .venv\\Scripts\\activate.bat")
        print("  2. If packages are missing, install them into your virtual environment:")
        print("       python -m pip install -r requirements.txt")
        print("=" * 76 + "\n")
        return False

    return True


def _verify_model_assets(project_root: Path) -> None:
    """Verify that required pretrained models are accessible."""
    karsl_model = project_root / "data" / "models" / "karsl" / "karsl_502.onnx"
    if not karsl_model.exists():
        print(f"[!] WARNING: Foundation model file not found at: {karsl_model}")
        print("    Live recognition using KArSL-502 may fail. Please ensure model assets are present.\n")


def main() -> int:
    """Bootstrap and launch the application."""
    project_root = Path(__file__).resolve().parent

    # Ensure project root is the first entry in sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Ensure working directory is set to project root
    os.chdir(project_root)

    # Pre-flight environment check
    if not _verify_environment():
        return 1

    # Model asset presence check
    _verify_model_assets(project_root)

    # Delegate directly to the verified src.main entry point
    try:
        from src.main import main as app_main
        app_main()
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except KeyboardInterrupt:
        print("\n[*] Application shutdown requested by user (Ctrl+C). Exiting cleanly.")
        return 0
    except Exception as exc:
        print(f"\n[!] Unexpected application error during execution: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
