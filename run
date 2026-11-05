#!/usr/bin/env python3
"""Global application launcher for Vision-Based Sign Language Recognition.

Enables one-command launching from the project root or any project subfolder:
    python run
    python run.py

Automatically locates the real project root, dispatches execution to the
isolated Python 3.12 virtual environment (.venv), verifies model assets,
and launches the PySide6 desktop GUI.
"""

from __future__ import annotations

import os
import subprocess
import sys


def find_project_root() -> str | None:
    """Locate the project root directory by searching upward for project markers."""
    start_dirs: list[str] = []

    # 1. Script directory (if __file__ exists)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        start_dirs.append(script_dir)
    except NameError:
        pass

    # 2. Current working directory
    try:
        cwd = os.path.abspath(os.getcwd())
        if cwd not in start_dirs:
            start_dirs.append(cwd)
    except Exception:
        pass

    for start in start_dirs:
        cur = start
        while True:
            has_venv = os.path.isdir(os.path.join(cur, ".venv"))
            has_src = os.path.isfile(os.path.join(cur, "src", "main.py"))
            has_karsl = os.path.isfile(os.path.join(cur, "data", "models", "karsl", "karsl_502.onnx"))

            if (has_venv and has_src) or (has_src and has_karsl):
                return cur

            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    return None


def get_venv_python(project_root: str) -> str:
    """Return the absolute path to the project virtual environment Python binary."""
    venv_py = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_py):
        return venv_py
    posix_py = os.path.join(project_root, ".venv", "bin", "python")
    if os.path.isfile(posix_py):
        return posix_py
    return venv_py


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


def _verify_model_assets(project_root: str) -> None:
    """Verify that required pretrained models are accessible."""
    karsl_model = os.path.join(project_root, "data", "models", "karsl", "karsl_502.onnx")
    if not os.path.isfile(karsl_model):
        print(f"[!] WARNING: Foundation model file not found at: {karsl_model}")
        print("    Live recognition using KArSL-502 may fail. Please ensure model assets are present.\n")


def bootstrap_gui(project_root: str) -> int:
    """Execute the application inside the confirmed virtual environment."""
    os.chdir(project_root)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not _verify_environment():
        return 1

    _verify_model_assets(project_root)

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


def main() -> int:
    """Main launcher entry point with automatic root discovery and venv dispatch."""
    project_root = find_project_root()
    if not project_root:
        print("\n" + "=" * 76, file=sys.stderr)
        print(" [!] ERROR: Project Root Not Found", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        print("Could not locate the 'Vision-Based Sign Language Recognition' root directory.", file=sys.stderr)
        print(f"Current Working Directory: {os.getcwd()}", file=sys.stderr)
        print("Please run this command from within the project directory tree.", file=sys.stderr)
        print("=" * 76 + "\n", file=sys.stderr)
        return 1

    venv_python = get_venv_python(project_root)
    if not os.path.isfile(venv_python):
        print("\n" + "=" * 76, file=sys.stderr)
        print(" [!] ERROR: Python Virtual Environment (.venv) Not Found", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        print(f"Expected virtual environment at: {venv_python}", file=sys.stderr)
        print("\nPlease create the virtual environment and install dependencies:", file=sys.stderr)
        print(f"  cd \"{project_root}\"", file=sys.stderr)
        print("  py -3.12 -m venv .venv", file=sys.stderr)
        print("  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", file=sys.stderr)
        print("=" * 76 + "\n", file=sys.stderr)
        return 1

    current_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(os.path.abspath(venv_python))

    if current_exe == target_exe:
        return bootstrap_gui(project_root)

    # Avoid recursion
    if os.environ.get("_ARSL_DISPATCH_ACTIVE") == "1":
        print(f"[!] ERROR: Recursion detected during virtual environment dispatch ({sys.executable}). Aborting.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["_ARSL_DISPATCH_ACTIVE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    root_run_script = os.path.join(project_root, "run.py")
    cmd = [venv_python, root_run_script] + sys.argv[1:]

    try:
        res = subprocess.run(cmd, cwd=project_root, env=env)
        if res.returncode in (0, 3221225477, -1073741819, -1):
            return 0
        return res.returncode
    except KeyboardInterrupt:
        print("\n[*] Application shutdown requested by user (Ctrl+C). Exiting cleanly.")
        return 0
    except Exception as exc:
        print(f"\n[!] Failed to launch virtual environment process: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
