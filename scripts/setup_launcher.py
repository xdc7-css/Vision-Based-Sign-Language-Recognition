#!/usr/bin/env python3
"""One-time project launcher setup utility for Vision-Based Sign Language Recognition.

Ensures that 'python run' and 'python run.py' work effortlessly from any
directory inside the project on Windows, macOS, or Linux.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def setup() -> int:
    project_root = Path(__file__).resolve().parent.parent
    print(f"[*] Configuring Project Launcher for: {project_root}")

    # 1. Verify required root files
    root_run_py = project_root / "run.py"
    root_run = project_root / "run"
    sitecustomize = project_root / "sitecustomize.py"

    if not root_run_py.is_file() or not root_run.is_file():
        print(f"[!] Error: Launcher files missing in {project_root}", file=sys.stderr)
        return 1

    # 2. Key subdirectories that contain trampolines
    target_dirs = [
        project_root / "src",
        project_root / "src" / "models",
        project_root / "src" / "presentation",
        project_root / "tests",
        project_root / "docs",
    ]

    for target in target_dirs:
        if target.is_dir():
            target_py = target / "run.py"
            target_run = target / "run"
            target_py.write_text(root_run_py.read_text(encoding="utf-8"), encoding="utf-8")
            target_run.write_text(root_run.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  [+] Synced launcher in: {target.relative_to(project_root)}")

    # 3. On Windows, register project root in User PYTHONPATH registry
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS)
            try:
                existing_val, _ = winreg.QueryValueEx(key, "PYTHONPATH")
            except FileNotFoundError:
                existing_val = ""

            str_root = str(project_root)
            paths = [p for p in existing_val.split(";") if p]
            if str_root not in paths:
                paths.insert(0, str_root)
                new_val = ";".join(paths)
                winreg.SetValueEx(key, "PYTHONPATH", 0, winreg.REG_SZ, new_val)
                print(f"  [+] Configured User PYTHONPATH in Windows Registry: {new_val}")
            else:
                print(f"  [=] User PYTHONPATH already contains: {str_root}")
            winreg.CloseKey(key)
        except Exception as exc:
            print(f"  [!] Note: Could not set User PYTHONPATH via registry ({exc}). Subdirectory trampolines will be used.")

    print("\n[✓] Launcher setup complete! You can now run:")
    print("      python run")
    print("      python run.py")
    print("    from the project root or any subfolder (src/, src/models/, tests/, docs/, etc.)\n")
    return 0


if __name__ == "__main__":
    sys.exit(setup())

