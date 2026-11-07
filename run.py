#!/usr/bin/env python3
"""Global dual-mode interactive launcher for ArSL Studio.

Enables one-command launching from the project root or any subfolder:
    python run
    python run.py

Presents an interactive choice:
    [1] Python Desktop — PySide6
    [2] Local Web — React + FastAPI
    [0] Exit
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


def find_project_root() -> str | None:
    """Locate the project root directory by searching upward for project markers."""
    start_dirs: list[str] = []

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        start_dirs.append(script_dir)
    except NameError:
        pass

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
            has_web = os.path.isdir(os.path.join(cur, "web"))
            has_backend = os.path.isdir(os.path.join(cur, "backend"))

            if (has_venv and has_src) or (has_src and has_backend) or (has_src and has_web):
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
    return sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a local TCP port is already open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def wait_for_backend(url: str = "http://127.0.0.1:8000/api/health", timeout_secs: float = 15.0) -> bool:
    """Wait until backend HTTP server responds with status 200."""
    t0 = time.time()
    while time.time() - t0 < timeout_secs:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def run_desktop(project_root: str) -> int:
    """Launch the original PySide6 desktop application."""
    print("\n[*] Initializing ArSL Studio Desktop (PySide6)...")
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from src.main import main as app_main
        app_main()
        return 0
    except KeyboardInterrupt:
        print("\n[*] Desktop application closed by user.")
        return 0
    except Exception as exc:
        print(f"\n[!] Error launching desktop application: {exc}", file=sys.stderr)
        return 1


def run_web(project_root: str, port: int = 8000) -> int:
    """Launch the Local Web application (FastAPI + React)."""
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("\n" + "=" * 50)
    print(" ArSL STUDIO — Starting Local Web Application")
    print("=" * 50)

    # 1. Verify/build web frontend bundle
    dist_index = os.path.join(project_root, "web", "dist", "index.html")
    if not os.path.isfile(dist_index):
        print("[*] Building production React frontend bundle...")
        build_res = subprocess.run(
            ["npm", "run", "build"],
            cwd=os.path.join(project_root, "web"),
            shell=True,
        )
        if build_res.returncode != 0:
            print("[!] Warning: React build failed. You can run 'npm run dev' inside web/ for live development.")

    # 2. Check port conflict
    if is_port_in_use(port):
        print(f"[!] Warning: Port {port} is already in use by another application.")
        print(f"    Please stop the conflicting process or choose another port.")
        return 1

    local_url = f"http://127.0.0.1:{port}"
    print(f"[*] Starting local FastAPI server at {local_url}...")
    print(f"[*] Access URL: {local_url}")
    print("[*] Press Ctrl+C in this terminal window to stop the server.\n")

    # Launch browser in a background timer
    def open_browser():
        time.sleep(1.2)
        if wait_for_backend(f"{local_url}/api/health", timeout_secs=10.0):
            print(f"[*] Opening browser to {local_url}...")
            webbrowser.open(local_url)

    import threading
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # Run Uvicorn directly
    try:
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=port, log_level="info")
        return 0
    except KeyboardInterrupt:
        print("\n[*] Web server shutdown requested by user (Ctrl+C). Exiting cleanly.")
        return 0
    except Exception as exc:
        print(f"\n[!] Web server runtime error: {exc}", file=sys.stderr)
        return 1


def display_menu() -> str:
    """Display the clean interactive CLI menu requested by user."""
    banner = """
========================================
 ArSL STUDIO
 Arabic Sign Language Recognition
========================================

 Select Application Mode:

 [1] Python Desktop — PySide6
 [2] Local Web — React + FastAPI
 [0] Exit
"""
    print(banner)
    try:
        choice = input(" Enter your choice: ").strip()
        return choice
    except (KeyboardInterrupt, EOFError):
        print("\n[*] Exiting cleanly.")
        return "0"


def main() -> int:
    """Global launcher entry point with automatic root discovery and venv dispatch."""
    project_root = find_project_root()
    if not project_root:
        print("[!] ERROR: Could not locate project root directory.", file=sys.stderr)
        return 1

    venv_python = get_venv_python(project_root)

    # If running outside the isolated virtualenv, dispatch directly into it
    current_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(os.path.abspath(venv_python))

    if current_exe != target_exe and os.path.isfile(venv_python) and os.environ.get("_ARSL_DISPATCH_ACTIVE") != "1":
        env = os.environ.copy()
        env["_ARSL_DISPATCH_ACTIVE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [venv_python, os.path.join(project_root, "run.py")] + sys.argv[1:]
        try:
            res = subprocess.run(cmd, cwd=project_root, env=env)
            return res.returncode
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print(f"[!] Error dispatching to virtual environment: {e}", file=sys.stderr)
            return 1

    # In case user provided CLI flags: e.g. python run --desktop or python run --web
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--desktop", "-d", "1"):
            return run_desktop(project_root)
        elif arg in ("--web", "-w", "2"):
            return run_web(project_root)
        elif arg in ("--exit", "-e", "0"):
            print("\n[*] Goodbye!\n")
            return 0
        elif arg in ("--help", "-h"):
            print("Usage: python run [1|--desktop] [2|--web] [0|--exit]")
            return 0

    # Interactive Menu Loop
    while True:
        choice = display_menu()

        if choice == "1":
            return run_desktop(project_root)
        elif choice == "2":
            return run_web(project_root)
        elif choice == "0":
            print("\n[*] Goodbye!\n")
            return 0
        else:
            print("\n[!] Invalid selection. Please enter 1, 2, or 0.")
            time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
