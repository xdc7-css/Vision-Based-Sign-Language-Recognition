"""Scoped Python startup hook for Vision-Based Sign Language Recognition.

Enables 'python run' and 'python run.py' to work seamlessly from ANY
arbitrary nested directory within the project, even if no local run.py exists.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _hook() -> None:
    # Check if first argument matches 'run' or 'run.py'
    if not sys.argv:
        return
    
    script_arg = os.path.basename(sys.argv[0])
    if script_arg not in ("run", "run.py"):
        return

    # Avoid recursive loops if dispatch is already in progress
    if os.environ.get("_ARSL_DISPATCH_ACTIVE") == "1":
        return

    # Locate project root by ascending from current working directory
    cwd = os.path.abspath(os.getcwd())
    cur = cwd
    project_root: str | None = None

    while True:
        has_venv = os.path.isdir(os.path.join(cur, ".venv"))
        has_src = os.path.isfile(os.path.join(cur, "src", "main.py"))
        has_karsl = os.path.isfile(os.path.join(cur, "data", "models", "karsl", "karsl_502.onnx"))

        if (has_venv and has_src) or (has_src and has_karsl):
            project_root = cur
            break

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # If the user is not inside this project directory tree, do nothing
    if not project_root:
        return

    # Locate the virtual environment python
    venv_py = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(venv_py):
        # Let the root run.py handle reporting the missing .venv
        venv_py = sys.executable

    root_run_script = os.path.join(project_root, "run.py")
    if not os.path.isfile(root_run_script):
        return

    env = os.environ.copy()
    env["_ARSL_DISPATCH_ACTIVE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [venv_py, root_run_script] + sys.argv[1:]

    try:
        res = subprocess.run(cmd, cwd=project_root, env=env)
        os._exit(res.returncode)
    except KeyboardInterrupt:
        os._exit(0)
    except Exception as exc:
        sys.stderr.write(f"\n[!] Launcher startup hook error: {exc}\n")
        os._exit(1)


try:
    _hook()
except Exception:
    pass
