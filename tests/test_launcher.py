"""Automated tests for the ArSL Studio dual-mode interactive launcher."""

import os
import sys
import pytest

from run import find_project_root, get_venv_python, is_port_in_use, main


def test_find_project_root():
    root = find_project_root()
    assert root is not None
    assert os.path.isdir(os.path.join(root, "src"))
    assert os.path.isfile(os.path.join(root, "run.py"))


def test_get_venv_python():
    root = find_project_root()
    venv_py = get_venv_python(root)
    assert os.path.isfile(venv_py)


def test_is_port_in_use():
    # An improbable port like 59876 should not be in use
    assert is_port_in_use(59876) is False


def test_launcher_cli_exit(monkeypatch):
    # Test exiting cleanly when passed '0'
    monkeypatch.setattr(sys, "argv", ["run.py", "0"])
    code = main()
    assert code == 0


def test_launcher_cli_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--help"])
    code = main()
    assert code == 0

