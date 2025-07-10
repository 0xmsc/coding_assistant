import subprocess
import pytest
from pathlib import Path

from coding_assistant.config import Config
from coding_assistant.sandbox import sandbox


def test_write_without_sandbox(tmp_path):
    test_file = tmp_path / "test.txt"

    with open(test_file, "w") as f:
        f.write("Hello, world!")

    with open(test_file, "r") as f:
        assert f.read() == "Hello, world!"


def test_write_with_sandbox_in_tmp(tmp_path):
    sandbox([tmp_path])

    test_file = tmp_path / "test.txt"

    with open(test_file, "w") as f:
        f.write("Hello, world!")

    with open(test_file, "r") as f:
        assert f.read() == "Hello, world!"


def test_write_in_non_allowed_directory(tmp_path):
    sandbox([])

    test_file = tmp_path / "test.txt"

    with pytest.raises(PermissionError):
        with open(test_file, "w") as f:
            f.write("Hello, world!")


def test_sandbox_fails_on_nonexistent_dir():
    with pytest.raises(FileNotFoundError):
        sandbox([Path("/does-not-exist")])


def test_run_binaries_with_sandbox():
    sandbox([])
    subprocess.check_call(["git", "help"])
    subprocess.check_call(["npm", "help"])
    subprocess.check_call(["uvx", "--help"])
