import subprocess
import pytest
from pathlib import Path

from coding_assistant.config import Config
from coding_assistant.sandbox import sandbox


def test_write_without_sandbox():
    # Try to create a file in tmp, should work
    with open("/tmp/test_sandbox.txt", "w") as f:
        f.write("Hello, world!")

    with open("/tmp/test_sandbox.txt", "r") as f:
        assert f.read() == "Hello, world!"


def test_write_with_sandbox_in_tmp():
    sandbox([Path("/tmp")])

    with open("/tmp/test_sandbox.txt", "w") as f:
        f.write("Hello, world!")

    with open("/tmp/test_sandbox.txt", "r") as f:
        assert f.read() == "Hello, world!"


def test_write_with_sandbox_in_home():
    sandbox([])

    with pytest.raises(PermissionError):
        with open("/tmp/test_sandbox.txt", "w") as f:
            f.write("Hello, world!")


def test_sandbox_fails_on_nonexistent_dir():
    with pytest.raises(FileNotFoundError):
        sandbox([Path("/does-not-exist")])


def test_run_binaries_with_sandbox(tmp_path):
    sandbox([Path("/tmp")])
    subprocess.check_call(["git", "help"], cwd="/tmp")
    subprocess.check_call(["npm", "help"], cwd="/tmp")
    subprocess.check_call(["uvx", "--help"], cwd="/tmp")
