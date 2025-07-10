import pytest

from coding_assistant.config import Config
from coding_assistant.sandbox import sandbox


def test_write_without_sandbox():
    # Try to create a file in tmp, should work
    with open("/tmp/test_sandbox.txt", "w") as f:
        f.write("Hello, world!")

    with open("/tmp/test_sandbox.txt", "r") as f:
        assert f.read() == "Hello, world!"


def test_write_with_sandbox_in_tmp():
    # Sandbox such that we can write to /tmp
    sandbox("/tmp")

    with open("/tmp/test_sandbox.txt", "w") as f:
        f.write("Hello, world!")

    with open("/tmp/test_sandbox.txt", "r") as f:
        assert f.read() == "Hello, world!"


def test_write_with_sandbox_in_home():
    # Sandbox such that we cannot write to /tmp
    sandbox("/home")

    with pytest.raises(PermissionError):
        with open("/tmp/test_sandbox.txt", "w") as f:
            f.write("Hello, world!")
