import multiprocessing
import subprocess
from pathlib import Path

import pytest

from coding_assistant.sandbox import sandbox


def _run_in_sandbox(test_func, *args):
    """
    Run a test function in a separate process to avoid sandbox affecting pytest cleanup.
    Returns True if the test passed, False if it failed.
    """

    def wrapper(queue, func, *args):
        try:
            func(*args)
            queue.put(("success", None))
        except Exception as e:
            queue.put(("error", str(e)))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue, test_func, *args))
    process.start()
    process.join()

    if not queue.empty():
        result_type, message = queue.get()
        if result_type == "success":
            return True
        else:
            print(f"Child process failed: {message}")
            return False
    else:
        print("Child process failed without error message")
        return False


def _test_writable_directory_allows_write(writable_dir):
    """Test function to run in child process."""
    sandbox(readable_directories=[], writable_directories=[writable_dir])

    test_file = writable_dir / "test.txt"
    with open(test_file, "w") as f:
        f.write("Hello, world!")

    with open(test_file, "r") as f:
        assert f.read() == "Hello, world!"


def _test_readable_directory_denies_write(readable_dir, existing_file):
    """Test function to run in child process."""
    sandbox(readable_directories=[readable_dir], writable_directories=[])

    # Should be able to read existing file
    with open(existing_file, "r") as f:
        assert f.read() == "Existing content"

    # Should NOT be able to write new file
    new_file = readable_dir / "new.txt"
    try:
        with open(new_file, "w") as f:
            f.write("New content")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected

    # Should NOT be able to modify existing file
    try:
        with open(existing_file, "w") as f:
            f.write("Modified content")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected


def _test_directory_access_separation(readable_dir, writable_dir, forbidden_dir, existing_file):
    """Test function to run in child process."""
    sandbox(readable_directories=[readable_dir], writable_directories=[writable_dir])

    # Can read from readable directory
    with open(existing_file, "r") as f:
        assert f.read() == "Read-only content"

    # Cannot write to readable directory
    try:
        with open(readable_dir / "new.txt", "w") as f:
            f.write("Should fail")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected

    # Can write to writable directory
    writable_file = writable_dir / "writable.txt"
    with open(writable_file, "w") as f:
        f.write("Writable content")

    with open(writable_file, "r") as f:
        assert f.read() == "Writable content"

    # Cannot access forbidden directory
    try:
        with open(forbidden_dir / "forbidden.txt", "w") as f:
            f.write("Should fail")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected


def _test_write_denied_in_non_allowed_directory(tmp_path):
    """Test function to run in child process."""
    sandbox(readable_directories=[], writable_directories=[])

    test_file = tmp_path / "test.txt"
    try:
        with open(test_file, "w") as f:
            f.write("Hello, world!")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected


def _test_nested_directory_permissions(readable_child, writable_child, existing_file):
    """Test function to run in child process."""
    sandbox(readable_directories=[readable_child], writable_directories=[writable_child])

    # Can read from readable child
    with open(existing_file, "r") as f:
        assert f.read() == "Child content"

    # Cannot write to readable child
    try:
        with open(readable_child / "new.txt", "w") as f:
            f.write("Should fail")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected

    # Can write to writable child
    writable_file = writable_child / "new.txt"
    with open(writable_file, "w") as f:
        f.write("Child writable")

    with open(writable_file, "r") as f:
        assert f.read() == "Child writable"


def _test_multiple_readable_directories(dir1, dir2, file1, file2):
    """Test function to run in child process."""
    sandbox(readable_directories=[dir1, dir2], writable_directories=[])

    # Should be able to read from both
    with open(file1, "r") as f:
        assert f.read() == "Content 1"
    with open(file2, "r") as f:
        assert f.read() == "Content 2"

    # Should not be able to write to either
    try:
        with open(dir1 / "new1.txt", "w") as f:
            f.write("Should fail")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected

    try:
        with open(dir2 / "new2.txt", "w") as f:
            f.write("Should fail")
        raise AssertionError("Expected PermissionError but write succeeded")
    except PermissionError:
        pass  # Expected


def _test_multiple_writable_directories(dir1, dir2):
    """Test function to run in child process."""
    sandbox(readable_directories=[], writable_directories=[dir1, dir2])

    # Should be able to write to both
    file1 = dir1 / "file1.txt"
    file2 = dir2 / "file2.txt"

    with open(file1, "w") as f:
        f.write("Content 1")
    with open(file2, "w") as f:
        f.write("Content 2")

    # Should be able to read from both
    with open(file1, "r") as f:
        assert f.read() == "Content 1"
    with open(file2, "r") as f:
        assert f.read() == "Content 2"


def _test_run_binaries_with_sandbox():
    """Test function to run in child process."""
    sandbox(readable_directories=[], writable_directories=[])
    subprocess.check_call(["git", "help"])
    subprocess.check_call(["npm", "help"])
    subprocess.check_call(["uvx", "--help"])


def test_write_without_sandbox(tmp_path):
    """Test that writing works without any sandbox restrictions."""
    test_file = tmp_path / "test.txt"

    with open(test_file, "w") as f:
        f.write("Hello, world!")

    with open(test_file, "r") as f:
        assert f.read() == "Hello, world!"


def test_writable_directory_allows_write(tmp_path):
    """Test that directories marked as writable allow writing."""
    writable_dir = tmp_path / "writable"
    writable_dir.mkdir()

    assert _run_in_sandbox(_test_writable_directory_allows_write, writable_dir)


def test_readable_directory_allows_read_but_not_write(tmp_path):
    """Test that directories marked as readable allow reading but not writing."""
    readable_dir = tmp_path / "readable"
    readable_dir.mkdir()

    # Create a file before applying sandbox
    existing_file = readable_dir / "existing.txt"
    with open(existing_file, "w") as f:
        f.write("Existing content")

    assert _run_in_sandbox(_test_readable_directory_denies_write, readable_dir, existing_file)


def test_directory_access_separation(tmp_path):
    """Test that readable and writable directories are properly separated."""
    readable_dir = tmp_path / "readable"
    writable_dir = tmp_path / "writable"
    forbidden_dir = tmp_path / "forbidden"

    readable_dir.mkdir()
    writable_dir.mkdir()
    forbidden_dir.mkdir()

    # Create existing file in readable directory
    existing_file = readable_dir / "existing.txt"
    with open(existing_file, "w") as f:
        f.write("Read-only content")

    assert _run_in_sandbox(_test_directory_access_separation, readable_dir, writable_dir, forbidden_dir, existing_file)


def test_write_in_non_allowed_directory(tmp_path):
    """Test that writing fails in directories not explicitly allowed."""
    assert _run_in_sandbox(_test_write_denied_in_non_allowed_directory, tmp_path)


def test_sandbox_fails_on_nonexistent_readable_dir():
    """Test that sandbox fails when readable directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Directory .* does not exist"):
        sandbox(readable_directories=[Path("/does-not-exist")], writable_directories=[])


def test_sandbox_fails_on_nonexistent_writable_dir():
    """Test that sandbox fails when writable directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Directory .* does not exist"):
        sandbox(readable_directories=[], writable_directories=[Path("/does-not-exist")])


def test_nested_directory_permissions(tmp_path):
    """Test that nested directories inherit proper permissions."""
    parent_dir = tmp_path / "parent"
    readable_child = parent_dir / "readable_child"
    writable_child = parent_dir / "writable_child"

    parent_dir.mkdir()
    readable_child.mkdir()
    writable_child.mkdir()

    # Create existing file in readable child
    existing_file = readable_child / "existing.txt"
    with open(existing_file, "w") as f:
        f.write("Child content")

    assert _run_in_sandbox(_test_nested_directory_permissions, readable_child, writable_child, existing_file)


def test_run_binaries_with_sandbox():
    """Test that common binaries still work with sandbox applied."""
    assert _run_in_sandbox(_test_run_binaries_with_sandbox)


def test_multiple_readable_directories(tmp_path):
    """Test that multiple readable directories work correctly."""
    dir1 = tmp_path / "readable1"
    dir2 = tmp_path / "readable2"
    dir1.mkdir()
    dir2.mkdir()

    # Create files in both directories
    file1 = dir1 / "file1.txt"
    file2 = dir2 / "file2.txt"

    with open(file1, "w") as f:
        f.write("Content 1")
    with open(file2, "w") as f:
        f.write("Content 2")

    assert _run_in_sandbox(_test_multiple_readable_directories, dir1, dir2, file1, file2)


def test_multiple_writable_directories(tmp_path):
    """Test that multiple writable directories work correctly."""
    dir1 = tmp_path / "writable1"
    dir2 = tmp_path / "writable2"
    dir1.mkdir()
    dir2.mkdir()

    assert _run_in_sandbox(_test_multiple_writable_directories, dir1, dir2)
