import argparse
import logging
import subprocess
import sys
from pathlib import Path

from landlock import FSAccess, Ruleset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_READABLE_DIRECTORIES = [
    "/mnt/wsl",
    "~/.ssh",
    "~/.rustup",
    "~/.config",
    "~/.local",
    "~/.cache",
    "~/.cargo",
    "~/.local/bin",
    "~/.cfg",
]

DEFAULT_WRITABLE_DIRECTORIES = [
    "~/.npm",
    "~/.cache/uv",
    "~/.local/share/uv",
    "/tmp",
    "/dev/shm",
    "~/.cache/coding_assistant",
    "~/.cache/nvim",
    "~/.local/state/nvim",
]


def _get_read_only_rule():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


def _get_read_write_file_rule():
    return FSAccess.WRITE_FILE | FSAccess.READ_FILE


def _get_read_only_file_rule():
    return FSAccess.READ_FILE


def sandbox(readable_directories: list[Path], writable_directories: list[Path]):
    rs = Ruleset()

    # System directories
    rs.allow(Path("/dev/null"), rules=_get_read_write_file_rule())
    rs.allow(Path("/dev/urandom"), rules=_get_read_only_file_rule())
    rs.allow(Path("/usr"), rules=_get_read_only_rule())
    rs.allow(Path("/lib"), rules=_get_read_only_rule())
    rs.allow(Path("/etc"), rules=_get_read_only_rule())
    rs.allow(Path("/proc"), rules=_get_read_only_rule())
    rs.allow(Path("/run"), rules=_get_read_only_rule())
    rs.allow(Path("/sys"), rules=_get_read_only_rule())

    # Standard directories
    for path in DEFAULT_READABLE_DIRECTORIES:
        p = Path(path).expanduser()
        if p.exists():
            rs.allow(p, rules=_get_read_only_rule())

    for path in DEFAULT_WRITABLE_DIRECTORIES:
        p = Path(path).expanduser()
        if p.exists():
            rs.allow(p, rules=FSAccess.all())

    # Allow each directory passed in the directories list
    for directory in readable_directories:
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        rs.allow(directory, rules=_get_read_only_rule())

    for directory in writable_directories:
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        rs.allow(directory, rules=FSAccess.all())

    rs.apply()


def main():
    """Main function for CLI usage of sandbox."""
    parser = argparse.ArgumentParser(
        description="Run a command in a sandboxed environment with restricted filesystem access"
    )
    parser.add_argument(
        "--readable-directories",
        type=str,
        nargs="*",
        default=[],
        help="Directories that should be readable (space-separated paths)",
    )
    parser.add_argument(
        "--writable-directories",
        type=str,
        nargs="*",
        default=[],
        help="Directories that should be writable (space-separated paths)",
    )
    parser.add_argument(
        "command",
        nargs="+",
        help="Command and arguments to execute in the sandbox",
    )

    args = parser.parse_args()

    # Convert string paths to Path objects
    readable_dirs = [Path(d).resolve() for d in args.readable_directories]
    writable_dirs = [Path(d).resolve() for d in args.writable_directories]

    # Apply sandbox
    sandbox(readable_directories=readable_dirs, writable_directories=writable_dirs)

    # Execute the command
    result = subprocess.run(args.command, capture_output=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
