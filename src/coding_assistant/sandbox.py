import argparse
import logging
import subprocess
import sys
from pathlib import Path

from landlock import FSAccess, Ruleset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_READABLE_DIRECTORIES = [
    "/usr",
    "/lib",
    "/etc",
    "/dev/urandom",
    "/proc",
    "/run",
    "/sys",
    "/mnt/wsl",
    # To commit.
    "~/.ssh",
    "~/.rustup",
    "~/.config",
    "~/.local",
    "~/.cache",
    # To run uv.
    "~/.cargo",
    "~/.local/bin",
    "~/.cfg",
]

DEFAULT_WRITABLE_DIRECTORIES = [
    "/tmp",
    "/dev/null",
    "/dev/shm",
    # To install MCP servers.
    "~/.npm",
    "~/.cache/uv",
    "~/.local/share/uv",
    # Trace files.
    "~/.cache/coding_assistant",
    # prompt_toolkit allows using nvim to edit the prompt.
    "~/.cache/nvim",
    "~/.local/state/nvim",
]


def _get_read_only_rule():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


def _get_read_write_file_rule():
    return FSAccess.WRITE_FILE | FSAccess.READ_FILE


def _get_read_only_file_rule():
    return FSAccess.READ_FILE


def _to_paths(list):
    return [Path(entry).expanduser().resolve() for entry in list]


def allow_read(rs: Ruleset, paths: list[Path]):
    for path in paths:
        if not path.exists():
            continue

        if path.is_dir():
            rs.allow(path, rules=_get_read_only_rule())
        else:
            rs.allow(path, rules=_get_read_only_file_rule())


def allow_write(rs: Ruleset, paths: list[Path]):
    for path in paths:
        if not path.exists():
            continue

        if path.is_dir():
            rs.allow(path, rules=FSAccess.all())
        else:
            rs.allow(path, rules=_get_read_write_file_rule())


def sandbox(readable_directories: list[Path], writable_directories: list[Path], include_defaults: bool = True):
    rs = Ruleset()

    for d in readable_directories:
        if not d.exists():
            raise FileNotFoundError(f"Directory {d} does not exist.")
    for d in writable_directories:
        if not d.exists():
            raise FileNotFoundError(f"Directory {d} does not exist.")

    writable_directories = _to_paths(writable_directories)
    if include_defaults:
        writable_directories.extend(_to_paths(DEFAULT_WRITABLE_DIRECTORIES))
    writable_directories = list(set(writable_directories))

    readable_directories = _to_paths(readable_directories)
    if include_defaults:
        readable_directories.extend(_to_paths(DEFAULT_READABLE_DIRECTORIES))
    readable_directories = list(set(readable_directories) - set(writable_directories))

    logger.info(f"Writable sandbox directories: {writable_directories}")
    logger.info(f"Readable sandbox directories: {readable_directories}")

    allow_write(rs, writable_directories)
    allow_read(rs, readable_directories)

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
