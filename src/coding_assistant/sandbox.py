import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from landlock import FSAccess, Ruleset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_READABLE_PATHS = [
    "/usr",
    "/lib",
    "/etc",
    "/dev/urandom",
    "/proc",
    "/run",
    "/sys",
    "/mnt/wsl",
    "/opt",
    "~/.ssh",
    "~/.rustup",
    "~/.config",
    "~/.local",
    "~/.cache",
    "~/.cargo",
    "~/.local/bin",
    "~/.cfg",
]

DEFAULT_WRITABLE_PATHS = [
    "/tmp",
    "/dev/null",
    "/dev/shm",
    "~/.npm",
    "~/.cache/uv",
    "~/.local/share/uv",
    "~/.cache/coding_assistant",
    # prompt_toolkit allows using nvim to edit the prompt.
    "~/.cache/nvim",
    "~/.local/state/nvim",
]


def _get_read_only_rule() -> FSAccess:
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


def _get_read_write_file_rule() -> FSAccess:
    return FSAccess.WRITE_FILE | FSAccess.READ_FILE


def _get_read_only_file_rule() -> FSAccess:
    return FSAccess.READ_FILE


def _to_paths(items: Sequence[str | Path]) -> list[Path]:
    return [Path(entry).expanduser().resolve() for entry in items]


def allow_read(rs: Ruleset, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue

        if path.is_dir():
            rs.allow(path, rules=_get_read_only_rule())
        else:
            rs.allow(path, rules=_get_read_only_file_rule())


def allow_write(rs: Ruleset, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue

        if path.is_dir():
            rs.allow(path, rules=FSAccess.all())
        else:
            rs.allow(path, rules=_get_read_write_file_rule())


def sandbox(readable_paths: list[Path], writable_paths: list[Path], include_defaults: bool = False) -> None:
    rs = Ruleset()

    if include_defaults:
        writable_paths.extend(_to_paths(DEFAULT_WRITABLE_PATHS))
        readable_paths.extend(_to_paths(DEFAULT_READABLE_PATHS))

    writable_paths_list = list(set(_to_paths(writable_paths)))
    readable_paths_list = list(set(_to_paths(readable_paths)) - set(writable_paths_list))

    logger.info(f"Writable sandbox directories: {writable_paths_list}")
    logger.info(f"Readable sandbox directories: {readable_paths_list}")

    allow_write(rs, writable_paths_list)
    allow_read(rs, readable_paths_list)

    rs.apply()


def main() -> None:
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

    readable_dirs = [Path(d).resolve() for d in args.readable_directories]
    writable_dirs = [Path(d).resolve() for d in args.writable_directories]

    sandbox(readable_paths=readable_dirs, writable_paths=writable_dirs)

    result = subprocess.run(args.command, capture_output=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
