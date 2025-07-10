import logging
from pathlib import Path
from landlock import Ruleset, FSAccess
from typing import List

logger = logging.getLogger(__name__)


def _get_read_only_rule():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


def sandbox(directories: list[Path]):
    rs = Ruleset()

    # System directories
    rs.allow(Path("/dev"), rules=FSAccess.all())
    rs.allow(Path("/usr"), rules=_get_read_only_rule())
    rs.allow(Path("/lib"), rules=_get_read_only_rule())
    rs.allow(Path("/etc"), rules=_get_read_only_rule())
    rs.allow(Path("/proc"), rules=_get_read_only_rule())
    rs.allow(Path("/run"), rules=_get_read_only_rule())
    rs.allow(Path("/sys"), rules=_get_read_only_rule())
    rs.allow(Path("/mnt/wsl"), rules=_get_read_only_rule())

    # User directories
    rs.allow(Path("~/.npm").expanduser(), rules=FSAccess.all())
    rs.allow(Path("~/.cache/uv").expanduser(), rules=FSAccess.all())
    rs.allow(Path("~/.local/share/uv").expanduser(), rules=FSAccess.all())

    rs.allow(Path("~/.cargo").expanduser(), rules=_get_read_only_rule())
    rs.allow(Path("~/.local/bin").expanduser(), rules=_get_read_only_rule())
    rs.allow(Path("~/.config").expanduser(), rules=_get_read_only_rule())
    rs.allow(Path("~/.cfg").expanduser(), rules=_get_read_only_rule())

    # Allow each directory passed in the directories list
    for directory in directories:
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")

        rs.allow(directory, rules=FSAccess.all())

    rs.apply()
