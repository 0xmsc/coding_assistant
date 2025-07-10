import logging
from pathlib import Path
from landlock import Ruleset, FSAccess

logger = logging.getLogger(__name__)


def _get_read_only_rule():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


import os

def sandbox(working_directory: Path):
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
    rs.allow(Path("~/.cache").expanduser(), rules=FSAccess.all())
    rs.allow(Path("~/.local/share/uv").expanduser(), rules=FSAccess.all())

    rs.allow(Path("~/.cargo").expanduser(), rules=_get_read_only_rule())
    rs.allow(Path("~/.local/bin").expanduser(), rules=_get_read_only_rule())
    rs.allow(Path("~/.config").expanduser(), rules=_get_read_only_rule())
    rs.allow(Path("~/.cfg").expanduser(), rules=_get_read_only_rule())

    # Allow the project Python virtual environment directory recursively (read/execute)
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path is not None:
        rs.allow(Path(venv_path), rules=_get_read_only_rule())
    else:
        # fallback: check for standard .venv directory in project root
        if Path('.venv').exists():
            rs.allow(Path('.venv'), rules=_get_read_only_rule())

    # And finally, the working directory
    rs.allow(working_directory, rules=FSAccess.all())

    rs.apply()

