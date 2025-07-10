import logging
from pathlib import Path
from landlock import Ruleset, FSAccess

logger = logging.getLogger(__name__)


def _get_read_only_rule():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


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

    # And finally, the working directory
    rs.allow(working_directory, rules=FSAccess.all())

    rs.apply()
