import logging
from pathlib import Path
from landlock import Ruleset, FSAccess

logger = logging.getLogger(__name__)


def _get_read_only_rule():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


def sandbox(working_directory: Path):
    logger.info(f"Sandboxing to working directory: {working_directory}")

    rs = Ruleset()

    rs.allow(Path("/usr"), rules=FSAccess.all())
    rs.allow(Path("/lib"), rules=FSAccess.all())
    rs.allow(Path("/etc"), rules=FSAccess.all())
    rs.allow(Path("/dev"), rules=FSAccess.all())
    rs.allow(Path("/proc"), rules=FSAccess.all())
    rs.allow(Path("/run"), rules=FSAccess.all())
    rs.allow(Path("/sys"), rules=FSAccess.all())
    rs.allow(Path("/mnt/wsl"), rules=FSAccess.all())

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
