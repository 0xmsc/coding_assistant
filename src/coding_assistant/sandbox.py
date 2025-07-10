from pathlib import Path
from landlock import Ruleset, FSAccess


def _get_read_only_fsaccess():
    return FSAccess.EXECUTE | FSAccess.READ_DIR | FSAccess.READ_FILE


def sandbox(working_directory: Path):
    rs = Ruleset()
    rs.allow("/", rules=_get_read_only_fsaccess())
    rs.allow(working_directory, rules=FSAccess.all())
    rs.apply()
