from pathlib import Path

import pytest

from coding_assistant.manager.db import exclusive_database_owner


def test_database_owner_lock_rejects_a_second_manager(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"

    with exclusive_database_owner(database_path):
        with pytest.raises(RuntimeError, match="Another manager already owns"):
            with exclusive_database_owner(database_path):
                pytest.fail("A second manager acquired the same database lock.")

    with exclusive_database_owner(database_path):
        pass
