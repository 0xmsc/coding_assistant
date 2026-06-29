from __future__ import annotations

import logging
from pathlib import Path

import pytest

from coding_assistant.infra.logging import setup_logging


def test_setup_logging_writes_to_stderr_and_session_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        setup_logging(console=True)

        logging.getLogger("coding_assistant.tests").error("visible manager error")
        for handler in root_logger.handlers:
            handler.flush()

        captured = capsys.readouterr()
        log_files = list((tmp_path / "coding_assistant" / "sessions").glob("*/session.log"))

        assert "visible manager error" in captured.err
        assert len(log_files) == 1
        assert "visible manager error" in log_files[0].read_text(encoding="utf-8")
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
