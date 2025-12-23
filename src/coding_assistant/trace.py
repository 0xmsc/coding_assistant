import json5
from datetime import datetime
from pathlib import Path
import logging
from typing import Any

from coding_assistant.paths import get_app_cache_dir

_trace_dir: Path | None = None
_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger(__name__)


def get_default_trace_dir() -> Path:
    return get_app_cache_dir() / "traces" / _session_id


def enable_tracing(directory: Path, clear: bool = False) -> None:
    global _trace_dir

    _trace_dir = directory
    _trace_dir.mkdir(parents=True, exist_ok=True)

    if clear:
        for f in _trace_dir.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)

    logger.info(f"Tracing to {_trace_dir}")


def trace_enabled() -> bool:
    return _trace_dir is not None


def _get_trace_path(name: str) -> Path:
    if _trace_dir is None:
        raise RuntimeError("Tracing is not enabled")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = _trace_dir / f"{timestamp}_{name}"
    return path


def trace_data(name: str, content: str) -> None:
    if _trace_dir is None:
        return

    trace_file = _get_trace_path(name)
    with open(trace_file, "w") as f:
        f.write(content)


def trace_json(name: str, data: Any) -> None:
    if _trace_dir is None:
        return

    # Ensure the name ends with .json5
    if name.endswith(".json"):
        name = name.removesuffix(".json") + ".json5"

    trace_file = _get_trace_path(name)

    if not name.endswith(".json5"):
        raise ValueError("trace_json only supports .json or .json5 extension")

    # Serialize to JSON5 string
    content = json5.dumps(data, indent=2)

    # Enhancing readability of multi-line strings by escaping literal newlines
    # as per JSON5 specification (backslash at end of line).
    content = content.replace("\\n", "\\\n")

    with open(trace_file, "w") as f:
        f.write(content)
