import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

def trace_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return bool(os.getenv("CODING_ASSISTANT_TRACE"))

def trace_data(name: str, data: Any) -> None:
    """
    Write data to a trace file if tracing is enabled.
    
    Args:
        name: The base name for the trace file.
        data: The data to trace (will be converted to JSON if it's a dict/list, or written as string).
    """
    if not trace_enabled():
        return

    trace_path = Path("/tmp/coding_assistant_trace")
    trace_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Handle both JSON-serializable data and raw strings
    if isinstance(data, (dict, list)):
        trace_file = trace_path / f"{timestamp}_{name}.json"
        with open(trace_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        trace_file = trace_path / f"{timestamp}_{name}.txt"
        with open(trace_file, "w") as f:
            f.write(str(data))
