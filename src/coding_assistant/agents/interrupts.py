import asyncio
import logging
import os
import signal
import sys
import threading
import traceback
from types import FrameType
from typing import Optional, Union, Any

logger = logging.getLogger(__name__)


class InterruptibleSection:
    def __init__(self) -> None:
        self._was_interrupted = 0
        self._original_handler: Optional[Union[signal._HANDLER, int, None]] = None

    @property
    def was_interrupted(self) -> bool:
        return self._was_interrupted > 0

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        self._was_interrupted += 1
        if self._was_interrupted > 5:
            sys.exit(1)

    def __enter__(self) -> "InterruptibleSection":
        self._original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        signal.signal(signal.SIGINT, self._original_handler)
