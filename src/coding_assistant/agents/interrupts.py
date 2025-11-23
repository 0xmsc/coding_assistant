import logging
import signal
from asyncio import AbstractEventLoop, Task
from enum import Enum
from types import FrameType
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class InterruptReason(str, Enum):
    USER_INTERRUPT = "user_interrupt"


class ToolCallCancellationManager:
    """Tracks tool-call tasks so they can be cancelled on user interrupts."""

    def __init__(self) -> None:
        self._tasks: set[Task[Any]] = set()

    def register_task(self, task: Task[Any]) -> None:
        self._tasks.add(task)
        task.add_done_callback(lambda finished_task: self._tasks.discard(finished_task))

    def cancel_all(self) -> None:
        for task in list(self._tasks):
            task.cancel()


class InterruptController:
    """Coordinates user interrupts, signal handling, and tool-task cancellation/cleanup."""

    def __init__(self, loop: AbstractEventLoop) -> None:
        self._loop = loop
        self._cancellation_manager = ToolCallCancellationManager()
        self._pending_cleanup: dict[str, Callable[[], Awaitable[None]] | None] = {}
        self._interrupt_reasons: list[InterruptReason] = []
        self._was_interrupted = 0
        self._original_handler: signal.Handlers | None = None

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT signals."""
        self._was_interrupted += 1
        self.request_interrupt()

    def __enter__(self) -> "InterruptController":
        """Set up SIGINT handler."""
        self._original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original SIGINT handler."""
        signal.signal(signal.SIGINT, self._original_handler)

    @property
    def was_interrupted(self) -> bool:
        return self._was_interrupted > 0

    def register_task(
        self,
        call_id: str,
        task: Task[Any],
        *,
        cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._cancellation_manager.register_task(task)
        self._pending_cleanup[call_id] = cleanup
        task.add_done_callback(lambda _: self._pending_cleanup.pop(call_id, None))

    def request_interrupt(self, reason: InterruptReason = InterruptReason.USER_INTERRUPT) -> None:
        self._loop.call_soon_threadsafe(self._handle_interrupt, reason)

    def _handle_interrupt(self, reason: InterruptReason) -> None:
        self._interrupt_reasons.append(reason)
        self._cancellation_manager.cancel_all()
        pending = list(self._pending_cleanup.values())
        self._pending_cleanup.clear()
        for cleanup in pending:
            if cleanup is not None:
                self._loop.create_task(self._run_cleanup(cleanup))

    async def _run_cleanup(self, cleanup: Callable[[], Awaitable[None]]) -> None:
        try:
            await cleanup()
        except Exception:  # pragma: no cover - defensive cleanup logging
            logger.exception("Error while running tool cleanup after interrupt")

    @property
    def has_pending_interrupt(self) -> bool:
        return bool(self._interrupt_reasons)

    def consume_interrupts(self) -> list[InterruptReason]:
        reasons = self._interrupt_reasons.copy()
        self._interrupt_reasons.clear()
        return reasons
