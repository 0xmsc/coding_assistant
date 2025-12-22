import signal
from asyncio import AbstractEventLoop, Task
from types import FrameType
from typing import Any, Callable


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
        self._was_interrupted = 0
        self._original_handler: Callable[[int, FrameType | None], Any] | int | None = None

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT signals."""
        self.request_interrupt()

    def __enter__(self) -> "InterruptController":
        """Set up SIGINT handler."""
        self._original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore original SIGINT handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)

    @property
    def was_interrupted(self) -> bool:
        return self._was_interrupted > 0

    def register_task(
        self,
        call_id: str,
        task: Task[Any],
    ) -> None:
        self._cancellation_manager.register_task(task)

    def request_interrupt(self) -> None:
        self._was_interrupted += 1
        self._loop.call_soon_threadsafe(self._handle_interrupt)

    def _handle_interrupt(self) -> None:
        self._cancellation_manager.cancel_all()

    @property
    def has_pending_interrupt(self) -> bool:
        return self._was_interrupted > 0
