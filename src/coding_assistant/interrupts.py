import signal
from asyncio import AbstractEventLoop, Task
from types import FrameType
from typing import Any, Callable


class ToolCallCancellationManager:
    """Tracks tool-call tasks so they can be cancelled on user interrupts."""

    def __init__(self) -> None:
        self._tasks: set[Task[Any]] = set()

    def register_task(self, task: Task[Any]) -> None:
        """Track a task and remove it automatically once it finishes."""
        self._tasks.add(task)
        task.add_done_callback(lambda finished_task: self._tasks.discard(finished_task))

    def cancel_all(self) -> None:
        """Cancel every currently tracked task."""
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
        """Translate SIGINT into the normal interrupt flow."""
        self.request_interrupt()

    def __enter__(self) -> "InterruptController":
        """Install the interrupt handler for the duration of the context."""
        self._original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore the previous SIGINT handler when the context exits."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)

    @property
    def was_interrupted(self) -> bool:
        """Return whether any interrupt has been requested so far."""
        return self._was_interrupted > 0

    def register_task(self, call_id: str, task: Task[Any]) -> None:
        """Track a tool task so it can be cancelled on interrupt."""
        self._cancellation_manager.register_task(task)

    def request_interrupt(self) -> None:
        """Record an interrupt and schedule cancellation on the event loop."""
        self._was_interrupted += 1
        self._loop.call_soon_threadsafe(self._handle_interrupt)

    def _handle_interrupt(self) -> None:
        """Cancel all tracked tasks after an interrupt request."""
        self._cancellation_manager.cancel_all()

    @property
    def has_pending_interrupt(self) -> bool:
        """Return whether an interrupt has been recorded."""
        return self._was_interrupted > 0
