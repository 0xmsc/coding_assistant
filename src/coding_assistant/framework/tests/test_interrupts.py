import asyncio
import os
import signal

import pytest

from coding_assistant.framework.interrupts import (
    InterruptController,
    ToolCallCancellationManager,
)


def test_interrupt_controller_catches_sigint():
    loop = asyncio.new_event_loop()
    try:
        with InterruptController(loop) as controller:
            os.kill(os.getpid(), signal.SIGINT)
        assert controller.was_interrupted
    finally:
        loop.close()


def test_interrupt_controller_handles_multiple_sigints():
    """Test that multiple SIGINTs are handled without exiting."""
    loop = asyncio.new_event_loop()
    try:
        with InterruptController(loop) as controller:
            for _ in range(6):
                os.kill(os.getpid(), signal.SIGINT)
        # Should have tracked all interrupts
        assert controller.was_interrupted
        assert controller._was_interrupted == 6
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_tool_call_cancellation_manager_cancel_all():
    loop = asyncio.get_running_loop()
    manager = ToolCallCancellationManager()

    async def wait_forever():
        await asyncio.Future()

    task = loop.create_task(wait_forever(), name="tool-task")
    manager.register_task(task)

    manager.cancel_all()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    # The done callback should have removed it from the set
    assert len(manager._tasks) == 0


@pytest.mark.asyncio
async def test_interrupt_controller_cancels_tasks():
    loop = asyncio.get_running_loop()
    controller = InterruptController(loop)

    async def wait_forever():
        await asyncio.Future()

    task = loop.create_task(wait_forever())
    controller.register_task("call-1", task)

    controller.request_interrupt()
    await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert controller.has_pending_interrupt
