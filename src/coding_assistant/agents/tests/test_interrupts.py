import asyncio
import os
import signal

import pytest

from coding_assistant.agents.interrupts import (
    InterruptController,
    InterruptReason,
    InterruptibleSection,
    ToolCallCancellationManager,
)


def test_interruptible_section_catches_sigint():
    with InterruptibleSection() as interruptible_section:
        os.kill(os.getpid(), signal.SIGINT)
    assert interruptible_section.was_interrupted


def test_interruptible_section_exits_after_too_many_sigints():
    with pytest.raises(SystemExit):
        with InterruptibleSection():
            for _ in range(6):
                os.kill(os.getpid(), signal.SIGINT)


@pytest.mark.asyncio
async def test_tool_call_cancellation_manager_cancel_all():
    loop = asyncio.get_running_loop()
    manager = ToolCallCancellationManager(loop)

    async def wait_forever():
        await asyncio.Future()

    task = manager.create_task(wait_forever(), name="tool-task")

    manager.cancel_all()
    await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert len(manager._tasks) == 0


@pytest.mark.asyncio
async def test_interrupt_controller_cancels_tasks_and_runs_cleanup():
    loop = asyncio.get_running_loop()
    controller = InterruptController(loop)
    cleanup_called = asyncio.Event()

    async def wait_forever():
        await asyncio.Future()

    async def cleanup():
        cleanup_called.set()

    task = controller.create_task("call-1", wait_forever(), cleanup=cleanup)

    controller.request_interrupt()
    await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    await asyncio.wait_for(cleanup_called.wait(), timeout=1)
    assert controller.has_pending_interrupt
    assert controller.consume_interrupts() == [InterruptReason.USER_INTERRUPT]
    assert not controller.has_pending_interrupt
