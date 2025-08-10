import os
import signal
import pytest
from coding_assistant.agents.interrupts import InterruptibleSection


def test_interruptible_section_catches_sigint():
    with InterruptibleSection() as interruptible_section:
        os.kill(os.getpid(), signal.SIGINT)
    assert interruptible_section.was_interrupted


def test_interruptible_section_exits_after_too_many_sigints():
    with pytest.raises(SystemExit):
        with InterruptibleSection():
            for _ in range(6):
                os.kill(os.getpid(), signal.SIGINT)
