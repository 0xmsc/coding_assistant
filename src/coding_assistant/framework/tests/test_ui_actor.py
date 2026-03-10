import asyncio

import pytest

from coding_assistant.ui import ActorUI, UI


class RecordingUI(UI):
    def __init__(self) -> None:
        self.events: list[str] = []

    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        self.events.append(f"start:{prompt_text}")
        await asyncio.sleep(0.05)
        self.events.append(f"end:{prompt_text}")
        return prompt_text

    async def confirm(self, prompt_text: str) -> bool:
        self.events.append(f"confirm:{prompt_text}")
        return True

    async def prompt(self, words: list[str] | None = None) -> str:
        self.events.append("prompt")
        return "ok"


@pytest.mark.asyncio
async def test_actor_ui_serializes_calls() -> None:
    ui = RecordingUI()
    actor_ui = ActorUI(ui, context_name="test")
    actor_ui.start()
    try:
        await asyncio.gather(actor_ui.ask("a"), actor_ui.ask("b"))
    finally:
        await actor_ui.stop()

    assert ui.events == ["start:a", "end:a", "start:b", "end:b"]
