import base64
from pathlib import Path

import pytest

from coding_assistant.framework.tests.test_agents import create_test_config
from coding_assistant.framework.tests.helpers import system_actor_scope_for_tests
from coding_assistant.llm.types import UserMessage
from coding_assistant.tools.tools import AgentTool
from coding_assistant.ui import NullUI


@pytest.mark.slow
@pytest.mark.asyncio
async def test_model_vision_recognizes_car_image() -> None:
    # NOTE: Download picture via `wget --output-document car.jpg https://upload.wikimedia.org/wikipedia/commons/0/01/SEAT_Leon_Mk4_IMG_4099.jpg`

    image_path = Path(__file__).with_name("car.jpg")
    image_bytes = image_path.read_bytes()

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    history = [UserMessage(content=[{"type": "image_url", "image_url": {"url": data_url}}])]

    config = create_test_config()
    ui = NullUI()
    async with system_actor_scope_for_tests(tools=[], ui=ui, context_name="test") as actors:
        tool = AgentTool(
            model=config.model,
            expert_model=config.expert_model,
            compact_conversation_at_tokens=config.compact_conversation_at_tokens,
            enable_ask_user=config.enable_ask_user,
            tools=[],
            history=history,
            ui=ui,
            actor_directory=actors.actor_directory,
            agent_actor_uri=actors.agent_actor_uri,
            tool_call_actor_uri=actors.tool_call_actor_uri,
            user_actor_uri=actors.user_actor_uri,
        )
        result = await tool.execute(
            parameters={
                "task": "What is the primary object in this image? Answer with exactly one lower-case word from this set: car, bicycle, motorcycle, bus, truck, person, dog, cat, building, tree, unknown.",
            }
        )
    assert result.content == "car"
