import base64
from pathlib import Path

import pytest

from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.framework.tests.test_agents import create_test_config
from coding_assistant.tools.tools import AgentTool
from coding_assistant.ui import NullUI

TEST_MODEL = "openrouter/openai/gpt-5-mini"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_model_vision_recognizes_car_image():
    # NOTE: Download picture via `wget --output-document car.jpg https://upload.wikimedia.org/wikipedia/commons/0/01/SEAT_Leon_Mk4_IMG_4099.jpg`

    image_path = Path(__file__).with_name("car.jpg")
    image_bytes = image_path.read_bytes()

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    history = []
    history.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]})

    config = create_test_config()
    tool = AgentTool(
        model=config.model,
        expert_model=config.expert_model,
        compact_conversation_at_tokens=config.compact_conversation_at_tokens,
        enable_ask_user=config.enable_ask_user,
        tools=[],
        history=history,
        progress_callbacks=NullProgressCallbacks(),
        ui=NullUI(),
        tool_callbacks=NullToolCallbacks(),
    )
    result = await tool.execute(
        parameters={
            "task": "What is the primary object in this image? Answer with exactly one lower-case word from this set: car, bicycle, motorcycle, bus, truck, person, dog, cat, building, tree, unknown.",
        }
    )
    assert result.content == "car"
