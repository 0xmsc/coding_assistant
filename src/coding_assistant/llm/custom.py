import asyncio
import json
import logging
import os
from collections.abc import Sequence
from typing import Literal

import httpx
from httpx_sse import aconnect_sse

from coding_assistant.llm.adapters import get_tools
from coding_assistant.llm.types import (
    Completion,
    BaseMessage,
    ProgressCallbacks,
    Tool,
    ToolCall,
    FunctionCall,
    AssistantMessage,
    message_to_dict,
)
from coding_assistant.trace import trace_json

logger = logging.getLogger(__name__)


def _get_base_url_and_api_key() -> tuple[str, str]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return ("https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"])
    else:
        return ("https://api.openai.com/v1", os.environ["OPENAI_API_KEY"])


def _merge_chunks(chunks: list[dict]) -> AssistantMessage:
    full_content = ""
    full_reasoning = ""
    full_tool_calls = {}

    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]

        if reasoning := delta.get("reasoning"):
            full_reasoning += reasoning

        if content := delta.get("content"):
            full_content += content

        for tc_chunk in delta.get("tool_calls", []):
            idx = tc_chunk["index"]

            tc = full_tool_calls.setdefault(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )

            if id := tc_chunk.get("id"):
                tc["id"] += id
            if function := tc_chunk.get("function"):
                if name := function.get("name"):
                    tc["function"]["name"] += name
                if arguments := function.get("arguments"):
                    tc["function"]["arguments"] += arguments

    final_tool_calls = []
    for _, item in sorted(full_tool_calls.items()):
        final_tool_calls.append(
            ToolCall(
                id=item["id"],
                function=FunctionCall(
                    name=item["function"]["name"],
                    arguments=item["function"]["arguments"],
                ),
            )
        )

    return AssistantMessage(
        role="assistant",
        content=full_content,
        reasoning_content=full_reasoning,
        tool_calls=final_tool_calls,
    )


async def _try_completion(
    messages: list[BaseMessage],
    tools: Sequence[Tool],
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    base_url, api_key = _get_base_url_and_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    provider_messages = [message_to_dict(m) for m in messages]
    provider_tools = await get_tools(tools)

    payload = {
        "model": model,
        "messages": provider_messages,
        "tools": provider_tools,
        "stream": True,
    }

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        async with aconnect_sse(client, "POST", "/chat/completions", json=payload) as source:
            chunks = []
            async for event in source.aiter_sse():
                if event.data == "[DONE]":
                    break

                chunk = json.loads(event.data)
                chunks.append(chunk)

                delta = chunk["choices"][0]["delta"]

                if reasoning := delta.get("reasoning"):
                    callbacks.on_reasoning_chunk(reasoning)

                if content := delta.get("content"):
                    callbacks.on_content_chunk(content)

            callbacks.on_chunks_end()

            # Merge all chunks into final message
            assistant_msg = _merge_chunks(chunks)

    trace_json(
        "completion.json5",
        {
            "model": model,
            "messages": provider_messages,
            "tools": provider_tools,
            "completion": message_to_dict(assistant_msg),
        },
    )

    return Completion(
        message=assistant_msg,
        tokens=0,  # TODO
    )


async def _try_completion_with_retry(
    messages: list[BaseMessage],
    tools: Sequence[Tool],
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await _try_completion(messages, tools, model, reasoning_effort, callbacks)
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} due to {e} for model {model}")
            await asyncio.sleep(0.5 + attempt)


async def complete(
    messages: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    callbacks: ProgressCallbacks,
):
    try:
        # Re-use reasoning parsing logic from LiteLLM adapter if available,
        # but we'll re-implement simple version to avoid circularity if needed.
        from coding_assistant.llm.litellm import _parse_model_and_reasoning

        model, reasoning_effort = _parse_model_and_reasoning(model)
        return await _try_completion_with_retry(messages, tools, model, reasoning_effort, callbacks)
    except Exception as e:
        logger.error(f"Error during model completion (OpenAI): {e}")
        raise e
