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


class APIConnectionError(Exception):
    pass


class APIError(Exception):
    pass


class RateLimitError(Exception):
    pass


class InternalServerError(Exception):
    pass


class BadRequestError(Exception):
    pass


def _map_internal_message_to_provider(msg: BaseMessage) -> dict:
    d = message_to_dict(msg)
    # Providers expect tool_calls to be null or a list, but we might have empty lists
    if "tool_calls" in d and not d["tool_calls"]:
        d.pop("tool_calls")
    return d


async def _try_completion(
    messages: list[BaseMessage],
    tools: Sequence[Tool],
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    # Detect if we should use OpenRouter
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY")

    is_openrouter = False
    if api_key:
        is_openrouter = True
    else:
        # Fallback to standard OpenAI if no OpenRouter key
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"

    headers = {"Authorization": f"Bearer {api_key}"}
    if is_openrouter:
        headers.update(
            {
                "HTTP-Referer": "https://github.com/0xmsc/coding-assistant",
                "X-Title": "Coding Assistant",
            }
        )

    provider_messages = [_map_internal_message_to_provider(m) for m in messages]
    provider_tools = await get_tools(tools)

    payload = {
        "model": model,
        "messages": provider_messages,
        "tools": provider_tools,
        "stream": True,
    }

    # reasoning_effort is only supported by O1/O3 models in OpenAI
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        async with aconnect_sse(client, "POST", "/chat/completions", json=payload) as source:
            full_content = []
            full_reasoning = []
            tool_calls_chunks = {}

            async for event in source.aiter_sse():
                if not event.data:
                    continue
                try:
                    chunk = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                if not chunk.get("choices"):
                    continue

                delta = chunk["choices"][0]["delta"]

                # Reasoning extraction (OpenRouter/OpenAI style)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    full_reasoning.append(reasoning)
                    callbacks.on_reasoning_chunk(reasoning)

                if delta.get("content"):
                    full_content.append(delta["content"])
                    callbacks.on_content_chunk(delta["content"])

                if delta.get("tool_calls"):
                    for tc_chunk in delta["tool_calls"]:
                        idx = tc_chunk["index"]
                        if idx not in tool_calls_chunks:
                            tool_calls_chunks[idx] = {
                                "id": tc_chunk.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }

                        if tc_chunk.get("id"):
                            tool_calls_chunks[idx]["id"] = tc_chunk["id"]
                        if tc_chunk.get("function", {}).get("name"):
                            tool_calls_chunks[idx]["function"]["name"] += tc_chunk["function"]["name"]
                        if tc_chunk.get("function", {}).get("arguments"):
                            tool_calls_chunks[idx]["function"]["arguments"] += tc_chunk["function"]["arguments"]

        callbacks.on_chunks_end()

    # Build final message
    final_tool_calls = []
    for idx in sorted(tool_calls_chunks.keys()):
        tc = tool_calls_chunks[idx]
        final_tool_calls.append(
            ToolCall(
                id=tc["id"], function=FunctionCall(name=tc["function"]["name"], arguments=tc["function"]["arguments"])
            )
        )

    content_str = "".join(full_content) if full_content else None
    reasoning_str = "".join(full_reasoning) if full_reasoning else None

    assistant_msg = AssistantMessage(
        role="assistant",
        content=content_str,
        reasoning_content=reasoning_str,
        tool_calls=final_tool_calls if final_tool_calls else None,
    )

    # Token count: dummy for now
    tokens = 0

    trace_json(
        "openai_completion.json",
        {
            "model": model,
            "messages": provider_messages,
            "tools": provider_tools,
            "completion": message_to_dict(assistant_msg),
        },
    )

    return Completion(
        message=assistant_msg,
        tokens=tokens,
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
        except (
            APIConnectionError,
            APIError,
            RateLimitError,
            InternalServerError,
            BadRequestError,
        ) as e:
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
