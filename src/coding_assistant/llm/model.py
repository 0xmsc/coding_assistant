import asyncio
import functools
import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, cast

import litellm

from coding_assistant.framework.callbacks import ProgressCallbacks
from coding_assistant.framework.models import (
    AssistantMessage,
    FunctionCall,
    LLMMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.trace import trace_data

logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.modify_params = True
litellm.drop_params = True


@dataclass
class Completion:
    message: LLMMessage
    tokens: int


def _map_litellm_message_to_internal(litellm_message: litellm.Message) -> LLMMessage:
    tool_calls = []
    if hasattr(litellm_message, "tool_calls") and litellm_message.tool_calls:
        for tc in litellm_message.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    function=FunctionCall(
                        name=tc.function.name or "",
                        arguments=tc.function.arguments or "",
                    ),
                )
            )

    return AssistantMessage(
        content=getattr(litellm_message, "content", None),
        reasoning_content=getattr(litellm_message, "reasoning_content", None),
        tool_calls=tool_calls,
    )


def _map_internal_message_to_litellm(msg: LLMMessage) -> dict:
    d: dict = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.name:
        d["name"] = msg.name

    if isinstance(msg, AssistantMessage):
        if msg.reasoning_content is not None:
            d["reasoning_content"] = msg.reasoning_content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
    elif isinstance(msg, ToolMessage):
        d["tool_call_id"] = msg.tool_call_id

    return d


@functools.cache
def _parse_model_and_reasoning(
    model: str,
) -> tuple[str, Literal["low", "medium", "high"] | None]:
    s = model.strip()
    m = re.match(r"^(.+?) \(([^)]*)\)$", s)

    if not m:
        return s, None

    base = m.group(1).strip()
    effort = m.group(2).strip().lower()

    if effort not in ("low", "medium", "high"):
        raise ValueError(f"Invalid reasoning effort level {effort} in {model}")

    effort = cast(Literal["low", "medium", "high"], effort)
    return base, effort


async def _try_completion(
    messages: list[LLMMessage],
    tools: list,
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    litellm_messages = [_map_internal_message_to_litellm(m) for m in messages]

    response = await litellm.acompletion(
        messages=litellm_messages,
        tools=tools,
        model=model,
        stream=True,
        reasoning_effort=reasoning_effort,
    )

    chunks = []

    async for chunk in response:
        if len(chunk["choices"]) > 0:
            delta = chunk["choices"][0]["delta"]
            if "reasoning" in delta and delta["reasoning"]:
                callbacks.on_reasoning_chunk(delta["reasoning"])

            if "content" in delta and delta["content"]:
                callbacks.on_content_chunk(delta["content"])

        # Drop created_at so that `ChunkProcessor` does not sort according to it.
        # It seems buggy and seems to create out-of-order chunks.
        chunk._hidden_params.pop("created_at", None)

        chunks.append(chunk)

    callbacks.on_chunks_end()

    completion = litellm.stream_chunk_builder(chunks)
    assert completion

    trace_data(
        "completion.json",
        json.dumps(
            {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "messages": litellm_messages,
                "tools": tools,
                "completion": completion.model_dump(),
            },
            indent=2,
            default=str,
        ),
    )

    litellm_message = completion["choices"][0]["message"]

    return Completion(
        message=_map_litellm_message_to_internal(litellm_message),
        tokens=completion["usage"]["total_tokens"],
    )


async def _try_completion_with_retry(
    messages: list[LLMMessage],
    tools: list,
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await _try_completion(messages, tools, model, reasoning_effort, callbacks)
        except (
            litellm.APIConnectionError,
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
        ) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} due to {e} for model {model}")
            await asyncio.sleep(0.5 + attempt)


async def complete(
    messages: list[LLMMessage] | list[dict],
    model: str,
    tools: list,
    callbacks: ProgressCallbacks,
):
    # Support both list[dict] and list[LLMMessage] during migration
    internal_messages: list[LLMMessage] = []
    for m in messages:
        if isinstance(m, (SystemMessage, UserMessage, AssistantMessage, ToolMessage)):
            internal_messages.append(m)
        elif isinstance(m, dict):
            # Fallback for dicts
            tool_calls = []
            if "tool_calls" in m:
                for tc in m["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            id=tc["id"],
                            function=FunctionCall(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"],
                            ),
                        )
                    )
            role = m["role"]
            content = m.get("content")
            name = m.get("name")
            if role == "system":
                internal_messages.append(SystemMessage(content=content or "", name=name))
            elif role == "user":
                internal_messages.append(UserMessage(content=content or "", name=name))
            elif role == "assistant":
                internal_messages.append(
                    AssistantMessage(
                        content=content,
                        reasoning_content=m.get("reasoning_content"),
                        tool_calls=tool_calls,
                        name=name,
                    )
                )
            elif role == "tool":
                internal_messages.append(
                    ToolMessage(
                        content=content or "",
                        tool_call_id=m.get("tool_call_id", ""),
                        name=name,
                    )
                )
            else:
                raise ValueError(f"Unknown role {role}")

    try:
        model, reasoning_effort = _parse_model_and_reasoning(model)
        return await _try_completion_with_retry(internal_messages, tools, model, reasoning_effort, callbacks)
    except Exception as e:
        logger.error(f"Error during model completion: {e}, last messages: {internal_messages[-5:]}")
        raise e
