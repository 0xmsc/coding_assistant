import dataclasses
import asyncio
import functools
import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Literal, cast

import httpx
from httpx_sse import aconnect_sse, SSEError

from coding_assistant.llm.adapters import get_tools
from coding_assistant.llm.types import (
    Completion,
    Usage,
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
    if os.environ.get("OPENAI_BASE_URL"):
        return (os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"])
    else:
        return ("https://api.openai.com/v1", os.environ["OPENAI_API_KEY"])


def _merge_chunks(chunks: list[dict]) -> AssistantMessage:
    full_content = ""
    full_reasoning = ""
    full_tool_calls: dict[int, dict] = {}
    full_reasoning_details: list[dict] = []

    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]

        if (reasoning := delta.get("reasoning")) or (reasoning := delta.get("reasoning_content")):
            full_reasoning += reasoning

        if content := delta.get("content"):
            full_content += content

        for tcc in delta.get("tool_calls", []):
            idx = tcc["index"]

            tc = full_tool_calls.setdefault(
                idx,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )

            if id := tcc.get("id"):
                tc["id"] += id
            if function := tcc.get("function"):
                if name := function.get("name"):
                    tc["function"]["name"] += name
                if arguments := function.get("arguments"):
                    tc["function"]["arguments"] += arguments

        # Openrouter specific field
        if reasoning_details := delta.get("reasoning_details"):
            for rdc in reasoning_details:
                # Try to merge with last
                if rdc["type"] in ("reasoning.text", "reasoning.summary"):
                    if (
                        full_reasoning_details
                        and (last := full_reasoning_details[-1])
                        and last["type"] == rdc["type"]
                        and last["index"] == rdc["index"]
                    ):
                        if text := rdc.get("text"):
                            last["text"] += text
                        if summary := rdc.get("summary"):
                            last["summary"] += summary
                        if signature := rdc.get("signature"):
                            last["signature"] = signature
                    else:
                        full_reasoning_details.append(rdc)
                else:
                    full_reasoning_details.append(rdc)

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
        content=full_content if full_content else None,
        reasoning_content=full_reasoning if full_reasoning else None,
        tool_calls=final_tool_calls,
        provider_specific_fields={
            "reasoning_details": full_reasoning_details,
        },
    )


def _extract_usage(chunks: list[dict]) -> Usage | None:
    if not chunks:
        return None

    if usage_chunk := chunks[-1].get("usage"):
        tokens = usage_chunk.get("total_tokens")
        cost = usage_chunk.get("cost")
        return Usage(tokens=tokens, cost=cost)

    return None


def _prepare_messages(messages: list[BaseMessage]) -> list[dict]:
    result = [message_to_dict(m) for m in messages]
    for m in result:
        if "provider_specific_fields" in m:
            for k, v in m["provider_specific_fields"].items():
                m[k] = v
            del m["provider_specific_fields"]
    return result


async def _try_completion(
    messages: list[BaseMessage],
    tools: Sequence[Tool],
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    base_url, api_key = _get_base_url_and_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    provider_messages = _prepare_messages(messages)
    provider_tools = await get_tools(tools)

    payload = {
        "model": model,
        "messages": provider_messages,
        "tools": provider_tools,
        "stream": True,
    }

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
        # TODO: Does OpenAI support this?
        # payload["reasoning"]["effort"] = reasoning_effort

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(30)) as client:
        async with aconnect_sse(client, "POST", "/chat/completions", json=payload) as source:
            chunks = []
            try:
                async for event in source.aiter_sse():
                    if event.data == "[DONE]":
                        break

                    chunk = json.loads(event.data)
                    chunks.append(chunk)

                    delta = chunk["choices"][0]["delta"]

                    if (reasoning := delta.get("reasoning")) or (reasoning := delta.get("reasoning_content")):
                        callbacks.on_reasoning_chunk(reasoning)

                    if content := delta.get("content"):
                        callbacks.on_content_chunk(content)
            except SSEError as e:
                response = source.response
                await response.aread()
                content = response.text
                logger.error(f"SSE error during completion: {e}, response {response}, {content}")
                raise

            callbacks.on_chunks_end()

            # Merge all chunks into final message
            message = _merge_chunks(chunks)
            usage = _extract_usage(chunks)

    trace_data = {
        "model": model,
        "chunks": chunks,
        "messages": provider_messages,
        "tools": provider_tools,
        "completion": message_to_dict(message),
    }

    if usage is not None:
        trace_data["usage"] = dataclasses.asdict(usage)

    trace_json("completion.json5", trace_data)

    return Completion(
        message=message,
        usage=usage,
    )


async def _try_completion_with_retry(
    messages: list[BaseMessage],
    tools: Sequence[Tool],
    model: str,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    callbacks: ProgressCallbacks,
):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return await _try_completion(messages, tools, model, reasoning_effort, callbacks)
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} due to {e} for model {model}")
            await asyncio.sleep(0.5 + attempt)


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


async def complete(
    messages: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    callbacks: ProgressCallbacks,
):
    try:
        model, reasoning_effort = _parse_model_and_reasoning(model)
        return await _try_completion_with_retry(messages, tools, model, reasoning_effort, callbacks)
    except Exception as e:
        logger.error(f"Error during model completion (OpenAI): {e}, last messages: {messages[-5:]}")
        raise e
