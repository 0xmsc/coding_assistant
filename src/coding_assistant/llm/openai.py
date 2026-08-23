from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx
from httpx_sse import SSEError, aconnect_sse

from coding_assistant.infra.trace import trace_json
from coding_assistant.llm.provider_config import resolve_provider_config
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    ToolCall,
    ToolDefinition,
    Usage,
    message_to_dict,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ProviderModel:
    """A provider model and its advertised reasoning efforts."""

    id: str
    reasoning_efforts: tuple[str, ...] = ()


async def _get_tools_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool definitions into the provider request payload."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        params = tool.parameters()
        fix_input_schema(params)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": params,
                },
            },
        )
    return result


def _get_base_url_and_api_key(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Resolve the API base URL and key from the configured provider env vars."""
    config = resolve_provider_config(env)
    return (config.base_url, config.api_key)


def _reasoning_efforts_from_model(
    item: dict[str, Any],
) -> tuple[str, ...]:
    reasoning = item.get("reasoning")
    if not isinstance(reasoning, dict):
        return ()

    efforts = reasoning.get("supported_efforts")
    return tuple(efforts) if isinstance(efforts, list) and all(isinstance(effort, str) for effort in efforts) else ()


async def list_models(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[ProviderModel]:
    """Return provider model IDs and any advertised reasoning capabilities."""
    if base_url is None or api_key is None:
        resolved_base_url, resolved_api_key = _get_base_url_and_api_key()
        base_url = base_url or resolved_base_url
        api_key = api_key or resolved_api_key
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, transport=transport, timeout=httpx.Timeout(15)
    ) as client:
        response = await client.get("/models")
        response.raise_for_status()

    decoded = response.json()
    if not isinstance(decoded, dict):
        raise ValueError("Provider model list response must be a JSON object.")
    data = decoded.get("data")
    if not isinstance(data, list):
        raise ValueError("Provider model list response must include a data array.")

    models: dict[str, ProviderModel] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_id = model_id.strip()
            models[model_id] = ProviderModel(
                id=model_id,
                reasoning_efforts=_reasoning_efforts_from_model(item),
            )
    return [models[model_id] for model_id in sorted(models)]


def _merge_chunks(chunks: list[dict[str, Any]]) -> AssistantMessage:
    """Collapse streamed provider chunks into one assistant message."""
    full_content = ""
    full_reasoning = ""
    full_tool_calls: dict[int, dict[str, Any]] = {}
    full_reasoning_details: list[dict[str, Any]] = []

    for chunk in chunks:
        if not chunk["choices"]:
            continue
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
                rdc_type = rdc.get("type")
                rdc_index = rdc.get("index")
                if rdc_type in ("reasoning.text", "reasoning.summary"):
                    if (
                        full_reasoning_details
                        and (last := full_reasoning_details[-1])
                        and last.get("type") == rdc_type
                        and last.get("index") == rdc_index
                    ):
                        if text := rdc.get("text"):
                            last["text"] = last.get("text", "") + text
                        if summary := rdc.get("summary"):
                            last["summary"] = last.get("summary", "") + summary
                        if signature := rdc.get("signature"):
                            last["signature"] = signature
                        if format_val := rdc.get("format"):
                            last["format"] = format_val
                        if id_val := rdc.get("id"):
                            last["id"] = id_val
                    else:
                        full_reasoning_details.append(dict(rdc))
                else:
                    full_reasoning_details.append(dict(rdc))

    if not full_reasoning and full_reasoning_details:
        extracted = [text for d in full_reasoning_details if (text := d.get("text") or d.get("summary"))]
        if extracted:
            full_reasoning = "".join(extracted)

    final_tool_calls = []
    for _, item in sorted(full_tool_calls.items()):
        final_tool_calls.append(
            ToolCall(
                id=item["id"],
                function=FunctionCall(
                    name=item["function"]["name"],
                    arguments=item["function"]["arguments"],
                ),
            ),
        )

    provider_specific_fields: dict[str, Any] = {}
    if full_reasoning_details:
        provider_specific_fields["reasoning_details"] = full_reasoning_details

    return AssistantMessage(
        role="assistant",
        content=full_content if full_content else None,
        reasoning_content=full_reasoning if full_reasoning else None,
        tool_calls=final_tool_calls,
        provider_specific_fields=provider_specific_fields,
    )


def _extract_usage(chunks: list[dict[str, Any]]) -> Usage | None:
    """Read usage information from the final streamed chunk when present."""
    if not chunks:
        return None

    if usage_chunk := chunks[-1].get("usage"):
        tokens = usage_chunk.get("total_tokens")
        cost = usage_chunk.get("cost")
        return Usage(tokens=tokens, cost=cost)

    return None


def _prepare_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Convert internal messages into the provider request payload shape."""
    result = [message_to_dict(m) for m in messages]
    for m in result:
        if "provider_specific_fields" in m:
            for k, v in m["provider_specific_fields"].items():
                m[k] = v
            del m["provider_specific_fields"]
    return result


async def _try_completion(
    messages: Sequence[BaseMessage],
    tools: Sequence[ToolDefinition],
    model: str,
    reasoning_effort: str | None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> AsyncIterator[ReasoningDeltaEvent | ContentDeltaEvent | CompletionEvent]:
    """Perform one streaming chat completion request against the provider."""
    if base_url is None or api_key is None:
        resolved_base_url, resolved_api_key = _get_base_url_and_api_key()
        base_url = base_url or resolved_base_url
        api_key = api_key or resolved_api_key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    provider_messages = _prepare_messages(messages)
    provider_tools = await _get_tools_payload(tools)

    payload: dict[str, Any] = {
        "model": model,
        "messages": provider_messages,
        "tools": provider_tools,
        "stream": True,
    }

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, transport=transport, timeout=httpx.Timeout(60)
    ) as client:
        async with aconnect_sse(client, "POST", "/chat/completions", json=payload) as source:
            chunks: list[dict[str, Any]] = []
            try:
                async for event in source.aiter_sse():
                    if event.data == "[DONE]":
                        break

                    chunk = json.loads(event.data)
                    chunks.append(chunk)

                    if not chunk["choices"]:
                        continue
                    delta = chunk["choices"][0]["delta"]

                    if (reasoning := delta.get("reasoning")) or (reasoning := delta.get("reasoning_content")):
                        yield ReasoningDeltaEvent(content=reasoning)
                    elif reasoning_details := delta.get("reasoning_details"):
                        for rdc in reasoning_details:
                            if text := rdc.get("text"):
                                yield ReasoningDeltaEvent(content=text)
                            elif summary := rdc.get("summary"):
                                yield ReasoningDeltaEvent(content=summary)

                    if content := delta.get("content"):
                        yield ContentDeltaEvent(content=content)
            except SSEError as e:
                response = source.response
                await response.aread()
                content = response.text
                logger.error(f"SSE error during completion: {e}, response {response}, {content}")
                raise

            # Merge all chunks into final message
            message = _merge_chunks(chunks)
            usage = _extract_usage(chunks)

    trace_data: dict[str, Any] = {
        "model": model,
        "chunks": chunks,
        "messages": provider_messages,
        "tools": provider_tools,
        "completion": message_to_dict(message),
    }

    if usage is not None:
        trace_data["usage"] = dataclasses.asdict(usage)

    trace_json("completion.json5", trace_data)

    yield CompletionEvent(
        completion=Completion(
            message=message,
            usage=usage,
        ),
    )


def _parse_model_and_reasoning(
    model: str,
) -> tuple[str, str | None]:
    """Split `model (effort)` syntax into the provider model and reasoning effort."""
    s = model.strip()
    m = re.match(r"^(.+?) \(([^)]*)\)$", s)

    if not m or not m.group(2).strip():
        return s, None

    return m.group(1).strip(), m.group(2).strip()


def fix_input_schema(input_schema: dict[str, Any]) -> None:
    """Remove schema features that some OpenAI-compatible providers reject."""
    for prop in input_schema.get("properties", {}).values():
        fmt = prop.get("format")
        if fmt == "uri":
            prop.pop("format", None)


async def stream_completion(
    messages: Sequence[BaseMessage],
    tools: Sequence[ToolDefinition],
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    retry_delay: float | None = None,
) -> AsyncIterator[ContentDeltaEvent | ReasoningDeltaEvent | ModelRetryEvent | StatusEvent | CompletionEvent]:
    """Retry transient HTTP failures before surfacing the completion error."""
    model, reasoning_effort = _parse_model_and_reasoning(model)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async for event in _try_completion(
                messages,
                tools,
                model,
                reasoning_effort,
                transport=transport,
                base_url=base_url,
                api_key=api_key,
            ):
                yield event
            return
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} due to {e} for model {model}")
            yield ModelRetryEvent()
            sleep_time = retry_delay if retry_delay is not None else (0.5 + attempt)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
