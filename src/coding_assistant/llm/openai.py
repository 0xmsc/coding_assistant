import asyncio
import logging
import os
from collections.abc import Sequence
from typing import Literal

from openai import AsyncOpenAI, APIConnectionError, APIError, RateLimitError, InternalServerError, BadRequestError

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


def _map_internal_message_to_openai(msg: BaseMessage) -> dict:
    d = message_to_dict(msg)
    # OpenAI and OpenRouter expect tool_calls to be null or a list, but we might have empty lists
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

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    openai_messages = [_map_internal_message_to_openai(m) for m in messages]
    openai_tools = await get_tools(tools)

    extra_headers = {}
    if is_openrouter:
        extra_headers["HTTP-Referer"] = "https://github.com/0xmsc/coding-assistant"
        extra_headers["X-Title"] = "Coding Assistant"

    kwargs = {
        "model": model,
        "messages": openai_messages,
        "tools": openai_tools,
        "stream": True,
        "extra_headers": extra_headers,
    }

    # reasoning_effort is only supported by O1/O3 models in OpenAI
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    response = await client.chat.completions.create(**kwargs)

    full_content = []
    full_reasoning = []
    tool_calls_chunks = {}
    role = "assistant"

    async for chunk in response:
        if not chunk.choices:
            continue
        
        delta = chunk.choices[0].delta
        
        # OpenRouter/OpenAI reasoning extraction
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            full_reasoning.append(reasoning)
            callbacks.on_reasoning_chunk(reasoning)

        if delta.content:
            full_content.append(delta.content)
            callbacks.on_content_chunk(delta.content)

        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index
                if idx not in tool_calls_chunks:
                    tool_calls_chunks[idx] = {
                        "id": tc_chunk.id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                
                if tc_chunk.id:
                    tool_calls_chunks[idx]["id"] = tc_chunk.id
                if tc_chunk.function.name:
                    tool_calls_chunks[idx]["function"]["name"] += tc_chunk.function.name
                if tc_chunk.function.arguments:
                    tool_calls_chunks[idx]["function"]["arguments"] += tc_chunk.function.arguments

    callbacks.on_chunks_end()

    # Build final message
    final_tool_calls = []
    for idx in sorted(tool_calls_chunks.keys()):
        tc = tool_calls_chunks[idx]
        final_tool_calls.append(
            ToolCall(
                id=tc["id"],
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"]
                )
            )
        )

    content_str = "".join(full_content) if full_content else None
    reasoning_str = "".join(full_reasoning) if full_reasoning else None

    assistant_msg = AssistantMessage(
        role="assistant",
        content=content_str,
        reasoning_content=reasoning_str,
        tool_calls=final_tool_calls
    )

    # Note: Usage info might not be available in streaming unless requested or supported by provider
    # For now we use a dummy or partial token count if available. 
    # OpenRouter often doesn't provide it in the last chunk of a stream without specific flags.
    tokens = 0 

    trace_json(
        "openai_completion.json",
        {
            "model": model,
            "messages": openai_messages,
            "tools": openai_tools,
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
