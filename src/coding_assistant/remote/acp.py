from __future__ import annotations

import json
from typing import Any

JsonObject = dict[str, Any]

JSONRPC_VERSION = "2.0"
ACP_PROTOCOL_VERSION = 1
STOP_REASON_END_TURN = "end_turn"
STOP_REASON_CANCELLED = "cancelled"

ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_SERVER = -32000


def parse_jsonrpc_message(data: str | bytes) -> JsonObject:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("JSON-RPC payload must be an object.")
    return payload


def jsonrpc_request(message_id: int | str, method: str, params: JsonObject | None = None) -> str:
    payload: JsonObject = {
        "jsonrpc": JSONRPC_VERSION,
        "id": message_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload)


def jsonrpc_notification(method: str, params: JsonObject | None = None) -> str:
    payload: JsonObject = {
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload)


def jsonrpc_result(message_id: int | str, result: JsonObject | None) -> str:
    return json.dumps(
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": message_id,
            "result": result,
        },
    )


def jsonrpc_error(message_id: int | str | None, code: int, message: str) -> str:
    return json.dumps(
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": message_id,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )


def text_block(text: str) -> JsonObject:
    return {"type": "text", "text": text}


def tool_content_text(text: str) -> list[JsonObject]:
    return [{"type": "content", "content": text_block(text)}]


def initialize_result(*, agent_name: str, agent_title: str, agent_version: str) -> JsonObject:
    return {
        "protocolVersion": ACP_PROTOCOL_VERSION,
        "agentCapabilities": {
            "loadSession": False,
            "promptCapabilities": {
                "image": True,
                "embeddedContext": True,
            },
        },
        "agentInfo": {
            "name": agent_name,
            "title": agent_title,
            "version": agent_version,
        },
        "authMethods": [],
    }


def prompt_content_from_acp(prompt_blocks: list[JsonObject]) -> str | list[JsonObject]:
    if len(prompt_blocks) == 1 and prompt_blocks[0].get("type") == "text":
        text = prompt_blocks[0].get("text")
        if not isinstance(text, str):
            raise ValueError("ACP text content must include a string 'text' field.")
        return text

    converted: list[JsonObject] = []
    for block in prompt_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("ACP text content must include a string 'text' field.")
            converted.append(text_block(text))
            continue

        if block_type == "image":
            mime_type = block.get("mimeType")
            if not isinstance(mime_type, str):
                raise ValueError("ACP image content must include a string 'mimeType' field.")
            uri = block.get("uri")
            data = block.get("data")
            if isinstance(uri, str):
                image_url = uri
            elif isinstance(data, str):
                image_url = f"data:{mime_type};base64,{data}"
            else:
                raise ValueError("ACP image content must include either 'uri' or 'data'.")
            converted.append({"type": "image_url", "image_url": {"url": image_url}})
            continue

        if block_type == "resource":
            resource = block.get("resource")
            if not isinstance(resource, dict):
                raise ValueError("ACP resource content must include a 'resource' object.")
            converted.append(text_block(_render_embedded_resource(resource)))
            continue

        if block_type == "resource_link":
            converted.append(text_block(_render_resource_link(block)))
            continue

        raise ValueError(f"Unsupported ACP content block type: {block_type}")

    return converted


def _render_embedded_resource(resource: JsonObject) -> str:
    uri = resource.get("uri", "resource")
    mime_type = resource.get("mimeType", "application/octet-stream")
    if isinstance(resource.get("text"), str):
        return f"Embedded resource {uri} ({mime_type}):\n{resource['text']}"
    return f"Embedded binary resource {uri} ({mime_type})."


def _render_resource_link(block: JsonObject) -> str:
    uri = block.get("uri", "resource")
    name = block.get("name") or block.get("title") or uri
    description = block.get("description")
    if isinstance(description, str) and description:
        return f"Resource link {name}: {uri}\n{description}"
    return f"Resource link {name}: {uri}"


def agent_message_update(session_id: str, text: str) -> str:
    return jsonrpc_notification(
        "session/update",
        {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": text_block(text),
            },
        },
    )


def tool_call_update_notification(session_id: str, update: JsonObject) -> str:
    return jsonrpc_notification(
        "session/update",
        {
            "sessionId": session_id,
            "update": update,
        },
    )


def tool_call_notification(
    *,
    tool_call_id: str,
    title: str,
    kind: str,
    status: str = "pending",
    raw_input: JsonObject | None = None,
) -> JsonObject:
    update: JsonObject = {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": title,
        "kind": kind,
        "status": status,
    }
    if raw_input is not None:
        update["rawInput"] = raw_input
    return update


def tool_call_lifecycle_update(
    *,
    tool_call_id: str,
    status: str,
    title: str | None = None,
    kind: str | None = None,
    raw_input: Any | None = None,
    raw_output: Any | None = None,
    content_text: str | None = None,
) -> JsonObject:
    update: JsonObject = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": status,
    }
    if title is not None:
        update["title"] = title
    if kind is not None:
        update["kind"] = kind
    if raw_input is not None:
        update["rawInput"] = raw_input
    if raw_output is not None:
        update["rawOutput"] = raw_output
    if content_text:
        update["content"] = tool_content_text(content_text)
    return update
