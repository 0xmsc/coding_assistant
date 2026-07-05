from __future__ import annotations

import json

import pytest

from coding_assistant.remote.jsonrpc import (
    ERROR_INVALID_REQUEST,
    ACP_PROTOCOL_VERSION,
    initialize_response,
    jsonrpc_result_required,
    params_from_payload,
    response_id_from_payload,
    session_id_from_params,
)


def test_response_id_from_payload_accepts_request_ids() -> None:
    assert response_id_from_payload({"id": 12}) == 12
    assert response_id_from_payload({"id": "abc"}) == "abc"
    assert response_id_from_payload({"id": None}) is None
    assert response_id_from_payload({"id": []}) is None


def test_params_from_payload_requires_object_params() -> None:
    assert params_from_payload({}) == {}
    assert params_from_payload({"params": None}) == {}
    assert params_from_payload({"params": {"x": 1}}) == {"x": 1}

    with pytest.raises(ValueError, match="Request params must be an object."):
        params_from_payload({"params": []})


def test_session_id_from_params_requires_non_empty_string() -> None:
    assert session_id_from_params({"sessionId": "sess_1"}) == "sess_1"

    with pytest.raises(ValueError, match="Request params must include sessionId."):
        session_id_from_params({})


def test_jsonrpc_result_required_rejects_notifications() -> None:
    payload = json.loads(jsonrpc_result_required(None, {"ok": True}))

    assert payload["error"]["code"] == ERROR_INVALID_REQUEST
    assert payload["error"]["message"] == "Method must be a request."


def test_initialize_response_negotiates_protocol_and_capabilities() -> None:
    result = initialize_response(
        requested_protocol_version=ACP_PROTOCOL_VERSION + 10,
        agent_name="coding-assistant",
        agent_title="Coding Assistant",
        capabilities={"loadSession": True},
    )

    assert result["protocolVersion"] == ACP_PROTOCOL_VERSION
    assert result["agentCapabilities"] == {"loadSession": True}
    assert result["agentInfo"]["name"] == "coding-assistant"
