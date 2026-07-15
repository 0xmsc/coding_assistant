from __future__ import annotations

import pytest

from coding_assistant.remote.jsonrpc import (
    ACP_PROTOCOL_VERSION,
    initialize_response,
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


@pytest.mark.parametrize("requested_version", [0, -1, True])
def test_initialize_response_rejects_incompatible_protocol_versions(requested_version: int) -> None:
    with pytest.raises(ValueError, match="No compatible protocol version"):
        initialize_response(
            requested_protocol_version=requested_version,
            agent_name="coding-assistant",
            agent_title="Coding Assistant",
        )
