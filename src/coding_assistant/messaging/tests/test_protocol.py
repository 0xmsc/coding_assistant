from datetime import datetime
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import StartTask, DisplayMessage


def test_envelope_creation() -> None:
    payload = StartTask(prompt="Hello world")
    envelope = Envelope(sender="ui", recipient="orchestrator", payload=payload)

    assert envelope.sender == "ui"
    assert envelope.recipient == "orchestrator"
    assert envelope.payload.prompt == "Hello world"
    assert isinstance(envelope.correlation_id, str)
    assert isinstance(envelope.timestamp, datetime)


def test_envelope_serialization() -> None:
    payload = DisplayMessage(content="Thinking...", role="assistant")
    envelope = Envelope(sender="orchestrator", recipient="ui", payload=payload)

    # Test JSON serialization
    json_data = envelope.model_dump_json()
    assert "Thinking..." in json_data

    # Test deserialization
    new_envelope = Envelope[DisplayMessage].model_validate_json(json_data)
    assert new_envelope.payload.content == "Thinking..."
    assert new_envelope.correlation_id == envelope.correlation_id


def test_trace_ids_persistent() -> None:
    payload = StartTask(prompt="Task")
    env1 = Envelope(sender="a", recipient="b", payload=payload)

    # simulate a reply
    env2 = Envelope(
        sender="b",
        recipient="a",
        payload=DisplayMessage(content="Working"),
        correlation_id=env1.correlation_id,
        trace_id=env1.trace_id,
        parent_id=env1.trace_id,
    )

    assert env1.trace_id == env2.trace_id
    assert env2.parent_id == env1.trace_id
    assert env1.correlation_id == env2.correlation_id
