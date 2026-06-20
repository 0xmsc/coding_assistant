from __future__ import annotations

from pathlib import Path

import pytest

from coding_assistant.llm.types import (
    AssistantMessage,
    FunctionCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.manager.store import SessionNotFoundError, SessionStore, StaleSessionCommitError
from coding_assistant.manager.workspace import WorkspaceMissingError, WorkspacePaths


def test_create_list_and_load_session_by_scope(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )

    created = store.create_session(
        scope_id="scope-a",
        title="First session",
        metadata={"messageCount": 1},
        messages=[SystemMessage(content="system")],
    )
    store.create_session(scope_id="scope-b", messages=[SystemMessage(content="other")])

    sessions = store.list_sessions(scope_id="scope-a")
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert [session.session_id for session in sessions] == [created.record.session_id]
    assert loaded.record == created.record
    assert loaded.messages == [SystemMessage(content="system")]
    assert loaded.workspace == tmp_path / "workspaces" / created.record.session_id
    assert loaded.workspace.is_dir()


def test_load_session_rejects_cross_scope_access(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])

    with pytest.raises(SessionNotFoundError):
        store.load_session(scope_id="scope-b", session_id=created.record.session_id)


def test_commit_messages_appends_json_payloads_and_increments_version(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])
    tool_call = ToolCall(id="call-1", function=FunctionCall(name="echo_tool", arguments='{"text": "hi"}'))

    new_version = store.commit_messages(
        scope_id="scope-a",
        session_id=created.record.session_id,
        base_version=0,
        title="Updated title",
        metadata={"messageCount": 4},
        messages=[
            UserMessage(content="hello"),
            AssistantMessage(content=None, tool_calls=[tool_call]),
            ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:hi"),
        ],
    )
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert new_version == 1
    assert loaded.record.version == 1
    assert loaded.record.title == "Updated title"
    assert loaded.record.metadata == {"messageCount": 4}
    assert loaded.messages == [
        SystemMessage(content="system"),
        UserMessage(content="hello"),
        AssistantMessage(content=None, tool_calls=[tool_call]),
        ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:hi"),
    ]


def test_commit_messages_rejects_stale_base_version(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])

    store.commit_messages(
        scope_id="scope-a",
        session_id=created.record.session_id,
        base_version=0,
        messages=[UserMessage(content="first")],
    )

    with pytest.raises(StaleSessionCommitError):
        store.commit_messages(
            scope_id="scope-a",
            session_id=created.record.session_id,
            base_version=0,
            messages=[UserMessage(content="stale")],
        )

    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)
    assert loaded.record.version == 1
    assert loaded.messages == [UserMessage(content="first")]


def test_rename_session_updates_title_without_changing_version(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])

    renamed = store.rename_session(scope_id="scope-a", session_id=created.record.session_id, title="Renamed")
    cleared = store.rename_session(scope_id="scope-a", session_id=created.record.session_id, title=None)

    assert renamed.title == "Renamed"
    assert renamed.version == 0
    assert cleared.title is None
    assert cleared.version == 0
    assert store.load_session(scope_id="scope-a", session_id=created.record.session_id).messages == [
        SystemMessage(content="system"),
    ]


def test_rename_session_rejects_cross_scope_access(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])

    with pytest.raises(SessionNotFoundError):
        store.rename_session(scope_id="scope-b", session_id=created.record.session_id, title="Bad")


def test_update_session_metadata_merges_without_changing_version(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(
        scope_id="scope-a",
        messages=[SystemMessage(content="system")],
        metadata={"model": "test-model", "messageCount": 1},
    )

    updated = store.update_session_metadata(
        scope_id="scope-a",
        session_id=created.record.session_id,
        metadata={"model": "alternate-model"},
    )
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert updated.version == 0
    assert loaded.record.metadata == {"model": "alternate-model", "messageCount": 1}
    assert loaded.messages == [SystemMessage(content="system")]


def test_update_session_metadata_rejects_cross_scope_access(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])

    with pytest.raises(SessionNotFoundError):
        store.update_session_metadata(
            scope_id="scope-b",
            session_id=created.record.session_id,
            metadata={"model": "bad-model"},
        )


def test_load_session_fails_when_workspace_is_missing(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])
    created.workspace.rmdir()

    with pytest.raises(WorkspaceMissingError):
        store.load_session(scope_id="scope-a", session_id=created.record.session_id)


def test_workspace_paths_are_derived_from_session_id(tmp_path: Path) -> None:
    workspaces = WorkspacePaths(root=tmp_path / "workspaces")

    path = workspaces.create_for_session("sess_abc")

    assert path == tmp_path / "workspaces" / "sess_abc"
    assert workspaces.require_for_session("sess_abc") == path
