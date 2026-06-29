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
from coding_assistant.core.session_updates import SessionAttachment
from coding_assistant.manager.store import SessionNotFoundError, SessionStore, StaleSessionCommitError
from coding_assistant.manager.workspace import WorkspaceMissingError, WorkspacePaths


def test_create_list_and_load_session_by_scope(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
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
    assert loaded.workspace == tmp_path / "sessions" / created.record.session_id / "workspace"
    assert loaded.attachments == tmp_path / "sessions" / created.record.session_id / "attachments"
    assert loaded.workspace.is_dir()
    assert loaded.attachments.is_dir()


def test_create_and_load_session_private_worker_env(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )

    created = store.create_session(
        scope_id="scope-a",
        messages=[],
        worker_env={"APPS_API_TOKEN": "secret-token"},
    )
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert created.worker_env == {"APPS_API_TOKEN": "secret-token"}
    assert loaded.worker_env == {"APPS_API_TOKEN": "secret-token"}
    assert store.list_sessions(scope_id="scope-a")[0].metadata == {}


def test_commit_messages_persists_attachments_separately(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])

    store.commit_messages(
        scope_id="scope-a",
        session_id=created.record.session_id,
        base_version=0,
        messages=[UserMessage(content="hello")],
        attachments=[
            SessionAttachment(
                attachment_id="att_1",
                name="photo.png",
                mime_type="image/png",
                size=4,
                path="/attachments/att_1-photo.png",
                sha256="abc",
            )
        ],
    )
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert loaded.message_records[1].message == UserMessage(content="hello")
    assert loaded.message_records[1].public is True
    assert loaded.attachment_records[0].sequence == 1
    assert loaded.attachment_records[0].attachment.name == "photo.png"
    assert loaded.attachment_records[0].attachment.path == "/attachments/att_1-photo.png"


def test_attachment_ids_can_repeat_across_sessions(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    first = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])
    second = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])
    attachment = SessionAttachment(
        attachment_id="att_same",
        name="photo.png",
        mime_type="image/png",
        size=4,
        path="/attachments/att_same-photo.png",
        sha256="abc",
    )

    store.commit_messages(
        scope_id="scope-a",
        session_id=first.record.session_id,
        base_version=0,
        messages=[],
        attachments=[attachment],
    )
    store.commit_messages(
        scope_id="scope-a",
        session_id=second.record.session_id,
        base_version=0,
        messages=[],
        attachments=[attachment],
    )

    assert (
        store.load_session(scope_id="scope-a", session_id=first.record.session_id).attachment_records[0].attachment
        == attachment
    )
    assert (
        store.load_session(scope_id="scope-a", session_id=second.record.session_id).attachment_records[0].attachment
        == attachment
    )


def test_create_session_uses_reserved_workspace(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    reservation = store.reserve_session_workspace()
    marker = reservation.workspace / ".agents" / "marker.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("prepared", encoding="utf-8")

    created = store.create_session(
        scope_id="scope-a",
        messages=[SystemMessage(content="system")],
        reserved_workspace=reservation,
    )
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert created.record.session_id == reservation.session_id
    assert created.workspace == reservation.workspace
    assert loaded.workspace == reservation.workspace
    assert marker.read_text(encoding="utf-8") == "prepared"


def test_load_session_rejects_cross_scope_access(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])

    with pytest.raises(SessionNotFoundError):
        store.load_session(scope_id="scope-b", session_id=created.record.session_id)


def test_commit_messages_appends_json_payloads_and_increments_version(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
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
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
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
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
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
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])

    with pytest.raises(SessionNotFoundError):
        store.rename_session(scope_id="scope-b", session_id=created.record.session_id, title="Bad")


def test_update_session_metadata_merges_without_changing_version(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
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
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
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
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    created = store.create_session(scope_id="scope-a", messages=[])
    created.workspace.rmdir()

    with pytest.raises(WorkspaceMissingError):
        store.load_session(scope_id="scope-a", session_id=created.record.session_id)


def test_workspace_paths_are_derived_from_session_id(tmp_path: Path) -> None:
    workspaces = WorkspacePaths(root=tmp_path / "sessions")

    paths = workspaces.create_for_session("sess_abc")

    assert paths.root == tmp_path / "sessions" / "sess_abc"
    assert paths.workspace == tmp_path / "sessions" / "sess_abc" / "workspace"
    assert paths.attachments == tmp_path / "sessions" / "sess_abc" / "attachments"
    assert workspaces.require_for_session("sess_abc") == paths
