import coding_assistant.trace
import pytest
from coding_assistant.trace import enable_tracing, trace_enabled, trace_data, trace_json


@pytest.fixture(autouse=True)
def reset_tracing():
    coding_assistant.trace._trace_dir = None


def test_tracing_toggle(tmp_path):
    assert not trace_enabled()
    enable_tracing(tmp_path)
    assert trace_enabled()


def test_trace_data_creates_file(tmp_path):
    trace_dir = tmp_path / "traces"
    enable_tracing(trace_dir)

    trace_data("test.json", '{"key": "value"}')

    assert trace_dir.exists()

    files = list(trace_dir.glob("*_test.json"))
    assert len(files) == 1
    assert files[0].read_text() == '{"key": "value"}'


def test_trace_data_disabled_does_nothing():
    assert not trace_enabled()

    trace_data("test.json", '{"key": "value"}')
    # No way to check path easily if disabled, but shouldn't crash


def test_trace_clear_directory(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "old_trace.json").write_text("old content")

    enable_tracing(trace_dir, clear=True)

    assert not (trace_dir / "old_trace.json").exists()


def test_trace_without_clear_keeps_files(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "old_trace.json").write_text("old content")

    enable_tracing(trace_dir, clear=False)

    assert (trace_dir / "old_trace.json").exists()
    assert (trace_dir / "old_trace.json").read_text() == "old content"


def test_trace_json_creates_json5_file(tmp_path):
    trace_dir = tmp_path / "traces"
    enable_tracing(trace_dir)

    data = {"key": "value", "multi": "line\nstring"}
    trace_json("test.json", data)

    files = list(trace_dir.glob("*_test.json5"))
    assert len(files) == 1
    content = files[0].read_text()
    assert 'key: "value"' in content
    assert 'multi: "line\\\nstring"' in content
