import coding_assistant.trace
import pytest
from coding_assistant.trace import enable_tracing, trace_enabled, trace_data, get_default_trace_dir


@pytest.fixture(autouse=True)
def reset_tracing():
    # Global state reset
    coding_assistant.trace._trace_enabled = False


def test_tracing_toggle():
    assert not trace_enabled()
    enable_tracing()
    assert trace_enabled()


def test_trace_data_creates_file(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("CODING_ASSISTANT_TRACE_DIR", str(trace_dir))
    enable_tracing()

    trace_data("test.json", '{"key": "value"}')

    assert trace_dir.exists()
    assert get_default_trace_dir() == trace_dir

    # The file should have a timestamp prefix
    files = list(trace_dir.glob("*_test.json"))
    assert len(files) == 1
    assert files[0].read_text() == '{"key": "value"}'


def test_trace_data_disabled_does_nothing(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("CODING_ASSISTANT_TRACE_DIR", str(trace_dir))

    trace_data("test.json", '{"key": "value"}')

    assert not trace_dir.exists()


def test_trace_clear_directory(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("CODING_ASSISTANT_TRACE_DIR", str(trace_dir))
    trace_dir.mkdir(parents=True)
    (trace_dir / "old_trace.json").write_text("old content")

    # Enable tracing with clear=True
    enable_tracing(clear=True)

    assert not (trace_dir / "old_trace.json").exists()


def test_trace_without_clear_keeps_files(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("CODING_ASSISTANT_TRACE_DIR", str(trace_dir))
    trace_dir.mkdir(parents=True)
    (trace_dir / "old_trace.json").write_text("old content")

    # Enable tracing with clear=False
    enable_tracing(clear=False)

    assert (trace_dir / "old_trace.json").exists()
    assert (trace_dir / "old_trace.json").read_text() == "old content"
