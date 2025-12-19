import pytest
from coding_assistant.trace import enable_tracing, trace_enabled, trace_data


@pytest.fixture(autouse=True)
def reset_tracing():
    # Global state reset
    import coding_assistant.trace

    coding_assistant.trace._trace_enabled = False


def test_tracing_toggle():
    assert not trace_enabled()
    enable_tracing()
    assert trace_enabled()


def test_trace_data_creates_file(tmp_path, monkeypatch):
    enable_tracing()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    trace_data("test.json", '{"key": "value"}')

    trace_path = tmp_path / "coding_assistant" / "traces"
    assert trace_path.exists()

    files = list(trace_path.glob("*_test.json"))
    assert len(files) == 1
    assert files[0].read_text() == '{"key": "value"}'


def test_trace_data_disabled_does_nothing(tmp_path, monkeypatch):
    # Ensure disabled
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    trace_data("test.json", '{"key": "value"}')

    trace_path = tmp_path / "coding_assistant" / "traces"
    assert not trace_path.exists()
