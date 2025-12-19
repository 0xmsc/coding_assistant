from prompt_toolkit.document import Document
from coding_assistant.ui import SlashCompleter


def test_slash_completer():
    words = ["/exit", "/compact", "/clear"]
    completer = SlashCompleter(words)

    # Test at start of line
    doc = Document("/", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    assert [c.text for c in completions] == ["/exit", "/compact", "/clear"]

    doc = Document("/e", cursor_position=2)
    completions = list(completer.get_completions(doc, None))
    assert [c.text for c in completions] == ["/exit"]

    # Test NOT at start of line
    doc = Document(" /", cursor_position=2)
    completions = list(completer.get_completions(doc, None))
    assert [c.text for c in completions] == []

    doc = Document("Hello /e", cursor_position=8)
    completions = list(completer.get_completions(doc, None))
    assert [c.text for c in completions] == []

    # Test without slash
    doc = Document("e", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    assert [c.text for c in completions] == []
