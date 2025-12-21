from coding_assistant.callbacks import ParagraphBuffer


def test_paragraph_buffer_empty():
    buffer = ParagraphBuffer()
    assert buffer.push("") == []
    assert buffer.flush() is None


def test_paragraph_buffer_single_paragraph():
    buffer = ParagraphBuffer()
    assert buffer.push("Hello world") == []
    assert buffer.flush() == "Hello world"


def test_paragraph_buffer_double_newline():
    buffer = ParagraphBuffer()
    assert buffer.push("First paragraph\n\nSecond") == ["First paragraph"]
    assert buffer.push(" paragraph") == []
    assert buffer.flush() == "Second paragraph"


def test_paragraph_buffer_multiple_paragraphs():
    buffer = ParagraphBuffer()
    chunks = ["Para 1\n\nPara 2\n", "\nPara 3", "\n\nPara 4"]

    assert buffer.push(chunks[0]) == ["Para 1"]
    assert buffer.push(chunks[1]) == ["Para 2"]
    assert buffer.push(chunks[2]) == ["Para 3"]
    assert buffer.flush() == "Para 4"


def test_paragraph_buffer_strip():
    buffer = ParagraphBuffer()
    # first push returns the first paragraph
    assert buffer.push("  \n\n  Spaced Para  ") == ["  "]
    # flush returns the second paragraph stripped
    assert buffer.flush() == "Spaced Para"


def test_paragraph_buffer_split_newline():
    buffer = ParagraphBuffer()
    assert buffer.push("Para 1\n") == []
    assert buffer.push("\nPara 2") == ["Para 1"]
    assert buffer.flush() == "Para 2"


def test_paragraph_buffer_many_newlines():
    buffer = ParagraphBuffer()
    assert buffer.push("Para 1\n\n\nPara 2") == ["Para 1"]
    assert buffer.flush() == "Para 2"


def test_paragraph_buffer_code_fence():
    buffer = ParagraphBuffer()
    code_chunk = "Here is code:\n\n```python\ndef hello():\n\n    print('world')\n```"
    assert buffer.push(code_chunk) == ["Here is code:"]
    # The entire code fence should be in the buffer still,
    # because the push returned only the first paragraph and kept the rest.
    # Wait, the current logic would keep the code fence in buffer if it's not followed by \n\n.
    assert buffer.flush() == "```python\ndef hello():\n\n    print('world')\n```"


def test_paragraph_buffer_code_fence_split():
    buffer = ParagraphBuffer()
    # It should split because the code fence is closed
    assert buffer.push("```python\ncode\n```\n\nNext para") == ["```python\ncode\n```"]
    assert buffer.flush() == "Next para"


def test_paragraph_buffer_incomplete_code_fence():
    buffer = ParagraphBuffer()
    assert buffer.push("```python\n") == []
    assert buffer.push("\n\nin code\n\n") == []
    assert buffer.push("```") == []  # closed now
    assert buffer.push("\n\nAfter") == ["```python\n\n\nin code\n\n```"]
    assert buffer.flush() == "After"
