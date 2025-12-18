from unittest.mock import patch, call
from coding_assistant.callbacks import DenseProgressCallbacks, DenseProgressOutputType

def test_dense_callbacks_empty_line_logic():
    cb = DenseProgressCallbacks()
    
    with patch("coding_assistant.callbacks.print") as mock_print:
        # 1. Start agent
        cb.on_agent_start("TestAgent", "gpt-4")
        
        # 2. First reasoning chunk - should trigger _ensure_empty_line
        # _ensure_empty_line when last was 'agent' should just print one empty line (if not reasoning/content)
        # Actually my implementation of _ensure_empty_line:
        # def _ensure_empty_line(self):
        #     if self._last_output_type in ("reasoning", "content"):
        #         print()
        #     print()
        cb.on_reasoning_chunk("Thinking")
        
        # 3. Second reasoning chunk - should NOT trigger _ensure_empty_line
        cb.on_reasoning_chunk(" more")
        
        # 4. Content chunk - should trigger _ensure_empty_line
        cb.on_content_chunk("Hello")
        
        # 5. Tool start
        cb.on_tool_start("TestAgent", "call_1", "test_tool", {"arg": 1})
        
        # 6. Content chunk after tool - should trigger _ensure_empty_line
        cb.on_content_chunk("Result")

        # Capture calls to print
        # call() with no args is print() which is an empty line
        print_calls = [c for c in mock_print.call_args_list]
        
        # Check sequences
        # Initial agent start has a print() then agent info
        assert call() in print_calls
        
        # Reason check: should have triggered _ensure_empty_line
        # Since last was 'agent', _ensure_empty_line prints one print()
        # Then on_reasoning_chunk prints the chunk
        
        # Verify content block after reasoning block has an extra newline
        # 1. on_reasoning_chunk sets type to 'reasoning'
        # 2. on_content_chunk sees type is 'reasoning', calls _ensure_empty_line
        # 3. _ensure_empty_line sees type is 'reasoning', calls print() (to end line) AND print() (empty line)
        
        # Let's find the sequence where reasoning switched to content
        found_double_newline = False
        for i in range(len(print_calls) - 2):
            if print_calls[i] == call() and print_calls[i+1] == call():
                found_double_newline = True
                break
        assert found_double_newline, "Expected double newline when switching from reasoning to content"

if __name__ == "__main__":
    test_dense_callbacks_empty_line_logic()
    print("Test passed!")
