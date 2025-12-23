import os
import pytest
from coding_assistant.llm.litellm import complete as litellm_complete
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import UserMessage, BaseMessage
from coding_assistant.framework.callbacks import ProgressCallbacks

class _RealCB(ProgressCallbacks):
    def __init__(self):
        self.content = ""
        self.done = False
    def on_content_chunk(self, chunk: str): self.content += chunk
    def on_reasoning_chunk(self, chunk: str): pass
    def on_chunks_end(self): self.done = True

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
async def test_litellm_openrouter_integration():
    model = "openrouter/x-ai/grok-code-fast-1"
    messages = [UserMessage(content="Respond with the word 'HELLO' and nothing else.")]
    cb = _RealCB()
    
    completion = await litellm_complete(messages, model=model, tools=[], callbacks=cb)
    
    assert "HELLO" in completion.message.content.upper()
    assert cb.done

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
async def test_openai_openrouter_integration():
    # Note: our openai adapter strips the 'openrouter/' prefix or uses the model as is.
    # OpenRouter models usually don't need the prefix if the base_url is set.
    model = "x-ai/grok-code-fast-1"
    messages = [UserMessage(content="Respond with the word 'WORLD' and nothing else.")]
    cb = _RealCB()
    
    completion = await openai_complete(messages, model=model, tools=[], callbacks=cb)
    
    assert "WORLD" in completion.message.content.upper()
    assert cb.done
