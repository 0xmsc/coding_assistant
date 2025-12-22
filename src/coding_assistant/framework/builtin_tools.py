from pydantic import BaseModel, Field
from coding_assistant.framework.types import Tool
from coding_assistant.framework.results import FinishTaskResult, CompactConversationResult


class FinishTaskSchema(BaseModel):
    result: str = Field(
        description="The result of the work on the task. The work of the agent is evaluated based on this result."
    )
    summary: str = Field(
        description="A concise summary of the conversation the agent and the client had. The summary must be a single paragraph, without line breaks. There should be enough context such that the work could be continued based on this summary. It should be possible to evaluate your result using only your input parameters and this summary.",
    )


class FinishTaskTool(Tool):
    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Signals that the assigned task is complete. This tool must be called eventually to terminate the agent's execution loop. This tool shall not be called when there are still open questions for the client."

    def parameters(self) -> dict:
        return FinishTaskSchema.model_json_schema()

    async def execute(self, parameters) -> FinishTaskResult:
        return FinishTaskResult(
            result=parameters["result"],
            summary=parameters["summary"],
        )


class CompactConversationSchema(BaseModel):
    summary: str = Field(description="A summary of the conversation so far.")


class CompactConversationTool(Tool):
    def name(self) -> str:
        return "compact_conversation"

    def description(self) -> str:
        return "Give the framework a summary of your conversation with the client so far. The work should be continuable based on this summary. This means that you need to include all the results you have already gathered so far. Additionally, you should include the next steps you had planned. When the user tells you to call this tool, you must do so immediately! This can also be called when you have reached a milestone and you think that the context you've gathered so far will not be relevant anymore going forward."

    def parameters(self) -> dict:
        return CompactConversationSchema.model_json_schema()

    async def execute(self, parameters) -> CompactConversationResult:
        return CompactConversationResult(summary=parameters["summary"])
