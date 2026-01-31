import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from pathlib import Path
import os

from coding_assistant.api.manager import SessionManager
from coding_assistant.api.models import AnswerResponse, ConfirmationResponse, Envelope, StartCommand, InterruptCommand

logger = logging.getLogger(__name__)


class SessionCreate(BaseModel):
    session_id: str
    working_directory: str


def create_app(session_manager: SessionManager) -> FastAPI:
    app = FastAPI(title="Coding Assistant API")

    @app.post("/sessions")
    async def create_session(data: SessionCreate) -> dict[str, str]:
        # In a real scenario,...
        return {"session_id": data.session_id, "status": "created"}

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()

        # In a real flow, the UI/Working Dir might come from the initial POST
        active = session_manager.create_session(
            session_id=session_id, websocket=websocket, working_directory=Path(os.getcwd())
        )

        try:
            while True:
                data = await websocket.receive_json()
                envelope = Envelope.model_validate(data)
                payload = envelope.payload

                if isinstance(payload, StartCommand):
                    if active.task and not active.task.done():
                        logger.warning(f"Task already running for session {session_id}")
                        continue

                    # Start the agent loop in the background
                    async def run_agent() -> None:
                        async with active.session:
                            await active.session.run_agent(task=payload.task)

                    active.task = asyncio.create_task(run_agent())

                elif isinstance(payload, (AnswerResponse, ConfirmationResponse)):
                    await active.response_queue.put(payload)

                elif isinstance(payload, InterruptCommand):
                    if active.task:
                        active.task.cancel()
                        logger.info(f"Task interrupted for session {session_id}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"Error in websocket loop: {e}")
        finally:
            await session_manager.cleanup_session(session_id)

    return app
