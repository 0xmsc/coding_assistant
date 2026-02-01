import asyncio
import abc
import logging
from typing import Optional

from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ActorMessage

logger = logging.getLogger(__name__)


class BaseActor(abc.ABC):
    """
    Abstract base class for all actors in the system.
    Each actor runs its own message loop and processes envelopes from its mailbox.
    """

    def __init__(self, address: str):
        self.address = address
        self.mailbox: asyncio.Queue[Envelope[ActorMessage]] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the actor's message processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.debug(f"Actor {self.address} started.")

    async def stop(self) -> None:
        """Stop the actor's message processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug(f"Actor {self.address} stopped.")

    async def _run_loop(self) -> None:
        """Internal loop to process messages from the mailbox."""
        while self._running:
            try:
                envelope = await self.mailbox.get()
                logger.debug(
                    f"Actor {self.address} processing {type(envelope.payload).__name__} "
                    f"from {envelope.sender} [trace: {envelope.trace_id}, cid: {envelope.correlation_id}]"
                )
                await self.receive(envelope)
                self.mailbox.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in actor {self.address} while processing message: {e}")

    @abc.abstractmethod
    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        """
        Handle an incoming envelope.
        Must be implemented by subclasses.
        """
        pass

    async def tell(self, envelope: Envelope[ActorMessage]) -> None:
        """Put an envelope into the actor's mailbox."""
        await self.mailbox.put(envelope)
