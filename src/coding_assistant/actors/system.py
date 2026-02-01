import asyncio
import logging

from coding_assistant.actors.base import BaseActor
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ActorMessage

logger = logging.getLogger(__name__)


class ActorSystem:
    """
    Central registry and dispatcher for actors.
    Handles routing missions between local actors.
    """

    def __init__(self) -> None:
        self._actors: dict[str, BaseActor] = {}
        self._futures: dict[str, asyncio.Future[Envelope[ActorMessage]]] = {}

    def register(self, actor: BaseActor) -> None:
        """Register a new actor with the system."""
        if actor.address in self._actors:
            raise ValueError(f"Actor with address {actor.address} already registered.")
        self._actors[actor.address] = actor
        logger.debug(f"Registered actor: {actor.address}")

    def unregister(self, address: str) -> None:
        """Unregister an actor."""
        if address in self._actors:
            del self._actors[address]
            logger.debug(f"Unregistered actor: {address}")

    async def send(self, envelope: Envelope[ActorMessage]) -> None:
        """Dispatch an envelope to the recipient actor."""
        target = envelope.recipient
        logger.info(
            f"System routing {type(envelope.payload).__name__} from {envelope.sender} to {target} "
            f"[trace: {envelope.trace_id}, cid: {envelope.correlation_id}]"
        )

        if target in self._actors:
            await self._actors[target].tell(envelope)

        # Reverting to explicit dispatch and letting receivers update futures?
        # No, let's just make the system registry smarter.

        if target not in self._actors:
            # If it's not an actor, check if it's a response to an 'ask'
            if envelope.correlation_id in self._futures:
                future = self._futures.pop(envelope.correlation_id)
                if not future.done():
                    future.set_result(envelope)
                return
            logger.warning(f"Message sent to unknown recipient: {target}")

    async def ask(self, envelope: Envelope[ActorMessage], timeout: float = 30.0) -> Envelope[ActorMessage]:
        """
        Send a message and wait for a response with the same correlation_id.
        """
        future = asyncio.get_running_loop().create_future()
        self._futures[envelope.correlation_id] = future

        await self.send(envelope)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            if envelope.correlation_id in self._futures:
                del self._futures[envelope.correlation_id]
            raise TimeoutError(f"Actor 'ask' timed out after {timeout}s for cid {envelope.correlation_id}")

    async def shutdown(self) -> None:
        """Stop all registered actors."""
        for actor in self._actors.values():
            await actor.stop()
        self._actors.clear()
        self._futures.clear()
