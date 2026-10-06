"""In-process event bus, for pushing live changes to the UI.

Each subscriber gets its own bounded queue: the proxy must never wait on a browser tab,
nor grow memory without bound for one. A subscriber that falls a whole queue behind is
cut off rather than silently losing events — a lost "hold" would leave a call waiting
that no page shows. Its stream ends, and the client reconnects to a fresh snapshot.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator


#: The last event a subscriber that fell too far behind receives: its stream ends there.
OVERFLOWED = {"type": "overflowed"}


class EventBus:
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self.closed = False

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)

    def publish(self, kind: str, **payload: Any) -> None:
        """Deliver an event to every subscriber. Must be called on the event loop."""
        self._put({"type": kind, "at": time.time(), **payload})

    def close(self) -> None:
        """End every subscription. A subscriber reads `None`, and should stop.

        An event stream never ends by itself, and a server waits for open responses before
        it stops — so without this, one open browser tab keeps the proxy from exiting.
        """
        self.closed = True
        self._put(None)

    def _put(self, event: dict[str, Any] | None) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                _drain(queue)
                if event is not None:
                    self._subscribers.discard(queue)
                    queue.put_nowait(OVERFLOWED)
                    continue
            queue.put_nowait(event)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any] | None]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=self.maxsize)
        if self.closed:
            queue.put_nowait(None)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)


def _drain(queue: asyncio.Queue[dict[str, Any] | None]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
