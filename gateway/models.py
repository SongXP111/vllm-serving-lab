"""Request lifecycle state machine and context tracking for the Gateway."""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional


class RequestState(str, Enum):
    """Explicit 3-state lifecycle for an LLM inference request in the Gateway.

    1. NOT_STARTED: Request received by Gateway, waiting for backend connection
       or first response headers/chunk. HTTP status code has NOT been committed
       to the client yet (can still return standard HTTP 502/503/504 JSON error).
    2. STREAMING: HTTP 200 OK and `text/event-stream` headers have been committed
       to the client, and SSE chunks are actively being passthrough-streamed.
       Cannot change HTTP status code anymore; mid-stream errors must be emitted
       as SSE events.
    3. FINISHED: Request lifecycle has terminated (completed normally, failed,
       timed out, or aborted by client disconnect), and the upstream connection
       has been closed/returned to the pool.
    """

    NOT_STARTED = "not_started"
    STREAMING = "streaming"
    FINISHED = "finished"


@dataclass
class RequestContext:
    """Tracks the lifecycle, timing, and cleanup state of a single request."""

    request_id: str
    is_stream: bool
    state: RequestState = RequestState.NOT_STARTED
    created_at: float = field(default_factory=time.monotonic)
    first_byte_at: Optional[float] = None
    finished_at: Optional[float] = None
    chunks_forwarded: int = 0
    client_disconnected: bool = False
    upstream_closed: bool = False
    error: Optional[str] = None

    def mark_streaming(self) -> None:
        """Transition from NOT_STARTED -> STREAMING."""
        if self.state == RequestState.NOT_STARTED:
            self.state = RequestState.STREAMING
            if self.first_byte_at is None:
                self.first_byte_at = time.monotonic()

    def mark_finished(
        self,
        *,
        upstream_closed: bool = True,
        client_disconnected: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Transition from NOT_STARTED or STREAMING -> FINISHED."""
        self.state = RequestState.FINISHED
        self.upstream_closed = upstream_closed
        if client_disconnected:
            self.client_disconnected = True
        if error is not None:
            self.error = error
        if self.finished_at is None:
            self.finished_at = time.monotonic()
