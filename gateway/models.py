"""Request lifecycle state machine and context tracking for the Gateway."""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional


class RequestState(str, Enum):
    """Explicit 3-state lifecycle for an LLM inference request in the Gateway.

    1. NOT_STARTED: Request received by Gateway (waiting in admission queue or
       waiting for backend connection / first response chunk). HTTP status code
       has NOT been committed to the client yet (can still return standard HTTP
       429/502/503/504 JSON error).
    2. STREAMING: HTTP 200 OK and `text/event-stream` headers have been committed
       to the client, and SSE chunks are actively being passthrough-streamed.
       Cannot change HTTP status code anymore; mid-stream errors must be emitted
       as SSE events.
    3. FINISHED: Request lifecycle has terminated (completed normally, failed,
       timed out, rejected by admission control, or aborted by client disconnect),
       and both the admission slot and upstream connection have been released.
    """

    NOT_STARTED = "not_started"
    STREAMING = "streaming"
    FINISHED = "finished"


@dataclass
class RequestContext:
    """Tracks the lifecycle, admission queueing, timing, and cleanup of a request."""

    request_id: str
    is_stream: bool
    tenant_id: str = "default"
    priority: str = "normal"
    state: RequestState = RequestState.NOT_STARTED
    created_at: float = field(default_factory=time.monotonic)
    deadline_at: Optional[float] = None
    enqueued_at: Optional[float] = None
    admitted_at: Optional[float] = None
    first_byte_at: Optional[float] = None
    finished_at: Optional[float] = None
    queue_wait_ms: float = 0.0
    was_queued: bool = False
    slot_acquired: bool = False
    slot_released: bool = False
    chunks_forwarded: int = 0
    client_disconnected: bool = False
    upstream_closed: bool = False
    error: Optional[str] = None

    def remaining_deadline(self) -> Optional[float]:
        """Return remaining seconds until request deadline, or None if no deadline."""
        if self.deadline_at is None:
            return None
        return self.deadline_at - time.monotonic()

    def mark_admitted(self, *, was_queued: bool) -> None:
        """Record that the request acquired an in-flight backend slot."""
        now = time.monotonic()
        self.admitted_at = now
        self.was_queued = was_queued
        self.slot_acquired = True
        if was_queued and self.enqueued_at is not None:
            self.queue_wait_ms = round((now - self.enqueued_at) * 1000.0, 3)
        else:
            self.queue_wait_ms = 0.0

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
