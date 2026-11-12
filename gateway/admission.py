"""Admission Control, Bounded Queue, Fairness & Backpressure (M1 Step 2).

Implements the core Flow Control concepts inspired by `llm-d EPP`:
1. Saturation Detection (Work-conserving fast path when `in_flight < max_inflight`).
2. Flow Classification (`FlowKey = (tenant_id, priority)`).
3. Per-Tenant Quota & Token Bucket Rate Limiting (`429 Too Many Requests`).
4. Bounded Queue (`queue_capacity`) & Overload Backpressure (`503 Service Unavailable`).
5. Priority Bands (`HIGH` > `NORMAL` > `LOW`) with Anti-Starvation (Consecutive Cap + Aging).
6. Per-Tenant Round-Robin Fairness within each priority band (prevents Noisy Neighbor).
7. End-to-end Deadline & Client Disconnect cleanup in both Queued and In-Flight states.
8. First-class observability metrics: `queue_depth`, `queue_wait`, `reject`, and `cancel`.
"""
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
import time
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

from gateway.config import GatewayConfig
from gateway.models import RequestContext


class PriorityLevel(IntEnum):
    """Priority tiers ordered from highest (0) to lowest (2)."""

    HIGH = 0
    NORMAL = 1
    LOW = 2

    @classmethod
    def from_str(cls, raw: Optional[str]) -> "PriorityLevel":
        if not raw:
            return cls.NORMAL
        val = str(raw).strip().lower()
        if val in ("high", "realtime", "vip", "critical", "0"):
            return cls.HIGH
        if val in ("low", "batch", "background", "offline", "2"):
            return cls.LOW
        return cls.NORMAL

    def label(self) -> str:
        return self.name.lower()


class AdmissionRejectedError(Exception):
    """Raised when a request is rejected by Admission Control (429 / 503 / 504)."""

    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.retry_after = retry_after


class TokenBucket:
    """Per-tenant Token Bucket rate limiter."""

    def __init__(self, rate_rps: float, burst_capacity: float) -> None:
        self.rate_rps = rate_rps
        self.burst_capacity = max(1.0, burst_capacity)
        self.tokens = self.burst_capacity
        self.last_refill = time.monotonic()

    def try_consume(self, amount: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_refill)
        self.last_refill = now
        self.tokens = min(self.burst_capacity, self.tokens + elapsed * self.rate_rps)
        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        deficit = amount - self.tokens
        retry_after = deficit / self.rate_rps if self.rate_rps > 0 else 1.0
        return False, round(max(0.05, retry_after), 3)


@dataclass
class QueueEntry:
    """Represents a single waiting request inside the Gateway bounded queue."""

    ctx: RequestContext
    priority: PriorityLevel
    tenant_id: str
    enqueued_at: float = field(default_factory=time.monotonic)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    granted: bool = False
    cancelled: bool = False


@dataclass
class AdmissionStats:
    """Observability metrics for Admission Control & Bounded Queue verification."""

    total_arrived: int = 0
    total_admitted_immediately: int = 0
    total_queued: int = 0
    max_observed_inflight: int = 0
    max_observed_queue_len: int = 0
    rejected_400_invalid: int = 0
    rejected_429_quota: int = 0
    rejected_429_rate_limit: int = 0
    rejected_503_queue_full: int = 0
    rejected_503_queue_timeout: int = 0
    rejected_504_deadline_exceeded: int = 0
    cancelled_in_queue: int = 0
    cancelled_in_flight: int = 0
    queue_wait_samples_ms: List[float] = field(default_factory=list)
    dispatched_by_priority: Dict[str, int] = field(
        default_factory=lambda: {"high": 0, "normal": 0, "low": 0}
    )
    dispatched_by_tenant: Dict[str, int] = field(default_factory=dict)
    dispatch_order: List[Dict[str, Any]] = field(default_factory=list)


class AdmissionController:
    """Manages in-flight concurrency, bounded priority/fairness queues, and quotas."""

    def __init__(self, config: GatewayConfig) -> None:
        self.cfg = config
        self.in_flight: int = 0
        self.queued_count: int = 0
        self.tenant_active: Dict[str, int] = defaultdict(int)
        self.tenant_buckets: Dict[str, TokenBucket] = {}

        # Per-priority -> Per-tenant FIFO queue + Round-Robin tenant rotation ring
        self.priority_queues: Dict[PriorityLevel, Dict[str, Deque[QueueEntry]]] = {
            p: defaultdict(deque) for p in PriorityLevel
        }
        self.tenant_rr_order: Dict[PriorityLevel, Deque[str]] = {
            p: deque() for p in PriorityLevel
        }

        # Anti-starvation counters across priority bands
        self.consecutive_high_dispatches: int = 0
        self.consecutive_normal_dispatches: int = 0

        self.stats = AdmissionStats()

    def record_invalid_request(self) -> None:
        """Record an invalid client request rejected before entering admission queue."""
        self.stats.rejected_400_invalid += 1

    def record_inflight_cancel(self) -> None:
        """Record a request cancelled while executing in-flight."""
        self.stats.cancelled_in_flight += 1

    def _get_token_bucket(self, tenant_id: str) -> TokenBucket:
        bucket = self.tenant_buckets.get(tenant_id)
        if bucket is None:
            bucket = TokenBucket(
                rate_rps=self.cfg.tenant_rate_limit_rps,
                burst_capacity=self.cfg.tenant_burst_capacity,
            )
            self.tenant_buckets[tenant_id] = bucket
        return bucket

    def _record_dispatch(self, ctx: RequestContext) -> None:
        prio = ctx.priority
        self.stats.queue_wait_samples_ms.append(ctx.queue_wait_ms)
        self.stats.dispatched_by_priority[prio] = (
            self.stats.dispatched_by_priority.get(prio, 0) + 1
        )
        self.stats.dispatched_by_tenant[ctx.tenant_id] = (
            self.stats.dispatched_by_tenant.get(ctx.tenant_id, 0) + 1
        )
        self.stats.dispatch_order.append(
            {
                "request_id": ctx.request_id,
                "tenant_id": ctx.tenant_id,
                "priority": prio,
                "was_queued": ctx.was_queued,
                "queue_wait_ms": ctx.queue_wait_ms,
            }
        )

    def _band_has_entries(self, prio: PriorityLevel) -> bool:
        rr = self.tenant_rr_order[prio]
        queues = self.priority_queues[prio]
        while rr:
            t_id = rr[0]
            q = queues.get(t_id)
            while q and (q[0].cancelled or q[0].granted):
                q.popleft()
            if q:
                return True
            rr.popleft()
            queues.pop(t_id, None)
        return False

    def _oldest_wait_in_band(self, prio: PriorityLevel, now: float) -> float:
        if not self._band_has_entries(prio):
            return 0.0
        oldest = 0.0
        for t_id in self.tenant_rr_order[prio]:
            q = self.priority_queues[prio].get(t_id)
            if q:
                oldest = max(oldest, now - q[0].enqueued_at)
        return oldest

    def _select_next_band(self) -> Optional[PriorityLevel]:
        non_empty = [p for p in PriorityLevel if self._band_has_entries(p)]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            chosen = non_empty[0]
            if chosen != PriorityLevel.HIGH:
                self.consecutive_high_dispatches = 0
            return chosen

        now = time.monotonic()
        # 1. Aging check: if LOW or NORMAL has waited >= starvation_aging_seconds,
        # promote it immediately to prevent starvation.
        if self.cfg.starvation_aging_seconds > 0:
            for lower_p in (PriorityLevel.LOW, PriorityLevel.NORMAL):
                if lower_p in non_empty:
                    if (
                        self._oldest_wait_in_band(lower_p, now)
                        >= self.cfg.starvation_aging_seconds
                    ):
                        self.consecutive_high_dispatches = 0
                        if lower_p == PriorityLevel.LOW:
                            self.consecutive_normal_dispatches = 0
                        return lower_p

        # 2. Weighted Fair / Consecutive Cap across priority bands:
        # Allow at most `max_consecutive_high_priority` HIGH dispatches while
        # lower-priority queues are waiting, then force 1 turn for lower priority.
        if PriorityLevel.HIGH in non_empty:
            max_high = max(1, self.cfg.max_consecutive_high_priority)
            if self.consecutive_high_dispatches < max_high:
                self.consecutive_high_dispatches += 1
                return PriorityLevel.HIGH
            # Grant a fair anti-starvation turn to NORMAL or LOW
            self.consecutive_high_dispatches = 0
            lower_bands = [p for p in non_empty if p != PriorityLevel.HIGH]
            if len(lower_bands) == 1:
                return lower_bands[0]
            if self.consecutive_normal_dispatches < 2:
                self.consecutive_normal_dispatches += 1
                return PriorityLevel.NORMAL
            self.consecutive_normal_dispatches = 0
            return PriorityLevel.LOW

        # Only NORMAL and LOW are non-empty
        if self.consecutive_normal_dispatches < 2:
            self.consecutive_normal_dispatches += 1
            return PriorityLevel.NORMAL
        self.consecutive_normal_dispatches = 0
        return PriorityLevel.LOW

    def _pop_next_entry(self) -> Optional[QueueEntry]:
        chosen_band = self._select_next_band()
        if chosen_band is None:
            return None

        rr = self.tenant_rr_order[chosen_band]
        queues = self.priority_queues[chosen_band]
        while rr:
            tenant_id = rr.popleft()
            q = queues.get(tenant_id)
            while q and (q[0].cancelled or q[0].granted):
                q.popleft()
            if not q:
                queues.pop(tenant_id, None)
                continue

            entry = q.popleft()
            # Clean any trailing cancelled entries for this tenant
            while q and (q[0].cancelled or q[0].granted):
                q.popleft()
            if q:
                # Put tenant at the back of the Round-Robin ring for next turn
                rr.append(tenant_id)
            else:
                queues.pop(tenant_id, None)

            self.queued_count = max(0, self.queued_count - 1)
            return entry

        return None

    def _remove_queued_entry(self, entry: QueueEntry) -> None:
        if entry.cancelled or entry.granted:
            return
        entry.cancelled = True
        q = self.priority_queues[entry.priority].get(entry.tenant_id)
        if q:
            try:
                q.remove(entry)
            except ValueError:
                pass
            if not q:
                self.priority_queues[entry.priority].pop(entry.tenant_id, None)
                try:
                    self.tenant_rr_order[entry.priority].remove(entry.tenant_id)
                except ValueError:
                    pass
        self.queued_count = max(0, self.queued_count - 1)

    async def acquire_slot(
        self,
        ctx: RequestContext,
        is_disconnected_cb: Callable[[], Awaitable[bool]],
    ) -> None:
        """Decide whether to Accept immediately, Queue, or Reject (429/503/504)."""
        self.stats.total_arrived += 1
        tenant_id = ctx.tenant_id
        prio = PriorityLevel.from_str(ctx.priority)
        ctx.priority = prio.label()

        # 0. Pre-check request deadline
        rem_deadline = ctx.remaining_deadline()
        if rem_deadline is not None and rem_deadline <= 0:
            self.stats.rejected_504_deadline_exceeded += 1
            raise AdmissionRejectedError(
                504,
                "deadline_exceeded",
                "Request deadline already expired before admission",
            )

        # 1. Per-tenant Rate Limit check -> HTTP 429
        if self.cfg.tenant_rate_limit_rps > 0:
            bucket = self._get_token_bucket(tenant_id)
            allowed, retry_after = bucket.try_consume(1.0)
            if not allowed:
                self.stats.rejected_429_rate_limit += 1
                raise AdmissionRejectedError(
                    429,
                    "rate_limit_exceeded",
                    f"Tenant '{tenant_id}' rate limit exceeded ({self.cfg.tenant_rate_limit_rps} RPS)",
                    retry_after=retry_after,
                )

        # 2. Per-tenant Concurrency Quota check -> HTTP 429
        tenant_limit = self.cfg.tenant_quotas.get(
            tenant_id, self.cfg.tenant_max_concurrency
        )
        if tenant_limit > 0 and self.tenant_active[tenant_id] >= tenant_limit:
            self.stats.rejected_429_quota += 1
            raise AdmissionRejectedError(
                429,
                "tenant_quota_exceeded",
                f"Tenant '{tenant_id}' active request quota exceeded ({tenant_limit})",
                retry_after=0.5,
            )

        # 3. Work-conserving Fast Path: Backend is NOT saturated and queue is empty
        if self.in_flight < self.cfg.max_inflight and self.queued_count == 0:
            self.in_flight += 1
            self.tenant_active[tenant_id] += 1
            self.stats.total_admitted_immediately += 1
            self.stats.max_observed_inflight = max(
                self.stats.max_observed_inflight, self.in_flight
            )
            ctx.mark_admitted(was_queued=False)
            self._record_dispatch(ctx)
            return

        # 4. Backend is saturated: Check Bounded Queue capacity -> HTTP 503
        if self.queued_count >= self.cfg.queue_capacity:
            self.stats.rejected_503_queue_full += 1
            raise AdmissionRejectedError(
                503,
                "server_overloaded_queue_full",
                (
                    f"System overloaded: bounded queue is full "
                    f"(queued={self.queued_count}/{self.cfg.queue_capacity}, "
                    f"in_flight={self.in_flight}/{self.cfg.max_inflight})"
                ),
                retry_after=1.0,
            )

        # 5. Enqueue into Priority + Per-Tenant Fair Queue
        entry = QueueEntry(ctx=ctx, priority=prio, tenant_id=tenant_id)
        ctx.enqueued_at = entry.enqueued_at
        t_queue = self.priority_queues[prio][tenant_id]
        if not t_queue and tenant_id not in self.tenant_rr_order[prio]:
            self.tenant_rr_order[prio].append(tenant_id)
        t_queue.append(entry)

        self.queued_count += 1
        self.tenant_active[tenant_id] += 1
        self.stats.total_queued += 1
        self.stats.max_observed_queue_len = max(
            self.stats.max_observed_queue_len, self.queued_count
        )

        # 6. Wait in queue with deadline, queue_timeout, and client disconnect watch
        wait_limit = self.cfg.queue_timeout
        deadline_limited = False
        if rem_deadline is not None and rem_deadline < wait_limit:
            wait_limit = max(0.0, rem_deadline)
            deadline_limited = True

        expire_at = entry.enqueued_at + wait_limit

        try:
            while not entry.event.is_set():
                if await is_disconnected_cb():
                    self.stats.cancelled_in_queue += 1
                    if entry.granted:
                        ctx.slot_acquired = True
                        self.release_slot(ctx)
                    else:
                        self._remove_queued_entry(entry)
                        self.tenant_active[tenant_id] = max(
                            0, self.tenant_active[tenant_id] - 1
                        )
                    raise AdmissionRejectedError(
                        499,
                        "client_closed_request",
                        "Client disconnected while waiting in Gateway queue",
                    )
                now = time.monotonic()
                remaining = expire_at - now
                if remaining <= 0:
                    if entry.event.is_set():
                        break
                    # Queue wait timed out or deadline expired!
                    self._remove_queued_entry(entry)
                    self.tenant_active[tenant_id] = max(
                        0, self.tenant_active[tenant_id] - 1
                    )
                    if deadline_limited:
                        self.stats.rejected_504_deadline_exceeded += 1
                        raise AdmissionRejectedError(
                            504,
                            "deadline_exceeded",
                            f"Request deadline expired while waiting in queue ({wait_limit:.3f}s)",
                        )
                    self.stats.rejected_503_queue_timeout += 1
                    raise AdmissionRejectedError(
                        503,
                        "queue_timeout_exceeded",
                        f"Timed out waiting in Gateway bounded queue after {wait_limit:.3f}s",
                        retry_after=1.0,
                    )

                try:
                    await asyncio.wait_for(
                        entry.event.wait(), timeout=min(0.015, remaining)
                    )
                except asyncio.TimeoutError:
                    continue

        except asyncio.CancelledError:
            self.stats.cancelled_in_queue += 1
            if entry.granted:
                # A slot was just handed to this entry as it got cancelled;
                # mark acquired and immediately release it to the next waiter!
                ctx.slot_acquired = True
                self.release_slot(ctx)
            else:
                self._remove_queued_entry(entry)
                self.tenant_active[tenant_id] = max(
                    0, self.tenant_active[tenant_id] - 1
                )
            raise

    def release_slot(self, ctx: RequestContext) -> None:
        """Idempotently release an in-flight slot and wake up the next queued request."""
        if not ctx.slot_acquired or ctx.slot_released:
            return
        ctx.slot_released = True
        self.tenant_active[ctx.tenant_id] = max(
            0, self.tenant_active[ctx.tenant_id] - 1
        )

        # Direct handoff: if another request is waiting in the bounded queue,
        # transfer this in-flight slot directly to it (prevents barging).
        while True:
            next_entry = self._pop_next_entry()
            if next_entry is None:
                self.in_flight = max(0, self.in_flight - 1)
                return
            if not next_entry.cancelled:
                next_entry.granted = True
                next_entry.ctx.mark_admitted(was_queued=True)
                self._record_dispatch(next_entry.ctx)
                next_entry.event.set()
                return

    def snapshot(self) -> Dict[str, Any]:
        """Return live state + first-class queue_depth/queue_wait/reject/cancel metrics."""
        samples = self.stats.queue_wait_samples_ms
        queued_samples = [w for w in samples if w > 0.0]
        avg_wait = (
            round(sum(queued_samples) / len(queued_samples), 3)
            if queued_samples
            else 0.0
        )
        max_wait = round(max(samples), 3) if samples else 0.0

        reject_total = (
            self.stats.rejected_400_invalid
            + self.stats.rejected_429_quota
            + self.stats.rejected_429_rate_limit
            + self.stats.rejected_503_queue_full
            + self.stats.rejected_503_queue_timeout
            + self.stats.rejected_504_deadline_exceeded
        )
        cancel_total = self.stats.cancelled_in_queue + self.stats.cancelled_in_flight

        return {
            "in_flight": self.in_flight,
            "max_inflight": self.cfg.max_inflight,
            "queued_count": self.queued_count,
            "queue_depth": self.queued_count,
            "queue_capacity": self.cfg.queue_capacity,
            "tenant_active": dict(self.tenant_active),
            "metrics": {
                "queue_depth": {
                    "current": self.queued_count,
                    "max_observed": self.stats.max_observed_queue_len,
                    "capacity": self.cfg.queue_capacity,
                },
                "queue_wait": {
                    "admitted_count": len(samples),
                    "queued_admitted_count": len(queued_samples),
                    "avg_queued_wait_ms": avg_wait,
                    "max_wait_ms": max_wait,
                },
                "reject": {
                    "total": reject_total,
                    "invalid_400": self.stats.rejected_400_invalid,
                    "quota_429": self.stats.rejected_429_quota,
                    "rate_limit_429": self.stats.rejected_429_rate_limit,
                    "queue_full_503": self.stats.rejected_503_queue_full,
                    "queue_timeout_503": self.stats.rejected_503_queue_timeout,
                    "deadline_504": self.stats.rejected_504_deadline_exceeded,
                },
                "cancel": {
                    "total": cancel_total,
                    "in_queue": self.stats.cancelled_in_queue,
                    "in_flight": self.stats.cancelled_in_flight,
                },
            },
            "stats": {
                "total_arrived": self.stats.total_arrived,
                "total_admitted_immediately": self.stats.total_admitted_immediately,
                "total_queued": self.stats.total_queued,
                "max_observed_inflight": self.stats.max_observed_inflight,
                "max_observed_queue_len": self.stats.max_observed_queue_len,
                "rejected_400_invalid": self.stats.rejected_400_invalid,
                "rejected_429_quota": self.stats.rejected_429_quota,
                "rejected_429_rate_limit": self.stats.rejected_429_rate_limit,
                "rejected_503_queue_full": self.stats.rejected_503_queue_full,
                "rejected_503_queue_timeout": self.stats.rejected_503_queue_timeout,
                "rejected_504_deadline_exceeded": self.stats.rejected_504_deadline_exceeded,
                "cancelled_in_queue": self.stats.cancelled_in_queue,
                "cancelled_in_flight": self.stats.cancelled_in_flight,
                "dispatched_by_priority": dict(self.stats.dispatched_by_priority),
                "dispatched_by_tenant": dict(self.stats.dispatched_by_tenant),
                "dispatch_order": list(self.stats.dispatch_order),
            },
        }
