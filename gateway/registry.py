"""Replica Registry for M2 Multi-Instance Routing.

Maintains per-replica state:
- endpoint & health status (with circuit breaker support)
- live Gateway-tracked `inflight` concurrency
- backend queue & capacity hints (`max_inflight`, `backend_waiting_queue`,
  `kv_cache_usage_perc`, `kv_cache_capacity_blocks`)
- observed real vLLM Prometheus metrics (`observed_vllm_prefix_cache_hit_rate`),
  strictly separated from the Router's estimated prefix affinity.
"""
from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional


@dataclass
class ReplicaConfig:
    """Static / initial configuration for a backend vLLM replica."""

    replica_id: str
    endpoint: str
    max_inflight: int = 4
    kv_cache_capacity_blocks: int = 1024


@dataclass
class ReplicaState:
    """Mutable runtime state and capacity hints for a single vLLM backend replica."""

    replica_id: str
    endpoint: str
    healthy: bool = True
    inflight: int = 0
    max_inflight: int = 4
    backend_waiting_queue: int = 0
    backend_running_requests: int = 0
    kv_cache_usage_perc: float = 0.0
    kv_cache_capacity_blocks: int = 1024

    # Circuit breaker / health tracking
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    last_health_check_at: Optional[float] = None
    last_metrics_sync_at: Optional[float] = None

    # Real observed vLLM prefix cache counters (populated ONLY from real vLLM /metrics,
    # NEVER from Gateway's estimated token-prefix affinity!)
    observed_vllm_prefix_cache_queries: Optional[float] = None
    observed_vllm_prefix_cache_hits: Optional[float] = None

    # Cumulative Gateway-side dispatch counters
    total_dispatched: int = 0
    total_completed: int = 0
    total_errors: int = 0
    max_observed_inflight: int = 0

    def is_available(self, now: Optional[float] = None) -> bool:
        """Return True if replica is healthy and its circuit breaker is closed."""
        ts = time.monotonic() if now is None else now
        return self.healthy and ts >= self.circuit_open_until

    def effective_queue_load(self) -> int:
        """Combined in-flight count at Gateway plus observed backend waiting queue."""
        return self.inflight + max(0, self.backend_waiting_queue)

    def load_ratio(self) -> float:
        """Normalized load ratio in [0.0, +inf) relative to `max_inflight` hint."""
        return self.effective_queue_load() / max(1, self.max_inflight)

    def observed_vllm_prefix_cache_hit_rate(self) -> Optional[float]:
        """Return real observed vLLM prefix cache hit rate, or None if unverified."""
        if (
            self.observed_vllm_prefix_cache_queries is None
            or self.observed_vllm_prefix_cache_hits is None
            or self.observed_vllm_prefix_cache_queries <= 0
        ):
            return None
        rate = (
            self.observed_vllm_prefix_cache_hits
            / self.observed_vllm_prefix_cache_queries
        )
        return round(max(0.0, min(1.0, rate)), 4)

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        ts = time.monotonic() if now is None else now
        return {
            "replica_id": self.replica_id,
            "endpoint": self.endpoint,
            "healthy": self.healthy,
            "available": self.is_available(ts),
            "inflight": self.inflight,
            "max_inflight": self.max_inflight,
            "backend_waiting_queue": self.backend_waiting_queue,
            "backend_running_requests": self.backend_running_requests,
            "kv_cache_usage_perc": round(self.kv_cache_usage_perc, 4),
            "kv_cache_capacity_blocks": self.kv_cache_capacity_blocks,
            "load_ratio": round(self.load_ratio(), 4),
            "consecutive_failures": self.consecutive_failures,
            "circuit_open": ts < self.circuit_open_until,
            "total_dispatched": self.total_dispatched,
            "total_completed": self.total_completed,
            "total_errors": self.total_errors,
            "max_observed_inflight": self.max_observed_inflight,
            "observed_vllm_metrics": {
                "synced": self.last_metrics_sync_at is not None,
                "prefix_cache_queries": self.observed_vllm_prefix_cache_queries,
                "prefix_cache_hits": self.observed_vllm_prefix_cache_hits,
                "prefix_cache_hit_rate": self.observed_vllm_prefix_cache_hit_rate(),
            },
        }


_PROM_METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_vllm_prometheus_metrics(metrics_text: str) -> Dict[str, float]:
    """Parse key vLLM Prometheus metrics from `/metrics` exposition text."""
    parsed: Dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_METRIC_RE.match(line)
        if not match:
            continue
        name, val_str = match.group(1), match.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue
        # Aggregate if multiple label series exist for the same metric name
        parsed[name] = parsed.get(name, 0.0) + val
    return parsed


class ReplicaRegistry:
    """Registry managing all backend vLLM replicas, health, load, and capacity hints."""

    def __init__(
        self,
        replicas: Optional[List[ReplicaConfig]] = None,
        *,
        circuit_breaker_failure_threshold: int = 3,
        circuit_breaker_cooldown_seconds: float = 5.0,
    ) -> None:
        self._replicas: Dict[str, ReplicaState] = {}
        self.circuit_breaker_failure_threshold = max(
            1, circuit_breaker_failure_threshold
        )
        self.circuit_breaker_cooldown_seconds = max(
            0.0, circuit_breaker_cooldown_seconds
        )
        if replicas:
            for r in replicas:
                self.register(
                    ReplicaState(
                        replica_id=r.replica_id,
                        endpoint=r.endpoint.rstrip("/"),
                        max_inflight=r.max_inflight,
                        kv_cache_capacity_blocks=r.kv_cache_capacity_blocks,
                    )
                )

    def register(self, state: ReplicaState) -> None:
        state.endpoint = state.endpoint.rstrip("/")
        self._replicas[state.replica_id] = state

    def unregister(self, replica_id: str) -> Optional[ReplicaState]:
        return self._replicas.pop(replica_id, None)

    def get(self, replica_id: str) -> Optional[ReplicaState]:
        return self._replicas.get(replica_id)

    def all_replicas(self) -> List[ReplicaState]:
        return list(self._replicas.values())

    def available_replicas(self, now: Optional[float] = None) -> List[ReplicaState]:
        ts = time.monotonic() if now is None else now
        return [r for r in self._replicas.values() if r.is_available(ts)]

    def total_max_inflight(self) -> int:
        return sum(r.max_inflight for r in self._replicas.values() if r.healthy)

    def set_health(self, replica_id: str, healthy: bool) -> None:
        replica = self._replicas.get(replica_id)
        if replica is None:
            raise KeyError(f"Unknown replica_id: {replica_id}")
        now = time.monotonic()
        replica.healthy = healthy
        replica.last_health_check_at = now
        if healthy:
            replica.consecutive_failures = 0
            replica.circuit_open_until = 0.0

    def update_capacity_hints(
        self,
        replica_id: str,
        *,
        max_inflight: Optional[int] = None,
        backend_waiting_queue: Optional[int] = None,
        backend_running_requests: Optional[int] = None,
        kv_cache_usage_perc: Optional[float] = None,
        kv_cache_capacity_blocks: Optional[int] = None,
    ) -> None:
        """Update runtime capacity/load hints for a replica."""
        replica = self._replicas.get(replica_id)
        if replica is None:
            raise KeyError(f"Unknown replica_id: {replica_id}")
        if max_inflight is not None:
            replica.max_inflight = max(1, int(max_inflight))
        if backend_waiting_queue is not None:
            replica.backend_waiting_queue = max(0, int(backend_waiting_queue))
        if backend_running_requests is not None:
            replica.backend_running_requests = max(0, int(backend_running_requests))
        if kv_cache_usage_perc is not None:
            replica.kv_cache_usage_perc = max(0.0, min(1.0, float(kv_cache_usage_perc)))
        if kv_cache_capacity_blocks is not None:
            replica.kv_cache_capacity_blocks = max(1, int(kv_cache_capacity_blocks))

    def update_from_vllm_metrics(
        self, replica_id: str, metrics_text: str
    ) -> Dict[str, Any]:
        """Sync real vLLM `/metrics` telemetry into ReplicaState.

        Explicitly populates `observed_vllm_prefix_cache_*` fields for later
        ground-truth verification against the Router's estimated affinity.
        """
        replica = self._replicas.get(replica_id)
        if replica is None:
            raise KeyError(f"Unknown replica_id: {replica_id}")
        parsed = parse_vllm_prometheus_metrics(metrics_text)
        now = time.monotonic()
        replica.last_metrics_sync_at = now

        if "vllm:num_requests_waiting" in parsed:
            replica.backend_waiting_queue = int(parsed["vllm:num_requests_waiting"])
        if "vllm:num_requests_running" in parsed:
            replica.backend_running_requests = int(parsed["vllm:num_requests_running"])

        for kv_key in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            if kv_key in parsed:
                replica.kv_cache_usage_perc = max(0.0, min(1.0, parsed[kv_key]))
                break

        for q_key in (
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_queries",
            "vllm:gpu_prefix_cache_queries_total",
            "vllm:gpu_prefix_cache_queries",
        ):
            if q_key in parsed:
                replica.observed_vllm_prefix_cache_queries = parsed[q_key]
                break

        for h_key in (
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_hits",
            "vllm:gpu_prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_hits",
        ):
            if h_key in parsed:
                replica.observed_vllm_prefix_cache_hits = parsed[h_key]
                break

        return replica.to_dict(now)

    def acquire_replica(self, replica_id: str) -> None:
        """Increment live in-flight counter when Gateway dispatches to a replica."""
        replica = self._replicas.get(replica_id)
        if replica is None:
            return
        replica.inflight += 1
        replica.total_dispatched += 1
        replica.max_observed_inflight = max(
            replica.max_observed_inflight, replica.inflight
        )

    def release_replica(self, replica_id: str, *, failed: bool = False) -> None:
        """Decrement live in-flight counter and update failure/circuit state."""
        replica = self._replicas.get(replica_id)
        if replica is None:
            return
        replica.inflight = max(0, replica.inflight - 1)
        if failed:
            replica.total_errors += 1
            replica.consecutive_failures += 1
            if (
                replica.consecutive_failures
                >= self.circuit_breaker_failure_threshold
                and self.circuit_breaker_cooldown_seconds > 0
            ):
                replica.circuit_open_until = (
                    time.monotonic() + self.circuit_breaker_cooldown_seconds
                )
        else:
            replica.total_completed += 1
            replica.consecutive_failures = 0

    def snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        replicas_list = [r.to_dict(now) for r in self._replicas.values()]
        return {
            "total_replicas": len(replicas_list),
            "available_replicas": sum(1 for r in replicas_list if r["available"]),
            "total_inflight": sum(r["inflight"] for r in replicas_list),
            "replicas": replicas_list,
        }
