"""Gateway configuration for M1 (Async Proxy + Admission Control) & M2 (Multi-Replica Routing)."""
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional

from gateway.registry import ReplicaConfig


@dataclass
class GatewayConfig:
    """Configuration for the async LLM Serving Gateway.

    Step 1 - Timeout & Connection Pool dimensions (mapped to LLM inference physics):
    - connect_timeout: TCP handshake timeout with backend vLLM.
    - write_timeout: Timeout for sending prompt payload to backend.
    - read_timeout: Max silence interval between consecutive chunks
      (bounds TTFT before first token, and bounds TPOT/jitter during streaming).
    - pool_timeout: Max wait time to acquire a connection from the httpx pool.

    Step 2 - Admission Control, Bounded Queue, Fairness & Backpressure:
    - max_inflight: Global max concurrent requests allowed to execute on backends at once.
    - queue_capacity: Max waiting requests held in the Gateway bounded queue when
      saturated (`in_flight >= max_inflight`). Requests beyond this return HTTP 503.
    - queue_timeout: Max seconds a request may wait in the bounded queue before 503.
    - default_deadline: Optional end-to-end request deadline (seconds).
    - tenant_max_concurrency / tenant_quotas: Per-tenant active request quota (429).
    - tenant_rate_limit_rps / tenant_burst_capacity: Per-tenant Token Bucket (429).
    - max_consecutive_high_priority / starvation_aging_seconds: Anti-starvation guards.

    M2 - Multi-Replica Registry & Routing Policy (`Filter -> Score -> Pick`):
    - replicas: Optional explicit list of `ReplicaConfig`. If omitted, resolves from
      `GATEWAY_BACKEND_URLS` (comma-separated) or falls back to single `backend_url`.
    - routing_policy: One of `random`, `least-load`, `prefix-aware`, `cache+load-aware`.
    - prefix_block_size: Pseudo-token window size matching vLLM APC block_size (default 16).
    - prefix_weight / load_weight: Scoring weights for `cache+load-aware`.
    - hotspot_load_threshold / kv_hotspot_threshold: Thresholds where `cache+load-aware`
      attenuates prefix affinity to prevent hotspot saturation on one warm replica.
    - kv_overload_threshold: Hard KV usage threshold for `CapacityFilter`.
    - circuit_breaker_failure_threshold / circuit_breaker_cooldown_seconds: Health circuit breaker.
    """

    backend_url: str = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8000")
    connect_timeout: float = float(os.environ.get("GATEWAY_CONNECT_TIMEOUT", "3.0"))
    read_timeout: float = float(os.environ.get("GATEWAY_READ_TIMEOUT", "30.0"))
    write_timeout: float = float(os.environ.get("GATEWAY_WRITE_TIMEOUT", "5.0"))
    pool_timeout: float = float(os.environ.get("GATEWAY_POOL_TIMEOUT", "3.0"))
    max_connections: int = int(os.environ.get("GATEWAY_MAX_CONNECTIONS", "100"))
    max_keepalive_connections: int = int(
        os.environ.get("GATEWAY_MAX_KEEPALIVE_CONNECTIONS", "20")
    )
    api_key: str = os.environ.get("VLLM_API_KEY", "")

    # M1 Step 2: Admission Control & Flow Control settings
    max_inflight: int = int(os.environ.get("GATEWAY_MAX_INFLIGHT", "4"))
    queue_capacity: int = int(os.environ.get("GATEWAY_QUEUE_CAPACITY", "20"))
    queue_timeout: float = float(os.environ.get("GATEWAY_QUEUE_TIMEOUT", "15.0"))
    default_deadline: float = float(os.environ.get("GATEWAY_DEFAULT_DEADLINE", "0.0"))
    tenant_max_concurrency: int = int(
        os.environ.get("GATEWAY_TENANT_MAX_CONCURRENCY", "0")
    )
    tenant_quotas: Dict[str, int] = field(default_factory=dict)
    tenant_rate_limit_rps: float = float(
        os.environ.get("GATEWAY_TENANT_RATE_LIMIT_RPS", "0.0")
    )
    tenant_burst_capacity: float = float(
        os.environ.get("GATEWAY_TENANT_BURST_CAPACITY", "10.0")
    )
    max_consecutive_high_priority: int = int(
        os.environ.get("GATEWAY_MAX_CONSECUTIVE_HIGH", "3")
    )
    starvation_aging_seconds: float = float(
        os.environ.get("GATEWAY_STARVATION_AGING_SECONDS", "2.0")
    )

    # M2: Multi-Instance Registry & Prefix/Load-Aware Routing settings
    replicas: List[ReplicaConfig] = field(default_factory=list)
    routing_policy: str = os.environ.get("GATEWAY_ROUTING_POLICY", "least-load")
    prefix_block_size: int = int(os.environ.get("GATEWAY_PREFIX_BLOCK_SIZE", "16"))
    prefix_weight: float = float(os.environ.get("GATEWAY_PREFIX_WEIGHT", "0.55"))
    load_weight: float = float(os.environ.get("GATEWAY_LOAD_WEIGHT", "0.45"))
    hotspot_load_threshold: float = float(
        os.environ.get("GATEWAY_HOTSPOT_LOAD_THRESHOLD", "0.75")
    )
    kv_hotspot_threshold: float = float(
        os.environ.get("GATEWAY_KV_HOTSPOT_THRESHOLD", "0.85")
    )
    kv_overload_threshold: float = float(
        os.environ.get("GATEWAY_KV_OVERLOAD_THRESHOLD", "0.95")
    )
    circuit_breaker_failure_threshold: int = int(
        os.environ.get("GATEWAY_CB_FAILURE_THRESHOLD", "3")
    )
    circuit_breaker_cooldown_seconds: float = float(
        os.environ.get("GATEWAY_CB_COOLDOWN_SECONDS", "5.0")
    )
    router_rng_seed: Optional[int] = None

    def resolved_replicas(self) -> List[ReplicaConfig]:
        """Return configured replicas, or fall back to env/single `backend_url`."""
        if self.replicas:
            return list(self.replicas)
        env_urls = os.environ.get("GATEWAY_BACKEND_URLS", "").strip()
        if env_urls:
            urls = [u.strip() for u in env_urls.split(",") if u.strip()]
            per_replica_cap = max(1, self.max_inflight // max(1, len(urls)))
            return [
                ReplicaConfig(
                    replica_id=f"replica-{idx}",
                    endpoint=url.rstrip("/"),
                    max_inflight=per_replica_cap,
                )
                for idx, url in enumerate(urls)
            ]
        return [
            ReplicaConfig(
                replica_id="replica-0",
                endpoint=self.backend_url.rstrip("/"),
                max_inflight=self.max_inflight,
            )
        ]
