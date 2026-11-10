"""Gateway configuration for M1 Step 1 & Step 2 (Async Proxy + Admission Control)."""
from dataclasses import dataclass, field
import os
from typing import Dict


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
    - max_inflight: Max concurrent requests allowed to execute on the backend at once.
    - queue_capacity: Max waiting requests held in the Gateway bounded queue when
      backend is saturated (`in_flight >= max_inflight`). Requests beyond this are
      immediately shed with HTTP 503.
    - queue_timeout: Max seconds a request may wait in the bounded queue before
      being shed with HTTP 503.
    - default_deadline: Optional end-to-end request deadline (seconds). 0.0 means
      no global deadline unless specified via `X-Request-Deadline` header.
    - tenant_max_concurrency: Max total active (in-flight + queued) requests per
      tenant. 0 means unlimited. Exceeding this returns HTTP 429.
    - tenant_quotas: Optional per-tenant override map `{tenant_id: max_concurrency}`.
    - tenant_rate_limit_rps: Per-tenant Token Bucket refill rate (requests/sec).
      0.0 means unlimited. Exceeding this returns HTTP 429.
    - tenant_burst_capacity: Per-tenant Token Bucket burst size.
    - max_consecutive_high_priority: Anti-starvation cap: max consecutive HIGH
      priority dispatches before granting 1 turn to waiting NORMAL/LOW queues.
    - starvation_aging_seconds: If a lower-priority request waits longer than this
      threshold, it is promoted to prevent permanent starvation.
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

    # Step 2: Admission Control & Flow Control settings
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
