"""Prometheus Metrics (Counter / Gauge / Histogram) & Request Correlation Store (Step 4).

Implements standard Prometheus Exposition Format 0.0.4 metrics aligned with
vLLM Production Metrics conventions:
- Counters:
  - `gateway_requests_total{status_code,stream,policy}`
  - `gateway_rejections_total{reason}`
  - `gateway_router_policy_selections_total{policy}`
  - `gateway_router_decisions_total{policy,replica}`
  - `gateway_cancel_total{stage}`
- Gauges:
  - `gateway_queue_depth`
  - `gateway_inflight_requests`
  - `gateway_replica_inflight{replica}`
  - `gateway_replica_healthy{replica}`
  - `gateway_router_estimated_prefix_affinity_avg` (explicitly marked as estimate)
  - `gateway_replica_observed_vllm_prefix_cache_hit_rate{replica}` (ground truth)
- Histograms (base SI unit `seconds`, compatible with PromQL `histogram_quantile`):
  - `gateway_queue_wait_seconds`
  - `gateway_ttft_seconds`
  - `gateway_itl_seconds`
  - `gateway_e2e_seconds`
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


QUEUE_WAIT_BUCKETS: Tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    15.0,
)
TTFT_BUCKETS: Tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
ITL_BUCKETS: Tuple[float, ...] = (
    0.002,
    0.005,
    0.01,
    0.02,
    0.035,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)
E2E_BUCKETS: Tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


class PrometheusHistogram:
    """Thread/async-safe Prometheus cumulative histogram in seconds."""

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float],
        max_samples: int = 2000,
    ) -> None:
        self.name = name
        self.help_text = help_text
        self.buckets = tuple(sorted(buckets))
        self._bucket_counts: List[int] = [0] * (len(self.buckets) + 1)
        self.sum_value: float = 0.0
        self.count: int = 0
        self._recent_samples: Deque[float] = deque(maxlen=max_samples)

    def observe(self, value_seconds: float) -> None:
        val = max(0.0, float(value_seconds))
        self.sum_value += val
        self.count += 1
        self._recent_samples.append(val)
        for idx, upper in enumerate(self.buckets):
            if val <= upper:
                self._bucket_counts[idx] += 1
        # +Inf bucket always increments
        self._bucket_counts[-1] += 1

    def percentile_ms(self, p: float) -> float:
        if not self._recent_samples:
            return 0.0
        ordered = sorted(self._recent_samples)
        idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
        return round(ordered[idx] * 1000.0, 2)

    def render_prometheus(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        for idx, upper in enumerate(self.buckets):
            lines.append(
                f'{self.name}_bucket{{le="{upper:g}"}} {self._bucket_counts[idx]}'
            )
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._bucket_counts[-1]}')
        lines.append(f"{self.name}_sum {self.sum_value:.6f}")
        lines.append(f"{self.name}_count {self.count}")
        return lines

    def summary(self) -> Dict[str, Any]:
        avg_ms = (
            round((self.sum_value / self.count) * 1000.0, 2) if self.count > 0 else 0.0
        )
        return {
            "count": self.count,
            "sum_seconds": round(self.sum_value, 6),
            "avg_ms": avg_ms,
            "p50_ms": self.percentile_ms(50),
            "p95_ms": self.percentile_ms(95),
            "p99_ms": self.percentile_ms(99),
        }


@dataclass
class RequestTraceRecord:
    """Correlates a Gateway request with its downstream backend vLLM request."""

    gateway_request_id: str
    traceparent: str
    tenant_id: str
    priority: str
    is_stream: bool
    status_code: int
    routing_policy: Optional[str] = None
    backend_replica_id: Optional[str] = None
    backend_endpoint: Optional[str] = None
    backend_request_id: Optional[str] = None
    estimated_prefix_affinity: float = 0.0
    queue_wait_ms: float = 0.0
    ttft_ms: Optional[float] = None
    itl_avg_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    e2e_ms: float = 0.0
    output_chunks: int = 0
    error: Optional[str] = None
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gateway_request_id": self.gateway_request_id,
            "traceparent": self.traceparent,
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "is_stream": self.is_stream,
            "status_code": self.status_code,
            "routing_policy": self.routing_policy,
            "backend_replica_id": self.backend_replica_id,
            "backend_endpoint": self.backend_endpoint,
            "backend_request_id": self.backend_request_id,
            "estimated_prefix_affinity": self.estimated_prefix_affinity,
            "queue_wait_ms": round(self.queue_wait_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "itl_avg_ms": (
                round(self.itl_avg_ms, 2) if self.itl_avg_ms is not None else None
            ),
            "itl_p99_ms": (
                round(self.itl_p99_ms, 2) if self.itl_p99_ms is not None else None
            ),
            "e2e_ms": round(self.e2e_ms, 2),
            "output_chunks": self.output_chunks,
            "error": self.error,
        }


class GatewayMetricsCollector:
    """Centralized metrics & trace correlation store for the Gateway."""

    def __init__(self, max_traces: int = 500) -> None:
        # Counters
        self.requests_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.policy_replica_decisions: Dict[Tuple[str, str], int] = defaultdict(int)

        # Histograms (all in seconds)
        self.queue_wait_seconds = PrometheusHistogram(
            "gateway_queue_wait_seconds",
            "Time spent waiting in the Gateway bounded admission queue in seconds.",
            QUEUE_WAIT_BUCKETS,
        )
        self.ttft_seconds = PrometheusHistogram(
            "gateway_ttft_seconds",
            "End-to-end Time To First Token observed at the Gateway in seconds.",
            TTFT_BUCKETS,
        )
        self.itl_seconds = PrometheusHistogram(
            "gateway_itl_seconds",
            "Inter-token / inter-chunk latency observed during SSE streaming in seconds.",
            ITL_BUCKETS,
        )
        self.e2e_seconds = PrometheusHistogram(
            "gateway_e2e_seconds",
            "Total end-to-end request duration observed at the Gateway in seconds.",
            E2E_BUCKETS,
        )

        # Recent correlated request traces (`gateway_request_id` <-> `backend_request_id`)
        self.traces: Deque[RequestTraceRecord] = deque(maxlen=max_traces)
        self.traces_by_id: Dict[str, RequestTraceRecord] = {}

    def record_request_completed(
        self,
        *,
        status_code: int,
        is_stream: bool,
        policy: Optional[str],
        queue_wait_ms: float,
        e2e_ms: float,
        ttft_ms: Optional[float] = None,
        itl_samples_ms: Optional[Sequence[float]] = None,
    ) -> None:
        pol_label = policy or "none"
        stream_label = "true" if is_stream else "false"
        self.requests_total[(str(status_code), stream_label, pol_label)] += 1
        self.queue_wait_seconds.observe(queue_wait_ms / 1000.0)
        self.e2e_seconds.observe(e2e_ms / 1000.0)
        if ttft_ms is not None and status_code == 200:
            self.ttft_seconds.observe(ttft_ms / 1000.0)
        if itl_samples_ms:
            for itl_ms in itl_samples_ms:
                self.itl_seconds.observe(itl_ms / 1000.0)

    def record_routing_decision(self, policy: str, replica_id: str) -> None:
        self.policy_replica_decisions[(policy, replica_id)] += 1

    def record_trace(self, trace: RequestTraceRecord) -> None:
        if len(self.traces) == self.traces.maxlen and self.traces:
            evicted = self.traces[0]
            self.traces_by_id.pop(evicted.gateway_request_id, None)
        self.traces.append(trace)
        self.traces_by_id[trace.gateway_request_id] = trace

    def render_prometheus(
        self,
        *,
        admission_snapshot: Dict[str, Any],
        router_snapshot: Dict[str, Any],
    ) -> str:
        m = admission_snapshot["metrics"]
        lines: List[str] = []

        # 1. `gateway_requests_total` (Counter)
        lines.extend(
            [
                "# HELP gateway_requests_total Total HTTP requests handled by Gateway chat completions.",
                "# TYPE gateway_requests_total counter",
            ]
        )
        for (status_code, stream_lbl, pol_lbl), count in sorted(
            self.requests_total.items()
        ):
            lines.append(
                f'gateway_requests_total{{status_code="{status_code}",stream="{stream_lbl}",policy="{pol_lbl}"}} {count}'
            )

        # 2. `gateway_inflight_requests` & backward-compatible `gateway_in_flight` (Gauge)
        in_flight = admission_snapshot["in_flight"]
        lines.extend(
            [
                "# HELP gateway_inflight_requests Current number of in-flight requests executing on backends.",
                "# TYPE gateway_inflight_requests gauge",
                f"gateway_inflight_requests {in_flight}",
                "# HELP gateway_in_flight Alias for gateway_inflight_requests.",
                "# TYPE gateway_in_flight gauge",
                f"gateway_in_flight {in_flight}",
            ]
        )

        # 3. `gateway_queue_depth` (Gauge)
        lines.extend(
            [
                "# HELP gateway_queue_depth Current number of requests waiting in Gateway bounded queue.",
                "# TYPE gateway_queue_depth gauge",
                f"gateway_queue_depth {m['queue_depth']['current']}",
                "# HELP gateway_queue_depth_max_observed Peak observed Gateway queue depth.",
                "# TYPE gateway_queue_depth_max_observed gauge",
                f"gateway_queue_depth_max_observed {m['queue_depth']['max_observed']}",
                "# HELP gateway_queue_wait_avg_ms Backward-compatible average queue wait ms gauge (prefer gateway_queue_wait_seconds histogram).",
                "# TYPE gateway_queue_wait_avg_ms gauge",
                f"gateway_queue_wait_avg_ms {m['queue_wait']['avg_queued_wait_ms']}",
                "# HELP gateway_queue_wait_max_ms Backward-compatible max queue wait ms gauge.",
                "# TYPE gateway_queue_wait_max_ms gauge",
                f"gateway_queue_wait_max_ms {m['queue_wait']['max_wait_ms']}",
            ]
        )

        # 4. Standard Histograms in seconds (`_bucket`, `_sum`, `_count`)
        lines.extend(self.queue_wait_seconds.render_prometheus())
        lines.extend(self.ttft_seconds.render_prometheus())
        lines.extend(self.itl_seconds.render_prometheus())
        lines.extend(self.e2e_seconds.render_prometheus())

        # 5. `gateway_rejections_total` (canonical) & `gateway_reject_total` (legacy alias)
        reject_reasons = (
            "invalid_400",
            "quota_429",
            "rate_limit_429",
            "queue_full_503",
            "queue_timeout_503",
            "deadline_504",
        )
        lines.extend(
            [
                "# HELP gateway_rejections_total Total rejected requests before inference by reason.",
                "# TYPE gateway_rejections_total counter",
            ]
        )
        for reason in reject_reasons:
            lines.append(
                f'gateway_rejections_total{{reason="{reason}"}} {m["reject"][reason]}'
            )
        lines.extend(
            [
                "# HELP gateway_reject_total Backward-compatible alias for gateway_rejections_total.",
                "# TYPE gateway_reject_total counter",
            ]
        )
        for reason in reject_reasons:
            lines.append(
                f'gateway_reject_total{{reason="{reason}"}} {m["reject"][reason]}'
            )

        # 6. `gateway_cancel_total`
        lines.extend(
            [
                "# HELP gateway_cancel_total Total cancelled requests by lifecycle stage.",
                "# TYPE gateway_cancel_total counter",
                f'gateway_cancel_total{{stage="in_queue"}} {m["cancel"]["in_queue"]}',
                f'gateway_cancel_total{{stage="in_flight"}} {m["cancel"]["in_flight"]}',
            ]
        )

        # 7. Routing policy selections & per-replica metrics
        lines.extend(
            [
                "# HELP gateway_router_policy_selections_total Total routing selections per policy.",
                "# TYPE gateway_router_policy_selections_total counter",
            ]
        )
        for pol, count in sorted(router_snapshot["stats"]["decisions_by_policy"].items()):
            lines.append(
                f'gateway_router_policy_selections_total{{policy="{pol}"}} {count}'
            )

        lines.extend(
            [
                "# HELP gateway_router_decisions_total Total routing decisions by policy and replica.",
                "# TYPE gateway_router_decisions_total counter",
            ]
        )
        for (pol, rid), count in sorted(self.policy_replica_decisions.items()):
            lines.append(
                f'gateway_router_decisions_total{{policy="{pol}",replica="{rid}"}} {count}'
            )

        lines.extend(
            [
                "# HELP gateway_router_estimated_prefix_affinity_avg Estimated token-prefix affinity (Gateway estimate only, NOT real vLLM cache hits).",
                "# TYPE gateway_router_estimated_prefix_affinity_avg gauge",
                f"gateway_router_estimated_prefix_affinity_avg {router_snapshot['stats']['avg_estimated_prefix_affinity']}",
                "# HELP gateway_replica_inflight Current in-flight requests per backend replica.",
                "# TYPE gateway_replica_inflight gauge",
            ]
        )
        for rep in router_snapshot["registry"]["replicas"]:
            rid = rep["replica_id"]
            lines.append(f'gateway_replica_inflight{{replica="{rid}"}} {rep["inflight"]}')

        lines.extend(
            [
                "# HELP gateway_replica_healthy Health status per backend replica (1=available, 0=unavailable).",
                "# TYPE gateway_replica_healthy gauge",
            ]
        )
        for rep in router_snapshot["registry"]["replicas"]:
            rid = rep["replica_id"]
            avail = 1 if rep["available"] else 0
            lines.append(f'gateway_replica_healthy{{replica="{rid}"}} {avail}')

        lines.extend(
            [
                "# HELP gateway_replica_observed_vllm_prefix_cache_hit_rate Ground-truth vLLM prefix cache hit rate scraped from real vLLM /metrics.",
                "# TYPE gateway_replica_observed_vllm_prefix_cache_hit_rate gauge",
            ]
        )
        for rep in router_snapshot["registry"]["replicas"]:
            hit_rate = rep["observed_vllm_metrics"]["prefix_cache_hit_rate"]
            if hit_rate is not None:
                rid = rep["replica_id"]
                lines.append(
                    f'gateway_replica_observed_vllm_prefix_cache_hit_rate{{replica="{rid}"}} {hit_rate}'
                )

        return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "requests_total": {
                f"{st}|stream={sr}|policy={pol}": cnt
                for (st, sr, pol), cnt in self.requests_total.items()
            },
            "histograms": {
                "queue_wait": self.queue_wait_seconds.summary(),
                "ttft": self.ttft_seconds.summary(),
                "itl": self.itl_seconds.summary(),
                "e2e": self.e2e_seconds.summary(),
            },
            "recent_traces_count": len(self.traces),
        }
