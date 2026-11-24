"""Production-style Async LLM Serving Gateway (M1, M2 & Step 4 Observability).

Responsibilities:
- M1 Step 1:
  1. Async forwarding for `POST /v1/chat/completions` (`stream=false` JSON and
     `stream=true` zero-buffer chunk-by-chunk SSE passthrough).
  2. Connection pool reuse via FastAPI `lifespan` and `httpx.AsyncClient`.
  3. Granular 4D timeouts (`connect`, `read`, `write`, `pool`) & `X-Request-ID`.
  4. Explicit 3-state request lifecycle (`NOT_STARTED` -> `STREAMING` -> `FINISHED`).
  5. Client disconnect detection with immediate upstream cancellation (`aclose()`).
- M1 Step 2:
  6. Admission Control (`max_inflight`) & Bounded Queue (`queue_capacity`).
  7. Tenant quota & Token Bucket rate limit (`429`) separated from overload (`503`).
  8. Priority tiers (`high` > `normal` > `low`) with anti-starvation + Per-Tenant Fairness.
  9. End-to-end `deadline` and cancellation cleanup across Queued & In-Flight states.
- M2:
  10. `ReplicaRegistry` tracking endpoint, health/circuit breaker, live `inflight`,
      backend queue & KV capacity hints, plus real observed vLLM metrics.
  11. `InstanceRouter` (`Filter -> Score -> Pick`) with four switchable policies:
      `random`, `least-load`, `prefix-aware`, and `cache+load-aware`.
- Step 4 (Observability & Evidence Chain):
  12. Standard Prometheus Counter / Gauge / Histogram metrics:
      `gateway_requests_total`, `gateway_queue_depth`, `gateway_queue_wait_seconds`,
      `gateway_inflight_requests`, `gateway_rejections_total`,
      `gateway_router_policy_selections_total`, `gateway_router_decisions_total`,
      `gateway_ttft_seconds`, `gateway_itl_seconds`, `gateway_e2e_seconds`.
  13. End-to-end request correlation (`gateway_request_id` <-> `backend_request_id`
      + W3C `traceparent` propagation & `/admin/traces`).
"""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

from gateway.admission import AdmissionController, AdmissionRejectedError
from gateway.config import GatewayConfig
from gateway.discovery import DNSResolverFn, HeadlessDNSReplicaDiscoverer
from gateway.metrics import GatewayMetricsCollector, RequestTraceRecord
from gateway.models import RequestContext
from gateway.registry import ReplicaRegistry
from gateway.router import InstanceRouter, NoHealthyReplicaError, RoutingPolicy

logger = logging.getLogger("gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _ensure_traceparent(incoming: Optional[str]) -> str:
    """Propagate incoming W3C traceparent or synthesize a fresh 00-<trace>-<span>-01."""
    if incoming and len(incoming.split("-")) == 4:
        return incoming.strip()
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{span_id}-01"


def _extract_backend_request_id_from_bytes(
    payload: bytes, *, is_sse: bool
) -> Optional[str]:
    """Extract backend vLLM completion ID (e.g. `chatcmpl-...`) from JSON or SSE bytes."""
    if not payload:
        return None
    try:
        text = payload.decode("utf-8", errors="ignore")
        if not is_sse:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj.get("id"):
                return str(obj["id"])
            return None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("data:") and "[DONE]" not in line:
                json_part = line[5:].strip()
                if not json_part:
                    continue
                obj = json.loads(json_part)
                if isinstance(obj, dict) and obj.get("id"):
                    return str(obj["id"])
    except Exception:
        return None
    return None


def create_gateway_app(
    config: Optional[GatewayConfig] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    dns_resolver: Optional[DNSResolverFn] = None,
) -> FastAPI:
    """Create the FastAPI Gateway application."""
    cfg = config or GatewayConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=cfg.connect_timeout,
            read=cfg.read_timeout,
            write=cfg.write_timeout,
            pool=cfg.pool_timeout,
        )
        limits = httpx.Limits(
            max_connections=cfg.max_connections,
            max_keepalive_connections=cfg.max_keepalive_connections,
        )
        resolved_replicas = cfg.resolved_replicas()
        registry = ReplicaRegistry(
            replicas=resolved_replicas,
            circuit_breaker_failure_threshold=cfg.circuit_breaker_failure_threshold,
            circuit_breaker_cooldown_seconds=cfg.circuit_breaker_cooldown_seconds,
        )
        router = InstanceRouter(
            registry=registry,
            policy=RoutingPolicy.from_str(cfg.routing_policy),
            prefix_block_size=cfg.prefix_block_size,
            prefix_weight=cfg.prefix_weight,
            load_weight=cfg.load_weight,
            hotspot_load_threshold=cfg.hotspot_load_threshold,
            kv_hotspot_threshold=cfg.kv_hotspot_threshold,
            kv_overload_threshold=cfg.kv_overload_threshold,
            rng_seed=cfg.router_rng_seed,
        )

        app.state.config = cfg
        app.state.admission = AdmissionController(cfg)
        app.state.registry = registry
        app.state.router = router
        app.state.metrics = GatewayMetricsCollector()
        app.state.draining = False
        app.state.discoverer: Optional[HeadlessDNSReplicaDiscoverer] = None
        if cfg.k8s_headless_service:
            discoverer = HeadlessDNSReplicaDiscoverer(
                registry=registry,
                router=router,
                service_host=cfg.k8s_headless_service,
                backend_port=cfg.k8s_backend_port,
                per_replica_max_inflight=cfg.per_replica_max_inflight,
                poll_interval_seconds=cfg.k8s_discovery_interval_seconds,
                resolver=dns_resolver,
            )
            await discoverer.sync_once()
            discoverer.start()
            app.state.discoverer = discoverer

        app.state.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            transport=transport,
        )
        app.state.active_requests: Dict[str, RequestContext] = {}
        app.state.recent_requests: Dict[str, RequestContext] = {}
        logger.info(
            "Gateway started (replicas=%d, policy=%s, max_inflight=%d, queue_capacity=%d, k8s_dns=%s)",
            len(registry.all_replicas()),
            router.policy.value,
            cfg.max_inflight,
            cfg.queue_capacity,
            cfg.k8s_headless_service or "disabled",
        )
        try:
            yield
        finally:
            app.state.draining = True
            if app.state.discoverer is not None:
                await app.state.discoverer.stop()
            await app.state.http_client.aclose()
            logger.info("Gateway HTTP connection pool closed.")

    app = FastAPI(title="vLLM Serving Gateway", version="0.4.0", lifespan=lifespan)

    def _build_upstream_headers(request: Request, ctx: RequestContext) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "X-Request-ID": ctx.request_id,
            "X-Gateway-Request-ID": ctx.request_id,
            "traceparent": ctx.traceparent,
            "X-Tenant-ID": ctx.tenant_id,
            "X-Priority": ctx.priority,
            "Content-Type": "application/json",
        }
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        elif cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def _error_json_response(
        status_code: int,
        message: str,
        error_type: str,
        request_id: str,
        retry_after: Optional[float] = None,
        traceparent: Optional[str] = None,
    ) -> JSONResponse:
        headers: Dict[str, str] = {
            "X-Request-ID": request_id,
            "X-Gateway-Request-ID": request_id,
        }
        if traceparent:
            headers["traceparent"] = traceparent
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(
            status_code=status_code,
            headers=headers,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "request_id": request_id,
                }
            },
        )

    def _extract_deadline_seconds(
        request: Request, body: Dict[str, Any]
    ) -> Optional[float]:
        raw = (
            request.headers.get("X-Request-Deadline-Seconds")
            or request.headers.get("X-Timeout-Seconds")
            or body.get("deadline_seconds")
        )
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        if cfg.default_deadline > 0:
            return cfg.default_deadline
        return None

    def _release_replica_once(
        registry: ReplicaRegistry, ctx: RequestContext, *, failed: bool = False
    ) -> None:
        if ctx.routed_replica_id is not None and not ctx.replica_released:
            ctx.replica_released = True
            registry.release_replica(ctx.routed_replica_id, failed=failed)

    def _record_observability(
        metrics: GatewayMetricsCollector,
        ctx: RequestContext,
        *,
        status_code: int,
        ttft_ms: Optional[float] = None,
        itl_samples_ms: Optional[List[float]] = None,
    ) -> None:
        end_ts = ctx.finished_at or time.monotonic()
        e2e_ms = max(0.0, (end_ts - ctx.created_at) * 1000.0)
        metrics.record_request_completed(
            status_code=status_code,
            is_stream=ctx.is_stream,
            policy=ctx.routing_policy,
            queue_wait_ms=ctx.queue_wait_ms,
            e2e_ms=e2e_ms,
            ttft_ms=ttft_ms,
            itl_samples_ms=itl_samples_ms,
        )
        itl_avg: Optional[float] = None
        itl_p99: Optional[float] = None
        if itl_samples_ms:
            itl_avg = sum(itl_samples_ms) / len(itl_samples_ms)
            ordered = sorted(itl_samples_ms)
            idx = min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))
            itl_p99 = ordered[idx]

        trace = RequestTraceRecord(
            gateway_request_id=ctx.request_id,
            traceparent=ctx.traceparent,
            tenant_id=ctx.tenant_id,
            priority=ctx.priority,
            is_stream=ctx.is_stream,
            status_code=status_code,
            routing_policy=ctx.routing_policy,
            backend_replica_id=ctx.routed_replica_id,
            backend_endpoint=ctx.routed_endpoint,
            backend_request_id=ctx.backend_request_id,
            estimated_prefix_affinity=ctx.estimated_prefix_affinity,
            queue_wait_ms=ctx.queue_wait_ms,
            ttft_ms=ttft_ms,
            itl_avg_ms=itl_avg,
            itl_p99_ms=itl_p99,
            e2e_ms=e2e_ms,
            output_chunks=ctx.chunks_forwarded,
            error=ctx.error,
        )
        metrics.record_trace(trace)
        logger.info("request_trace %s", json.dumps(trace.to_dict(), ensure_ascii=False))

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> Dict[str, str]:
        """Lightweight liveness probe (NEVER coupled to queue/load latency)."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        """Readiness probe: 200 when not draining and >=1 healthy backend replica exists."""
        if getattr(request.app.state, "draining", False):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "gateway_draining"},
            )
        registry: ReplicaRegistry = request.app.state.registry
        avail = registry.available_replicas()
        if not avail:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "no_available_backend_replicas",
                    "available_replicas": 0,
                },
            )
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "available_replicas": len(avail)},
        )

    @app.post("/admin/drain")
    async def gateway_drain(request: Request) -> Dict[str, Any]:
        request.app.state.draining = True
        return {
            "status": "draining",
            "in_flight": request.app.state.admission.in_flight,
        }

    @app.get("/admin/discovery")
    async def discovery_status(request: Request) -> Dict[str, Any]:
        discoverer = getattr(request.app.state, "discoverer", None)
        if discoverer is None:
            return {"enabled": False}
        return discoverer.snapshot()

    @app.get("/admin/admission")
    async def admission_status(request: Request) -> Dict[str, Any]:
        admission: AdmissionController = request.app.state.admission
        return admission.snapshot()

    @app.get("/admin/routing")
    async def routing_status(request: Request) -> Dict[str, Any]:
        router: InstanceRouter = request.app.state.router
        return router.snapshot()

    @app.get("/admin/traces")
    async def list_traces(request: Request, limit: int = 50) -> Dict[str, Any]:
        metrics: GatewayMetricsCollector = request.app.state.metrics
        items = [t.to_dict() for t in list(metrics.traces)[-max(1, limit) :]]
        return {
            "count": len(items),
            "histograms": metrics.snapshot()["histograms"],
            "traces": items,
        }

    @app.get("/admin/traces/{gateway_request_id}")
    async def get_trace_by_id(gateway_request_id: str, request: Request) -> Response:
        metrics: GatewayMetricsCollector = request.app.state.metrics
        trace = metrics.traces_by_id.get(gateway_request_id)
        if trace is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Trace not found for gateway_request_id={gateway_request_id}",
                        "type": "trace_not_found",
                    }
                },
            )
        return JSONResponse(status_code=200, content=trace.to_dict())

    @app.post("/admin/routing/policy")
    async def set_routing_policy(request: Request) -> Response:
        router: InstanceRouter = request.app.state.router
        try:
            payload = await request.json()
            policy_name = payload.get("policy") if isinstance(payload, dict) else None
            new_policy = router.set_policy(policy_name)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_policy"}},
            )
        return JSONResponse(
            status_code=200,
            content={"active_policy": new_policy.value},
        )

    @app.post("/admin/replicas/{replica_id}/health")
    async def update_replica_health(replica_id: str, request: Request) -> Response:
        registry: ReplicaRegistry = request.app.state.registry
        try:
            payload = await request.json()
            healthy = bool(payload.get("healthy", True))
            registry.set_health(replica_id, healthy)
            replica = registry.get(replica_id)
            return JSONResponse(status_code=200, content=replica.to_dict())
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": str(exc), "type": "replica_not_found"}},
            )
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request"}},
            )

    @app.post("/admin/replicas/sync-metrics")
    async def sync_replicas_vllm_metrics(request: Request) -> Dict[str, Any]:
        """Actively scrape real `/metrics` from every replica to sync vLLM ground truth."""
        registry: ReplicaRegistry = request.app.state.registry
        client: httpx.AsyncClient = request.app.state.http_client
        results: Dict[str, Any] = {}
        for replica in registry.all_replicas():
            try:
                resp = await client.get(f"{replica.endpoint}/metrics")
                if resp.status_code == 200:
                    results[replica.replica_id] = registry.update_from_vllm_metrics(
                        replica.replica_id, resp.text
                    )
                else:
                    results[replica.replica_id] = {
                        "error": f"HTTP {resp.status_code}"
                    }
            except httpx.HTTPError as exc:
                results[replica.replica_id] = {"error": str(exc)}
        return {"synced_replicas": results}

    @app.get("/metrics")
    async def prometheus_metrics(request: Request) -> Response:
        admission: AdmissionController = request.app.state.admission
        router: InstanceRouter = request.app.state.router
        metrics: GatewayMetricsCollector = request.app.state.metrics
        body_text = metrics.render_prometheus(
            admission_snapshot=admission.snapshot(),
            router_snapshot=router.snapshot(),
        )
        return Response(
            content=body_text,
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        traceparent = _ensure_traceparent(request.headers.get("traceparent"))
        client: httpx.AsyncClient = request.app.state.http_client
        registry: ReplicaRegistry = request.app.state.registry
        ctx = RequestContext(
            request_id=request_id, is_stream=False, traceparent=traceparent
        )
        headers = _build_upstream_headers(request, ctx)

        avail = registry.available_replicas()
        if not avail:
            return _error_json_response(
                503,
                "No healthy backend replicas available",
                "no_healthy_replicas",
                request_id,
                traceparent=traceparent,
            )
        target_url = f"{avail[0].endpoint}/v1/models"
        try:
            resp = await client.get(target_url, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
                headers={
                    "X-Request-ID": request_id,
                    "X-Gateway-Request-ID": request_id,
                    "traceparent": traceparent,
                    "X-Routed-Replica": avail[0].replica_id,
                },
            )
        except httpx.TimeoutException as exc:
            return _error_json_response(
                504,
                f"Backend timeout on /v1/models: {exc}",
                "gateway_timeout",
                request_id,
                traceparent=traceparent,
            )
        except httpx.HTTPError as exc:
            return _error_json_response(
                502,
                f"Backend unreachable on /v1/models: {exc}",
                "bad_gateway",
                request_id,
                traceparent=traceparent,
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        request_id = (
            request.headers.get("X-Gateway-Request-ID")
            or request.headers.get("X-Request-ID")
            or f"req-{uuid.uuid4().hex[:12]}"
        )
        traceparent = _ensure_traceparent(request.headers.get("traceparent"))
        admission: AdmissionController = request.app.state.admission
        registry: ReplicaRegistry = request.app.state.registry
        router: InstanceRouter = request.app.state.router
        metrics: GatewayMetricsCollector = request.app.state.metrics

        # Parse and validate JSON body while still in NOT_STARTED state
        try:
            body = await request.json()
        except Exception as exc:
            admission.record_invalid_request()
            invalid_ctx = RequestContext(
                request_id=request_id,
                is_stream=False,
                traceparent=traceparent,
                error="invalid_request_error",
            )
            invalid_ctx.mark_finished(upstream_closed=True, error="invalid_request_error")
            _record_observability(metrics, invalid_ctx, status_code=400)
            return _error_json_response(
                400,
                f"Invalid JSON request body: {exc}",
                "invalid_request_error",
                request_id,
                traceparent=traceparent,
            )

        if (
            not isinstance(body, dict)
            or not body.get("model")
            or "messages" not in body
            or not isinstance(body.get("messages"), list)
        ):
            admission.record_invalid_request()
            invalid_ctx = RequestContext(
                request_id=request_id,
                is_stream=False,
                traceparent=traceparent,
                error="invalid_request_error",
            )
            invalid_ctx.mark_finished(upstream_closed=True, error="invalid_request_error")
            _record_observability(metrics, invalid_ctx, status_code=400)
            return _error_json_response(
                400,
                "Invalid request schema: 'model' (str) and 'messages' (list) are required",
                "invalid_request_error",
                request_id,
                traceparent=traceparent,
            )

        policy_override = request.headers.get("X-Routing-Policy")
        if policy_override:
            try:
                RoutingPolicy.from_str(policy_override)
            except ValueError as exc:
                admission.record_invalid_request()
                invalid_ctx = RequestContext(
                    request_id=request_id,
                    is_stream=bool(body.get("stream", False)),
                    traceparent=traceparent,
                    error="invalid_routing_policy",
                )
                invalid_ctx.mark_finished(
                    upstream_closed=True, error="invalid_routing_policy"
                )
                _record_observability(metrics, invalid_ctx, status_code=400)
                return _error_json_response(
                    400,
                    str(exc),
                    "invalid_routing_policy",
                    request_id,
                    traceparent=traceparent,
                )

        is_stream = bool(body.get("stream", False))
        tenant_id = (
            request.headers.get("X-Tenant-ID")
            or body.get("tenant_id")
            or body.get("user")
            or "default"
        )
        priority = (
            request.headers.get("X-Priority")
            or body.get("priority")
            or "normal"
        )

        ctx = RequestContext(
            request_id=request_id,
            is_stream=is_stream,
            tenant_id=str(tenant_id),
            priority=str(priority),
            traceparent=traceparent,
        )
        deadline_sec = _extract_deadline_seconds(request, body)
        if deadline_sec is not None:
            ctx.deadline_at = ctx.created_at + deadline_sec

        request.app.state.active_requests[request_id] = ctx
        request.app.state.recent_requests[request_id] = ctx

        # =====================================================================
        # M1 Step 2: Admission Control (Accept / Queue / Reject 429, 503, 504)
        # =====================================================================
        try:
            await admission.acquire_slot(ctx, request.is_disconnected)
        except AdmissionRejectedError as exc:
            ctx.mark_finished(
                upstream_closed=True,
                client_disconnected=(exc.status_code == 499),
                error=exc.error_type,
            )
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=exc.status_code)
            return _error_json_response(
                exc.status_code,
                exc.message,
                exc.error_type,
                request_id,
                retry_after=exc.retry_after,
                traceparent=traceparent,
            )
        except asyncio.CancelledError:
            ctx.mark_finished(
                upstream_closed=True,
                client_disconnected=True,
                error="cancelled_in_queue",
            )
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=499)
            raise

        # Check remaining deadline right after acquiring an in-flight slot
        rem_deadline = ctx.remaining_deadline()
        if rem_deadline is not None and rem_deadline <= 0:
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error="deadline_exceeded")
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=504)
            return _error_json_response(
                504,
                "Request deadline expired immediately after queue wait",
                "deadline_exceeded",
                request_id,
                traceparent=traceparent,
            )

        # =====================================================================
        # M2: Late-Binding Instance Selection (Filter -> Score -> Pick)
        # =====================================================================
        try:
            decision = router.select_replica(
                request_id=request_id,
                body=body,
                policy_override=policy_override,
            )
        except NoHealthyReplicaError as exc:
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error="no_healthy_replicas")
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=503)
            return _error_json_response(
                503,
                str(exc),
                "no_healthy_replicas",
                request_id,
                retry_after=1.0,
                traceparent=traceparent,
            )

        target_replica = decision.replica
        registry.acquire_replica(target_replica.replica_id)
        metrics.record_routing_decision(
            decision.policy.value, target_replica.replica_id
        )
        ctx.routed_replica_id = target_replica.replica_id
        ctx.routed_endpoint = target_replica.endpoint
        ctx.routing_policy = decision.policy.value
        ctx.estimated_prefix_affinity = decision.estimated_prefix_affinity

        routing_headers = {
            "X-Request-ID": request_id,
            "X-Gateway-Request-ID": request_id,
            "traceparent": traceparent,
            "X-Queue-Wait-Ms": f"{ctx.queue_wait_ms:.2f}",
            "X-Routed-Replica": target_replica.replica_id,
            "X-Routing-Policy": decision.policy.value,
            "X-Estimated-Prefix-Affinity": f"{decision.estimated_prefix_affinity:.4f}",
        }

        effective_timeout = (
            min(cfg.read_timeout, rem_deadline)
            if rem_deadline is not None
            else cfg.read_timeout
        )

        client: httpx.AsyncClient = request.app.state.http_client
        headers = _build_upstream_headers(request, ctx)
        target_url = f"{target_replica.endpoint}/v1/chat/completions"

        # =====================================================================
        # Branch A: Non-streaming mode (stream=false)
        # =====================================================================
        if not is_stream:
            failed = False
            resp_status = 500
            try:
                upstream_resp = await asyncio.wait_for(
                    client.post(target_url, json=body, headers=headers),
                    timeout=effective_timeout,
                )
                resp_status = upstream_resp.status_code
                ctx.backend_request_id = (
                    _extract_backend_request_id_from_bytes(
                        upstream_resp.content, is_sse=False
                    )
                    or upstream_resp.headers.get("X-Request-ID")
                )
                ctx.mark_finished(upstream_closed=True)
                resp_headers = dict(routing_headers)
                if ctx.backend_request_id:
                    resp_headers["X-Backend-Request-ID"] = ctx.backend_request_id
                ttft_ms = (
                    (ctx.finished_at - ctx.created_at) * 1000.0
                    if ctx.finished_at is not None and resp_status == 200
                    else None
                )
                _record_observability(
                    metrics, ctx, status_code=resp_status, ttft_ms=ttft_ms
                )
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "application/json"
                    ),
                    headers=resp_headers,
                )
            except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
                failed = True
                ctx.mark_finished(upstream_closed=True, error=f"timeout: {exc}")
                _record_observability(metrics, ctx, status_code=504)
                return _error_json_response(
                    504,
                    f"Upstream vLLM timed out: {exc}",
                    "gateway_timeout",
                    request_id,
                    traceparent=traceparent,
                )
            except httpx.HTTPError as exc:
                failed = True
                ctx.mark_finished(upstream_closed=True, error=f"http_error: {exc}")
                _record_observability(metrics, ctx, status_code=502)
                return _error_json_response(
                    502,
                    f"Upstream vLLM error: {exc}",
                    "bad_gateway",
                    request_id,
                    traceparent=traceparent,
                )
            except asyncio.CancelledError:
                admission.record_inflight_cancel()
                ctx.mark_finished(
                    upstream_closed=True, client_disconnected=True, error="cancelled"
                )
                _record_observability(metrics, ctx, status_code=499)
                raise
            finally:
                _release_replica_once(registry, ctx, failed=failed)
                admission.release_slot(ctx)
                request.app.state.active_requests.pop(request_id, None)

        # =====================================================================
        # Branch B: Streaming mode (stream=true)
        # 3-State Machine: NOT_STARTED -> STREAMING -> FINISHED
        # =====================================================================
        upstream_req = client.build_request(
            "POST", target_url, json=body, headers=headers
        )
        upstream_resp: Optional[httpx.Response] = None

        try:
            # Phase 1 (NOT_STARTED): Connect and obtain upstream response headers
            upstream_resp = await asyncio.wait_for(
                client.send(upstream_req, stream=True),
                timeout=effective_timeout,
            )

            if upstream_resp.status_code != 200:
                error_bytes = await upstream_resp.aread()
                await upstream_resp.aclose()
                _release_replica_once(
                    registry, ctx, failed=(upstream_resp.status_code >= 500)
                )
                admission.release_slot(ctx)
                ctx.mark_finished(
                    upstream_closed=True,
                    error=f"upstream_status_{upstream_resp.status_code}",
                )
                request.app.state.active_requests.pop(request_id, None)
                _record_observability(
                    metrics, ctx, status_code=upstream_resp.status_code
                )
                return Response(
                    content=error_bytes,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "application/json"
                    ),
                    headers=routing_headers,
                )

            # Peek the very first chunk while still in NOT_STARTED state!
            chunk_iter = upstream_resp.aiter_raw()
            rem_now = ctx.remaining_deadline()
            first_token_timeout = (
                min(cfg.read_timeout, max(0.001, rem_now))
                if rem_now is not None
                else cfg.read_timeout
            )
            first_chunk = await asyncio.wait_for(
                anext(chunk_iter, b""),
                timeout=first_token_timeout,
            )

        except StopAsyncIteration:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            _release_replica_once(registry, ctx, failed=True)
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error="empty_stream")
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=502)
            return _error_json_response(
                502,
                "Upstream closed stream before first token",
                "bad_gateway",
                request_id,
                traceparent=traceparent,
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            _release_replica_once(registry, ctx, failed=True)
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error=f"pre_stream_timeout: {exc}")
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=504)
            return _error_json_response(
                504,
                f"Upstream timed out before first token (TTFT/deadline timeout): {exc}",
                "gateway_timeout",
                request_id,
                traceparent=traceparent,
            )
        except httpx.HTTPError as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            _release_replica_once(registry, ctx, failed=True)
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error=f"pre_stream_error: {exc}")
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=502)
            return _error_json_response(
                502,
                f"Upstream failed before first token: {exc}",
                "bad_gateway",
                request_id,
                traceparent=traceparent,
            )
        except asyncio.CancelledError:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            admission.record_inflight_cancel()
            _release_replica_once(registry, ctx, failed=False)
            admission.release_slot(ctx)
            ctx.mark_finished(
                upstream_closed=True,
                client_disconnected=True,
                error="cancelled_before_first_token",
            )
            request.app.state.active_requests.pop(request_id, None)
            _record_observability(metrics, ctx, status_code=499)
            raise

        # Phase 2 (STREAMING): First chunk arrived! Commit HTTP 200 OK headers.
        ctx.mark_streaming()
        first_chunk_ttft_ms = (
            (ctx.first_byte_at - ctx.created_at) * 1000.0
            if ctx.first_byte_at is not None
            else 0.0
        )
        ctx.backend_request_id = (
            _extract_backend_request_id_from_bytes(first_chunk, is_sse=True)
            or upstream_resp.headers.get("X-Request-ID")
        )

        stream_headers = {
            **routing_headers,
            "X-Gateway-TTFT-Ms": f"{first_chunk_ttft_ms:.2f}",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if ctx.backend_request_id:
            stream_headers["X-Backend-Request-ID"] = ctx.backend_request_id

        async def sse_passthrough_generator() -> AsyncIterator[bytes]:
            client_left = False
            stream_err: Optional[str] = None
            itl_samples_ms: List[float] = []
            last_chunk_ts = ctx.first_byte_at or time.monotonic()
            try:
                if first_chunk:
                    if await request.is_disconnected():
                        client_left = True
                        return
                    ctx.chunks_forwarded += 1
                    yield first_chunk

                while True:
                    if await request.is_disconnected():
                        client_left = True
                        logger.info(
                            "Client disconnected mid-stream (request_id=%s)", request_id
                        )
                        break

                    rem_step = ctx.remaining_deadline()
                    if rem_step is not None and rem_step <= 0:
                        raise httpx.ReadTimeout("Request deadline expired mid-stream")

                    step_timeout = (
                        min(cfg.read_timeout, rem_step)
                        if rem_step is not None
                        else cfg.read_timeout
                    )
                    try:
                        chunk = await asyncio.wait_for(
                            anext(chunk_iter), timeout=step_timeout
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as t_exc:
                        raise httpx.ReadTimeout(
                            f"Mid-stream chunk timeout: {t_exc}"
                        ) from t_exc

                    if await request.is_disconnected():
                        client_left = True
                        break
                    if chunk:
                        now_ts = time.monotonic()
                        if b"[DONE]" not in chunk:
                            itl_samples_ms.append((now_ts - last_chunk_ts) * 1000.0)
                            last_chunk_ts = now_ts
                            if ctx.backend_request_id is None:
                                ctx.backend_request_id = (
                                    _extract_backend_request_id_from_bytes(
                                        chunk, is_sse=True
                                    )
                                )
                        ctx.chunks_forwarded += 1
                        yield chunk

            except asyncio.CancelledError:
                client_left = True
                logger.info(
                    "Streaming task cancelled by client disconnect (request_id=%s)",
                    request_id,
                )
                raise
            except (httpx.HTTPError, httpx.StreamError) as exc:
                stream_err = f"mid_stream_error: {exc}"
                logger.error(
                    "Upstream failed mid-stream (request_id=%s): %s", request_id, exc
                )
                err_payload = json.dumps(
                    {
                        "error": {
                            "message": f"Upstream stream interrupted: {exc}",
                            "type": "mid_stream_error",
                            "request_id": request_id,
                        }
                    }
                )
                yield f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8")
            finally:
                # Phase 3 (FINISHED): Always close upstream stream, release replica
                # in-flight count, release admission slot, and record final metrics/trace!
                await upstream_resp.aclose()
                if client_left:
                    admission.record_inflight_cancel()
                _release_replica_once(registry, ctx, failed=(stream_err is not None))
                admission.release_slot(ctx)
                ctx.mark_finished(
                    upstream_closed=True,
                    client_disconnected=client_left,
                    error=stream_err,
                )
                request.app.state.active_requests.pop(request_id, None)
                _record_observability(
                    metrics,
                    ctx,
                    status_code=499 if client_left else 200,
                    ttft_ms=first_chunk_ttft_ms,
                    itl_samples_ms=itl_samples_ms,
                )

        return StreamingResponse(
            sse_passthrough_generator(),
            status_code=200,
            media_type="text/event-stream",
            headers=stream_headers,
        )

    return app


app = create_gateway_app()
