"""Production-style Async LLM Serving Gateway (M1 Step 1/2 & M2 Multi-Replica Routing).

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
  12. Strict semantic separation between Gateway-side *estimated* token-prefix
      affinity (`X-Estimated-Prefix-Affinity`) and real vLLM KV cache hits.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

from gateway.admission import AdmissionController, AdmissionRejectedError
from gateway.config import GatewayConfig
from gateway.models import RequestContext
from gateway.registry import ReplicaRegistry
from gateway.router import InstanceRouter, NoHealthyReplicaError, RoutingPolicy

logger = logging.getLogger("gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_gateway_app(
    config: Optional[GatewayConfig] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
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
        app.state.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            transport=transport,
        )
        app.state.active_requests: Dict[str, RequestContext] = {}
        app.state.recent_requests: Dict[str, RequestContext] = {}
        logger.info(
            "Gateway started (replicas=%d, policy=%s, max_inflight=%d, queue_capacity=%d)",
            len(resolved_replicas),
            router.policy.value,
            cfg.max_inflight,
            cfg.queue_capacity,
        )
        try:
            yield
        finally:
            await app.state.http_client.aclose()
            logger.info("Gateway HTTP connection pool closed.")

    app = FastAPI(title="vLLM Serving Gateway", version="0.3.0", lifespan=lifespan)

    def _build_upstream_headers(request: Request, ctx: RequestContext) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "X-Request-ID": ctx.request_id,
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
    ) -> JSONResponse:
        headers: Dict[str, str] = {"X-Request-ID": request_id}
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

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/admission")
    async def admission_status(request: Request) -> Dict[str, Any]:
        admission: AdmissionController = request.app.state.admission
        return admission.snapshot()

    @app.get("/admin/routing")
    async def routing_status(request: Request) -> Dict[str, Any]:
        router: InstanceRouter = request.app.state.router
        return router.snapshot()

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
        snap = admission.snapshot()
        r_snap = router.snapshot()
        m = snap["metrics"]
        lines = [
            "# HELP gateway_in_flight Current number of requests executing on backends.",
            "# TYPE gateway_in_flight gauge",
            f"gateway_in_flight {snap['in_flight']}",
            "# HELP gateway_queue_depth Current number of requests waiting in Gateway queue.",
            "# TYPE gateway_queue_depth gauge",
            f"gateway_queue_depth {m['queue_depth']['current']}",
            "# HELP gateway_queue_depth_max_observed Peak observed Gateway queue depth.",
            "# TYPE gateway_queue_depth_max_observed gauge",
            f"gateway_queue_depth_max_observed {m['queue_depth']['max_observed']}",
            "# HELP gateway_queue_wait_avg_ms Average queue wait time in ms for queued requests.",
            "# TYPE gateway_queue_wait_avg_ms gauge",
            f"gateway_queue_wait_avg_ms {m['queue_wait']['avg_queued_wait_ms']}",
            "# HELP gateway_queue_wait_max_ms Max observed queue wait time in ms.",
            "# TYPE gateway_queue_wait_max_ms gauge",
            f"gateway_queue_wait_max_ms {m['queue_wait']['max_wait_ms']}",
            "# HELP gateway_reject_total Total rejected requests before entering inference.",
            "# TYPE gateway_reject_total counter",
            f'gateway_reject_total{{reason="invalid_400"}} {m["reject"]["invalid_400"]}',
            f'gateway_reject_total{{reason="quota_429"}} {m["reject"]["quota_429"]}',
            f'gateway_reject_total{{reason="rate_limit_429"}} {m["reject"]["rate_limit_429"]}',
            f'gateway_reject_total{{reason="queue_full_503"}} {m["reject"]["queue_full_503"]}',
            f'gateway_reject_total{{reason="queue_timeout_503"}} {m["reject"]["queue_timeout_503"]}',
            f'gateway_reject_total{{reason="deadline_504"}} {m["reject"]["deadline_504"]}',
            "# HELP gateway_cancel_total Total cancelled requests by lifecycle stage.",
            "# TYPE gateway_cancel_total counter",
            f'gateway_cancel_total{{stage="in_queue"}} {m["cancel"]["in_queue"]}',
            f'gateway_cancel_total{{stage="in_flight"}} {m["cancel"]["in_flight"]}',
            "# HELP gateway_router_estimated_prefix_affinity_avg Estimated token-prefix affinity (Gateway estimate only, NOT real vLLM cache hits).",
            "# TYPE gateway_router_estimated_prefix_affinity_avg gauge",
            f"gateway_router_estimated_prefix_affinity_avg {r_snap['stats']['avg_estimated_prefix_affinity']}",
            "# HELP gateway_replica_inflight Current in-flight requests per backend replica.",
            "# TYPE gateway_replica_inflight gauge",
        ]
        for rep in r_snap["registry"]["replicas"]:
            rid = rep["replica_id"]
            lines.append(f'gateway_replica_inflight{{replica="{rid}"}} {rep["inflight"]}')
        lines.append("# HELP gateway_replica_healthy Health status per backend replica (1=available, 0=unavailable).")
        lines.append("# TYPE gateway_replica_healthy gauge")
        for rep in r_snap["registry"]["replicas"]:
            rid = rep["replica_id"]
            avail = 1 if rep["available"] else 0
            lines.append(f'gateway_replica_healthy{{replica="{rid}"}} {avail}')
        lines.append("# HELP gateway_router_decisions_total Total routing decisions by replica.")
        lines.append("# TYPE gateway_router_decisions_total counter")
        for rid, count in r_snap["stats"]["decisions_by_replica"].items():
            lines.append(f'gateway_router_decisions_total{{replica="{rid}"}} {count}')
        lines.append("# HELP gateway_replica_observed_vllm_prefix_cache_hit_rate Ground-truth vLLM prefix cache hit rate scraped from real vLLM /metrics.")
        lines.append("# TYPE gateway_replica_observed_vllm_prefix_cache_hit_rate gauge")
        for rep in r_snap["registry"]["replicas"]:
            hit_rate = rep["observed_vllm_metrics"]["prefix_cache_hit_rate"]
            if hit_rate is not None:
                rid = rep["replica_id"]
                lines.append(
                    f'gateway_replica_observed_vllm_prefix_cache_hit_rate{{replica="{rid}"}} {hit_rate}'
                )

        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        client: httpx.AsyncClient = request.app.state.http_client
        registry: ReplicaRegistry = request.app.state.registry
        ctx = RequestContext(request_id=request_id, is_stream=False)
        headers = _build_upstream_headers(request, ctx)

        avail = registry.available_replicas()
        if not avail:
            return _error_json_response(
                503, "No healthy backend replicas available", "no_healthy_replicas", request_id
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
                    "X-Routed-Replica": avail[0].replica_id,
                },
            )
        except httpx.TimeoutException as exc:
            return _error_json_response(
                504, f"Backend timeout on /v1/models: {exc}", "gateway_timeout", request_id
            )
        except httpx.HTTPError as exc:
            return _error_json_response(
                502, f"Backend unreachable on /v1/models: {exc}", "bad_gateway", request_id
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        admission: AdmissionController = request.app.state.admission
        registry: ReplicaRegistry = request.app.state.registry
        router: InstanceRouter = request.app.state.router

        # Parse and validate JSON body while still in NOT_STARTED state
        # Invalid requests are rejected BEFORE touching the admission queue or backend!
        try:
            body = await request.json()
        except Exception as exc:
            admission.record_invalid_request()
            return _error_json_response(
                400, f"Invalid JSON request body: {exc}", "invalid_request_error", request_id
            )

        if (
            not isinstance(body, dict)
            or not body.get("model")
            or "messages" not in body
            or not isinstance(body.get("messages"), list)
        ):
            admission.record_invalid_request()
            return _error_json_response(
                400,
                "Invalid request schema: 'model' (str) and 'messages' (list) are required",
                "invalid_request_error",
                request_id,
            )

        policy_override = request.headers.get("X-Routing-Policy")
        if policy_override:
            try:
                RoutingPolicy.from_str(policy_override)
            except ValueError as exc:
                admission.record_invalid_request()
                return _error_json_response(
                    400, str(exc), "invalid_routing_policy", request_id
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
            return _error_json_response(
                exc.status_code,
                exc.message,
                exc.error_type,
                request_id,
                retry_after=exc.retry_after,
            )
        except asyncio.CancelledError:
            ctx.mark_finished(
                upstream_closed=True,
                client_disconnected=True,
                error="cancelled_in_queue",
            )
            request.app.state.active_requests.pop(request_id, None)
            raise

        # Check remaining deadline right after acquiring an in-flight slot
        rem_deadline = ctx.remaining_deadline()
        if rem_deadline is not None and rem_deadline <= 0:
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error="deadline_exceeded")
            request.app.state.active_requests.pop(request_id, None)
            return _error_json_response(
                504,
                "Request deadline expired immediately after queue wait",
                "deadline_exceeded",
                request_id,
            )

        # =====================================================================
        # M2: Late-Binding Instance Selection (Filter -> Score -> Pick)
        # Runs AFTER Admission Control so load signals are fresh & accurate!
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
            return _error_json_response(
                503,
                str(exc),
                "no_healthy_replicas",
                request_id,
                retry_after=1.0,
            )

        target_replica = decision.replica
        registry.acquire_replica(target_replica.replica_id)
        ctx.routed_replica_id = target_replica.replica_id
        ctx.routing_policy = decision.policy.value
        ctx.estimated_prefix_affinity = decision.estimated_prefix_affinity

        routing_headers = {
            "X-Request-ID": request_id,
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
            try:
                upstream_resp = await asyncio.wait_for(
                    client.post(target_url, json=body, headers=headers),
                    timeout=effective_timeout,
                )
                ctx.mark_finished(upstream_closed=True)
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "application/json"
                    ),
                    headers=routing_headers,
                )
            except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
                failed = True
                ctx.mark_finished(upstream_closed=True, error=f"timeout: {exc}")
                return _error_json_response(
                    504,
                    f"Upstream vLLM timed out: {exc}",
                    "gateway_timeout",
                    request_id,
                )
            except httpx.HTTPError as exc:
                failed = True
                ctx.mark_finished(upstream_closed=True, error=f"http_error: {exc}")
                return _error_json_response(
                    502,
                    f"Upstream vLLM error: {exc}",
                    "bad_gateway",
                    request_id,
                )
            except asyncio.CancelledError:
                admission.record_inflight_cancel()
                ctx.mark_finished(
                    upstream_closed=True, client_disconnected=True, error="cancelled"
                )
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

            # If backend rejected request with non-200 status before streaming,
            # read error body, close upstream stream, release replica & slot, and return.
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
            return _error_json_response(
                502, "Upstream closed stream before first token", "bad_gateway", request_id
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            _release_replica_once(registry, ctx, failed=True)
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error=f"pre_stream_timeout: {exc}")
            request.app.state.active_requests.pop(request_id, None)
            return _error_json_response(
                504,
                f"Upstream timed out before first token (TTFT/deadline timeout): {exc}",
                "gateway_timeout",
                request_id,
            )
        except httpx.HTTPError as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            _release_replica_once(registry, ctx, failed=True)
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error=f"pre_stream_error: {exc}")
            request.app.state.active_requests.pop(request_id, None)
            return _error_json_response(
                502,
                f"Upstream failed before first token: {exc}",
                "bad_gateway",
                request_id,
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
            raise

        # Phase 2 (STREAMING): First chunk arrived! Commit HTTP 200 OK headers.
        ctx.mark_streaming()

        async def sse_passthrough_generator() -> AsyncIterator[bytes]:
            client_left = False
            stream_err: Optional[str] = None
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
                # in-flight count, AND release the admission slot!
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

        return StreamingResponse(
            sse_passthrough_generator(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                **routing_headers,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_gateway_app()
