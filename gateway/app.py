"""Production-style Async LLM Serving Gateway (M1 Step 1 & Step 2).

Responsibilities:
- Step 1:
  1. Single-backend async forwarding for `POST /v1/chat/completions` (both
     `stream=false` JSON and `stream=true` chunk-by-chunk SSE passthrough).
  2. Zero full-response buffering in streaming mode (`aiter_raw()`).
  3. Connection pool reuse via FastAPI `lifespan` and `httpx.AsyncClient`.
  4. Granular 4D timeouts (`connect`, `read`, `write`, `pool`).
  5. End-to-end `request_id` generation and propagation (`X-Request-ID`).
  6. Explicit 3-state request lifecycle (`NOT_STARTED` -> `STREAMING` -> `FINISHED`).
  7. Client disconnect detection (`request.is_disconnected()` & `CancelledError`)
     with immediate upstream cancellation (`await upstream_resp.aclose()`).
- Step 2:
  8. Admission Control (`max_inflight=4`) & Bounded Queue (`queue_capacity=20`).
  9. Flow Control: `tenant_id` quota & Token Bucket rate limit (`429 Too Many Requests`)
     strictly separated from system overload backpressure (`503 Service Unavailable`).
  10. Priority tiers (`high` > `normal` > `low`) with anti-starvation cap/aging
      and Per-Tenant Round-Robin fairness within each priority band.
  11. End-to-end `deadline` and cancellation cleanup across both Queued and
      In-Flight states.
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
        app.state.config = cfg
        app.state.admission = AdmissionController(cfg)
        app.state.http_client = httpx.AsyncClient(
            base_url=cfg.backend_url,
            timeout=timeout,
            limits=limits,
            transport=transport,
        )
        app.state.active_requests: Dict[str, RequestContext] = {}
        app.state.recent_requests: Dict[str, RequestContext] = {}
        logger.info(
            "Gateway started (backend=%s, max_inflight=%d, queue_capacity=%d)",
            cfg.backend_url,
            cfg.max_inflight,
            cfg.queue_capacity,
        )
        try:
            yield
        finally:
            await app.state.http_client.aclose()
            logger.info("Gateway HTTP connection pool closed.")

    app = FastAPI(title="vLLM Serving Gateway", version="0.2.0", lifespan=lifespan)

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

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/admission")
    async def admission_status(request: Request) -> Dict[str, Any]:
        admission: AdmissionController = request.app.state.admission
        return admission.snapshot()

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        client: httpx.AsyncClient = request.app.state.http_client
        ctx = RequestContext(request_id=request_id, is_stream=False)
        headers = _build_upstream_headers(request, ctx)
        try:
            resp = await client.get("/v1/models", headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
                headers={"X-Request-ID": request_id},
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

        # Parse JSON body while still in NOT_STARTED state
        try:
            body = await request.json()
        except Exception as exc:
            return _error_json_response(
                400, f"Invalid JSON request body: {exc}", "invalid_request_error", request_id
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

        admission: AdmissionController = request.app.state.admission

        # =====================================================================
        # Step 2: Admission Control (Accept / Queue / Reject 429 or 503 or 504)
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

        effective_timeout = (
            min(cfg.read_timeout, rem_deadline)
            if rem_deadline is not None
            else cfg.read_timeout
        )

        client: httpx.AsyncClient = request.app.state.http_client
        headers = _build_upstream_headers(request, ctx)

        # =====================================================================
        # Branch A: Non-streaming mode (stream=false)
        # =====================================================================
        if not is_stream:
            try:
                upstream_resp = await asyncio.wait_for(
                    client.post("/v1/chat/completions", json=body, headers=headers),
                    timeout=effective_timeout,
                )
                ctx.mark_finished(upstream_closed=True)
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "application/json"
                    ),
                    headers={
                        "X-Request-ID": request_id,
                        "X-Queue-Wait-Ms": f"{ctx.queue_wait_ms:.2f}",
                    },
                )
            except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
                ctx.mark_finished(upstream_closed=True, error=f"timeout: {exc}")
                return _error_json_response(
                    504,
                    f"Upstream vLLM timed out: {exc}",
                    "gateway_timeout",
                    request_id,
                )
            except httpx.HTTPError as exc:
                ctx.mark_finished(upstream_closed=True, error=f"http_error: {exc}")
                return _error_json_response(
                    502,
                    f"Upstream vLLM error: {exc}",
                    "bad_gateway",
                    request_id,
                )
            except asyncio.CancelledError:
                ctx.mark_finished(
                    upstream_closed=True, client_disconnected=True, error="cancelled"
                )
                raise
            finally:
                admission.release_slot(ctx)
                request.app.state.active_requests.pop(request_id, None)

        # =====================================================================
        # Branch B: Streaming mode (stream=true)
        # 3-State Machine: NOT_STARTED -> STREAMING -> FINISHED
        # =====================================================================
        upstream_req = client.build_request(
            "POST", "/v1/chat/completions", json=body, headers=headers
        )
        upstream_resp: Optional[httpx.Response] = None

        try:
            # Phase 1 (NOT_STARTED): Connect and obtain upstream response headers
            upstream_resp = await asyncio.wait_for(
                client.send(upstream_req, stream=True),
                timeout=effective_timeout,
            )

            # If backend rejected request with non-200 status before streaming,
            # read error body, close upstream stream, release slot, and return.
            if upstream_resp.status_code != 200:
                error_bytes = await upstream_resp.aread()
                await upstream_resp.aclose()
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
                    headers={"X-Request-ID": request_id},
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
            admission.release_slot(ctx)
            ctx.mark_finished(upstream_closed=True, error="empty_stream")
            request.app.state.active_requests.pop(request_id, None)
            return _error_json_response(
                502, "Upstream closed stream before first token", "bad_gateway", request_id
            )
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
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
                # Phase 3 (FINISHED): Always close upstream stream AND release
                # the in-flight admission slot so the next queued request wakes up!
                await upstream_resp.aclose()
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
                "X-Request-ID": request_id,
                "X-Queue-Wait-Ms": f"{ctx.queue_wait_ms:.2f}",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_gateway_app()
