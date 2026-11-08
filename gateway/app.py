"""Production-style Async LLM Serving Gateway (M1 Step 1).

Responsibilities in Step 1:
1. Single-backend async forwarding for `POST /v1/chat/completions` (both
   `stream=false` JSON and `stream=true` chunk-by-chunk SSE passthrough).
2. Zero full-response buffering in streaming mode (`aiter_raw()`).
3. Connection pool reuse via FastAPI `lifespan` and `httpx.AsyncClient`.
4. Granular 4D timeouts (`connect`, `read`, `write`, `pool`).
5. End-to-end `request_id` generation and propagation (`X-Request-ID`).
6. Explicit 3-state request lifecycle (`NOT_STARTED` -> `STREAMING` -> `FINISHED`):
   - Pre-first-token failure (`NOT_STARTED`): returns standard HTTP 502/504 JSON.
   - Mid-stream failure (`STREAMING`, after HTTP 200 headers are committed):
     emits an SSE error event frame and cleanly closes the stream.
7. Client disconnect detection (`request.is_disconnected()` & `CancelledError`)
   with immediate upstream cancellation (`await upstream_resp.aclose()`).
"""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
from typing import AsyncIterator, Dict, Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

from gateway.config import GatewayConfig
from gateway.models import RequestContext, RequestState

logger = logging.getLogger("gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_gateway_app(
    config: Optional[GatewayConfig] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> FastAPI:
    """Create the FastAPI Gateway application.

    Args:
        config: Gateway configuration (backend URL, timeouts, pool limits).
        transport: Optional custom httpx transport (used in unit/failure tests
            to inject mock backend behaviors without real network sockets).
    """
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
        app.state.http_client = httpx.AsyncClient(
            base_url=cfg.backend_url,
            timeout=timeout,
            limits=limits,
            transport=transport,
        )
        # Track active and recent request contexts for observability & testing
        app.state.active_requests: Dict[str, RequestContext] = {}
        app.state.recent_requests: Dict[str, RequestContext] = {}
        logger.info("Gateway started (backend=%s)", cfg.backend_url)
        try:
            yield
        finally:
            await app.state.http_client.aclose()
            logger.info("Gateway HTTP connection pool closed.")

    app = FastAPI(title="vLLM Serving Gateway", version="0.1.0", lifespan=lifespan)

    def _build_upstream_headers(request: Request, request_id: str) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "X-Request-ID": request_id,
            "Content-Type": "application/json",
        }
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        elif cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def _error_json_response(
        status_code: int, message: str, error_type: str, request_id: str
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            headers={"X-Request-ID": request_id},
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "request_id": request_id,
                }
            },
        )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        client: httpx.AsyncClient = request.app.state.http_client
        headers = _build_upstream_headers(request, request_id)
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
        ctx = RequestContext(request_id=request_id, is_stream=is_stream)
        request.app.state.active_requests[request_id] = ctx
        request.app.state.recent_requests[request_id] = ctx

        client: httpx.AsyncClient = request.app.state.http_client
        headers = _build_upstream_headers(request, request_id)

        # =====================================================================
        # Branch A: Non-streaming mode (stream=false)
        # =====================================================================
        if not is_stream:
            try:
                upstream_resp = await client.post(
                    "/v1/chat/completions", json=body, headers=headers
                )
                ctx.mark_finished(upstream_closed=True)
                request.app.state.active_requests.pop(request_id, None)
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "application/json"
                    ),
                    headers={"X-Request-ID": request_id},
                )
            except httpx.TimeoutException as exc:
                ctx.mark_finished(upstream_closed=True, error=f"timeout: {exc}")
                request.app.state.active_requests.pop(request_id, None)
                return _error_json_response(
                    504,
                    f"Upstream vLLM timed out: {exc}",
                    "gateway_timeout",
                    request_id,
                )
            except httpx.HTTPError as exc:
                ctx.mark_finished(upstream_closed=True, error=f"http_error: {exc}")
                request.app.state.active_requests.pop(request_id, None)
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
                request.app.state.active_requests.pop(request_id, None)
                raise

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
            upstream_resp = await client.send(upstream_req, stream=True)

            # If backend rejected request with non-200 status before streaming,
            # read error body, close upstream stream, and return HTTP status directly.
            if upstream_resp.status_code != 200:
                error_bytes = await upstream_resp.aread()
                await upstream_resp.aclose()
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
            # Why? If the backend crashes or times out BEFORE emitting the 1st token,
            # we have NOT committed HTTP 200 OK headers to the client yet, so we can
            # still return a clean HTTP 502 / 504 JSON error response!
            chunk_iter = upstream_resp.aiter_raw()
            first_chunk = await anext(chunk_iter, b"")

        except StopAsyncIteration:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            ctx.mark_finished(upstream_closed=True, error="empty_stream")
            request.app.state.active_requests.pop(request_id, None)
            return _error_json_response(
                502, "Upstream closed stream before first token", "bad_gateway", request_id
            )
        except httpx.TimeoutException as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
            ctx.mark_finished(upstream_closed=True, error=f"pre_stream_timeout: {exc}")
            request.app.state.active_requests.pop(request_id, None)
            return _error_json_response(
                504,
                f"Upstream timed out before first token (TTFT timeout): {exc}",
                "gateway_timeout",
                request_id,
            )
        except httpx.HTTPError as exc:
            if upstream_resp is not None:
                await upstream_resp.aclose()
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
            ctx.mark_finished(
                upstream_closed=True,
                client_disconnected=True,
                error="cancelled_before_first_token",
            )
            request.app.state.active_requests.pop(request_id, None)
            raise

        # Phase 2 (STREAMING): First chunk arrived! Commit HTTP 200 OK headers.
        # Once StreamingResponse starts, HTTP status code 200 is sent on the wire
        # and CANNOT be changed to 502/504 anymore.
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

                async for chunk in chunk_iter:
                    if await request.is_disconnected():
                        client_left = True
                        logger.info(
                            "Client disconnected mid-stream (request_id=%s)", request_id
                        )
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
                # Mid-stream failure AFTER HTTP 200 headers were already sent!
                # We cannot return a 502 HTTP status code now; instead emit an
                # explicit SSE error frame so the client does not hang forever.
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
                # Phase 3 (FINISHED): Always close upstream response to release
                # the connection back to the pool (or abort the in-flight vLLM stream
                # so vLLM frees GPU KV Cache blocks immediately).
                await upstream_resp.aclose()
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
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_gateway_app()
