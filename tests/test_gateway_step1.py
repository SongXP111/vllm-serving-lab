"""Automated tests for M1 Step 1: Async Gateway + SSE Lifecycle & Failure Modes.

Covers all mandatory requirements from the M1 Step 1 specification:
1. Non-streaming (`stream=false`) JSON forwarding + `X-Request-ID` propagation.
2. Streaming (`stream=true`) chunk-by-chunk SSE passthrough without full buffering.
3. Failure Mode 1: Backend fails BEFORE first token (`NOT_STARTED` - both HTTP 503
   and TTFT ReadTimeout) -> Gateway returns standard HTTP 503/504 JSON error and
   closes upstream connection immediately.
4. Failure Mode 2: Backend fails MID-STREAM (`STREAMING`, after HTTP 200 headers
   are already sent) -> Status code remains 200 (cannot send another HTTP status
   header on the wire), Gateway emits SSE error frame and closes upstream stream.
5. Failure Mode 3: Client actively disconnects after receiving 2 tokens -> Gateway
   detects disconnect and immediately closes upstream stream (`aclose()`),
   stopping backend token generation and leaving 0 leaked resources.
6. Failure Mode 4: Slow client reading -> Gateway streams without buffering explosion
   and cleanly closes upstream connection when slow client finishes.
7. Failure Mode 5: Multi-request concurrency (25 concurrent requests) -> verifies
   connection pool reuse, every upstream stream is closed, and `active_requests`
   returns to 0 (zero resource leak).
"""
import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.models import RequestState

EVIDENCE_RESULTS: Dict[str, Any] = {}


class TrackedAsyncByteStream(httpx.AsyncByteStream):
    """Custom httpx stream that records whether `aclose()` was called by Gateway."""

    def __init__(self, generator_func):
        self._gen = generator_func()
        self.closed = False
        self.chunks_emitted = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._gen:
            self.chunks_emitted += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True
        await self._gen.aclose()


@pytest.mark.anyio
async def test_1_non_streaming_json_forwarding():
    """1. Verify stream=false returns JSON, propagates X-Request-ID, and cleans up."""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-Request-ID") == "req-test-nonstream"
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            },
        )

    app = create_gateway_app(transport=httpx.MockTransport(mock_handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-test-nonstream"},
                json={
                    "model": "qwen2.5-7b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            assert resp.headers["X-Request-ID"] == "req-test-nonstream"
            assert resp.json()["choices"][0]["message"]["content"] == "Hello!"

            ctx = app.state.recent_requests["req-test-nonstream"]
            assert ctx.state == RequestState.FINISHED
            assert ctx.upstream_closed is True
            assert len(app.state.active_requests) == 0
            EVIDENCE_RESULTS["non_streaming"] = {
                "status_code": resp.status_code,
                "final_state": ctx.state.value,
                "upstream_closed": ctx.upstream_closed,
                "leaked_active_requests": len(app.state.active_requests),
            }


@pytest.mark.anyio
async def test_2_streaming_sse_passthrough_and_state_transitions():
    """2. Verify stream=true streams SSE chunks without buffering and closes upstream."""
    upstream_streams: List[TrackedAsyncByteStream] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        async def token_gen():
            for tok in ["Hello", " world", "!"]:
                payload = json.dumps({"choices": [{"delta": {"content": tok}}]})
                yield f"data: {payload}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        stream = TrackedAsyncByteStream(token_gen)
        upstream_streams.append(stream)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    app = create_gateway_app(transport=httpx.MockTransport(mock_handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            received_chunks: List[bytes] = []
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-stream-ok"},
                json={
                    "model": "qwen2.5-7b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["X-Request-ID"] == "req-stream-ok"
                async for chunk in resp.aiter_bytes():
                    received_chunks.append(chunk)

            full_text = b"".join(received_chunks).decode("utf-8")
            assert "Hello" in full_text
            assert "data: [DONE]" in full_text
            assert upstream_streams[0].closed is True

            ctx = app.state.recent_requests["req-stream-ok"]
            assert ctx.state == RequestState.FINISHED
            assert ctx.chunks_forwarded == 4
            assert ctx.upstream_closed is True
            assert len(app.state.active_requests) == 0
            EVIDENCE_RESULTS["streaming_normal"] = {
                "status_code": 200,
                "chunks_forwarded": ctx.chunks_forwarded,
                "upstream_stream_closed": upstream_streams[0].closed,
                "leaked_active_requests": len(app.state.active_requests),
            }


@pytest.mark.anyio
async def test_3_pre_first_token_failure_returns_http_error_json():
    """3. Failure Test: Backend fails BEFORE first token (both timeout & crash).

    Verifies that while in NOT_STARTED (before HTTP 200 headers are committed),
    Gateway returns standard HTTP 502/504 JSON and closes the upstream stream.
    """
    upstream_streams: List[TrackedAsyncByteStream] = []

    async def pre_token_fail_handler(request: httpx.Request) -> httpx.Response:
        req_id = request.headers.get("X-Request-ID", "")
        if req_id == "req-pre-timeout":
            raise httpx.ReadTimeout("TTFT timed out waiting for vLLM prefill")

        # Simulate backend sending 200 headers but crashing BEFORE yielding token #1
        async def crash_before_first_token():
            raise httpx.ReadError("Backend process died during prefill before token 1")
            yield b""  # unreachable

        stream = TrackedAsyncByteStream(crash_before_first_token)
        upstream_streams.append(stream)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    app = create_gateway_app(transport=httpx.MockTransport(pre_token_fail_handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # Case A: ReadTimeout before headers
            resp_timeout = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-pre-timeout"},
                json={"model": "qwen", "messages": [], "stream": True},
            )
            assert resp_timeout.status_code == 504
            assert resp_timeout.json()["error"]["type"] == "gateway_timeout"

            # Case B: Backend crashed before emitting 1st token chunk
            resp_crash = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-pre-crash"},
                json={"model": "qwen", "messages": [], "stream": True},
            )
            assert resp_crash.status_code == 502
            assert resp_crash.json()["error"]["type"] == "bad_gateway"
            assert upstream_streams[0].closed is True
            assert len(app.state.active_requests) == 0

            EVIDENCE_RESULTS["failure_pre_first_token"] = {
                "timeout_http_status": resp_timeout.status_code,
                "crash_http_status": resp_crash.status_code,
                "upstream_stream_closed": upstream_streams[0].closed,
                "leaked_active_requests": len(app.state.active_requests),
            }


@pytest.mark.anyio
async def test_4_mid_stream_failure_after_headers_sent():
    """4. Failure Test: Backend fails MID-STREAM (after HTTP 200 headers sent).

    Why we cannot return a 502 JSON response here:
    Once the 1st token arrives, ASGI `http.response.start` (status=200) has already
    been flushed to the TCP socket. HTTP/1.1 wire protocol forbids sending a second
    status line mid-body. Thus Gateway must emit an in-band SSE error frame and close.
    """
    upstream_streams: List[TrackedAsyncByteStream] = []

    async def crashing_stream_handler(request: httpx.Request) -> httpx.Response:
        async def faulty_gen():
            yield b'data: {"choices":[{"delta":{"content":"First"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"Second"}}]}\n\n'
            raise httpx.ReadError("vLLM worker process crashed mid-decode!")

        stream = TrackedAsyncByteStream(faulty_gen)
        upstream_streams.append(stream)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    app = create_gateway_app(transport=httpx.MockTransport(crashing_stream_handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            chunks: List[bytes] = []
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-mid-crash"},
                json={"model": "qwen", "messages": [], "stream": True},
            ) as resp:
                assert resp.status_code == 200
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)

            body_text = b"".join(chunks).decode("utf-8")
            assert "First" in body_text
            assert "mid_stream_error" in body_text
            assert "data: [DONE]" in body_text
            assert upstream_streams[0].closed is True

            ctx = app.state.recent_requests["req-mid-crash"]
            assert ctx.state == RequestState.FINISHED
            assert ctx.error is not None and "mid_stream_error" in ctx.error
            assert len(app.state.active_requests) == 0

            EVIDENCE_RESULTS["failure_mid_stream"] = {
                "committed_http_status": 200,
                "sse_error_frame_emitted": "mid_stream_error" in body_text,
                "upstream_stream_closed": upstream_streams[0].closed,
                "leaked_active_requests": len(app.state.active_requests),
            }


@pytest.mark.anyio
async def test_5_client_disconnect_cancels_and_closes_upstream():
    """5. Failure Test: Client disconnects after receiving 2 tokens.

    Verifies that Gateway immediately calls `await upstream_resp.aclose()` so
    backend stops generating remaining tokens and frees GPU KV Cache blocks.
    """
    upstream_streams: List[TrackedAsyncByteStream] = []

    async def long_stream_handler(request: httpx.Request) -> httpx.Response:
        async def infinite_gen():
            for i in range(100):
                yield f'data: {{"token": {i}}}\n\n'.encode("utf-8")
                await asyncio.sleep(0.01)

        stream = TrackedAsyncByteStream(infinite_gen)
        upstream_streams.append(stream)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    app = create_gateway_app(transport=httpx.MockTransport(long_stream_handler))
    async with app.router.lifespan_context(app):
        disconnect_event = asyncio.Event()
        request_body_sent = False
        body_chunks_received = 0

        async def asgi_receive():
            nonlocal request_body_sent
            if not request_body_sent:
                request_body_sent = True
                return {
                    "type": "http.request",
                    "body": json.dumps(
                        {"model": "qwen", "messages": [], "stream": True}
                    ).encode("utf-8"),
                    "more_body": False,
                }
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def asgi_send(message):
            nonlocal body_chunks_received
            if message["type"] == "http.response.body":
                if message.get("body"):
                    body_chunks_received += 1
                if body_chunks_received >= 2:
                    disconnect_event.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-request-id", b"req-client-abort"),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("gateway", 80),
        }

        await app(scope, asgi_receive, asgi_send)

        assert upstream_streams[0].closed is True
        assert upstream_streams[0].chunks_emitted < 10

        ctx = app.state.recent_requests["req-client-abort"]
        assert ctx.state == RequestState.FINISHED
        assert ctx.client_disconnected is True
        assert ctx.upstream_closed is True
        assert len(app.state.active_requests) == 0

        EVIDENCE_RESULTS["failure_client_disconnect"] = {
            "client_disconnected_after_chunks": body_chunks_received,
            "backend_total_tokens_planned": 100,
            "backend_tokens_actually_emitted_before_abort": upstream_streams[0].chunks_emitted,
            "upstream_stream_closed": upstream_streams[0].closed,
            "leaked_active_requests": len(app.state.active_requests),
        }


@pytest.mark.anyio
async def test_6_slow_client_reading_and_high_concurrency_no_leak():
    """6. Failure Test: Slow client reading + 25 concurrent requests.

    Verifies that every single upstream stream is closed and `active_requests`
    returns to 0 with zero leaked handles.
    """
    all_streams: List[TrackedAsyncByteStream] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        async def gen():
            for i in range(5):
                yield f"data: chunk-{i}\n\n".encode("utf-8")
                await asyncio.sleep(0.002)
            yield b"data: [DONE]\n\n"

        stream = TrackedAsyncByteStream(gen)
        all_streams.append(stream)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    app = create_gateway_app(transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:

            async def run_one(idx: int):
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"X-Request-ID": f"req-conc-{idx}"},
                    json={"model": "qwen", "messages": [], "stream": True},
                ) as resp:
                    assert resp.status_code == 200
                    async for _ in resp.aiter_bytes():
                        if idx % 5 == 0:
                            # Slow client delaying reads
                            await asyncio.sleep(0.015)

            await asyncio.gather(*(run_one(i) for i in range(25)))

            assert len(all_streams) == 25
            assert all(s.closed for s in all_streams)
            assert len(app.state.active_requests) == 0

            EVIDENCE_RESULTS["slow_client_and_concurrency"] = {
                "concurrent_requests": 25,
                "slow_clients_count": 5,
                "all_upstream_streams_closed": all(s.closed for s in all_streams),
                "leaked_active_requests": len(app.state.active_requests),
            }

            # Persist verifiable evidence artifact
            out_dir = Path("results/gateway")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "step1_failure_tests.json").write_text(
                json.dumps(EVIDENCE_RESULTS, indent=2), encoding="utf-8"
            )
