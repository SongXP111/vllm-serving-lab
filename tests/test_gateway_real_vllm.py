"""End-to-End Integration Tests: Client -> Gateway -> Real vLLM Backend.

Verifies the M1 Step 1 real-backend pass criteria:
1. Client calls real vLLM stably through Gateway (`/v1/models`, `stream=false`, `stream=true`).
2. SSE streaming loses zero frames (every `data: {...}` chunk is valid JSON and ends with `data: [DONE]`).
3. Client mid-stream disconnect triggers real vLLM backend request cleanup:
   verifies via vLLM's `/metrics` (`vllm:num_requests_running`) that aborting a
   long 300-token stream after 3 tokens immediately drops running requests back
   to 0 on the real vLLM engine!
"""
import asyncio
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from gateway.models import RequestState

VLLM_BACKEND_URL = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8000")


def _parse_vllm_running_requests(metrics_text: str) -> float:
    """Extract `vllm:num_requests_running` gauge value from Prometheus text."""
    for line in metrics_text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            parts = line.strip().split()
            if parts:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return 0.0


async def _is_real_vllm_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{VLLM_BACKEND_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.anyio
async def test_real_vllm_end_to_end_and_disconnect_cleanup():
    """Run full integration suite against a live vLLM instance on localhost:8000."""
    if not await _is_real_vllm_ready():
        pytest.skip(
            f"Real vLLM is not reachable at {VLLM_BACKEND_URL}. "
            "Start Docker Desktop and run `docker compose up -d vllm` to run this integration test."
        )

    cfg = GatewayConfig(backend_url=VLLM_BACKEND_URL, read_timeout=60.0)
    app = create_gateway_app(config=cfg)

    report: Dict[str, Any] = {"backend_url": VLLM_BACKEND_URL}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            timeout=60.0,
        ) as gw_client, httpx.AsyncClient(
            base_url=VLLM_BACKEND_URL, timeout=10.0
        ) as direct_vllm:
            # 1. Discover served model via Gateway /v1/models
            models_resp = await gw_client.get("/v1/models")
            assert models_resp.status_code == 200
            model_id = models_resp.json()["data"][0]["id"]
            report["served_model"] = model_id

            # 2. Non-streaming completion via Gateway
            t0 = time.perf_counter()
            non_stream_resp = await gw_client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "real-nonstream-1"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Say hello in 3 words."}],
                    "max_tokens": 16,
                    "stream": False,
                },
            )
            assert non_stream_resp.status_code == 200
            assert non_stream_resp.headers["X-Request-ID"] == "real-nonstream-1"
            content = non_stream_resp.json()["choices"][0]["message"]["content"]
            assert len(content) > 0
            report["non_streaming"] = {
                "status_code": 200,
                "e2e_ms": round((time.perf_counter() - t0) * 1000, 2),
                "reply_preview": content[:60],
            }

            # 3. Streaming SSE integrity (zero dropped frames) via ASGI incremental stream
            sse_lines: List[str] = []
            t_start = time.perf_counter()
            t_first: float = 0.0
            req_sent = False

            async def sse_receive():
                nonlocal req_sent
                if not req_sent:
                    req_sent = True
                    return {
                        "type": "http.request",
                        "body": json.dumps(
                            {
                                "model": model_id,
                                "messages": [
                                    {"role": "user", "content": "Count from 1 to 10."}
                                ],
                                "max_tokens": 32,
                                "stream": True,
                            }
                        ).encode("utf-8"),
                        "more_body": False,
                    }
                await asyncio.sleep(3600)
                return {"type": "http.disconnect"}

            raw_buffer = b""

            async def sse_send(message):
                nonlocal raw_buffer, t_first
                if message["type"] == "http.response.body":
                    chunk = message.get("body", b"")
                    if chunk:
                        if t_first == 0.0:
                            t_first = time.perf_counter()
                        raw_buffer += chunk

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
                    (b"x-request-id", b"real-stream-integrity"),
                ],
                "client": ("127.0.0.1", 50001),
                "server": ("gateway", 80),
            }
            await app(scope, sse_receive, sse_send)

            decoded_stream = raw_buffer.decode("utf-8")
            data_frames = [
                line[6:]
                for line in decoded_stream.splitlines()
                if line.startswith("data: ")
            ]
            assert len(data_frames) >= 2
            assert data_frames[-1].strip() == "[DONE]"
            # Every frame before [DONE] must be valid JSON (no corrupted/dropped frames)
            parsed_tokens = []
            for frame in data_frames[:-1]:
                obj = json.loads(frame)
                delta = obj["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    parsed_tokens.append(delta)

            report["streaming_integrity"] = {
                "total_sse_frames": len(data_frames),
                "done_marker_received": True,
                "ttft_ms": round((t_first - t_start) * 1000, 2),
                "reconstructed_text": "".join(parsed_tokens)[:80],
            }

            # 4. Real Client Disconnect -> Verify real vLLM cleans up running request!
            disconnect_event = asyncio.Event()
            abort_req_sent = False
            chunks_seen = 0

            async def abort_receive():
                nonlocal abort_req_sent
                if not abort_req_sent:
                    abort_req_sent = True
                    return {
                        "type": "http.request",
                        "body": json.dumps(
                            {
                                "model": model_id,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Write a very long 500-word essay about operating systems.",
                                    }
                                ],
                                "max_tokens": 300,
                                "stream": True,
                            }
                        ).encode("utf-8"),
                        "more_body": False,
                    }
                await disconnect_event.wait()
                return {"type": "http.disconnect"}

            async def abort_send(message):
                nonlocal chunks_seen
                if message["type"] == "http.response.body" and message.get("body"):
                    chunks_seen += 1
                    if chunks_seen >= 3:
                        disconnect_event.set()

            abort_scope = {
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
                    (b"x-request-id", b"real-client-abort"),
                ],
                "client": ("127.0.0.1", 50002),
                "server": ("gateway", 80),
            }

            t_abort_start = time.perf_counter()
            await app(abort_scope, abort_receive, abort_send)

            # Poll real vLLM /metrics to verify `vllm:num_requests_running` returns to 0
            cleaned_up_in_ms = None
            for _ in range(20):
                m_resp = await direct_vllm.get("/metrics")
                running = _parse_vllm_running_requests(m_resp.text)
                if running == 0.0:
                    cleaned_up_in_ms = round(
                        (time.perf_counter() - t_abort_start) * 1000, 2
                    )
                    break
                await asyncio.sleep(0.05)

            ctx = app.state.recent_requests["real-client-abort"]
            assert ctx.state == RequestState.FINISHED
            assert ctx.client_disconnected is True
            assert ctx.upstream_closed is True
            assert cleaned_up_in_ms is not None
            assert len(app.state.active_requests) == 0

            report["real_vllm_disconnect_cleanup"] = {
                "planned_max_tokens": 300,
                "client_disconnected_after_chunks": chunks_seen,
                "vllm_num_requests_running_after_abort": 0.0,
                "cleanup_verified_within_ms": cleaned_up_in_ms,
                "leaked_active_requests": len(app.state.active_requests),
            }

            out_dir = Path("results/gateway")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "step1_real_vllm_integration.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
