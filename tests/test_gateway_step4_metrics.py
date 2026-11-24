"""Automated unit tests for Step 4: Prometheus Metrics (Counter/Gauge/Histogram)
and End-to-End Request Correlation (`gateway_request_id` <-> `backend_request_id`).
"""
import asyncio
import json
from typing import Dict, List

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from gateway.registry import ReplicaConfig


@pytest.mark.anyio
async def test_step4_prometheus_metrics_and_histograms():
    """Verify all required Step 4 Prometheus Counter, Gauge, and Histogram families."""

    async def mock_vllm(request: httpx.Request) -> httpx.Response:
        # Verify Gateway propagates both gateway_request_id and W3C traceparent upstream!
        assert request.headers.get("X-Gateway-Request-ID") == request.headers.get(
            "X-Request-ID"
        )
        assert request.headers.get("traceparent", "").startswith("00-")

        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            async def gen():
                yield b'data: {"id":"chatcmpl-stream-99","choices":[{"delta":{"content":"Hi"}}]}\n\n'
                await asyncio.sleep(0.015)
                yield b'data: {"id":"chatcmpl-stream-99","choices":[{"delta":{"content":" there"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            class SimpleStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    async for c in gen():
                        yield c

                async def aclose(self):
                    pass

            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=SimpleStream(),
            )
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-json-42",
                "choices": [{"message": {"content": "Hello"}}],
            },
        )

    cfg = GatewayConfig(
        max_inflight=4,
        queue_capacity=10,
        tenant_quotas={"throttled": 1},
        replicas=[
            ReplicaConfig("replica-0", "http://replica-0:8000", max_inflight=2),
            ReplicaConfig("replica-1", "http://replica-1:8000", max_inflight=2),
        ],
        routing_policy="least-load",
        router_rng_seed=42,
    )
    app = create_gateway_app(config=cfg, transport=httpx.MockTransport(mock_vllm))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # 1. Non-streaming request under `least-load` with custom traceparent
            custom_tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
            r_json = await client.post(
                "/v1/chat/completions",
                headers={
                    "X-Request-ID": "gw-req-json-1",
                    "traceparent": custom_tp,
                },
                json={
                    "model": "qwen2.5-7b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert r_json.status_code == 200
            assert r_json.headers["X-Gateway-Request-ID"] == "gw-req-json-1"
            assert r_json.headers["X-Backend-Request-ID"] == "chatcmpl-json-42"
            assert r_json.headers["traceparent"] == custom_tp

            # 2. Streaming request under `cache+load-aware`
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "X-Request-ID": "gw-req-stream-1",
                    "X-Routing-Policy": "cache+load-aware",
                },
                json={
                    "model": "qwen2.5-7b",
                    "messages": [{"role": "user", "content": "hi stream"}],
                    "stream": True,
                },
            ) as r_stream:
                assert r_stream.status_code == 200
                assert r_stream.headers["X-Gateway-Request-ID"] == "gw-req-stream-1"
                assert r_stream.headers["X-Backend-Request-ID"] == "chatcmpl-stream-99"
                assert float(r_stream.headers["X-Gateway-TTFT-Ms"]) >= 0.0
                async for _ in r_stream.aiter_bytes():
                    pass

            # 3. Invalid request -> 400 rejection counter
            r_bad = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "gw-req-bad-1"},
                json={"model": "qwen2.5-7b"},
            )
            assert r_bad.status_code == 400

            # 4. Verify `/admin/traces/{gateway_request_id}` end-to-end correlation!
            t_json = (await client.get("/admin/traces/gw-req-json-1")).json()
            assert t_json["gateway_request_id"] == "gw-req-json-1"
            assert t_json["backend_request_id"] == "chatcmpl-json-42"
            assert t_json["traceparent"] == custom_tp
            assert t_json["routing_policy"] == "least-load"
            assert t_json["backend_replica_id"] in ("replica-0", "replica-1")

            t_stream = (await client.get("/admin/traces/gw-req-stream-1")).json()
            assert t_stream["gateway_request_id"] == "gw-req-stream-1"
            assert t_stream["backend_request_id"] == "chatcmpl-stream-99"
            assert t_stream["routing_policy"] == "cache+load-aware"
            assert t_stream["ttft_ms"] is not None
            assert t_stream["itl_avg_ms"] is not None

            # 5. Verify Prometheus `/metrics` exposition format
            prom_resp = await client.get("/metrics")
            assert prom_resp.status_code == 200
            text = prom_resp.text

            # Required Counters
            assert 'gateway_requests_total{status_code="200",stream="false",policy="least-load"} 1' in text
            assert 'gateway_requests_total{status_code="200",stream="true",policy="cache+load-aware"} 1' in text
            assert 'gateway_requests_total{status_code="400",stream="false",policy="none"} 1' in text
            assert 'gateway_rejections_total{reason="invalid_400"} 1' in text
            assert 'gateway_router_policy_selections_total{policy="least-load"} 1' in text
            assert 'gateway_router_policy_selections_total{policy="cache+load-aware"} 1' in text

            # Required Gauges
            assert "gateway_queue_depth 0" in text
            assert "gateway_inflight_requests 0" in text

            # Required Histograms (seconds base unit with _bucket, _sum, _count)
            for hist_name in (
                "gateway_queue_wait_seconds",
                "gateway_ttft_seconds",
                "gateway_itl_seconds",
                "gateway_e2e_seconds",
            ):
                assert f"# TYPE {hist_name} histogram" in text
                assert f'{hist_name}_bucket{{le="+Inf"}}' in text
                assert f"{hist_name}_sum " in text
                assert f"{hist_name}_count " in text

            assert "gateway_queue_wait_seconds_count 3" in text
            assert "gateway_ttft_seconds_count 2" in text
            assert "gateway_itl_seconds_count 1" in text
            assert "gateway_e2e_seconds_count 3" in text
