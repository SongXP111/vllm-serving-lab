"""Unit tests for Step 5: Mock Replica Probes/Drain & Gateway K8s Headless DNS Discovery."""
import asyncio
import json
from typing import Set

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from mock_backend.server import create_mock_replica_app


@pytest.mark.anyio
async def test_mock_replica_cold_start_readiness_and_graceful_drain():
    """Verify Startup/Readiness/Liveness separation and graceful streaming drain."""
    app = create_mock_replica_app(startup_delay_sec=0.15)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mock-pod"
        ) as client:
            # 1. During simulated model cold start:
            #    Liveness (/healthz) MUST be 200, while /startupz & /readyz are 503!
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/startupz")).status_code == 503
            assert (await client.get("/readyz")).status_code == 503

            # 2. Wait for model weights to finish loading -> all 3 probes become 200!
            await asyncio.sleep(0.20)
            assert (await client.get("/startupz")).status_code == 200
            assert (await client.get("/readyz")).status_code == 200
            assert (await client.get("/healthz")).status_code == 200

            # 3. Toggle readiness off via /admin/readiness -> /readyz=503, /healthz STILL 200!
            await client.post("/admin/readiness", json={"ready": False})
            assert (await client.get("/readyz")).status_code == 503
            assert (await client.get("/healthz")).status_code == 200
            await client.post("/admin/readiness", json={"ready": True})
            assert (await client.get("/readyz")).status_code == 200

            # 4. Graceful drain: trigger /admin/drain while an SSE stream is running
            #    Wait: drive SSE via raw ASGI so it streams concurrently while we hit /admin/drain!
            stream_chunks = []
            first_chunk_arrived = asyncio.Event()
            body_sent = False

            async def asgi_receive():
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {
                        "type": "http.request",
                        "body": json.dumps(
                            {
                                "model": "mock-qwen2.5-7b-fp8",
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 5,
                                "token_delay_ms": 20.0,
                                "stream": True,
                            }
                        ).encode("utf-8"),
                        "more_body": False,
                    }
                await asyncio.sleep(3600)
                return {"type": "http.disconnect"}

            async def asgi_send(msg):
                if msg["type"] == "http.response.body" and msg.get("body"):
                    stream_chunks.append(msg["body"])
                    first_chunk_arrived.set()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 50000),
                "server": ("mock-pod", 8000),
            }
            stream_task = asyncio.create_task(app(scope, asgi_receive, asgi_send))
            await asyncio.wait_for(first_chunk_arrived.wait(), timeout=2.0)

            # Trigger Phase 1 preStop grace drain mid-stream:
            # /readyz MUST immediately become 503 for Endpoints/DNS removal, while
            # stray requests arriving during DNS propagation still succeed (200)!
            grace_resp = await client.post("/admin/drain?reject_new=false")
            assert grace_resp.status_code == 200
            assert grace_resp.json()["reject_new"] is False
            assert (await client.get("/readyz")).status_code == 503
            stray_req = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-qwen2.5-7b-fp8", "messages": [], "max_tokens": 1, "token_delay_ms": 1.0},
            )
            assert stray_req.status_code == 200

            # Trigger Phase 2 hard drain right before SIGTERM:
            # New requests arriving after propagation window are explicitly rejected (503).
            drain_resp = await client.post("/admin/drain?reject_new=true")
            assert drain_resp.status_code == 200
            assert drain_resp.json()["reject_new"] is True
            assert (await client.get("/readyz")).status_code == 503
            new_req = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-qwen2.5-7b-fp8", "messages": []},
            )
            assert new_req.status_code == 503

            # Existing in-flight stream MUST finish draining cleanly with [DONE]!
            await stream_task
            full_stream = b"".join(stream_chunks).decode("utf-8")
            assert "data: [DONE]" in full_stream
            assert app.state.rt.in_flight == 0


@pytest.mark.anyio
async def test_gateway_headless_dns_discovery_sync_and_drain():
    """Verify Gateway dynamically adds/removes K8s Pod IPs via Headless DNS."""
    current_ips: Set[str] = {"10.244.0.10", "10.244.0.11"}

    async def fake_dns(host: str, port: int) -> Set[str]:
        return set(current_ips)

    async def ok_backend(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"id": f"chatcmpl-{request.url.host}", "choices": [{"message": {"content": "ok"}}]},
        )

    cfg = GatewayConfig(
        k8s_headless_service="mock-vllm-headless.llm-serving.svc.cluster.local",
        k8s_backend_port=8000,
        k8s_discovery_interval_seconds=0.2,
        routing_policy="cache+load-aware",
    )
    app = create_gateway_app(
        config=cfg,
        transport=httpx.MockTransport(ok_backend),
        dns_resolver=fake_dns,
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/readyz")).status_code == 200
            assert len(app.state.registry.available_replicas()) == 2

            # Simulate Pod 10.244.0.10 failing readinessProbe and Pod 10.244.0.12 scaling up
            current_ips.remove("10.244.0.10")
            current_ips.add("10.244.0.12")
            await app.state.discoverer.sync_once()

            avail_ids = {r.replica_id for r in app.state.registry.available_replicas()}
            assert avail_ids == {"pod-10-244-0-11", "pod-10-244-0-12"}
            assert app.state.registry.get("pod-10-244-0-10") is None

            # Simulate 0 ready backend pods -> Gateway /readyz becomes 503 while /healthz stays 200!
            current_ips.clear()
            await app.state.discoverer.sync_once()
            assert (await client.get("/readyz")).status_code == 503
            assert (await client.get("/healthz")).status_code == 200

            disc = (await client.get("/admin/discovery")).json()
            assert disc["enabled"] is True
            assert len(disc["events"]) >= 2
