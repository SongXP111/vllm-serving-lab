"""Unit tests for Step 6: Dynamic Replica-Aware Admission Capacity & Queue Drainage."""
import asyncio
from typing import Set

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig


@pytest.mark.anyio
async def test_dynamic_max_inflight_scales_with_k8s_ready_replicas_and_drains_queue():
    """Verify Gateway scales effective_max_inflight with K8s replicas and wakes queued waiters."""
    current_ips: Set[str] = {"10.244.0.10"}
    release_backend = asyncio.Event()
    started_on_backend = asyncio.Event()
    active_backend_calls = 0
    max_observed_backend_calls = 0

    async def fake_dns(host: str, port: int) -> Set[str]:
        return set(current_ips)

    async def slow_backend(request: httpx.Request) -> httpx.Response:
        nonlocal active_backend_calls, max_observed_backend_calls
        active_backend_calls += 1
        max_observed_backend_calls = max(max_observed_backend_calls, active_backend_calls)
        if active_backend_calls >= 2:
            started_on_backend.set()
        try:
            await release_backend.wait()
            return httpx.Response(
                status_code=200,
                json={
                    "id": f"chatcmpl-{request.url.host}",
                    "choices": [{"message": {"content": "ok"}}],
                },
            )
        finally:
            active_backend_calls -= 1

    cfg = GatewayConfig(
        k8s_headless_service="mock-vllm-headless.llm-serving.svc.cluster.local",
        k8s_backend_port=8000,
        k8s_discovery_interval_seconds=60.0,  # manually triggered via sync_once
        max_inflight=8,
        per_replica_max_inflight=2,
        dynamic_max_inflight=True,
        queue_capacity=10,
        queue_timeout=10.0,
        routing_policy="least-load",
    )
    app = create_gateway_app(
        config=cfg,
        transport=httpx.MockTransport(slow_backend),
        dns_resolver=fake_dns,
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # 1 replica -> effective_max_inflight == 2
            assert app.state.admission.effective_max_inflight == 2
            ready_json = (await client.get("/readyz")).json()
            assert ready_json["effective_max_inflight"] == 2

            # Launch 4 concurrent requests: 2 run immediately, 2 sit in Gateway queue!
            req_tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "mock-qwen2.5-7b-fp8",
                            "messages": [{"role": "user", "content": f"q-{i}"}],
                            "stream": False,
                        },
                    )
                )
                for i in range(4)
            ]
            await asyncio.wait_for(started_on_backend.wait(), timeout=2.0)
            await asyncio.sleep(0.05)

            assert app.state.admission.in_flight == 2
            assert app.state.admission.queued_count == 2

            # Prometheus /metrics exposes both gateway_queue_depth=2 and effective ceiling=2
            metrics_text = (await client.get("/metrics")).text
            assert "gateway_queue_depth 2" in metrics_text
            assert "gateway_effective_max_inflight 2" in metrics_text

            # Simulate KEDA + HPA scaling mock-vllm from 1 -> 2 replicas!
            current_ips.add("10.244.0.11")
            await app.state.discoverer.sync_once()
            await asyncio.sleep(0.05)

            # Effective capacity jumps 2 -> 4 and IMMEDIATELY wakes both queued waiters!
            assert app.state.admission.effective_max_inflight == 4
            assert app.state.admission.in_flight == 4
            assert app.state.admission.queued_count == 0
            assert max_observed_backend_calls == 4

            # Simulate scale-down 2 -> 1 while 4 requests are still in flight:
            # Capacity contracts to 2 gracefully without cancelling in-flight requests.
            current_ips.remove("10.244.0.11")
            await app.state.discoverer.sync_once()
            assert app.state.admission.effective_max_inflight == 2

            release_backend.set()
            responses = await asyncio.gather(*req_tasks)
            assert all(r.status_code == 200 for r in responses)
            assert app.state.admission.in_flight == 0
            assert app.state.admission.queued_count == 0
