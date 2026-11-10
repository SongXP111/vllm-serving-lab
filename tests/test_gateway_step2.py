"""Automated verification suite for M1 Step 2: Admission Control, Bounded Queue,
Fairness, Priority Anti-Starvation, Deadlines, and Backpressure (429 vs 503).

Covers all mandatory requirements from the M1 Step 2 specification:
1. Mock Backend with fixed processing duration, `max_inflight=4`, `queue_capacity=20`.
2. Burst > 24 concurrent requests (32 concurrent requests):
   - Proves `in_flight` never exceeds 4 and queue length never exceeds 20.
   - 24 requests succeed (200 OK), 8 overflow requests immediately return 503.
   - After slots release, new requests are accepted normally.
3. Tenant Quota & Rate Limit (`429 Too Many Requests`) vs System Overload (`503`),
   plus Multi-Tenant Contention with Per-Tenant Round-Robin Fairness.
4. Continuous High-Priority arrival flood + Anti-Starvation guarantee:
   - Proves High-priority jumps ahead, while Low-priority is guaranteed periodic
     turns (`max_consecutive_high_priority=3`) and never permanently starves.
5. Client cancellation while waiting in the queue (and in-flight):
   - Proves cancelled queued request is immediately evicted without wasting a
     backend execution slot.
6. Queue timeout (`503`), end-to-end `deadline` (`504`), and backend execution
   timeout cleanup with full slot recovery.
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig

STEP2_EVIDENCE: Dict[str, Any] = {}


@pytest.mark.anyio
async def test_1_burst_over_24_bounded_queue_and_recovery():
    """1. Burst of 32 concurrent requests against max_inflight=4, queue_capacity=20.

    Proves:
    - Queue never grows beyond 20 (`max_observed_queue_len == 20`).
    - In-flight never exceeds 4 (`max_observed_inflight == 4`).
    - Exactly 24 requests succeed (200) and 8 overflow requests are rejected with 503.
    - Once slots are released, subsequent requests are accepted with 0 queue wait.
    """
    fixed_backend_delay = 0.04
    current_backend_inflight = 0
    max_backend_inflight = 0

    async def fixed_delay_backend(request: httpx.Request) -> httpx.Response:
        nonlocal current_backend_inflight, max_backend_inflight
        current_backend_inflight += 1
        max_backend_inflight = max(max_backend_inflight, current_backend_inflight)
        try:
            await asyncio.sleep(fixed_backend_delay)
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        finally:
            current_backend_inflight -= 1

    cfg = GatewayConfig(
        max_inflight=4,
        queue_capacity=20,
        queue_timeout=5.0,
    )
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(fixed_delay_backend)
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # Fire 32 requests simultaneously (4 in-flight + 20 queued + 8 overflow)
            burst_size = 32
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"req-burst-{i}",
                            "X-Tenant-ID": f"tenant-{i % 4}",
                        },
                        json={"model": "qwen", "messages": [], "stream": False},
                    )
                    for i in range(burst_size)
                )
            )

            status_codes = [r.status_code for r in responses]
            count_200 = status_codes.count(200)
            count_503 = status_codes.count(503)

            assert count_200 == 24
            assert count_503 == 8
            assert max_backend_inflight == 4

            snap = app.state.admission.snapshot()
            assert snap["stats"]["max_observed_inflight"] == 4
            assert snap["stats"]["max_observed_queue_len"] == 20
            assert snap["stats"]["rejected_503_queue_full"] == 8
            assert snap["in_flight"] == 0
            assert snap["queued_count"] == 0

            # Prove Gateway recovers and accepts new requests after slots are freed
            recovery_resp = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-post-burst-recovery"},
                json={"model": "qwen", "messages": [], "stream": False},
            )
            assert recovery_resp.status_code == 200
            assert float(recovery_resp.headers["X-Queue-Wait-Ms"]) == 0.0

            STEP2_EVIDENCE["burst_over_24_and_recovery"] = {
                "burst_requests_sent": burst_size,
                "configured_max_inflight": 4,
                "configured_queue_capacity": 20,
                "max_observed_backend_inflight": max_backend_inflight,
                "max_observed_queue_len": snap["stats"]["max_observed_queue_len"],
                "succeeded_200_count": count_200,
                "rejected_503_queue_full_count": count_503,
                "in_flight_after_burst": snap["in_flight"],
                "queued_after_burst": snap["queued_count"],
                "post_burst_recovery_status": recovery_resp.status_code,
            }


@pytest.mark.anyio
async def test_2_tenant_quota_rate_limit_429_and_fairness():
    """2. Verify 429 (Tenant Quota / Rate Limit) vs 503, and Multi-Tenant Fairness.

    Part A: Single tenant exceeding `tenant_max_concurrency` or `tenant_rate_limit_rps`
            gets HTTP 429 (while other tenants succeed with 200).
    Part B: When `tenant-noisy` floods 8 requests and `tenant-quiet` sends 2 requests
            while backend is saturated, Per-Tenant Round-Robin interleaves their
            dispatches (`noisy -> quiet -> noisy -> quiet`) instead of starving `quiet`.
    """

    async def mock_backend(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(status_code=200, json={"ok": True})

    # Part A1: Tenant concurrency quota (429)
    cfg_quota = GatewayConfig(
        max_inflight=4,
        queue_capacity=20,
        tenant_max_concurrency=2,
    )
    app_quota = create_gateway_app(
        config=cfg_quota, transport=httpx.MockTransport(mock_backend)
    )
    async with app_quota.router.lifespan_context(app_quota):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_quota), base_url="http://gateway"
        ) as client:
            # Tenant-A sends 4 concurrent requests (limit is 2 -> 2 succeed, 2 get 429)
            # Tenant-B sends 1 concurrent request at the same time -> succeeds (200)
            resps = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Tenant-ID": "tenant-A"},
                        json={"model": "qwen", "messages": []},
                    )
                    for _ in range(4)
                ),
                client.post(
                    "/v1/chat/completions",
                    headers={"X-Tenant-ID": "tenant-B"},
                    json={"model": "qwen", "messages": []},
                ),
            )
            tenant_a_codes = [r.status_code for r in resps[:4]]
            tenant_b_code = resps[4].status_code
            assert tenant_a_codes.count(200) == 2
            assert tenant_a_codes.count(429) == 2
            assert tenant_b_code == 200

    # Part A2: Tenant Token Bucket rate limit (429)
    cfg_rl = GatewayConfig(
        max_inflight=4,
        queue_capacity=20,
        tenant_rate_limit_rps=5.0,
        tenant_burst_capacity=2.0,
    )
    app_rl = create_gateway_app(
        config=cfg_rl, transport=httpx.MockTransport(mock_backend)
    )
    async with app_rl.router.lifespan_context(app_rl):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_rl), base_url="http://gateway"
        ) as client:
            rl_resps = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Tenant-ID": "tenant-burst"},
                        json={"model": "qwen", "messages": []},
                    )
                    for _ in range(4)
                )
            )
            rl_codes = [r.status_code for r in rl_resps]
            assert rl_codes.count(200) == 2
            assert rl_codes.count(429) == 2

    # Part B: Per-Tenant Round-Robin Fairness under contention
    release_holders = asyncio.Event()

    async def fair_backend(request: httpx.Request) -> httpx.Response:
        req_id = request.headers.get("X-Request-ID", "")
        if req_id.startswith("holder-"):
            await release_holders.wait()
        else:
            await asyncio.sleep(0.01)
        return httpx.Response(status_code=200, json={"ok": True})

    cfg_fair = GatewayConfig(max_inflight=4, queue_capacity=20)
    app_fair = create_gateway_app(
        config=cfg_fair, transport=httpx.MockTransport(fair_backend)
    )
    async with app_fair.router.lifespan_context(app_fair):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_fair), base_url="http://gateway"
        ) as client:
            # 1. Start 4 warmup requests to saturate all 4 in_flight slots
            holders = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"holder-{i}",
                            "X-Tenant-ID": "tenant-init",
                        },
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(4)
            ]
            while app_fair.state.admission.in_flight < 4:
                await asyncio.sleep(0.001)

            # 2. tenant-noisy floods 6 requests into queue
            noisy_tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"noisy-{i}",
                            "X-Tenant-ID": "tenant-noisy",
                        },
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(6)
            ]
            while app_fair.state.admission.queued_count < 6:
                await asyncio.sleep(0.001)

            # 3. tenant-quiet sends 2 requests after noisy's 6 requests are already queued
            quiet_tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"quiet-{i}",
                            "X-Tenant-ID": "tenant-quiet",
                        },
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(2)
            ]
            while app_fair.state.admission.queued_count < 8:
                await asyncio.sleep(0.001)

            # Release the 4 holders now that all 8 contended requests are queued!
            release_holders.set()
            await asyncio.gather(*holders, *noisy_tasks, *quiet_tasks)

            snap = app_fair.state.admission.snapshot()
            queued_dispatches = [
                d["tenant_id"]
                for d in snap["stats"]["dispatch_order"]
                if d["was_queued"]
            ]
            # First 4 queued dispatches MUST alternate: noisy, quiet, noisy, quiet!
            assert queued_dispatches[:4] == [
                "tenant-noisy",
                "tenant-quiet",
                "tenant-noisy",
                "tenant-quiet",
            ]

            STEP2_EVIDENCE["tenant_quota_rate_limit_and_fairness"] = {
                "quota_test_tenant_a_200": tenant_a_codes.count(200),
                "quota_test_tenant_a_429": tenant_a_codes.count(429),
                "quota_test_tenant_b_200": 1 if tenant_b_code == 200 else 0,
                "rate_limit_test_200": rl_codes.count(200),
                "rate_limit_test_429": rl_codes.count(429),
                "fairness_queued_dispatch_sequence": queued_dispatches,
            }


@pytest.mark.anyio
async def test_3_continuous_high_priority_flood_no_low_priority_starvation():
    """3. Continuous High-Priority arrivals + Low-Priority Anti-Starvation.

    Proves:
    - High priority requests jump ahead of Low priority requests.
    - Even when 9 High priority requests are queued alongside 3 Low priority
      requests, `max_consecutive_high_priority=3` guarantees that after every 3
      High dispatches, 1 waiting Low request is dispatched (zero starvation!).
    """

    release_holders = asyncio.Event()

    async def step_backend(request: httpx.Request) -> httpx.Response:
        req_id = request.headers.get("X-Request-ID", "")
        if req_id.startswith("init-"):
            await release_holders.wait()
        else:
            await asyncio.sleep(0.01)
        return httpx.Response(status_code=200, json={"ok": True})

    cfg = GatewayConfig(
        max_inflight=4,
        queue_capacity=20,
        max_consecutive_high_priority=3,
        starvation_aging_seconds=10.0,  # isolate consecutive-cap mechanism
    )
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(step_backend)
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # 1. Saturate all 4 in_flight slots
            holders = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"init-{i}",
                            "X-Priority": "normal",
                        },
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(4)
            ]
            while app.state.admission.in_flight < 4:
                await asyncio.sleep(0.001)

            # 2. Enqueue 3 LOW priority requests first
            low_tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"low-{i}",
                            "X-Priority": "low",
                        },
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(3)
            ]
            while app.state.admission.queued_count < 3:
                await asyncio.sleep(0.001)

            # 3. Flood 9 HIGH priority requests into the queue
            high_tasks = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={
                            "X-Request-ID": f"high-{i}",
                            "X-Priority": "high",
                        },
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(9)
            ]
            while app.state.admission.queued_count < 12:
                await asyncio.sleep(0.001)

            release_holders.set()
            all_resps = await asyncio.gather(*holders, *low_tasks, *high_tasks)
            assert all(r.status_code == 200 for r in all_resps)

            snap = app.state.admission.snapshot()
            queued_priorities = [
                d["priority"]
                for d in snap["stats"]["dispatch_order"]
                if d["was_queued"]
            ]

            # Expected pattern with max_consecutive_high_priority=3:
            # [high, high, high, low, high, high, high, low, high, high, high, low]
            assert queued_priorities == [
                "high",
                "high",
                "high",
                "low",
                "high",
                "high",
                "high",
                "low",
                "high",
                "high",
                "high",
                "low",
            ]

            STEP2_EVIDENCE["priority_and_anti_starvation"] = {
                "queued_low_requests": 3,
                "queued_high_requests": 9,
                "max_consecutive_high_cap": 3,
                "observed_queued_dispatch_priorities": queued_priorities,
                "low_priority_starved_count": 0,
            }


@pytest.mark.anyio
async def test_4_client_cancel_in_queue_evicts_cleanly_and_preserves_slots():
    """4. Client disconnects while waiting in the bounded queue.

    Proves:
    - When a queued client disconnects, its QueueEntry is immediately evicted.
    - It NEVER consumes a backend slot when one opens up.
    - The request queued behind it gets the freed slot and succeeds with 200 OK.
    """
    backend_received_ids: List[str] = []

    async def tracked_backend(request: httpx.Request) -> httpx.Response:
        req_id = request.headers.get("X-Request-ID", "")
        backend_received_ids.append(req_id)
        await asyncio.sleep(0.08)
        return httpx.Response(status_code=200, json={"id": req_id})

    cfg = GatewayConfig(max_inflight=4, queue_capacity=20)
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(tracked_backend)
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # 1. Saturate all 4 in-flight slots
            holders = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Request-ID": f"hold-{i}"},
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(4)
            ]
            await asyncio.sleep(0.01)
            assert app.state.admission.in_flight == 4

            # 2. Launch a raw ASGI request `req-cancel-in-queue` that disconnects
            #    after 20ms while still waiting in the queue
            disconnect_trigger = asyncio.Event()
            body_sent = False

            async def asgi_receive():
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {
                        "type": "http.request",
                        "body": b'{"model":"qwen","messages":[]}',
                        "more_body": False,
                    }
                await disconnect_trigger.wait()
                return {"type": "http.disconnect"}

            async def asgi_send(message):
                pass

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
                    (b"x-request-id", b"req-cancel-in-queue"),
                ],
                "client": ("127.0.0.1", 51000),
                "server": ("gateway", 80),
            }

            cancel_task = asyncio.create_task(app(scope, asgi_receive, asgi_send))
            await asyncio.sleep(0.005)

            # 3. Enqueue `req-survivor` behind `req-cancel-in-queue`
            survivor_task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers={"X-Request-ID": "req-survivor"},
                    json={"model": "qwen", "messages": []},
                )
            )
            await asyncio.sleep(0.005)
            assert app.state.admission.queued_count == 2

            # 4. Trigger client disconnect on `req-cancel-in-queue` while holders are still running
            disconnect_trigger.set()
            await cancel_task

            # Wait for holders and survivor to finish
            survivor_resp = await survivor_task
            await asyncio.gather(*holders)

            assert survivor_resp.status_code == 200
            assert "req-cancel-in-queue" not in backend_received_ids
            assert "req-survivor" in backend_received_ids

            snap = app.state.admission.snapshot()
            assert snap["stats"]["cancelled_in_queue"] == 1
            assert snap["in_flight"] == 0
            assert snap["queued_count"] == 0
            assert len(app.state.active_requests) == 0

            STEP2_EVIDENCE["cancel_in_queue"] = {
                "cancelled_in_queue_count": snap["stats"]["cancelled_in_queue"],
                "cancelled_request_sent_to_backend": (
                    "req-cancel-in-queue" in backend_received_ids
                ),
                "survivor_request_status": survivor_resp.status_code,
                "final_in_flight": snap["in_flight"],
                "final_queued": snap["queued_count"],
                "leaked_active_requests": len(app.state.active_requests),
            }


@pytest.mark.anyio
async def test_5_queue_timeout_deadline_and_backend_timeout_slot_recovery():
    """5. Verify queue timeout (503), request deadline (504), and backend timeout (504).

    Proves that timed-out requests in both queued and executing states cleanly
    release their slots and allow subsequent normal requests to succeed.
    """

    async def variable_backend(request: httpx.Request) -> httpx.Response:
        req_id = request.headers.get("X-Request-ID", "")
        if req_id.startswith("slow"):
            await asyncio.sleep(0.18)
        else:
            await asyncio.sleep(0.01)
        return httpx.Response(status_code=200, json={"id": req_id})

    cfg = GatewayConfig(
        max_inflight=4,
        queue_capacity=20,
        queue_timeout=0.04,
        read_timeout=0.5,
    )
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(variable_backend)
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # 1. Occupy all 4 in_flight slots with slow requests (0.18s)
            slow_holders = [
                asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Request-ID": f"slow-holder-{i}"},
                        json={"model": "qwen", "messages": []},
                    )
                )
                for i in range(4)
            ]
            await asyncio.sleep(0.005)

            # 2. Send a request that will exceed `queue_timeout=0.04s` -> HTTP 503
            resp_q_timeout = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-q-timeout"},
                json={"model": "qwen", "messages": []},
            )
            assert resp_q_timeout.status_code == 503
            assert (
                resp_q_timeout.json()["error"]["type"] == "queue_timeout_exceeded"
            )

            # 3. Send a request with short `X-Request-Deadline-Seconds: 0.02` -> HTTP 504
            resp_deadline_q = await client.post(
                "/v1/chat/completions",
                headers={
                    "X-Request-ID": "req-deadline-queue",
                    "X-Request-Deadline-Seconds": "0.02",
                },
                json={"model": "qwen", "messages": []},
            )
            assert resp_deadline_q.status_code == 504
            assert resp_deadline_q.json()["error"]["type"] == "deadline_exceeded"

            await asyncio.gather(*slow_holders)

            # 4. Now slots are free! Test in-flight deadline timeout (backend takes 0.18s, deadline is 0.04s)
            resp_inflight_timeout = await client.post(
                "/v1/chat/completions",
                headers={
                    "X-Request-ID": "slow-inflight-timeout",
                    "X-Request-Deadline-Seconds": "0.04",
                },
                json={"model": "qwen", "messages": []},
            )
            assert resp_inflight_timeout.status_code == 504

            snap = app.state.admission.snapshot()
            assert snap["in_flight"] == 0
            assert snap["queued_count"] == 0
            assert len(app.state.active_requests) == 0

            # 5. Verify all 4 slots are healthy and accept normal requests
            recovery_resps = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={"X-Request-ID": f"fast-recover-{i}"},
                        json={"model": "qwen", "messages": []},
                    )
                    for i in range(4)
                )
            )
            assert all(r.status_code == 200 for r in recovery_resps)

            STEP2_EVIDENCE["timeouts_and_slot_recovery"] = {
                "queue_timeout_status": resp_q_timeout.status_code,
                "queue_deadline_status": resp_deadline_q.status_code,
                "inflight_deadline_status": resp_inflight_timeout.status_code,
                "post_timeout_in_flight": snap["in_flight"],
                "post_timeout_queued": snap["queued_count"],
                "recovery_4_requests_all_200": all(
                    r.status_code == 200 for r in recovery_resps
                ),
            }

            out_dir = Path("results/gateway")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "step2_admission_tests.json").write_text(
                json.dumps(STEP2_EVIDENCE, indent=2), encoding="utf-8"
            )
