"""Automated verification suite for M2: ReplicaRegistry & Switchable Routing Policies.

Covers all M2 requirements:
1. ReplicaRegistry state management: endpoint, health, circuit breaker, live
   `inflight`, backend queue & KV capacity hints, and real vLLM `/metrics` sync
   (proving strict separation between Gateway estimated affinity and observed
   vLLM prefix cache hits).
2. Chained token-prefix block hashing (mirroring vLLM APC) & bounded LRU eviction.
3. Switchable policies (`random`, `least-load`, `prefix-aware`, `cache+load-aware`)
   via config, runtime `/admin/routing/policy`, and per-request header override.
4. Four-policy comparison on a shared-prefix multi-tenant workload.
5. Hotspot skew & KV-pressure spillover: proves `cache+load-aware` harvests warm
   prefix affinity at moderate load while spilling over before saturating one hot
   replica (unlike pure `prefix-aware`).
6. Streaming SSE + client disconnect cleanly releases both replica `inflight` and
   global admission slots (zero leak across replicas).
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from gateway.registry import ReplicaConfig
from gateway.router import (
    EstimatedPrefixCacheIndex,
    RoutingPolicy,
    compute_prefix_block_hashes,
)

M2_EVIDENCE: Dict[str, Any] = {}


@pytest.fixture(scope="module", autouse=True)
def persist_m2_evidence():
    yield
    if M2_EVIDENCE:
        out_dir = Path("results/gateway")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "m2_routing_tests.json").write_text(
            json.dumps(M2_EVIDENCE, indent=2), encoding="utf-8"
        )


def _make_long_prompt(prefix_tag: str, suffix: str) -> List[Dict[str, str]]:
    """Build a chat prompt with a realistic >48-token shared system prefix."""
    shared_system = (
        f"System domain {prefix_tag}: You are a senior distributed systems and AI "
        f"infrastructure architect analyzing vLLM serving latency, PagedAttention "
        f"physical KV cache block tables, continuous batching, chunked prefill "
        f"budgets, and prefix cache affinity across cluster replicas for {prefix_tag}."
    )
    return [
        {"role": "system", "content": shared_system},
        {"role": "user", "content": suffix},
    ]


def _three_replicas(max_inflight_each: int = 4) -> List[ReplicaConfig]:
    return [
        ReplicaConfig(
            replica_id="replica-0",
            endpoint="http://replica-0:8000",
            max_inflight=max_inflight_each,
            kv_cache_capacity_blocks=256,
        ),
        ReplicaConfig(
            replica_id="replica-1",
            endpoint="http://replica-1:8000",
            max_inflight=max_inflight_each,
            kv_cache_capacity_blocks=256,
        ),
        ReplicaConfig(
            replica_id="replica-2",
            endpoint="http://replica-2:8000",
            max_inflight=max_inflight_each,
            kv_cache_capacity_blocks=256,
        ),
    ]


@pytest.mark.anyio
async def test_1_replica_registry_health_circuit_breaker_and_vllm_metrics_sync():
    """1. Verify ReplicaRegistry health, circuit breaker, capacity hints, and
    strict separation between estimated prefix affinity and real vLLM cache hits.
    """
    fail_replica_0 = False

    async def multi_replica_backend(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if request.url.path == "/metrics":
            if host == "replica-0":
                metrics_body = (
                    "vllm:num_requests_running 2.0\n"
                    "vllm:num_requests_waiting 1.0\n"
                    "vllm:gpu_cache_usage_perc 0.42\n"
                    "vllm:prefix_cache_queries_total 1000.0\n"
                    "vllm:prefix_cache_hits_total 750.0\n"
                )
            else:
                metrics_body = (
                    "vllm:num_requests_running 0.0\n"
                    "vllm:num_requests_waiting 0.0\n"
                    "vllm:gpu_cache_usage_perc 0.10\n"
                    "vllm:prefix_cache_queries_total 200.0\n"
                    "vllm:prefix_cache_hits_total 40.0\n"
                )
            return httpx.Response(status_code=200, text=metrics_body)

        if host == "replica-0" and fail_replica_0:
            raise httpx.ConnectError("replica-0 connection refused")

        return httpx.Response(
            status_code=200,
            json={"replica": host, "choices": [{"message": {"content": "ok"}}]},
        )

    cfg = GatewayConfig(
        max_inflight=12,
        queue_capacity=20,
        replicas=_three_replicas(max_inflight_each=4),
        routing_policy="prefix-aware",
        circuit_breaker_failure_threshold=2,
        circuit_breaker_cooldown_seconds=30.0,
        router_rng_seed=42,
    )
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(multi_replica_backend)
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # Before syncing real vLLM metrics, observed_vllm_prefix_cache_hit_rate MUST be None
            # even after Gateway estimates prefix affinity!
            prompt = _make_long_prompt("tenant-alpha", "Explain prefix hashing")
            r1 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            assert r1.status_code == 200
            warm_replica = r1.headers["X-Routed-Replica"]

            r2 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            assert r2.status_code == 200
            assert r2.headers["X-Routed-Replica"] == warm_replica
            assert float(r2.headers["X-Estimated-Prefix-Affinity"]) == 1.0

            routing_before_sync = (await client.get("/admin/routing")).json()
            assert routing_before_sync["prefix_estimator"]["is_estimated_only"] is True
            for rep in routing_before_sync["registry"]["replicas"]:
                # Estimated affinity NEVER fabricates real vLLM cache hits!
                assert rep["observed_vllm_metrics"]["synced"] is False
                assert rep["observed_vllm_metrics"]["prefix_cache_hit_rate"] is None

            # Now actively sync real vLLM `/metrics`
            sync_resp = await client.post("/admin/replicas/sync-metrics")
            assert sync_resp.status_code == 200
            rep0 = app.state.registry.get("replica-0")
            assert rep0.backend_waiting_queue == 1
            assert rep0.kv_cache_usage_perc == pytest.approx(0.42)
            assert rep0.observed_vllm_prefix_cache_hit_rate() == pytest.approx(0.75)

            # Health toggle: mark replica-1 & replica-2 unhealthy, force traffic onto replica-0
            await client.post(
                "/admin/replicas/replica-1/health", json={"healthy": False}
            )
            await client.post(
                "/admin/replicas/replica-2/health", json={"healthy": False}
            )
            r_only_0 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            assert r_only_0.status_code == 200
            assert r_only_0.headers["X-Routed-Replica"] == "replica-0"

            # Trip circuit breaker on replica-0 after 2 consecutive failures
            fail_replica_0 = True
            e1 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            e2 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            assert e1.status_code == 502 and e2.status_code == 502
            assert not app.state.registry.get("replica-0").is_available()

            # With all 3 replicas unavailable -> 503 no_healthy_replicas
            e3 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            assert e3.status_code == 503
            assert e3.json()["error"]["type"] == "no_healthy_replicas"

            # Restore replica-1 -> traffic immediately recovers on replica-1
            await client.post(
                "/admin/replicas/replica-1/health", json={"healthy": True}
            )
            r_recovered = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt},
            )
            assert r_recovered.status_code == 200
            assert r_recovered.headers["X-Routed-Replica"] == "replica-1"

            M2_EVIDENCE["registry_and_ground_truth_separation"] = {
                "estimated_affinity_on_repeat_request": float(
                    r2.headers["X-Estimated-Prefix-Affinity"]
                ),
                "observed_hit_rate_before_vllm_sync": None,
                "observed_hit_rate_after_vllm_sync_replica_0": rep0.observed_vllm_prefix_cache_hit_rate(),
                "circuit_breaker_tripped_on_replica_0": True,
                "all_unhealthy_status_code": e3.status_code,
                "recovered_replica": r_recovered.headers["X-Routed-Replica"],
            }


def test_2_chained_prefix_block_hashing_and_lru_eviction():
    """2. Verify vLLM-style chained block hashing and bounded LRU eviction."""
    base_tokens = [f"tok{i}" for i in range(48)]  # 3 full blocks of 16 tokens
    prompt_a = " ".join(base_tokens)
    # Modify 1 token in block #1 (token index 20) -> block #0 matches, block #1 & #2 diverge!
    mutated = list(base_tokens)
    mutated[20] = "MUTATED"
    prompt_b = " ".join(mutated)
    # Short prompt < 16 tokens -> 0 full blocks (mirrors vLLM APC full-block indexing)
    prompt_short = "hello world"

    hashes_a = compute_prefix_block_hashes(prompt_a, block_size=16, namespace="qwen")
    hashes_b = compute_prefix_block_hashes(prompt_b, block_size=16, namespace="qwen")
    hashes_short = compute_prefix_block_hashes(prompt_short, block_size=16, namespace="qwen")

    assert len(hashes_a) == 3
    assert len(hashes_b) == 3
    assert len(hashes_short) == 0
    assert hashes_a[0] == hashes_b[0]
    assert hashes_a[1] != hashes_b[1]
    assert hashes_a[2] != hashes_b[2]  # chained dependency on block #1!

    # Test bounded LRU eviction on EstimatedPrefixCacheIndex
    index = EstimatedPrefixCacheIndex(default_max_blocks_per_replica=4)
    index.record_routed_prefix("replica-0", hashes_a, max_blocks=4)
    matched_b, ratio_b = index.estimate_affinity("replica-0", hashes_b)
    assert matched_b == 1
    assert ratio_b == pytest.approx(1 / 3, abs=1e-3)

    # Insert 3 brand-new blocks (capacity=4 -> evicts oldest 2 blocks from hashes_a)
    other_prompt = " ".join(f"other{i}" for i in range(48))
    hashes_other = compute_prefix_block_hashes(other_prompt, block_size=16, namespace="qwen")
    index.record_routed_prefix("replica-0", hashes_other, max_blocks=4)
    assert index.tracked_blocks_count("replica-0") == 4
    # Because record_routed_prefix refreshes root blocks last (leaf-first eviction),
    # foundation block hashes_a[0] survives while leaf blocks hashes_a[1], hashes_a[2] evict!
    matched_after_evict, _ = index.estimate_affinity("replica-0", hashes_a)
    assert matched_after_evict == 1

    M2_EVIDENCE["chained_hash_and_lru"] = {
        "block0_match_on_mid_prompt_mutation": hashes_a[0] == hashes_b[0],
        "block1_invalidated": hashes_a[1] != hashes_b[1],
        "block2_cascaded_invalidation": hashes_a[2] != hashes_b[2],
        "partial_block_under_16_tokens_count": len(hashes_short),
        "lru_capped_tracked_blocks": index.tracked_blocks_count("replica-0"),
        "foundation_block_retained_after_leaf_eviction": matched_after_evict == 1,
    }


@pytest.mark.anyio
async def test_3_switchable_policies_random_and_least_load():
    """3. Verify runtime policy switching and least-load steering away from busy replicas."""

    async def ok_backend(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"replica": request.url.host})

    cfg = GatewayConfig(
        max_inflight=12,
        queue_capacity=20,
        replicas=_three_replicas(max_inflight_each=4),
        routing_policy="random",
        router_rng_seed=7,
    )
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(ok_backend)
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # Policy 1: random spreads across replicas
            random_hits: List[str] = []
            for i in range(18):
                r = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen",
                        "messages": [{"role": "user", "content": f"hi {i}"}],
                    },
                )
                assert r.status_code == 200
                assert r.headers["X-Routing-Policy"] == "random"
                random_hits.append(r.headers["X-Routed-Replica"])
            assert len(set(random_hits)) == 3

            # Switch runtime policy to `least-load` via admin API
            sw = await client.post(
                "/admin/routing/policy", json={"policy": "least-load"}
            )
            assert sw.status_code == 200
            assert sw.json()["active_policy"] == "least-load"

            # Set asymmetric load hints: replica-0 has waiting=2, replica-1 has waiting=3, replica-2 is idle (0)
            app.state.registry.update_capacity_hints("replica-0", backend_waiting_queue=2)
            app.state.registry.update_capacity_hints("replica-1", backend_waiting_queue=3)
            app.state.registry.update_capacity_hints("replica-2", backend_waiting_queue=0)

            r_ll = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "user", "content": "q"}]},
            )
            assert r_ll.status_code == 200
            assert r_ll.headers["X-Routing-Policy"] == "least-load"
            assert r_ll.headers["X-Routed-Replica"] == "replica-2"

            # Invalid policy override rejected cleanly with 400 before admission queue
            r_bad_pol = await client.post(
                "/v1/chat/completions",
                headers={"X-Routing-Policy": "nonexistent-policy"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "q"}]},
            )
            assert r_bad_pol.status_code == 400


@pytest.mark.anyio
async def test_4_four_policy_comparison_on_shared_prefix_workload():
    """4. Compare all 4 policies (`random`, `least-load`, `prefix-aware`, `cache+load-aware`)
    on an identical workload of 3 distinct shared prefixes x 6 turns = 18 requests.
    """

    async def fast_backend(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"replica": request.url.host})

    # Fixed 18-request workload: 3 shared system prefixes interleaved across 6 rounds
    workload = []
    for round_idx in range(6):
        for prefix_id in ("sys-A", "sys-B", "sys-C"):
            workload.append(
                _make_long_prompt(prefix_id, f"Follow-up turn {round_idx} question?")
            )

    comparison: Dict[str, Any] = {}

    for policy_name in ("random", "least-load", "prefix-aware", "cache+load-aware"):
        cfg = GatewayConfig(
            max_inflight=12,
            queue_capacity=20,
            replicas=_three_replicas(max_inflight_each=4),
            routing_policy=policy_name,
            router_rng_seed=42,
        )
        app = create_gateway_app(
            config=cfg, transport=httpx.MockTransport(fast_backend)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway"
            ) as client:
                affinities: List[float] = []
                for idx, msgs in enumerate(workload):
                    resp = await client.post(
                        "/v1/chat/completions",
                        headers={"X-Request-ID": f"{policy_name}-{idx}"},
                        json={"model": "qwen2.5-7b", "messages": msgs},
                    )
                    assert resp.status_code == 200
                    affinities.append(
                        float(resp.headers["X-Estimated-Prefix-Affinity"])
                    )
                snap = app.state.router.snapshot()
                comparison[policy_name] = {
                    "avg_estimated_prefix_affinity": snap["stats"][
                        "avg_estimated_prefix_affinity"
                    ],
                    "estimated_affinity_nonzero_ratio": snap["stats"][
                        "estimated_affinity_nonzero_ratio"
                    ],
                    "decisions_by_replica": snap["stats"]["decisions_by_replica"],
                }

    # On warm follow-up rounds (15 of 18 requests have a previously seen prefix),
    # `prefix-aware` and `cache+load-aware` MUST beat `random` substantially in
    # estimated prefix affinity!
    assert (
        comparison["prefix-aware"]["avg_estimated_prefix_affinity"]
        > comparison["random"]["avg_estimated_prefix_affinity"]
    )
    assert (
        comparison["cache+load-aware"]["avg_estimated_prefix_affinity"]
        > comparison["random"]["avg_estimated_prefix_affinity"]
    )
    assert comparison["prefix-aware"]["estimated_affinity_nonzero_ratio"] >= 0.80
    assert comparison["cache+load-aware"]["estimated_affinity_nonzero_ratio"] >= 0.80

    M2_EVIDENCE["four_policy_shared_prefix_comparison"] = comparison


@pytest.mark.anyio
async def test_5_hotspot_skew_prefix_aware_vs_cache_load_aware_spillover():
    """5. Hotspot Skew & KV Pressure test:
    - Under a concurrent burst of 6 identical-prefix requests (each replica `max_inflight=4`),
      pure `prefix-aware` packs warm `replica-0` to 4/4 (100% saturation), whereas
      `cache+load-aware` (`hotspot_load_threshold=0.75`) harvests affinity up to 3/4
      and then spills over to idle replicas BEFORE saturating `replica-0`!
    - Under high KV pressure (`kv_cache_usage_perc=0.90`) on warm `replica-0`,
      `cache+load-aware` immediately spills over to protect `replica-0` from KV eviction.
    """
    hot_prompt = _make_long_prompt("HOT-AGENT", "Execute step in shared context")

    async def run_hotspot_burst(policy_name: str) -> Dict[str, Any]:
        release_burst = asyncio.Event()
        burst_started = 0

        async def burst_backend(request: httpx.Request) -> httpx.Response:
            nonlocal burst_started
            req_id = request.headers.get("X-Request-ID", "")
            if req_id.startswith("hot-burst-"):
                burst_started += 1
                await release_burst.wait()
            return httpx.Response(status_code=200, json={"replica": request.url.host})

        cfg = GatewayConfig(
            max_inflight=12,
            queue_capacity=20,
            replicas=_three_replicas(max_inflight_each=4),
            routing_policy=policy_name,
            prefix_weight=0.55,
            load_weight=0.45,
            hotspot_load_threshold=0.75,
            kv_hotspot_threshold=0.85,
            router_rng_seed=123,
        )
        app = create_gateway_app(
            config=cfg, transport=httpx.MockTransport(burst_backend)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway"
            ) as client:
                # 1. Warm up a single replica with `hot_prompt`
                r_warm = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Request-ID": "warmup-hot"},
                    json={"model": "qwen2.5-7b", "messages": hot_prompt},
                )
                assert r_warm.status_code == 200
                warm_id = r_warm.headers["X-Routed-Replica"]

                # 2. Launch 6 concurrent requests all carrying `hot_prompt`
                tasks = [
                    asyncio.create_task(
                        client.post(
                            "/v1/chat/completions",
                            headers={"X-Request-ID": f"hot-burst-{i}"},
                            json={"model": "qwen2.5-7b", "messages": hot_prompt},
                        )
                    )
                    for i in range(6)
                ]
                while burst_started < 6:
                    await asyncio.sleep(0.002)

                # Capture live in-flight distribution across replicas during peak burst!
                peak_inflight_by_replica = {
                    r.replica_id: r.inflight for r in app.state.registry.all_replicas()
                }
                release_burst.set()
                resps = await asyncio.gather(*tasks)
                assert all(r.status_code == 200 for r in resps)
                assert all(r.inflight == 0 for r in app.state.registry.all_replicas())

                return {
                    "warm_replica": warm_id,
                    "peak_inflight_by_replica": peak_inflight_by_replica,
                    "warm_replica_peak_inflight": peak_inflight_by_replica[warm_id],
                    "spilled_to_other_replicas_count": 6
                    - peak_inflight_by_replica[warm_id],
                }

    res_prefix_only = await run_hotspot_burst("prefix-aware")
    res_cache_load = await run_hotspot_burst("cache+load-aware")

    # Pure `prefix-aware` packs warm replica to full saturation (4/4 = 100%)
    assert res_prefix_only["warm_replica_peak_inflight"] == 4
    # `cache+load-aware` caps warm replica at 3/4 (75% hotspot threshold) and spills 3 requests!
    assert res_cache_load["warm_replica_peak_inflight"] == 3
    assert res_cache_load["spilled_to_other_replicas_count"] == 3

    # Also verify KV cache pressure spillover under `cache+load-aware`
    async def ok_backend(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"replica": request.url.host})

    cfg_kv = GatewayConfig(
        max_inflight=12,
        queue_capacity=20,
        replicas=_three_replicas(max_inflight_each=4),
        routing_policy="cache+load-aware",
        kv_hotspot_threshold=0.85,
        router_rng_seed=99,
    )
    app_kv = create_gateway_app(
        config=cfg_kv, transport=httpx.MockTransport(ok_backend)
    )
    async with app_kv.router.lifespan_context(app_kv):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_kv), base_url="http://gateway"
        ) as client:
            r0 = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": hot_prompt},
            )
            warm_kv_id = r0.headers["X-Routed-Replica"]
            # Simulate high KV cache pressure (90%) on the warm replica
            app_kv.state.registry.update_capacity_hints(
                warm_kv_id, kv_cache_usage_perc=0.90
            )
            r_spill = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": hot_prompt},
            )
            assert r_spill.status_code == 200
            assert r_spill.headers["X-Routed-Replica"] != warm_kv_id

    M2_EVIDENCE["hotspot_skew_and_kv_spillover"] = {
        "prefix_aware_burst": res_prefix_only,
        "cache_load_aware_burst": res_cache_load,
        "kv_pressure_90pct_spilled_away_from_warm_replica": (
            r_spill.headers["X-Routed-Replica"] != warm_kv_id
        ),
    }


@pytest.mark.anyio
async def test_6_streaming_disconnect_releases_replica_inflight_and_metrics():
    """6. Verify mid-stream client disconnect cleans up per-replica `inflight` & `/metrics`."""

    class TrackedStream(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            for i in range(50):
                yield f'data: {{"tok": {i}}}\n\n'.encode("utf-8")
                await asyncio.sleep(0.01)

        async def aclose(self):
            self.closed = True

    created_streams: List[TrackedStream] = []

    async def stream_backend(request: httpx.Request) -> httpx.Response:
        s = TrackedStream()
        created_streams.append(s)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=s,
        )

    cfg = GatewayConfig(
        max_inflight=12,
        queue_capacity=20,
        replicas=_three_replicas(max_inflight_each=4),
        routing_policy="cache+load-aware",
        router_rng_seed=42,
    )
    app = create_gateway_app(
        config=cfg, transport=httpx.MockTransport(stream_backend)
    )

    async with app.router.lifespan_context(app):
        disconnect_event = asyncio.Event()
        body_sent = False
        chunks_seen = 0

        async def asgi_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": json.dumps(
                        {
                            "model": "qwen2.5-7b",
                            "messages": _make_long_prompt("STREAM-SESS", "write essay"),
                            "stream": True,
                        }
                    ).encode("utf-8"),
                    "more_body": False,
                }
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def asgi_send(message):
            nonlocal chunks_seen
            if message["type"] == "http.response.body" and message.get("body"):
                chunks_seen += 1
                if chunks_seen >= 2:
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
                (b"x-request-id", b"req-m2-stream-abort"),
            ],
            "client": ("127.0.0.1", 52000),
            "server": ("gateway", 80),
        }

        await app(scope, asgi_receive, asgi_send)

        assert created_streams[0].closed is True
        assert app.state.admission.in_flight == 0
        assert all(r.inflight == 0 for r in app.state.registry.all_replicas())
        assert len(app.state.active_requests) == 0

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            prom = await client.get("/metrics")
            assert prom.status_code == 200
            assert "gateway_router_estimated_prefix_affinity_avg" in prom.text
            assert 'gateway_replica_inflight{replica="replica-0"} 0' in prom.text
            assert 'gateway_replica_healthy{replica="replica-0"} 1' in prom.text

        M2_EVIDENCE["streaming_disconnect_cleanup"] = {
            "upstream_stream_closed": created_streams[0].closed,
            "final_global_inflight": app.state.admission.in_flight,
            "final_per_replica_inflight": {
                r.replica_id: r.inflight for r in app.state.registry.all_replicas()
            },
            "leaked_active_requests": len(app.state.active_requests),
        }
