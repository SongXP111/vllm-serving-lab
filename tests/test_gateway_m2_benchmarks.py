"""M2 Multi-Replica Algorithm Benchmark & Pass-Criteria Suite (Part A: Mock Replicas).

IMPORTANT BENCHMARK DISCLAIMER:
This suite uses a physics-aware multi-replica Mock Simulator (`SimulatedVLLMCluster`)
purely for **multi-replica routing algorithm comparison** (Before/After relative gains
on identical workloads). Simulated `mock_rps` MUST NEVER be reported as real multi-GPU
vLLM TPS; real single-GPU RTX 5080 vLLM behavior is verified separately in
`tests/test_gateway_m2_real_vllm.py`.

Covers all 4 mandatory M2 test scenarios and compares `queue time`, `TTFT P99`,
`mock throughput`, `simulated replica prefix hit`, and `gateway estimated affinity`:
1. Shared System Prompt multi-turn workload (Before: `least-load` vs After: `cache+load-aware`).
2. Mixed Long Prefill + Short Chat workload.
3. Hot Prefix Skew onto one replica — explicitly exposing the **Cache Locality vs
   Load Balance conflict** across `random`, `least-load`, `prefix-aware`, `cache+load-aware`.
4. Unresponsive Replica (Hang -> `ReadTimeout 504` -> Circuit Breaker -> Auto Failover).
"""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from gateway.registry import ReplicaConfig
from gateway.router import compute_prefix_block_hashes, extract_prompt_text

BENCHMARK_RESULTS: Dict[str, Any] = {
    "_disclaimer": (
        "Mock Replica simulator results for multi-replica routing algorithm validation only. "
        "Do NOT interpret `mock_rps` as real multi-GPU vLLM TPS. Real single-GPU RTX 5080 "
        "vLLM verification is recorded separately in results/gateway/m2_real_vllm_verification.json."
    )
}


@pytest.fixture(scope="module", autouse=True)
def persist_m2_benchmark_results():
    yield
    if len(BENCHMARK_RESULTS) > 1:
        out_dir = Path("results/gateway")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "m2_mock_policy_comparison.json").write_text(
            json.dumps(BENCHMARK_RESULTS, indent=2), encoding="utf-8"
        )


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 2)


def _make_prompt_with_blocks(prefix_id: str, prefix_blocks: int, suffix: str) -> List[Dict[str, str]]:
    """Construct a prompt whose system message spans `prefix_blocks` x 16 pseudo-tokens."""
    words = [f"{prefix_id}_w{i}" for i in range(prefix_blocks * 16)]
    return [
        {"role": "system", "content": " ".join(words)},
        {"role": "user", "content": suffix},
    ]


@dataclass
class SimulatedReplicaNode:
    """Physics-aware Mock vLLM replica with local KV Prefix Cache & GPU execution slots."""

    replica_id: str
    gpu_exec_slots: int = 2
    kv_capacity_blocks: int = 256
    prefill_ms_per_miss_block: float = 4.5
    prefill_ms_per_hit_block: float = 0.2
    base_ttft_ms: float = 4.0
    decode_tokens: int = 3
    decode_ms_per_token: float = 6.0
    hang: bool = False

    sem: asyncio.Semaphore = field(init=False)
    kv_cache: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    local_waiting: int = 0
    local_running: int = 0
    peak_local_concurrent: int = 0
    sim_prefix_queries_blocks: int = 0
    sim_prefix_hits_blocks: int = 0
    requests_handled: int = 0

    def __post_init__(self) -> None:
        self.sem = asyncio.Semaphore(self.gpu_exec_slots)

    def simulated_prefix_hit_rate(self) -> float:
        if self.sim_prefix_queries_blocks <= 0:
            return 0.0
        return round(
            self.sim_prefix_hits_blocks / self.sim_prefix_queries_blocks, 4
        )


class SimulatedVLLMCluster:
    """Simulates a 3-replica vLLM cluster with realistic prefill miss/hit latency and queueing."""

    def __init__(
        self,
        nodes: List[SimulatedReplicaNode],
        *,
        block_size: int = 16,
    ) -> None:
        self.nodes: Dict[str, SimulatedReplicaNode] = {n.replica_id: n for n in nodes}
        self.block_size = block_size

    def make_transport(self) -> httpx.MockTransport:
        cluster = self

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host or "replica-0"
            node = cluster.nodes[host]

            if node.hang:
                # Simulate a completely hung / unresponsive backend replica
                await asyncio.sleep(5.0)
                return httpx.Response(status_code=504, json={"error": "hung"})

            body = json.loads(request.content.decode("utf-8"))
            model = str(body.get("model", ""))
            prompt_text = extract_prompt_text(body)
            block_hashes = compute_prefix_block_hashes(
                prompt_text, block_size=cluster.block_size, namespace=model
            )
            is_stream = bool(body.get("stream", False))

            arrived_at = time.monotonic()
            node.local_waiting += 1
            node.peak_local_concurrent = max(
                node.peak_local_concurrent, node.local_waiting + node.local_running
            )

            async def _acquire_and_prefill() -> Tuple[float, float, int, int]:
                await node.sem.acquire()
                node.local_waiting = max(0, node.local_waiting - 1)
                node.local_running += 1
                node.requests_handled += 1
                replica_queue_ms = (time.monotonic() - arrived_at) * 1000.0

                # Check local simulated GPU KV Prefix Cache
                total_blocks = len(block_hashes)
                hit_blocks = 0
                for h in block_hashes:
                    if h in node.kv_cache:
                        hit_blocks += 1
                    else:
                        break
                miss_blocks = max(0, total_blocks - hit_blocks)

                node.sim_prefix_queries_blocks += total_blocks
                node.sim_prefix_hits_blocks += hit_blocks

                # Simulate Prefill compute time (miss blocks are expensive; hit blocks are near-free!)
                prefill_ms = (
                    node.base_ttft_ms
                    + miss_blocks * node.prefill_ms_per_miss_block
                    + hit_blocks * node.prefill_ms_per_hit_block
                )
                await asyncio.sleep(prefill_ms / 1000.0)

                # Populate local GPU KV cache with LRU eviction
                now = time.monotonic()
                for h in reversed(block_hashes):
                    node.kv_cache[h] = now
                    node.kv_cache.move_to_end(h)
                while len(node.kv_cache) > node.kv_capacity_blocks:
                    node.kv_cache.popitem(last=False)

                return replica_queue_ms, prefill_ms, hit_blocks, total_blocks

            if not is_stream:
                replica_q_ms, prefill_ms, hit_b, tot_b = await _acquire_and_prefill()
                try:
                    await asyncio.sleep(
                        (node.decode_tokens * node.decode_ms_per_token) / 1000.0
                    )
                finally:
                    node.local_running = max(0, node.local_running - 1)
                    node.sem.release()
                return httpx.Response(
                    status_code=200,
                    headers={
                        "X-Sim-Replica-Queue-Ms": f"{replica_q_ms:.2f}",
                        "X-Sim-Prefill-Ms": f"{prefill_ms:.2f}",
                        "X-Sim-Hit-Blocks": str(hit_b),
                        "X-Sim-Total-Blocks": str(tot_b),
                    },
                    json={"choices": [{"message": {"content": "ok"}}]},
                )

            # Streaming mode: first token emits right after replica queue + Prefill!
            class SimSSEStream(httpx.AsyncByteStream):
                def __init__(self) -> None:
                    self._released = False
                    self._acquired = False

                def _release_once(self) -> None:
                    if self._acquired and not self._released:
                        self._released = True
                        node.local_running = max(0, node.local_running - 1)
                        node.sem.release()

                async def __aiter__(self):
                    try:
                        replica_q_ms, prefill_ms, hit_b, tot_b = (
                            await _acquire_and_prefill()
                        )
                        self._acquired = True
                        meta = json.dumps(
                            {
                                "replica": host,
                                "replica_queue_ms": round(replica_q_ms, 2),
                                "prefill_ms": round(prefill_ms, 2),
                                "hit_blocks": hit_b,
                                "total_blocks": tot_b,
                                "choices": [{"delta": {"content": "tok0"}}],
                            }
                        )
                        yield f"data: {meta}\n\n".encode("utf-8")
                        for t in range(1, node.decode_tokens):
                            await asyncio.sleep(node.decode_ms_per_token / 1000.0)
                            payload = json.dumps(
                                {"choices": [{"delta": {"content": f" tok{t}"}}]}
                            )
                            yield f"data: {payload}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                    finally:
                        self._release_once()

                async def aclose(self) -> None:
                    self._release_once()

            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=SimSSEStream(),
            )

        return httpx.MockTransport(handler)

    def aggregate_prefix_hit_rate(self) -> float:
        tot_q = sum(n.sim_prefix_queries_blocks for n in self.nodes.values())
        tot_h = sum(n.sim_prefix_hits_blocks for n in self.nodes.values())
        return round(tot_h / tot_q, 4) if tot_q > 0 else 0.0


async def _run_streaming_workload(
    *,
    policy: str,
    waves: List[List[Tuple[str, List[Dict[str, str]]]]],
    max_inflight_per_replica: int = 4,
    gpu_exec_slots: int = 2,
    kv_capacity_blocks: int = 256,
    hotspot_load_threshold: float = 0.50,
    prefill_ms_per_miss_block: float = 4.5,
    decode_ms_per_token: float = 6.0,
    warmup_request: Optional[Tuple[str, List[Dict[str, str]]]] = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Execute an identical multi-wave streaming workload against 3 Simulated Replicas."""
    nodes = [
        SimulatedReplicaNode(
            replica_id=f"replica-{i}",
            gpu_exec_slots=gpu_exec_slots,
            kv_capacity_blocks=kv_capacity_blocks,
            prefill_ms_per_miss_block=prefill_ms_per_miss_block,
            decode_ms_per_token=decode_ms_per_token,
        )
        for i in range(3)
    ]
    cluster = SimulatedVLLMCluster(nodes)
    replicas = [
        ReplicaConfig(
            replica_id=f"replica-{i}",
            endpoint=f"http://replica-{i}:8000",
            max_inflight=max_inflight_per_replica,
            kv_cache_capacity_blocks=kv_capacity_blocks,
        )
        for i in range(3)
    ]
    cfg = GatewayConfig(
        max_inflight=max_inflight_per_replica * 3,
        queue_capacity=40,
        replicas=replicas,
        routing_policy=policy,
        prefix_weight=0.55,
        load_weight=0.45,
        hotspot_load_threshold=hotspot_load_threshold,
        router_rng_seed=rng_seed,
    )
    app = create_gateway_app(config=cfg, transport=cluster.make_transport())

    ttft_samples_ms: List[float] = []
    gw_queue_samples_ms: List[float] = []
    replica_queue_samples_ms: List[float] = []
    total_queue_samples_ms: List[float] = []
    status_counts: Dict[int, int] = {}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            if warmup_request is not None:
                w_id, w_msgs = warmup_request
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"X-Request-ID": w_id},
                    json={"model": "qwen2.5-7b", "messages": w_msgs, "stream": True},
                ) as resp:
                    assert resp.status_code == 200
                    async for _ in resp.aiter_bytes():
                        pass

            # Reset simulated hit stats after optional warmup so measurement isolates test waves
            for n in nodes:
                n.sim_prefix_queries_blocks = 0
                n.sim_prefix_hits_blocks = 0
                n.peak_local_concurrent = 0
                n.requests_handled = 0

            async def send_one(req_id: str, msgs: List[Dict[str, str]]) -> None:
                # Drive ASGI directly so TTFT is recorded the exact moment the
                # 1st `http.response.body` SSE chunk leaves StreamingResponse
                # (httpx.ASGITransport buffers full responses before yielding).
                t0 = time.monotonic()
                body_bytes = json.dumps(
                    {"model": "qwen2.5-7b", "messages": msgs, "stream": True}
                ).encode("utf-8")
                body_sent = False
                resp_status = 500
                resp_headers: Dict[str, str] = {}
                first_seen = False
                rep_q_ms = 0.0

                async def asgi_receive():
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {
                            "type": "http.request",
                            "body": body_bytes,
                            "more_body": False,
                        }
                    await asyncio.sleep(3600)
                    return {"type": "http.disconnect"}

                async def asgi_send(message):
                    nonlocal resp_status, first_seen, rep_q_ms
                    if message["type"] == "http.response.start":
                        resp_status = message["status"]
                        for k, v in message.get("headers", []):
                            resp_headers[k.decode("latin1").lower()] = v.decode(
                                "latin1"
                            )
                    elif message["type"] == "http.response.body":
                        chunk = message.get("body", b"")
                        if resp_status == 200 and chunk and not first_seen:
                            ttft_ms = (time.monotonic() - t0) * 1000.0
                            first_seen = True
                            gw_q_ms = float(
                                resp_headers.get("x-queue-wait-ms", "0.0")
                            )
                            text = chunk.decode("utf-8", errors="ignore")
                            for line in text.splitlines():
                                if line.startswith("data: ") and "[DONE]" not in line:
                                    try:
                                        obj = json.loads(line[6:])
                                        rep_q_ms = float(
                                            obj.get("replica_queue_ms", 0.0)
                                        )
                                    except Exception:
                                        pass
                                    break
                            ttft_samples_ms.append(round(ttft_ms, 2))
                            gw_queue_samples_ms.append(round(gw_q_ms, 2))
                            replica_queue_samples_ms.append(round(rep_q_ms, 2))
                            total_queue_samples_ms.append(
                                round(gw_q_ms + rep_q_ms, 2)
                            )

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
                        (b"x-request-id", req_id.encode("latin1")),
                    ],
                    "client": ("127.0.0.1", 53000),
                    "server": ("gateway", 80),
                }
                await app(scope, asgi_receive, asgi_send)
                status_counts[resp_status] = status_counts.get(resp_status, 0) + 1

            wall_start = time.monotonic()
            for wave in waves:
                await asyncio.gather(*(send_one(rid, msgs) for rid, msgs in wave))
            wall_elapsed = max(1e-4, time.monotonic() - wall_start)

            snap = app.state.router.snapshot()
            completed_200 = status_counts.get(200, 0)
            return {
                "completed_200": completed_200,
                "wall_time_ms": round(wall_elapsed * 1000.0, 2),
                "mock_throughput_rps": round(completed_200 / wall_elapsed, 2),
                "queue_time_avg_ms": round(
                    sum(total_queue_samples_ms) / max(1, len(total_queue_samples_ms)), 2
                ),
                "queue_time_p99_ms": _percentile(total_queue_samples_ms, 99),
                "ttft_avg_ms": round(
                    sum(ttft_samples_ms) / max(1, len(ttft_samples_ms)), 2
                ),
                "ttft_p50_ms": _percentile(ttft_samples_ms, 50),
                "ttft_p99_ms": _percentile(ttft_samples_ms, 99),
                "simulated_replica_prefix_hit_rate": cluster.aggregate_prefix_hit_rate(),
                "gateway_estimated_prefix_affinity": snap["stats"][
                    "avg_estimated_prefix_affinity"
                ],
                "requests_by_replica": {
                    n.replica_id: n.requests_handled for n in nodes
                },
                "peak_local_concurrent_by_replica": {
                    n.replica_id: n.peak_local_concurrent for n in nodes
                },
            }


@pytest.mark.anyio
async def test_scenario_1_shared_system_prompt_multi_turn_before_after():
    """Scenario 1: Shared System Prompt multi-turn workload (Before: least-load vs After: cache+load-aware).

    6 distinct agents (12 prefix blocks each = 72 total working-set blocks) across
    3 replicas with finite per-replica KV capacity = 30 blocks (cluster aggregate = 90 blocks).
    - `least-load` sprays all 6 agents across all 3 replicas -> continuous LRU cache thrashing!
    - `cache+load-aware` shards 2 sticky agents per replica (24 <= 30 blocks) ->
      multiplies effective cluster KV capacity by 3x and slashes TTFT P50/avg!
    """
    waves: List[List[Tuple[str, List[Dict[str, str]]]]] = []
    # 4 turns x 6 distinct long-prefix agents (sent in waves of 3 concurrent requests) = 24 requests
    for turn in range(4):
        waves.append(
            [
                (
                    f"s1-t{turn}-agent{a}",
                    _make_prompt_with_blocks(f"agent{a}", 12, f"turn {turn} query"),
                )
                for a in (0, 1, 2)
            ]
        )
        waves.append(
            [
                (
                    f"s1-t{turn}-agent{a}",
                    _make_prompt_with_blocks(f"agent{a}", 12, f"turn {turn} query"),
                )
                for a in (3, 4, 5)
            ]
        )

    results: Dict[str, Any] = {}
    for pol in ("random", "least-load", "prefix-aware", "cache+load-aware"):
        results[pol] = await _run_streaming_workload(
            policy=pol,
            waves=waves,
            max_inflight_per_replica=4,
            gpu_exec_slots=2,
            kv_capacity_blocks=30,
            prefill_ms_per_miss_block=8.0,
            decode_ms_per_token=6.0,
            rng_seed=42,
        )

    before = results["least-load"]
    after = results["cache+load-aware"]

    # Before (least-load) vs After (cache+load-aware) assertions:
    assert (
        after["simulated_replica_prefix_hit_rate"]
        > before["simulated_replica_prefix_hit_rate"] + 0.30
    )
    assert after["ttft_p50_ms"] < before["ttft_p50_ms"]
    assert after["ttft_avg_ms"] < before["ttft_avg_ms"]
    assert after["mock_throughput_rps"] > before["mock_throughput_rps"]

    BENCHMARK_RESULTS["scenario_1_shared_system_prompt_multi_turn"] = {
        "before_policy": "least-load",
        "after_policy": "cache+load-aware",
        "policies": results,
        "before_vs_after_delta": {
            "prefix_hit_rate_before": before["simulated_replica_prefix_hit_rate"],
            "prefix_hit_rate_after": after["simulated_replica_prefix_hit_rate"],
            "ttft_avg_ms_before": before["ttft_avg_ms"],
            "ttft_avg_ms_after": after["ttft_avg_ms"],
            "ttft_p50_ms_before": before["ttft_p50_ms"],
            "ttft_p50_ms_after": after["ttft_p50_ms"],
            "ttft_p99_ms_before": before["ttft_p99_ms"],
            "ttft_p99_ms_after": after["ttft_p99_ms"],
            "mock_rps_before": before["mock_throughput_rps"],
            "mock_rps_after": after["mock_throughput_rps"],
        },
    }


@pytest.mark.anyio
async def test_scenario_2_mixed_long_and_short_workload():
    """Scenario 2: Mixed Long Prefill (16 blocks) + Short Chat (2 blocks) across 4 policies."""
    waves: List[List[Tuple[str, List[Dict[str, str]]]]] = []
    for turn in range(4):
        waves.append(
            [
                (
                    f"s2-t{turn}-long1",
                    _make_prompt_with_blocks("longDoc1", 16, f"summarize section {turn}"),
                ),
                (
                    f"s2-t{turn}-long2",
                    _make_prompt_with_blocks("longDoc2", 16, f"audit clause {turn}"),
                ),
                (
                    f"s2-t{turn}-short1",
                    _make_prompt_with_blocks(f"shortA{turn}", 2, "quick reply"),
                ),
                (
                    f"s2-t{turn}-short2",
                    _make_prompt_with_blocks(f"shortB{turn}", 2, "quick check"),
                ),
            ]
        )

    results: Dict[str, Any] = {}
    for pol in ("random", "least-load", "prefix-aware", "cache+load-aware"):
        results[pol] = await _run_streaming_workload(
            policy=pol,
            waves=waves,
            max_inflight_per_replica=4,
            gpu_exec_slots=2,
            prefill_ms_per_miss_block=4.0,
            decode_ms_per_token=5.0,
            rng_seed=42,
        )

    assert (
        results["cache+load-aware"]["simulated_replica_prefix_hit_rate"]
        > results["least-load"]["simulated_replica_prefix_hit_rate"]
    )
    assert (
        results["cache+load-aware"]["ttft_avg_ms"]
        < results["random"]["ttft_avg_ms"]
    )

    BENCHMARK_RESULTS["scenario_2_mixed_long_short"] = {"policies": results}


@pytest.mark.anyio
async def test_scenario_3_hotspot_prefix_conflict_cache_locality_vs_load_balance():
    """Scenario 3: Hot Prefix Skew onto one replica — Cache Locality vs Load Balance Conflict!

    Warm up `replica-0` with `hotDoc` (12 blocks), then fire a single simultaneous
    burst of 8 requests ALL sharing `hotDoc` (`gpu_exec_slots=2`, `max_inflight=8`).
    Demonstrates the fundamental conflict:
    - `prefix-aware`: maximizes simulated prefix hit (~100%), BUT packs all 8 requests
      onto `replica-0` -> worst queue time, worst TTFT P99, and lowest mock throughput!
    - `least-load`: spreads evenly (`3/3/2`) with lowest local queue time, BUT incurs
      cold misses on idle replicas (lower prefix hit).
    - `cache+load-aware`: harvests warm hits on `replica-0` up to hotspot threshold,
      then spills excess requests to idle replicas -> dramatically beats `prefix-aware`
      on queue time, TTFT P99, and throughput while beating `least-load` on prefix hit!
    """
    hot_msgs = _make_prompt_with_blocks("hotDoc", 12, "analyze hot prompt")
    burst_wave = [
        (f"s3-hot-{i}", _make_prompt_with_blocks("hotDoc", 12, f"hot query {i}"))
        for i in range(8)
    ]

    results: Dict[str, Any] = {}
    for pol in ("random", "least-load", "prefix-aware", "cache+load-aware"):
        results[pol] = await _run_streaming_workload(
            policy=pol,
            waves=[burst_wave],
            warmup_request=("s3-warmup", hot_msgs),
            max_inflight_per_replica=8,
            gpu_exec_slots=2,
            hotspot_load_threshold=0.35,  # ~3/8 inflight triggers hotspot spill
            prefill_ms_per_miss_block=4.0,
            decode_ms_per_token=10.0,
            rng_seed=7,
        )

    pa = results["prefix-aware"]
    ll = results["least-load"]
    cl = results["cache+load-aware"]

    # Prove the Cache Locality vs Load Balance conflict quantitatively:
    # 1. `prefix-aware` gets highest prefix hit, BUT stacks all 8 requests on 1 replica
    #    causing severe queue time & TTFT P99 degradation!
    assert max(pa["requests_by_replica"].values()) == 8
    assert pa["simulated_replica_prefix_hit_rate"] >= ll["simulated_replica_prefix_hit_rate"]
    assert pa["queue_time_p99_ms"] > cl["queue_time_p99_ms"]
    assert pa["ttft_p99_ms"] > cl["ttft_p99_ms"]
    assert cl["mock_throughput_rps"] > pa["mock_throughput_rps"]
    # 2. `cache+load-aware` spills load off the hot replica (max <= 4 instead of 8)
    assert max(cl["requests_by_replica"].values()) < 8
    # 3. `cache+load-aware` preserves better or equal prefix hit than `least-load`
    assert cl["simulated_replica_prefix_hit_rate"] >= ll["simulated_replica_prefix_hit_rate"]

    BENCHMARK_RESULTS["scenario_3_hotspot_locality_vs_load_conflict"] = {
        "conflict_summary": {
            "prefix_aware_max_reqs_on_one_replica": max(
                pa["requests_by_replica"].values()
            ),
            "prefix_aware_queue_p99_ms": pa["queue_time_p99_ms"],
            "prefix_aware_ttft_p99_ms": pa["ttft_p99_ms"],
            "prefix_aware_prefix_hit": pa["simulated_replica_prefix_hit_rate"],
            "least_load_queue_p99_ms": ll["queue_time_p99_ms"],
            "least_load_ttft_p99_ms": ll["ttft_p99_ms"],
            "least_load_prefix_hit": ll["simulated_replica_prefix_hit_rate"],
            "cache_load_aware_max_reqs_on_one_replica": max(
                cl["requests_by_replica"].values()
            ),
            "cache_load_aware_queue_p99_ms": cl["queue_time_p99_ms"],
            "cache_load_aware_ttft_p99_ms": cl["ttft_p99_ms"],
            "cache_load_aware_prefix_hit": cl["simulated_replica_prefix_hit_rate"],
        },
        "policies": results,
    }


@pytest.mark.anyio
async def test_scenario_4_unresponsive_replica_hang_timeout_and_failover():
    """Scenario 4: One replica stops responding (hangs -> 504 ReadTimeout -> circuit breaker -> failover)."""
    nodes = [
        SimulatedReplicaNode(
            replica_id=f"replica-{i}",
            gpu_exec_slots=2,
            prefill_ms_per_miss_block=2.0,
            decode_ms_per_token=3.0,
        )
        for i in range(3)
    ]
    cluster = SimulatedVLLMCluster(nodes)
    replicas = [
        ReplicaConfig(
            replica_id=f"replica-{i}",
            endpoint=f"http://replica-{i}:8000",
            max_inflight=4,
            kv_cache_capacity_blocks=256,
        )
        for i in range(3)
    ]
    cfg = GatewayConfig(
        max_inflight=12,
        queue_capacity=20,
        read_timeout=0.06,  # Fast timeout to detect hung replica in 60ms
        replicas=replicas,
        routing_policy="cache+load-aware",
        circuit_breaker_failure_threshold=2,
        circuit_breaker_cooldown_seconds=30.0,
        router_rng_seed=42,
    )
    app = create_gateway_app(config=cfg, transport=cluster.make_transport())

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            # 1. Warm up a prompt so one replica becomes its sticky warm replica
            prompt = _make_prompt_with_blocks("failoverDoc", 8, "initial turn")
            r_init = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen2.5-7b", "messages": prompt, "stream": False},
            )
            assert r_init.status_code == 200
            hung_id = r_init.headers["X-Routed-Replica"]

            # 2. Hang the warm replica completely! (Simulates GPU deadlock / hung worker)
            cluster.nodes[hung_id].hang = True

            # 3. Send 10 sequential requests for the same prompt:
            #    - First 2 hit the hung replica -> 504 timeout after 60ms -> trips circuit breaker!
            #    - Remaining 8 automatically fail over to healthy replicas and succeed (200 OK)!
            statuses: List[int] = []
            routed_ids: List[str] = []
            healthy_latencies_ms: List[float] = []

            for i in range(10):
                t0 = time.monotonic()
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Request-ID": f"s4-req-{i}"},
                    json={"model": "qwen2.5-7b", "messages": prompt, "stream": False},
                )
                elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
                statuses.append(resp.status_code)
                if resp.status_code == 200:
                    routed_ids.append(resp.headers["X-Routed-Replica"])
                    healthy_latencies_ms.append(elapsed_ms)

            assert statuses[:2] == [504, 504]
            assert statuses[2:] == [200] * 8
            assert hung_id not in routed_ids
            assert not app.state.registry.get(hung_id).is_available()
            assert app.state.registry.get(hung_id).inflight == 0
            assert app.state.admission.in_flight == 0

            BENCHMARK_RESULTS["scenario_4_unresponsive_replica_failover"] = {
                "hung_replica_id": hung_id,
                "read_timeout_ms": 60.0,
                "circuit_breaker_threshold": 2,
                "status_sequence": statuses,
                "timeouts_before_circuit_open": statuses.count(504),
                "post_failover_succeeded_200": statuses[2:].count(200),
                "failover_target_replicas": list(set(routed_ids)),
                "post_failover_latency_p99_ms": _percentile(healthy_latencies_ms, 99),
                "hung_replica_leaked_inflight": app.state.registry.get(hung_id).inflight,
            }
