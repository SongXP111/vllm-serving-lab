"""M2 Single-GPU RTX 5080 Real vLLM Verification (Part B).

Verifies real vLLM behavior through the M2 Routing Gateway against a live single-GPU
vLLM container on localhost:8000 (with `--enable-prefix-caching`):
1. Discovers served model via Gateway `/v1/models` and registers the real vLLM
   endpoint in `ReplicaRegistry`.
2. Sends a unique ~1500-token shared System Prompt across 1 Cold turn + 3 Warm
   follow-up turns (streaming SSE) to measure real RTX 5080 Cold vs Warm TTFT.
3. Calls `POST /admin/replicas/sync-metrics` before & after the workload to fetch
   ground-truth vLLM Prometheus counters (`vllm:prefix_cache_hits_total` /
   `vllm:prefix_cache_queries_total`), cross-checking Gateway's *estimated*
   prefix affinity against real GPU KV Cache prefix hits without conflating them.
"""
import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List
import uuid

import httpx
import pytest

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from gateway.registry import ReplicaConfig

VLLM_BACKEND_URL = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8000")


async def _is_real_vllm_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{VLLM_BACKEND_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


def _build_unique_long_system_prompt(salt: str) -> str:
    """Create a ~1500-token unique system prompt so Turn 0 is guaranteed a cold miss."""
    paragraph = (
        f"Session-Salt-{salt}: In production LLM serving systems such as vLLM, "
        "Automatic Prefix Caching (APC) divides prompt token sequences into fixed-size "
        "PagedAttention physical blocks (16 tokens per block) and indexes each full block "
        "using a chained hash of its parent block hash concatenated with its token IDs. "
        "When multiple multi-turn requests share a long system prompt or retrieved RAG "
        "context, cache hits bypass redundant quadratic Prefill matrix multiplications "
        "and drastically reduce Time To First Token (TTFT) on GPU accelerators. "
    )
    return "\n".join(f"[Section {i:02d}] {paragraph}" for i in range(18))


async def _asgi_stream_call(
    app,
    *,
    request_id: str,
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 16,
) -> Dict[str, Any]:
    """Stream directly over ASGI to measure true zero-buffered TTFT and headers."""
    body_bytes = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode("utf-8")
    body_sent = False
    status_code = 500
    resp_headers: Dict[str, str] = {}
    t_start = time.perf_counter()
    ttft_ms = 0.0
    first_seen = False
    sse_chunks = 0

    async def receive():
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

    async def send(message):
        nonlocal status_code, first_seen, ttft_ms, sse_chunks
        if message["type"] == "http.response.start":
            status_code = message["status"]
            for k, v in message.get("headers", []):
                resp_headers[k.decode("latin1").lower()] = v.decode("latin1")
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if status_code == 200 and chunk:
                sse_chunks += 1
                if not first_seen:
                    ttft_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                    first_seen = True

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
            (b"x-request-id", request_id.encode("latin1")),
        ],
        "client": ("127.0.0.1", 54000),
        "server": ("gateway", 80),
    }
    await app(scope, receive, send)
    e2e_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    return {
        "request_id": request_id,
        "status_code": status_code,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "sse_chunks": sse_chunks,
        "routed_replica": resp_headers.get("x-routed-replica"),
        "routing_policy": resp_headers.get("x-routing-policy"),
        "gateway_estimated_prefix_affinity": float(
            resp_headers.get("x-estimated-prefix-affinity", "0.0")
        ),
    }


@pytest.mark.anyio
async def test_m2_real_rtx5080_vllm_prefix_cache_cross_check():
    """Cross-check M2 Gateway estimated prefix affinity vs real RTX 5080 vLLM `/metrics`."""
    if not await _is_real_vllm_ready():
        pytest.skip(
            f"Real vLLM is not reachable at {VLLM_BACKEND_URL}. "
            "Start Docker Desktop and run `docker compose up -d vllm` first."
        )

    cfg = GatewayConfig(
        backend_url=VLLM_BACKEND_URL,
        read_timeout=60.0,
        max_inflight=4,
        queue_capacity=20,
        replicas=[
            ReplicaConfig(
                replica_id="replica-rtx5080",
                endpoint=VLLM_BACKEND_URL,
                max_inflight=4,
                kv_cache_capacity_blocks=2048,
            )
        ],
        routing_policy="cache+load-aware",
    )
    app = create_gateway_app(config=cfg)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            timeout=60.0,
        ) as gw_client:
            # 1. Discover real served model via Gateway
            models_resp = await gw_client.get("/v1/models")
            assert models_resp.status_code == 200
            model_id = models_resp.json()["data"][0]["id"]

            # 2. Sync baseline real vLLM `/metrics` BEFORE sending our experiment turns
            before_sync = (await gw_client.post("/admin/replicas/sync-metrics")).json()
            rep_before = before_sync["synced_replicas"]["replica-rtx5080"]
            q_before = (
                rep_before["observed_vllm_metrics"]["prefix_cache_queries"] or 0.0
            )
            h_before = rep_before["observed_vllm_metrics"]["prefix_cache_hits"] or 0.0

            # 3. Send 1 Cold Turn + 3 Warm Turns sharing a unique ~1500-token system prompt
            salt = uuid.uuid4().hex[:8]
            long_system = _build_unique_long_system_prompt(salt)
            questions = [
                "In one short sentence, what does APC stand for?",
                "In one short sentence, how many tokens are in a default block?",
                "In one short sentence, why is block hashing chained?",
                "In one short sentence, which latency metric does APC reduce?",
            ]

            turn_records: List[Dict[str, Any]] = []
            for idx, q in enumerate(questions):
                msgs = [
                    {"role": "system", "content": long_system},
                    {"role": "user", "content": q},
                ]
                rec = await _asgi_stream_call(
                    app,
                    request_id=f"m2-real-turn-{idx}",
                    model_id=model_id,
                    messages=msgs,
                    max_tokens=16,
                )
                assert rec["status_code"] == 200
                assert rec["routed_replica"] == "replica-rtx5080"
                assert rec["routing_policy"] == "cache+load-aware"
                rec["turn"] = idx
                rec["phase"] = "cold" if idx == 0 else "warm"
                turn_records.append(rec)

            # 4. Sync real vLLM `/metrics` AFTER the 4 turns for ground-truth delta!
            after_sync = (await gw_client.post("/admin/replicas/sync-metrics")).json()
            rep_after = after_sync["synced_replicas"]["replica-rtx5080"]
            q_after = (
                rep_after["observed_vllm_metrics"]["prefix_cache_queries"] or 0.0
            )
            h_after = rep_after["observed_vllm_metrics"]["prefix_cache_hits"] or 0.0

            delta_queries = round(q_after - q_before, 2)
            delta_hits = round(h_after - h_before, 2)
            delta_hit_rate = (
                round(delta_hits / delta_queries, 4) if delta_queries > 0 else None
            )

            cold_ttft = turn_records[0]["ttft_ms"]
            warm_ttfts = [r["ttft_ms"] for r in turn_records[1:]]
            warm_ttft_avg = round(sum(warm_ttfts) / len(warm_ttfts), 2)
            warm_est_affinities = [
                r["gateway_estimated_prefix_affinity"] for r in turn_records[1:]
            ]
            warm_est_affinity_avg = round(
                sum(warm_est_affinities) / len(warm_est_affinities), 4
            )

            # Verify Gateway estimated cold=0.0 and warm>0.90
            assert turn_records[0]["gateway_estimated_prefix_affinity"] == 0.0
            assert warm_est_affinity_avg >= 0.90
            # Verify real RTX 5080 vLLM achieved genuine TTFT speedup & real KV cache hits!
            assert warm_ttft_avg < cold_ttft
            assert delta_queries > 0
            assert delta_hits > 0
            assert delta_hit_rate is not None and delta_hit_rate >= 0.50

            # Verify zero leaked in-flight on real replica
            assert app.state.registry.get("replica-rtx5080").inflight == 0
            assert app.state.admission.in_flight == 0

            report = {
                "hardware_note": (
                    "Single-GPU RTX 5080 Laptop real vLLM verification "
                    "(verifies real APC behavior & TTFT speedup; not multi-GPU cluster TPS)."
                ),
                "backend_url": VLLM_BACKEND_URL,
                "served_model": model_id,
                "routing_policy": "cache+load-aware",
                "cold_turn_ttft_ms": cold_ttft,
                "warm_turns_ttft_ms": warm_ttfts,
                "warm_turns_ttft_avg_ms": warm_ttft_avg,
                "ttft_speedup_ratio": round(cold_ttft / max(0.1, warm_ttft_avg), 2),
                "gateway_estimated_warm_affinity_avg": warm_est_affinity_avg,
                "real_vllm_ground_truth_delta": {
                    "prefix_cache_queries_delta": delta_queries,
                    "prefix_cache_hits_delta": delta_hits,
                    "observed_vllm_delta_hit_rate": delta_hit_rate,
                    "kv_cache_usage_perc_after": rep_after["kv_cache_usage_perc"],
                },
                "turns": turn_records,
            }

            out_dir = Path("results/gateway")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "m2_real_vllm_verification.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
