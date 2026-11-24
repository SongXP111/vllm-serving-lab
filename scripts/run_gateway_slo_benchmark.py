#!/usr/bin/env python3
"""Controlled Single-Variable SLO, Goodput & Regression Benchmark Runner (Step 4).

Executes an identical, fixed-seed Poisson workload through the real M1/M2 Gateway
against the live single-GPU RTX 5080 vLLM server (`http://localhost:8000`):
- Strictly holds constant: arrival schedule (`seed=42`), input/output token lengths,
  model version, quantization dtype, and SLO thresholds.
- Changes ONLY ONE variable per run (e.g., `gateway.max_inflight`, `queue_capacity`,
  or `routing_policy`).
- Collects:
  - TTFT P50 / P99, TPOT P50 / P99, ITL P50 / P99, E2E P50 / P99
  - Raw Throughput (`output_tps`, `rps`) AND SLO-compliant **Goodput** (`goodput_rps`, `goodput_pct`)
  - Error rate & Rejection rate (`429` / `503` / `504`)
  - Real GPU Peak VRAM (MiB) via `nvidia-smi` background sampler
  - Explicit assumed cost per 1M output tokens (clearly marked as modeling assumption,
    NOT real cloud billing!)
  - Gateway Prometheus metrics snapshot (`/metrics`) & correlated trace samples
    (`gateway_request_id` <-> `backend_request_id`)
- Outputs a `summary.json` directly compatible with `scripts/check_regression.py`!
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.app import create_gateway_app
from gateway.config import GatewayConfig
from gateway.registry import ReplicaConfig


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 2)


def sample_nvidia_smi() -> Optional[Tuple[float, float, float]]:
    """Return `(used_vram_mib, total_vram_mib, gpu_util_pct)` or None."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        ).strip()
        first_line = out.splitlines()[0]
        used, total, util = [float(x.strip()) for x in first_line.split(",")[:3]]
        return used, total, util
    except Exception:
        return None


def build_fixed_workload(
    *,
    num_requests: int,
    seed: int,
    arrival_rate_rps: float,
    input_sections: int,
    unique_prefix_ratio: float,
    prompt_salt: str,
) -> List[Dict[str, Any]]:
    """Build a deterministic Poisson-arrival prompt workload with fixed seed.

    `prompt_salt` varies per run when testing uncached Prefill so prior runs do
    NOT pollute vLLM's Automatic Prefix Cache (cross-run cache isolation!), while
    keeping exact token counts and Poisson arrival offsets identical via `seed`.
    """
    rng = random.Random(seed)
    base_paragraph = (
        "In high-throughput LLM serving, continuous batching, PagedAttention KV blocks, "
        "chunked prefill budgets, and gateway admission control jointly determine the "
        "Pareto frontier between raw token throughput and tail latencies (TTFT P99 and "
        "ITL/TPOT P99). Increasing in-flight concurrency amortizes weight bandwidth reads "
        "during decode, but pushing past the hardware saturation knee inflates queuing "
        "delay and inter-token latency jitter. "
    )
    workload: List[Dict[str, Any]] = []
    t_offset = 0.0
    for idx in range(num_requests):
        if idx > 0 and arrival_rate_rps > 0:
            t_offset += rng.expovariate(arrival_rate_rps)
        # Deterministic salt per prompt + run to guarantee zero cross-run APC pollution
        if rng.random() < unique_prefix_ratio:
            salt = f"{prompt_salt}-s{seed}-u{idx:03d}-{rng.randint(100000, 999999)}"
        else:
            salt = f"{prompt_salt}-s{seed}-shared{idx % 3}"
        sys_text = "\n".join(
            f"[Doc {s:02d} {salt}] {base_paragraph}" for s in range(input_sections)
        )
        workload.append(
            {
                "index": idx,
                "arrival_offset_sec": round(t_offset, 4),
                "messages": [
                    {"role": "system", "content": sys_text},
                    {
                        "role": "user",
                        "content": (
                            f"Request #{idx}: Explain in 5 concise bullet points why "
                            "increasing batch concurrency trades off throughput against tail latency."
                        ),
                    },
                ],
            }
        )
    return workload


async def asgi_stream_request(
    app,
    *,
    request_id: str,
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Dict[str, Any]:
    """Send a single streaming request over ASGI directly for zero-buffered TTFT/ITL."""
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
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    itl_samples_ms: List[float] = []
    output_tokens = 0
    sse_buffer = ""

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
        nonlocal status_code, first_token_ts, last_token_ts, output_tokens, sse_buffer
        if message["type"] == "http.response.start":
            status_code = message["status"]
            for k, v in message.get("headers", []):
                resp_headers[k.decode("latin1").lower()] = v.decode("latin1")
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if status_code == 200 and chunk:
                now = time.perf_counter()
                sse_buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in sse_buffer:
                    line, sse_buffer = sse_buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:") or "[DONE]" in line:
                        continue
                    try:
                        obj = json.loads(line[5:].strip())
                        choices = obj.get("choices") or []
                        delta = choices[0].get("delta", {}) if choices else {}
                        content = delta.get("content")
                        if content:
                            output_tokens += 1
                            if first_token_ts is None:
                                first_token_ts = now
                            elif last_token_ts is not None:
                                itl_samples_ms.append((now - last_token_ts) * 1000.0)
                            last_token_ts = now
                    except Exception:
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
            (b"x-request-id", request_id.encode("latin1")),
            (b"x-gateway-request-id", request_id.encode("latin1")),
        ],
        "client": ("127.0.0.1", 55000),
        "server": ("gateway", 80),
    }

    await app(scope, receive, send)
    t_end = time.perf_counter()
    e2e_ms = (t_end - t_start) * 1000.0
    ttft_ms = (
        (first_token_ts - t_start) * 1000.0 if first_token_ts is not None else e2e_ms
    )
    tpot_ms = (
        ((t_end - first_token_ts) * 1000.0 / max(1, output_tokens - 1))
        if first_token_ts is not None and output_tokens > 1
        else 0.0
    )
    return {
        "request_id": request_id,
        "status_code": status_code,
        "queue_wait_ms": float(resp_headers.get("x-queue-wait-ms", "0.0")),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "itl_samples_ms": [round(x, 2) for x in itl_samples_ms],
        "e2e_ms": round(e2e_ms, 2),
        "output_tokens": output_tokens,
        "backend_request_id": resp_headers.get("x-backend-request-id"),
        "routed_replica": resp_headers.get("x-routed-replica"),
        "traceparent": resp_headers.get("traceparent"),
    }


async def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = GatewayConfig(
        backend_url=args.backend_url,
        read_timeout=60.0,
        max_inflight=args.max_inflight,
        queue_capacity=args.queue_capacity,
        queue_timeout=args.queue_timeout,
        replicas=[
            ReplicaConfig(
                replica_id="replica-rtx5080",
                endpoint=args.backend_url,
                max_inflight=args.max_inflight,
                kv_cache_capacity_blocks=2048,
            )
        ],
        routing_policy=args.routing_policy,
        router_rng_seed=args.seed,
    )
    app = create_gateway_app(config=cfg)

    effective_salt = args.prompt_salt or f"{args.run_label}-{int(time.time())}"
    workload = build_fixed_workload(
        num_requests=args.num_requests,
        seed=args.seed,
        arrival_rate_rps=args.arrival_rate,
        input_sections=args.input_sections,
        unique_prefix_ratio=args.unique_prefix_ratio,
        prompt_salt=effective_salt,
    )

    peak_vram_mib = 0.0
    total_vram_mib = 0.0
    gpu_util_samples: List[float] = []
    sampling_stop = asyncio.Event()

    async def gpu_sampler():
        nonlocal peak_vram_mib, total_vram_mib
        while not sampling_stop.is_set():
            sample = sample_nvidia_smi()
            if sample is not None:
                used, total, util = sample
                peak_vram_mib = max(peak_vram_mib, used)
                total_vram_mib = max(total_vram_mib, total)
                gpu_util_samples.append(util)
            await asyncio.sleep(0.25)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            timeout=60.0,
        ) as gw_client:
            models_resp = await gw_client.get("/v1/models")
            if models_resp.status_code != 200:
                raise RuntimeError(f"Failed to query /v1/models: {models_resp.text}")
            model_id = models_resp.json()["data"][0]["id"]

            # Warmup 1 short call to ensure CUDA graphs / kernels are warm
            await asgi_stream_request(
                app,
                request_id=f"{args.run_label}-warmup",
                model_id=model_id,
                messages=[{"role": "user", "content": "Say ok."}],
                max_tokens=4,
            )

            # Capture baseline real vLLM `/metrics` before workload for exact delta
            pre_sync = (await gw_client.post("/admin/replicas/sync-metrics")).json()
            pre_rep = pre_sync["synced_replicas"]["replica-rtx5080"]
            pre_q = pre_rep["observed_vllm_metrics"]["prefix_cache_queries"] or 0.0
            pre_h = pre_rep["observed_vllm_metrics"]["prefix_cache_hits"] or 0.0

            sampler_task = asyncio.create_task(gpu_sampler())
            wall_start = time.perf_counter()

            async def dispatch_item(item: Dict[str, Any]) -> Dict[str, Any]:
                if item["arrival_offset_sec"] > 0:
                    await asyncio.sleep(item["arrival_offset_sec"])
                return await asgi_stream_request(
                    app,
                    request_id=f"{args.run_label}-req-{item['index']:03d}",
                    model_id=model_id,
                    messages=item["messages"],
                    max_tokens=args.max_tokens,
                )

            results = await asyncio.gather(*(dispatch_item(w) for w in workload))
            wall_elapsed = max(1e-4, time.perf_counter() - wall_start)
            sampling_stop.set()
            await sampler_task

            # Scrape final Gateway `/metrics` and correlated `/admin/traces`
            prom_text = (await gw_client.get("/metrics")).text
            traces_payload = (await gw_client.get("/admin/traces?limit=5")).json()
            sync_payload = (await gw_client.post("/admin/replicas/sync-metrics")).json()
            post_rep = sync_payload["synced_replicas"]["replica-rtx5080"]
            post_q = post_rep["observed_vllm_metrics"]["prefix_cache_queries"] or 0.0
            post_h = post_rep["observed_vllm_metrics"]["prefix_cache_hits"] or 0.0
            vllm_delta_queries = round(post_q - pre_q, 2)
            vllm_delta_hits = round(post_h - pre_h, 2)
            vllm_delta_hit_rate = (
                round(vllm_delta_hits / vllm_delta_queries, 4)
                if vllm_delta_queries > 0
                else 0.0
            )

    # Compute full metrics
    total_reqs = len(results)
    ok_results = [r for r in results if r["status_code"] == 200]
    rejected_results = [r for r in results if r["status_code"] in (429, 503)]
    error_results = [r for r in results if r["status_code"] not in (200, 429, 503)]

    ttfts = [r["ttft_ms"] for r in ok_results]
    tpots = [r["tpot_ms"] for r in ok_results if r["tpot_ms"] > 0]
    all_itls = [x for r in ok_results for x in r["itl_samples_ms"]]
    e2es = [r["e2e_ms"] for r in ok_results]
    q_waits = [r["queue_wait_ms"] for r in ok_results]

    total_output_tokens = sum(r["output_tokens"] for r in ok_results)
    output_tps = round(total_output_tokens / wall_elapsed, 2)
    raw_rps = round(len(ok_results) / wall_elapsed, 2)

    # SLO Goodput check
    good_results = [
        r
        for r in ok_results
        if r["ttft_ms"] <= args.slo_ttft_ms
        and r["tpot_ms"] <= args.slo_tpot_ms
        and r["e2e_ms"] <= args.slo_e2e_ms
    ]
    goodput_rps = round(len(good_results) / wall_elapsed, 2)
    goodput_ratio_pct = round((len(good_results) / max(1, total_reqs)) * 100.0, 2)

    reject_rate_pct = round((len(rejected_results) / max(1, total_reqs)) * 100.0, 2)
    error_rate_pct = round(
        ((total_reqs - len(ok_results)) / max(1, total_reqs)) * 100.0, 2
    )

    # Explicit cost modeling based on clearly stated GPU hourly price assumption
    cost_per_million_usd = (
        round((args.assumed_gpu_hourly_usd / 3600.0) / output_tps * 1_000_000.0, 4)
        if output_tps > 0
        else 0.0
    )

    benchmark_entry = {
        "workload": args.workload_name,
        "concurrency": args.max_inflight,
        "num_requests": total_reqs,
        "completed_200": len(ok_results),
        "rejected_429_503": len(rejected_results),
        "errors_other": len(error_results),
        "wall_time_sec": round(wall_elapsed, 2),
        "output_tokens_total": total_output_tokens,
        "output_tps": output_tps,
        "raw_rps": raw_rps,
        "goodput_count": len(good_results),
        "goodput_rps": goodput_rps,
        "goodput_ratio_pct": goodput_ratio_pct,
        "p50_queue_wait_ms": percentile(q_waits, 50),
        "p99_queue_wait_ms": percentile(q_waits, 99),
        "p50_ttft_ms": percentile(ttfts, 50),
        "p99_ttft_ms": percentile(ttfts, 99),
        "p50_tpot_ms": percentile(tpots, 50),
        "p99_tpot_ms": percentile(tpots, 99),
        "p50_itl_ms": percentile(all_itls, 50),
        "p99_itl_ms": percentile(all_itls, 99),
        "p50_e2e_ms": percentile(e2es, 50),
        "p99_e2e_ms": percentile(e2es, 99),
        "reject_rate_pct": reject_rate_pct,
        "error_rate_pct": error_rate_pct,
        "peak_vram_mib": round(peak_vram_mib, 1),
        "total_vram_mib": round(total_vram_mib, 1),
        "mean_gpu_util_pct": (
            round(sum(gpu_util_samples) / len(gpu_util_samples), 1)
            if gpu_util_samples
            else 0.0
        ),
        "cost_per_million_tokens_usd": cost_per_million_usd,
    }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "experiment_controls": {
            "seed": args.seed,
            "arrival_distribution": f"poisson(rate={args.arrival_rate} rps)",
            "num_requests": args.num_requests,
            "input_sections": args.input_sections,
            "unique_prefix_ratio": args.unique_prefix_ratio,
            "max_tokens": args.max_tokens,
            "gateway_max_inflight": args.max_inflight,
            "gateway_queue_capacity": args.queue_capacity,
            "routing_policy": args.routing_policy,
            "slo_thresholds": {
                "ttft_ms": args.slo_ttft_ms,
                "tpot_ms": args.slo_tpot_ms,
                "e2e_ms": args.slo_e2e_ms,
            },
            "cost_assumption": (
                f"Assumed ${args.assumed_gpu_hourly_usd:.2f}/GPU-hr for relative "
                "cost-per-1M-tokens modeling ONLY (NOT real cloud billing)."
            ),
        },
        "model": {
            "model_name": model_id,
            "dtype": "FP8 (Dynamic)",
            "gpu": "NVIDIA GeForce RTX 5080 Laptop GPU",
        },
        # Format directly compatible with `scripts/check_regression.py`!
        # Note: keep fixed key `concurrency=1` in regression matching key when comparing
        # same workload with different Gateway configs (`--match-concurrency`).
        "benchmarks": [
            {
                **benchmark_entry,
                "concurrency": args.regression_key_concurrency,
                "actual_gateway_max_inflight": args.max_inflight,
            }
        ],
        "gateway_histograms": traces_payload.get("histograms", {}),
        "sample_correlated_traces": traces_payload.get("traces", [])[:3],
        "vllm_metrics_delta": {
            "prefix_cache_queries_delta": vllm_delta_queries,
            "prefix_cache_hits_delta": vllm_delta_hits,
            "observed_prefix_cache_hit_rate_delta": vllm_delta_hit_rate,
        },
        "vllm_metrics_snapshot": sync_payload.get("synced_replicas", {}),
        "gateway_prometheus_excerpt": [
            ln
            for ln in prom_text.splitlines()
            if ln.startswith(
                (
                    "gateway_requests_total",
                    "gateway_queue_depth ",
                    "gateway_inflight_requests ",
                    "gateway_rejections_total",
                    "gateway_router_policy_selections_total",
                    "gateway_queue_wait_seconds_sum",
                    "gateway_queue_wait_seconds_count",
                    "gateway_ttft_seconds_sum",
                    "gateway_ttft_seconds_count",
                    "gateway_itl_seconds_sum",
                    "gateway_itl_seconds_count",
                )
            )
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[{args.run_label}] Saved benchmark summary to {out_path}")
    print(json.dumps(benchmark_entry, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4 Controlled Single-Variable SLO & Goodput Benchmark Runner"
    )
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--workload-name", default="poisson_long_prefill_24")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--input-sections", type=int, default=14)
    parser.add_argument("--unique-prefix-ratio", type=float, default=1.0)
    parser.add_argument("--prompt-salt", default="")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=40)
    parser.add_argument("--queue-timeout", type=float, default=30.0)
    parser.add_argument("--routing-policy", default="cache+load-aware")
    parser.add_argument("--regression-key-concurrency", type=int, default=1)
    parser.add_argument("--slo-ttft-ms", type=float, default=650.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=20.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=2500.0)
    parser.add_argument("--assumed-gpu-hourly-usd", type=float, default=1.50)
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
