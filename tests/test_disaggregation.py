#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 9: Prefill-Decode Disaggregation Benchmark

Measures the isolation benefits of separating Prefill workers from Decode workers.
Tests interactive decode stability (TPOT jitter) while the system is bombarded by
bursts of heavy prefill requests.

Measures:
  - Interactive Decode TPOT Mean, P95, P99 (ms)
  - Heavy Prefill TTFT (ms)
  - Inter-token Latency Jitter (Standard Deviation of TPOT)
"""
import urllib.request
import urllib.error
import json
import time
import os
import sys
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL", "qwen2.5-7b-instruct-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "")
RESULT_DIR = "results/disaggregation"

NUM_PREFILL_BURSTS = 4
NUM_DECODE_CLIENTS = 4

LONG_PREFILL_PROMPT = (
    "请对以下长篇计算机体系结构文献进行深度精炼和摘要："
    + ("在多核异构加速计算架构中，计算流水线的指令发射效率与内存带宽访问延迟构成了主要的瓶颈特征。" * 30)
)
SHORT_INTERACTIVE_PROMPT = "请写一首关于量子纠缠与浩瀚星空的现代短诗，要求押韵，字数在100字左右。"


def send_interactive_decode(worker_id: str) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": f"[{worker_id}] {SHORT_INTERACTIVE_PROMPT}"}],
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 150,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(BASE_URL, data=data, headers=headers)
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    tpots = []
    tokens = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content", "")
            if content:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                elif last_token_time is not None:
                    tpots.append((now - last_token_time) * 1000)
                last_token_time = now
                tokens += 1

    stable_tpots = tpots[3:] if len(tpots) > 3 else tpots
    return {
        "worker_id": worker_id,
        "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else 0.0,
        "tpot_mean_ms": statistics.mean(stable_tpots) if stable_tpots else 0.0,
        "tpot_std_ms": statistics.stdev(stable_tpots) if len(stable_tpots) > 1 else 0.0,
        "tokens": tokens,
    }


def send_heavy_prefill(worker_id: str) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": f"[{worker_id}] {LONG_PREFILL_PROMPT}"}],
        "stream": False,
        "max_tokens": 30,
        "extra_body": {"pd_worker": "prefill"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(BASE_URL, data=data, headers=headers)
    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        _ = resp.read()
    elapsed = (time.perf_counter() - start_time) * 1000
    return {"worker_id": worker_id, "prefill_time_ms": elapsed}


def run_benchmark(arch_label: str) -> dict:
    print("=" * 70)
    print(f"   vLLM Serving Lab - Prefill-Decode Disaggregation Test ({arch_label.upper()})")
    print(f"   Endpoint    : {BASE_URL}")
    print(f"   Contenders  : {NUM_PREFILL_BURSTS} Heavy Prefill Bursts + {NUM_DECODE_CLIENTS} Streaming Decode Clients")
    print("=" * 70)

    print("\n🚀 Injecting concurrent prefill bursts and interactive streaming requests...")
    bench_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_PREFILL_BURSTS + NUM_DECODE_CLIENTS) as executor:
        # Launch heavy prefills and interactive decodes simultaneously
        prefill_futures = [executor.submit(send_heavy_prefill, f"prefill_{i+1}") for i in range(NUM_PREFILL_BURSTS)]
        decode_futures = [executor.submit(send_interactive_decode, f"decode_{i+1}") for i in range(NUM_DECODE_CLIENTS)]

        prefill_results = [f.result() for f in as_completed(prefill_futures)]
        decode_results = [f.result() for f in as_completed(decode_futures)]

    bench_elapsed = time.perf_counter() - bench_start

    mean_prefill_ms = statistics.mean([r["prefill_time_ms"] for r in prefill_results])
    decode_tpots = [r["tpot_mean_ms"] for r in decode_results]
    decode_stds = [r["tpot_std_ms"] for r in decode_results]
    decode_ttfts = [r["ttft_ms"] for r in decode_results]

    mean_decode_tpot = statistics.mean(decode_tpots) if decode_tpots else 0.0
    mean_decode_jitter = statistics.mean(decode_stds) if decode_stds else 0.0
    p95_decode_tpot = statistics.quantiles(decode_tpots, n=20)[18] if len(decode_tpots) >= 20 else max(decode_tpots)

    summary = {
        "experiment": "prefill_decode_disaggregation",
        "architecture": arch_label,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heavy_prefill_mean_ms": round(mean_prefill_ms, 2),
        "interactive_decode_tpot_mean_ms": round(mean_decode_tpot, 2),
        "interactive_decode_tpot_p95_ms": round(p95_decode_tpot, 2),
        "interactive_decode_jitter_std_ms": round(mean_decode_jitter, 2),
        "interactive_decode_ttft_mean_ms": round(statistics.mean(decode_ttfts), 2),
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_file = os.path.join(RESULT_DIR, f"{arch_label}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"📊 Architecture Summary ({arch_label.upper()}):")
    print(f"   • Heavy Prefill Latency       : {mean_prefill_ms:.1f} ms")
    print(f"   • Interactive Decode TPOT     : {mean_decode_tpot:.1f} ms")
    print(f"   • Decode Jitter (Stability)   : {mean_decode_jitter:.2f} ms std-dev")
    print(f"   📁 Saved report to: {out_file}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM PD Disaggregation Benchmark")
    parser.add_argument("--arch", type=str, default="disaggregated", choices=["disaggregated", "monolithic"],
                        help="Architecture under test")
    args = parser.parse_args()

    run_benchmark(args.arch)


if __name__ == "__main__":
    main()
