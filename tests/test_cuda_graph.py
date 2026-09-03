#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 6: CUDA/HIP Graph vs Eager Mode Benchmark

Evaluates the latency benefits of CUDA Graph operator capture during decoding iterations,
comparing captured graph execution vs eager mode kernel launch overhead.

Measures:
  - Inter-Token Latency (TPOT Mean, P95, P99)
  - Generation Throughput (tokens/s)
  - GPU Memory Footprint
"""
import urllib.request
import urllib.error
import json
import time
import os
import sys
import argparse
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL", "qwen2.5-7b-instruct-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "")
RESULT_DIR = "results/cuda-graph"

CONCURRENCIES = [1, 4]
ROUNDS_PER_CONCURRENCY = 2
MAX_TOKENS = 300
TPOT_WARMUP_TOKENS = 5

DECODE_PROMPT = (
    "请详细分析现代 GPU 架构中算子发射开销（Kernel Launch Overhead）对深度学习小批量推理延迟的影响，"
    "解释 CUDA Graph 的捕获（Capture）与重放（Replay）机制，以及它如何消除 CPU-GPU 驱动层同步瓶颈。"
)


def get_gpu_memory_used_mib() -> int:
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.check_output(cmd, universal_newlines=True).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            return int(lines[0])
    except Exception:
        pass
    return 0


def send_streaming_request(worker_id: str) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": f"[{worker_id}] {DECODE_PROMPT}"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
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
    tokens_generated = 0
    completion_tokens = None

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue

            chunk = json.loads(line[6:])
            usage = chunk.get("usage")
            if usage and "completion_tokens" in usage:
                completion_tokens = usage["completion_tokens"]

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                elif last_token_time is not None:
                    tpots.append((now - last_token_time) * 1000)
                last_token_time = now
                tokens_generated += 1

    end_time = time.perf_counter()
    total_tokens = completion_tokens or tokens_generated
    stable_tpots = tpots[TPOT_WARMUP_TOKENS:] if len(tpots) > TPOT_WARMUP_TOKENS else tpots

    return {
        "worker_id": worker_id,
        "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else 0.0,
        "tpot_mean_ms": statistics.mean(stable_tpots) if stable_tpots else 0.0,
        "tokens": total_tokens,
        "e2e_ms": (end_time - start_time) * 1000,
    }


def run_benchmark(state_label: str) -> dict:
    print("=" * 70)
    print(f"   vLLM Serving Lab - CUDA Graph Benchmark ({state_label.upper()})")
    print(f"   Model    : {DEFAULT_MODEL}")
    print(f"   Endpoint : {BASE_URL}")
    print("=" * 70)

    vram_used = get_gpu_memory_used_mib()
    print(f"📊 Baseline GPU VRAM Used: {vram_used} MiB\n")

    grid_results = []

    for c in CONCURRENCIES:
        print(f"▶️  Testing Concurrency c={c} ({ROUNDS_PER_CONCURRENCY} rounds)...")
        c_tpots = []
        c_ttfts = []
        c_tps_list = []

        for r in range(1, ROUNDS_PER_CONCURRENCY + 1):
            round_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as executor:
                futures = [executor.submit(send_streaming_request, f"c{c}_w{i+1}") for i in range(c)]
                results = [f.result() for f in as_completed(futures)]

            round_elapsed = time.perf_counter() - round_start
            total_tokens = sum(res["tokens"] for res in results)
            round_tps = total_tokens / round_elapsed if round_elapsed > 0 else 0.0

            round_tpots = [res["tpot_mean_ms"] for res in results if res["tpot_mean_ms"] > 0]
            if round_tpots:
                c_tpots.extend(round_tpots)
            c_ttfts.extend(res["ttft_ms"] for res in results)
            c_tps_list.append(round_tps)

            mean_tpot = statistics.mean(round_tpots) if round_tpots else 0.0
            print(f"   Round {r}: TPOT = {mean_tpot:.1f} ms | Throughput = {round_tps:.1f} tok/s")

        grid_results.append({
            "concurrency": c,
            "mean_tpot_ms": round(statistics.mean(c_tpots), 2) if c_tpots else 0.0,
            "p95_tpot_ms": round(statistics.quantiles(c_tpots, n=20)[18], 2) if len(c_tpots) >= 20 else round(max(c_tpots), 2) if c_tpots else 0.0,
            "throughput_tps": round(statistics.mean(c_tps_list), 2) if c_tps_list else 0.0,
            "mean_ttft_ms": round(statistics.mean(c_ttfts), 2) if c_ttfts else 0.0,
        })

    summary = {
        "experiment": "cuda_graph",
        "state": state_label,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_used_mib": vram_used,
        "results": grid_results,
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_file = os.path.join(RESULT_DIR, f"{state_label}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"📊 State Summary ({state_label.upper()}):")
    for row in grid_results:
        print(f"   • c={row['concurrency']}: TPOT = {row['mean_tpot_ms']:.1f} ms | Throughput = {row['throughput_tps']:.1f} tok/s")
    print(f"   📁 Saved report to: {out_file}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM CUDA Graph Benchmark")
    parser.add_argument("--state", type=str, default="cuda_graph", choices=["cuda_graph", "eager_mode"],
                        help="Execution mode (cuda_graph or eager_mode)")
    args = parser.parse_args()

    run_benchmark(args.state)


if __name__ == "__main__":
    main()
