#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 7: KV Cache Quantization Benchmark (FP8 vs Auto/BF16)

Evaluates the capacity doubling and throughput benefits of FP8 KV Cache quantization
(--kv-cache-dtype fp8_e4m3).

Measures:
  - Total Available GPU Blocks & Token Capacity (from /metrics)
  - Memory Footprint
  - Decoding Throughput (tokens/s) & TPOT (ms)
"""
import urllib.request
import urllib.error
import json
import time
import os
import sys
import re
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = os.environ.get("VLLM_PORT", "8000")
METRICS_URL = f"http://localhost:{PORT}/metrics"
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL", "qwen2.5-7b-instruct-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "")
RESULT_DIR = "results/kv-cache-quant"

NUM_WORKERS = 4
MAX_TOKENS = 250

PROMPT = (
    "请详细论述大语言模型（LLM）推理中 KV Cache 量化技术（如 FP8 E4M3 与 E5M2 格式）的基本原理，"
    "分析 Key 矩阵与 Value 矩阵在动态量化尺度（Per-token / Per-tensor Scaling）下的精度敏感性差异。"
)


def fetch_kv_metrics() -> dict:
    """Fetch Prometheus metrics to extract KV cache capacity details."""
    metrics = {"num_gpu_blocks": 0, "gpu_cache_usage_perc": 0.0, "kv_cache_tokens": 0}
    try:
        req = urllib.request.Request(METRICS_URL)
        if API_KEY:
            req.add_header("Authorization", f"Bearer {API_KEY}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                # Match num_gpu_blocks
                m_blocks = re.match(r'(?:vllm:)?num_gpu_blocks\{?[^}]*\}?\s+(\d+)', line)
                if m_blocks:
                    metrics["num_gpu_blocks"] = int(m_blocks.group(1))

                # Match gpu_cache_usage_perc
                m_usage = re.match(r'(?:vllm:)?gpu_cache_usage_perc\{?[^}]*\}?\s+([\d\.]+)', line)
                if m_usage:
                    metrics["gpu_cache_usage_perc"] = float(m_usage.group(1))

        # Default block size in vLLM is 16 tokens
        metrics["kv_cache_tokens"] = metrics["num_gpu_blocks"] * 16
    except Exception as e:
        print(f"⚠️ Could not read /metrics: {e}")
    return metrics


def send_streaming_request(worker_id: str) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": f"[{worker_id}] {PROMPT}"}],
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
    content_pieces = []

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
                content_pieces.append(content)

    end_time = time.perf_counter()
    total_tokens = completion_tokens or tokens_generated
    stable_tpots = tpots[3:] if len(tpots) > 3 else tpots

    return {
        "worker_id": worker_id,
        "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else 0.0,
        "tpot_mean_ms": statistics.mean(stable_tpots) if stable_tpots else 0.0,
        "tokens": total_tokens,
        "e2e_ms": (end_time - start_time) * 1000,
        "preview": "".join(content_pieces)[:60].strip(),
    }


def run_benchmark(dtype_label: str) -> dict:
    print("=" * 70)
    print(f"   vLLM Serving Lab - KV Cache Quantization Benchmark ({dtype_label.upper()})")
    print(f"   Model       : {DEFAULT_MODEL}")
    print(f"   KV Dtype    : {dtype_label}")
    print(f"   Concurrency : {NUM_WORKERS}")
    print("=" * 70)

    # 1. Capture KV Cache metrics before traffic
    before_metrics = fetch_kv_metrics()
    print("📊 KV Cache Pool Metrics:")
    print(f"   • Available GPU Blocks  : {before_metrics['num_gpu_blocks']} blocks")
    print(f"   • Total Token Capacity  : {before_metrics['kv_cache_tokens']} tokens")
    print(f"   • Initial KV Cache Used : {before_metrics['gpu_cache_usage_perc'] * 100:.1f}%\n")

    # 2. Run concurrent generation
    print(f"🚀 Running {NUM_WORKERS} concurrent generation requests...")
    bench_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(send_streaming_request, f"req_{i+1}") for i in range(NUM_WORKERS)]
        results = [f.result() for f in as_completed(futures)]

    bench_elapsed = time.perf_counter() - bench_start
    total_tokens = sum(r["tokens"] for r in results)
    tps = total_tokens / bench_elapsed if bench_elapsed > 0 else 0.0

    mean_ttft = statistics.mean([r["ttft_ms"] for r in results])
    mean_tpot = statistics.mean([r["tpot_mean_ms"] for r in results])

    after_metrics = fetch_kv_metrics()

    summary = {
        "experiment": "kv_cache_quantization",
        "kv_cache_dtype": dtype_label,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_gpu_blocks": before_metrics["num_gpu_blocks"],
        "kv_cache_token_capacity": before_metrics["kv_cache_tokens"],
        "generation_throughput_tps": round(tps, 2),
        "mean_ttft_ms": round(mean_ttft, 2),
        "mean_tpot_ms": round(mean_tpot, 2),
        "peak_kv_cache_used_perc": round(after_metrics["gpu_cache_usage_perc"] * 100, 2),
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_file = os.path.join(RESULT_DIR, f"{dtype_label}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"📊 State Summary ({dtype_label.upper()}):")
    print(f"   • KV Cache Blocks Capacity : {before_metrics['num_gpu_blocks']} blocks ({before_metrics['kv_cache_tokens']} tokens)")
    print(f"   • Generation Throughput    : {tps:.1f} tokens/s")
    print(f"   • Mean TPOT                : {mean_tpot:.1f} ms")
    print(f"   • Mean TTFT                : {mean_ttft:.1f} ms")
    print(f"   📁 Saved report to: {out_file}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM KV Cache Quantization Benchmark")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp8_e4m3", "fp8"],
                        help="KV Cache precision (auto or fp8_e4m3)")
    args = parser.parse_args()

    run_benchmark(args.dtype)


if __name__ == "__main__":
    main()
