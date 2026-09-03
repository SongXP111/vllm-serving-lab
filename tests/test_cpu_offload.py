#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 8: CPU Offloading & Swap Space Benchmark

Evaluates system resilience against KV Cache out-of-memory errors by monitoring
CPU Swap Space allocation and block swapping behavior during high-pressure workloads.

Measures:
  - CPU Cache Usage % and GPU Cache Usage % (from /metrics)
  - Number of CPU Blocks allocated
  - Request success rate under heavy concurrent sequence pressure
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
RESULT_DIR = "results/cpu-offload"

NUM_WORKERS = 8
MAX_TOKENS = 300

PROMPT = (
    "请详细分析现代大模型推理服务（如 vLLM）中的虚拟内存分页管理机制（PagedAttention），"
    "阐述当 GPU 显存耗尽时，系统如何借助 CPU 主机内存交换空间（Swap Space）进行 KV Cache 的置换与抢占，"
    "并评估 PCIe 总线带宽在块迁移（Block Migration）过程中的性能衰退特征。"
)


def fetch_cache_metrics() -> dict:
    metrics = {
        "num_gpu_blocks": 0,
        "num_cpu_blocks": 0,
        "gpu_cache_usage_perc": 0.0,
        "cpu_cache_usage_perc": 0.0,
    }
    try:
        req = urllib.request.Request(METRICS_URL)
        if API_KEY:
            req.add_header("Authorization", f"Bearer {API_KEY}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                m_gpu_b = re.match(r'(?:vllm:)?num_gpu_blocks\{?[^}]*\}?\s+(\d+)', line)
                if m_gpu_b:
                    metrics["num_gpu_blocks"] = int(m_gpu_b.group(1))

                m_cpu_b = re.match(r'(?:vllm:)?num_cpu_blocks\{?[^}]*\}?\s+(\d+)', line)
                if m_cpu_b:
                    metrics["num_cpu_blocks"] = int(m_cpu_b.group(1))

                m_gpu_u = re.match(r'(?:vllm:)?gpu_cache_usage_perc\{?[^}]*\}?\s+([\d\.]+)', line)
                if m_gpu_u:
                    metrics["gpu_cache_usage_perc"] = float(m_gpu_u.group(1))

                m_cpu_u = re.match(r'(?:vllm:)?cpu_cache_usage_perc\{?[^}]*\}?\s+([\d\.]+)', line)
                if m_cpu_u:
                    metrics["cpu_cache_usage_perc"] = float(m_cpu_u.group(1))
    except Exception as e:
        print(f"⚠️ Could not read /metrics: {e}")
    return metrics


def send_request(worker_id: str) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": f"[{worker_id}] {PROMPT}"}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(BASE_URL, data=data, headers=headers)
    start_time = time.perf_counter()
    tokens = 0
    err = None

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            usage = body.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
    except Exception as e:
        err = str(e)

    elapsed = time.perf_counter() - start_time
    return {
        "worker_id": worker_id,
        "tokens": tokens,
        "elapsed_s": elapsed,
        "error": err,
    }


def run_benchmark(swap_label: str) -> dict:
    print("=" * 70)
    print(f"   vLLM Serving Lab - CPU Offloading / Swap Space Benchmark ({swap_label.upper()})")
    print(f"   Model       : {DEFAULT_MODEL}")
    print(f"   Swap Config : {swap_label}")
    print("=" * 70)

    initial_metrics = fetch_cache_metrics()
    print("📊 Initial Memory Subsystem Metrics:")
    print(f"   • GPU KV Blocks Allocated : {initial_metrics['num_gpu_blocks']} blocks")
    print(f"   • CPU Swap Blocks Reserve : {initial_metrics['num_cpu_blocks']} blocks")
    print(f"   • GPU Cache Initial Usage : {initial_metrics['gpu_cache_usage_perc'] * 100:.1f}%")
    print(f"   • CPU Cache Initial Usage : {initial_metrics['cpu_cache_usage_perc'] * 100:.1f}%\n")

    print(f"🚀 Injecting {NUM_WORKERS} parallel heavy sequence requests to stress KV cache...")
    bench_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(send_request, f"stress_{i+1}") for i in range(NUM_WORKERS)]
        results = [f.result() for f in as_completed(futures)]

    bench_elapsed = time.perf_counter() - bench_start
    final_metrics = fetch_cache_metrics()

    completed = sum(1 for r in results if r["error"] is None)
    failed = sum(1 for r in results if r["error"] is not None)
    total_tokens = sum(r["tokens"] for r in results)
    tps = total_tokens / bench_elapsed if bench_elapsed > 0 else 0.0

    summary = {
        "experiment": "cpu_offloading_and_swap",
        "state": swap_label,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_gpu_blocks": initial_metrics["num_gpu_blocks"],
        "num_cpu_blocks": initial_metrics["num_cpu_blocks"],
        "gpu_cache_usage_peak_perc": round(final_metrics["gpu_cache_usage_perc"] * 100, 2),
        "cpu_cache_usage_peak_perc": round(final_metrics["cpu_cache_usage_perc"] * 100, 2),
        "completed_requests": completed,
        "failed_requests": failed,
        "success_rate_pct": round((completed / len(results)) * 100, 2),
        "generation_throughput_tps": round(tps, 2),
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_file = os.path.join(RESULT_DIR, f"{swap_label}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"📊 State Summary ({swap_label.upper()}):")
    print(f"   • CPU Swap Reserve Blocks  : {initial_metrics['num_cpu_blocks']} blocks")
    print(f"   • Peak GPU Cache Usage     : {final_metrics['gpu_cache_usage_perc'] * 100:.1f}%")
    print(f"   • Peak CPU Cache Usage     : {final_metrics['cpu_cache_usage_perc'] * 100:.1f}%")
    print(f"   • Request Success Rate     : {summary['success_rate_pct']}% ({completed}/{completed + failed})")
    print(f"   • System Throughput        : {tps:.1f} tokens/s")
    print(f"   📁 Saved report to: {out_file}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM CPU Offload Benchmark")
    parser.add_argument("--state", type=str, default="swap_enabled", choices=["swap_enabled", "swap_disabled"],
                        help="Swap space test state")
    args = parser.parse_args()

    run_benchmark(args.state)


if __name__ == "__main__":
    main()
