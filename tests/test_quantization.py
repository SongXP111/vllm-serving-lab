#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 3: Quantization Comparison (AWQ vs BF16)

Evaluates model weight memory footprint, available KV cache blocks/tokens,
inference performance (TTFT, TPOT, generation throughput), and qualitative output quality.
"""
import urllib.request
import urllib.error
import json
import time
import os
import re
import statistics
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

BASE_URL_COMPLETIONS = "http://localhost:8000/v1/chat/completions"
BASE_URL_METRICS = "http://localhost:8000/metrics"
RESULT_DIR = "results/quantization"

ROUNDS = 3
CONCURRENCY = 4
TPOT_WARMUP_TOKENS = 5

QUALITY_QUESTION = (
    "请用严密的逻辑回答：在一个绝热密闭的房间内，打开正在通电运转的电冰箱门，"
    "房间的整体温度最终会上升、下降还是不变？请一句话解释根本原因。"
)
BENCH_PROMPT = (
    "请深度分析人工智能在工业自动化及自动驾驶领域的应用现状、核心算法模型框架"
    "以及面临的伦理与安全挑战，写一篇关于未来的技术展望长文。"
)

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------
def get_metrics_info(served_model_name):
    """Fetch vLLM /metrics and nvidia-smi to get memory and KV cache statistics.

    Returns a dict with all collected metrics (safe defaults on failure).
    """
    info = {
        "num_gpu_blocks": 0,
        "kv_cache_tokens": 0,
        "gpu_memory_used_mib": 0,
        "model_weights_memory_mib": 0,
    }

    # 1. Parse vLLM /metrics
    try:
        req = urllib.request.Request(BASE_URL_METRICS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")

            for line in content.splitlines():
                # Match both `vllm:cache_config_info` and `vllm_cache_config_info`
                if re.match(r"vllm[_:]cache_config_info", line):
                    blocks_m = re.search(r'num_gpu_blocks="(\d+)"', line)
                    if blocks_m:
                        info["num_gpu_blocks"] = int(blocks_m.group(1))
                    tokens_m = re.search(r'kv_cache_size_tokens="(\d+)"', line)
                    if tokens_m:
                        info["kv_cache_tokens"] = int(tokens_m.group(1))

                # Model weights memory (bytes gauge, not always present)
                # Example: vllm:model_weights_memory_bytes 3456789012
                if re.match(r"vllm[_:]model_weights_memory_bytes\b", line):
                    val = line.strip().split()[-1]
                    try:
                        info["model_weights_memory_mib"] = round(
                            float(val) / (1024 * 1024)
                        )
                    except ValueError:
                        pass
    except Exception as e:
        print(f"   ⚠️ Could not parse /metrics: {e}")

    # 2. Parse nvidia-smi for total GPU memory (supplementary)
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, universal_newlines=True).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            info["gpu_memory_used_mib"] = int(lines[0])
    except Exception as e:
        print(f"   ⚠️ Could not run nvidia-smi: {e}")

    # 3. Estimate weight memory if /metrics didn't provide it
    if info["model_weights_memory_mib"] == 0 and info["gpu_memory_used_mib"] > 0:
        # Rough estimate: total VRAM - KV cache overhead - runtime overhead (~500 MiB)
        # This is imprecise but better than nothing
        pass  # leave as 0, we'll display "N/A" in that case

    return info


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def warmup(model_name):
    print("🔄 Sending warmup request...")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    }
    req = urllib.request.Request(
        BASE_URL_COMPLETIONS,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        print("   ✅ Warmup complete.\n")
    except Exception as e:
        print(f"   ⚠️ Warmup failed ({e})\n")


# ---------------------------------------------------------------------------
# Quality sample
# ---------------------------------------------------------------------------
def test_quality(model_name):
    """Ask a reasoning question and return the full answer for cross-model comparison."""
    print("🧠 Testing quality sample...")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": QUALITY_QUESTION}],
        "temperature": 0.1,
        "max_tokens": 300,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    req = urllib.request.Request(
        BASE_URL_COMPLETIONS,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            answer = data["choices"][0]["message"]["content"].strip()
            # Display first 150 chars
            display = answer.replace("\n", " ")[:150]
            print(f"   💬 Q: {QUALITY_QUESTION[:40]}...")
            print(f"   🤖 A: {display}...\n")
            return answer
    except Exception as e:
        print(f"   ⚠️ Quality test failed: {e}\n")
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Benchmark worker
# ---------------------------------------------------------------------------
def send_bench_request(worker_id, model_name, results_list):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": f"[{worker_id}] {BENCH_PROMPT}"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.3,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        BASE_URL_COMPLETIONS,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    request_start = time.perf_counter()
    last_time = None
    ttft_ms = None
    tpots = []
    tokens_generated = 0
    completion_tokens = None

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for raw_line in response:
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
                if "content" in delta and delta["content"]:
                    now = time.perf_counter()
                    if ttft_ms is None:
                        ttft_ms = (now - request_start) * 1000
                    if last_time is not None:
                        tpots.append((now - last_time) * 1000)
                    last_time = now
                    tokens_generated += 1
    except Exception as e:
        print(f"   ⚠️ [Worker {worker_id}] Error: {e}")
        return

    e2e_ms = (time.perf_counter() - request_start) * 1000
    stable_tpots = (
        tpots[TPOT_WARMUP_TOKENS:] if len(tpots) > TPOT_WARMUP_TOKENS else tpots
    )
    gen_tokens = completion_tokens if completion_tokens else tokens_generated

    with _lock:
        results_list.append({
            "worker_id": worker_id,
            "ttft_ms": ttft_ms,
            "tpots": stable_tpots,
            "gen_tokens": gen_tokens,
            "e2e_ms": e2e_ms,
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def run_one_round(round_id, model_name, concurrency):
    results_list = []
    start_time = time.perf_counter()
    print(f"── Round {round_id} (Concurrency={concurrency}) ──")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_bench_request, i + 1, model_name, results_list)
            for i in range(concurrency)
        ]
        for f in futures:
            f.result()
    duration = time.perf_counter() - start_time
    return results_list, duration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Quantization Comparison Test")
    parser.add_argument(
        "--model-label",
        type=str,
        required=True,
        help="Label e.g. 4B-BF16, 4B-AWQ, 8B-AWQ",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Served model name in vLLM e.g. qwen3-4b-awq",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"   vLLM Serving Lab - Experiment 3: Quantization ({args.model_label})")
    print("=" * 70)

    # ── Collect system metrics (after model loaded, before traffic) ────
    metrics = get_metrics_info(args.model_name)
    weight_mib = metrics["model_weights_memory_mib"]
    weight_display = f"{weight_mib} MiB ({weight_mib / 1024:.2f} GiB)" if weight_mib > 0 else "N/A (not exposed by this vLLM version)"

    print("📊 System & Memory Statistics:")
    print(f"    - Total GPU VRAM Used (nvidia-smi) : {metrics['gpu_memory_used_mib']} MiB")
    print(f"    - Model Weights Memory             : {weight_display}")
    print(f"    - Available KV Cache Blocks        : {metrics['num_gpu_blocks']} blocks")
    print(f"    - Available KV Cache Tokens        : {metrics['kv_cache_tokens']} tokens\n")

    # ── Warmup & quality ──────────────────────────────────────
    warmup(args.model_name)
    quality_answer = test_quality(args.model_name)

    # ── Benchmark rounds ──────────────────────────────────────
    all_ttfts = []
    all_tpots = []
    all_e2es = []
    total_gen_tokens = 0
    total_duration = 0.0

    print(f"🚀 Running benchmark across {ROUNDS} rounds (Concurrency={CONCURRENCY})...")
    for r in range(1, ROUNDS + 1):
        results, duration = run_one_round(r, args.model_name, CONCURRENCY)
        total_duration += duration
        for res in results:
            if res["ttft_ms"] is not None:
                all_ttfts.append(res["ttft_ms"])
            all_tpots.extend(res["tpots"])
            all_e2es.append(res["e2e_ms"])
            total_gen_tokens += res["gen_tokens"]
        if r < ROUNDS:
            time.sleep(1)

    # ── Statistics ────────────────────────────────────────────
    mean_ttft = statistics.mean(all_ttfts) if all_ttfts else 0.0
    p99_ttft = percentile(all_ttfts, 99)
    mean_tpot = statistics.mean(all_tpots) if all_tpots else 0.0
    p95_tpot = percentile(all_tpots, 95)
    p99_tpot = percentile(all_tpots, 99)
    p99_e2e = percentile(all_e2es, 99)
    gen_tps = total_gen_tokens / total_duration if total_duration > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"📈 Quantization Results Summary ({args.model_label}):")
    print("-" * 70)
    print(f" 💾 GPU VRAM (nvidia-smi)     : {metrics['gpu_memory_used_mib']} MiB")
    print(f" 💾 Model Weights Memory      : {weight_display}")
    print(f" 📦 Available KV Cache        : {metrics['num_gpu_blocks']} blocks ({metrics['kv_cache_tokens']} tokens)")
    print("-" * 70)
    print(f" ⏱️  TTFT (Mean / P99)         : {mean_ttft:7.2f} / {p99_ttft:7.2f} ms")
    print(f" ⏱️  TPOT (Mean / P95 / P99)   : {mean_tpot:7.2f} / {p95_tpot:7.2f} / {p99_tpot:7.2f} ms")
    print(f" ⏱️  E2E Latency P99           : {p99_e2e:7.2f} ms")
    print("-" * 70)
    print(f" 🚀 Generation Throughput     : {gen_tps:7.1f} tokens/s")
    print("-" * 70)
    print(f" 💡 Key Insight: AWQ primarily solves GPU memory capacity (expanding KV cache),")
    print(f"    not guaranteed faster than BF16 across all hardware and batch sizes.")
    print("=" * 70)

    # ── Save JSON report ──────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_file = os.path.join(RESULT_DIR, f"{args.model_label}.json")
    report_data = {
        "experiment": "quantization_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_label": args.model_label,
        "served_model_name": args.model_name,
        "gpu_memory_used_mib": metrics["gpu_memory_used_mib"],
        "model_weights_memory_mib": weight_mib,
        "num_gpu_kv_blocks": metrics["num_gpu_blocks"],
        "kv_cache_tokens_capacity": metrics["kv_cache_tokens"],
        "rounds": ROUNDS,
        "concurrency": CONCURRENCY,
        "ttft_mean_ms": round(mean_ttft, 2),
        "ttft_p99_ms": round(p99_ttft, 2),
        "tpot_mean_ms": round(mean_tpot, 2),
        "tpot_p95_ms": round(p95_tpot, 2),
        "tpot_p99_ms": round(p99_tpot, 2),
        "e2e_p99_ms": round(p99_e2e, 2),
        "total_gen_tokens": total_gen_tokens,
        "total_duration_sec": round(total_duration, 2),
        "generation_throughput_tps": round(gen_tps, 2),
        "quality_sample_question": QUALITY_QUESTION,
        "quality_sample_answer": quality_answer,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {result_file}\n")


if __name__ == "__main__":
    main()
