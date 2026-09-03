#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 5: Speculative Decoding Benchmark

Evaluates token generation speedup achieved by speculative decoding (N-gram or small Draft Model)
on code generation and structured text generation tasks.

Measures:
  - Generation Throughput (tokens/s)
  - Inter-Token Latency (TPOT Mean, P95, P99)
  - TTFT (Time To First Token)
"""
import urllib.request
import urllib.error
import json
import time
import os
import sys
import argparse
import statistics
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL", "qwen2.5-7b-instruct-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "")
RESULT_DIR = "results/speculative-decoding"

ROUNDS = 3
TPOT_WARMUP_TOKENS = 3

# Code & structured data prompts where speculative ngram / draft matching has high acceptance rates
PROMPTS = [
    (
        "Code Generation",
        "请用 Python 编写一个标准的二叉搜索树（Binary Search Tree）完整实现，包括插入 (insert)、"
        "查找 (search)、删除 (delete) 以及中序遍历 (inorder_traversal) 函数，并附带详细类型注解与单元测试。"
    ),
    (
        "JSON Extraction",
        "请生成包含 5 个虚拟服务器节点的 JSON 集群拓扑结构数据，每个节点必须包含 node_id, hostname, ip_address, "
        "cpu_cores, memory_gb, disk_gb, status, roles (list of string), tags (dict) 等字段。"
    ),
    (
        "Algorithm Documentation",
        "请详细编写快速排序（QuickSort）算法的技术文档，包括算法核心原理、分治思想（Divide and Conquer）、"
        "Python 递归代码实现、时间复杂度与空间复杂度推导，以及在最坏情况下的避免策略。"
    ),
]


def send_streaming_request(prompt: str, max_tokens: int = 400) -> dict:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,  # Greedy for reproducible evaluation
        "max_tokens": max_tokens,
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
    content_pieces = []
    completion_tokens_reported = None

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue

            chunk = json.loads(line[6:])
            usage = chunk.get("usage")
            if usage and "completion_tokens" in usage:
                completion_tokens_reported = usage["completion_tokens"]

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
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
    total_elapsed = end_time - start_time
    gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time) else total_elapsed

    final_token_count = completion_tokens_reported or tokens_generated
    tps = final_token_count / gen_time if gen_time > 0 else 0.0

    # Discard warmup tokens
    stable_tpots = tpots[TPOT_WARMUP_TOKENS:] if len(tpots) > TPOT_WARMUP_TOKENS else tpots

    return {
        "ttft_ms": ttft_ms,
        "tpot_mean_ms": statistics.mean(stable_tpots) if stable_tpots else 0.0,
        "tpot_p95_ms": statistics.quantiles(stable_tpots, n=20)[18] if len(stable_tpots) >= 20 else (max(stable_tpots) if stable_tpots else 0.0),
        "tpot_p99_ms": statistics.quantiles(stable_tpots, n=100)[98] if len(stable_tpots) >= 100 else (max(stable_tpots) if stable_tpots else 0.0),
        "tokens": final_token_count,
        "throughput_tps": tps,
        "answer_preview": "".join(content_pieces)[:80].strip(),
    }


def run_suite(state_label: str) -> dict:
    print("=" * 70)
    print(f"   vLLM Serving Lab - Speculative Decoding Benchmark ({state_label.upper()})")
    print(f"   Model    : {DEFAULT_MODEL}")
    print(f"   Endpoint : {BASE_URL}")
    print("=" * 70)

    task_results = []
    for category, prompt in PROMPTS:
        print(f"\n▶️  Testing [{category}]...")
        cat_tpots = []
        cat_tps = []
        cat_ttfts = []

        for r in range(1, ROUNDS + 1):
            res = send_streaming_request(prompt, max_tokens=350)
            cat_tpots.append(res["tpot_mean_ms"])
            cat_tps.append(res["throughput_tps"])
            cat_ttfts.append(res["ttft_ms"])
            print(f"   Round {r}: TPOT = {res['tpot_mean_ms']:.1f} ms | Throughput = {res['throughput_tps']:.1f} tok/s | TTFT = {res['ttft_ms']:.1f} ms")

        task_results.append({
            "category": category,
            "tpot_mean_ms": round(statistics.mean(cat_tpots), 2),
            "throughput_tps": round(statistics.mean(cat_tps), 2),
            "ttft_mean_ms": round(statistics.mean(cat_ttfts), 2),
        })

    overall_tpot = statistics.mean([t["tpot_mean_ms"] for t in task_results])
    overall_tps = statistics.mean([t["throughput_tps"] for t in task_results])
    overall_ttft = statistics.mean([t["ttft_mean_ms"] for t in task_results])

    summary = {
        "experiment": "speculative_decoding",
        "state": state_label,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_tpot_mean_ms": round(overall_tpot, 2),
        "overall_throughput_tps": round(overall_tps, 2),
        "overall_ttft_mean_ms": round(overall_ttft, 2),
        "tasks": task_results,
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_file = os.path.join(RESULT_DIR, f"{state_label}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"📊 State Summary ({state_label.upper()}):")
    print(f"   • Mean TPOT (Decode Latency) : {overall_tpot:.1f} ms/token")
    print(f"   • Generation Throughput      : {overall_tps:.1f} tokens/s")
    print(f"   • Mean TTFT                  : {overall_ttft:.1f} ms")
    print(f"   📁 Saved report to: {out_file}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM Speculative Decoding Benchmark")
    parser.add_argument("--state", type=str, default="enabled", choices=["enabled", "disabled"],
                        help="Speculative decoding test state (enabled or disabled)")
    args = parser.parse_args()

    run_suite(args.state)


if __name__ == "__main__":
    main()
