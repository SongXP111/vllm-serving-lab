#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 4: Scheduling Policy Demonstration (FCFS vs Priority)

Simulates queue congestion with background low-priority batch requests (priority=10)
and evaluates how quickly a high-priority VIP request (priority=0) is scheduled and served.

Measures:
  - VIP Request TTFT (ms)
  - VIP Request E2E Latency (ms)
  - Batch Requests Mean Latency (ms)
  - Overall Generation Throughput (tokens/s)
"""
import urllib.request
import urllib.error
import json
import time
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL", "qwen2.5-7b-instruct-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "")
RESULT_DIR = "results/scheduling-policy"

NUM_BATCH_WORKERS = 6
BATCH_MAX_TOKENS = 250
VIP_MAX_TOKENS = 30

BATCH_PROMPT = (
    "请详细论述分布式系统中的两阶段提交协议（2PC）和三阶段提交协议（3PC）的优缺点，"
    "分析它们在网络分区时可能面临的阻塞问题，并结合公式进行细致阐述。"
)
VIP_PROMPT = "请简短回答：大语言模型中的 KV Cache 的主要作用是什么？请用一句话概括。"


def send_streaming_request(worker_id: str, prompt: str, max_tokens: int, priority: int) -> dict:
    """Send a streaming request to vLLM with priority in extra_body."""
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "extra_body": {"priority": priority},
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(BASE_URL, data=data, headers=headers)

    submit_time = time.perf_counter()
    first_token_time = None
    tokens_generated = 0
    content_pieces = []
    error_msg = None

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue

                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_generated += 1
                    content_pieces.append(content)
    except Exception as e:
        error_msg = str(e)

    end_time = time.perf_counter()
    ttft_ms = (first_token_time - submit_time) * 1000 if first_token_time else None
    e2e_ms = (end_time - submit_time) * 1000

    return {
        "worker_id": worker_id,
        "priority": priority,
        "submit_time": submit_time,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tokens_generated": tokens_generated,
        "answer_preview": "".join(content_pieces)[:80].strip(),
        "error": error_msg,
    }


def run_test(policy_name: str, rounds: int = 3) -> dict:
    print("=" * 70)
    print(f"   vLLM Serving Lab - Scheduling Policy Test ({policy_name.upper()})")
    print(f"   Endpoint  : {BASE_URL}")
    print(f"   Model     : {DEFAULT_MODEL}")
    print(f"   Policy    : {policy_name}")
    print(f"   Contenders: {NUM_BATCH_WORKERS} Batch Requests (priority=10) vs 1 VIP Request (priority=0)")
    print("=" * 70)

    vip_ttfts = []
    vip_e2es = []
    batch_latencies = []
    total_tokens_all_rounds = 0
    total_wall_time = 0

    for r in range(1, rounds + 1):
        print(f"\n▶️  Round [{r}/{rounds}] running congestion test...")
        round_start = time.perf_counter()
        results = []

        with ThreadPoolExecutor(max_workers=NUM_BATCH_WORKERS + 1) as executor:
            # 1. First launch the low-priority batch workers to saturate the scheduler
            futures = []
            for i in range(NUM_BATCH_WORKERS):
                f = executor.submit(
                    send_streaming_request,
                    worker_id=f"batch_{i+1}",
                    prompt=f"[{r}-{i+1}] {BATCH_PROMPT}",
                    max_tokens=BATCH_MAX_TOKENS,
                    priority=10,
                )
                futures.append(f)

            # 2. Brief sleep (150ms) to ensure batch requests reach the server and begin executing
            time.sleep(0.15)

            # 3. Submit the VIP high-priority interactive request (priority=0)
            vip_future = executor.submit(
                send_streaming_request,
                worker_id="vip_user",
                prompt=f"[{r}-VIP] {VIP_PROMPT}",
                max_tokens=VIP_MAX_TOKENS,
                priority=0,
            )
            futures.append(vip_future)

            for f in as_completed(futures):
                res = f.result()
                results.append(res)

        round_elapsed = time.perf_counter() - round_start
        total_wall_time += round_elapsed

        # Extract VIP vs Batch results
        vip_res = next((res for res in results if res["worker_id"] == "vip_user"), None)
        batch_res = [res for res in results if res["worker_id"] != "vip_user"]

        if vip_res and vip_res["ttft_ms"] is not None:
            vip_ttfts.append(vip_res["ttft_ms"])
            vip_e2es.append(vip_res["e2e_ms"])
            print(f"   ⭐ [VIP Result] TTFT: {vip_res['ttft_ms']:.1f} ms | E2E: {vip_res['e2e_ms']:.1f} ms | Answer: {vip_res['answer_preview'][:50]}...")
        else:
            err = vip_res.get("error") if vip_res else "No response"
            print(f"   ❌ [VIP Failed] {err}")

        round_batch_e2es = [b["e2e_ms"] for b in batch_res if b["error"] is None]
        if round_batch_e2es:
            mean_batch_e2e = sum(round_batch_e2es) / len(round_batch_e2es)
            batch_latencies.extend(round_batch_e2es)
            print(f"   📦 [Batch Mean E2E] {mean_batch_e2e:.1f} ms across {len(round_batch_e2es)} tasks")

        round_tokens = sum(res["tokens_generated"] for res in results)
        total_tokens_all_rounds += round_tokens
        round_tps = round_tokens / round_elapsed if round_elapsed > 0 else 0
        print(f"   ⚡ [Throughput] {round_tps:.1f} tokens/s (Elapsed: {round_elapsed:.2f}s)")

    avg_vip_ttft = sum(vip_ttfts) / len(vip_ttfts) if vip_ttfts else 0
    avg_vip_e2e = sum(vip_e2es) / len(vip_e2es) if vip_e2es else 0
    avg_batch_e2e = sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0
    overall_tps = total_tokens_all_rounds / total_wall_time if total_wall_time > 0 else 0

    summary = {
        "experiment": "scheduling_policy",
        "policy": policy_name,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rounds": rounds,
        "batch_workers": NUM_BATCH_WORKERS,
        "vip_priority": 0,
        "batch_priority": 10,
        "vip_ttft_mean_ms": round(avg_vip_ttft, 2),
        "vip_ttft_rounds_ms": [round(t, 2) for t in vip_ttfts],
        "vip_e2e_mean_ms": round(avg_vip_e2e, 2),
        "batch_e2e_mean_ms": round(avg_batch_e2e, 2),
        "overall_generation_throughput_tps": round(overall_tps, 2),
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_file = os.path.join(RESULT_DIR, f"{policy_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"📊 Test Completed for Policy: {policy_name.upper()}")
    print(f"   • VIP TTFT (Mean)         : {avg_vip_ttft:.1f} ms")
    print(f"   • VIP E2E Latency (Mean)  : {avg_vip_e2e:.1f} ms")
    print(f"   • Batch Tasks Mean E2E    : {avg_batch_e2e:.1f} ms")
    print(f"   • Total System Throughput : {overall_tps:.1f} tokens/s")
    print(f"   📁 Saved report to: {out_file}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM Scheduling Policy Benchmark")
    parser.add_argument("--policy", type=str, default="fcfs", choices=["fcfs", "priority"],
                        help="Scheduling policy under test (default: fcfs)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of test iterations (default: 3)")
    args = parser.parse_args()

    run_test(args.policy, args.rounds)


if __name__ == "__main__":
    main()
