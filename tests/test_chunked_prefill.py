#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 2: Chunked Prefill Demonstration
Simulates simultaneous execution of Long Prefill requests and ongoing Decode-heavy requests.

Measures:
  - Long Request TTFT (Mean, P99)
  - Short Request TPOT (Mean, P95, P99) — measured as inter-token latency with warmup window
  - E2E Request Latency P99
  - Overall generation throughput (tokens/s, generation tokens only)
"""
import urllib.request
import urllib.error
import json
import time
import os
import statistics
import uuid
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
MODEL = "qwen3-8b-awq"
RESULT_DIR = "results/chunked-prefill"

ROUNDS = 3  # repeat the collision test to reduce variance
TPOT_WARMUP_TOKENS = 5  # discard the first N inter-token intervals (prefill tail noise)

# 构建一个长的预填充提示词 (约 1800 tokens)，每次带有 UUID 确保必触发真正 Prefill 计算
LONG_PROMPT_BASE = "你是一个计算机体系结构和算子架构专家。" + (
    "在深度学习和大模型服务中，预填充（Prefill）阶段属于计算密集型（Compute-bound），"
    "需要对所有输入的 Token 执行完整的 Self-Attention 矩阵乘法。 "
    "当并发请求中同时存在长文本预填充和短文本解码（Decode）时，长文本预填充会独占 GPU 的 CUDA 核心，"
    "导致正在进行逐字生成的短请求被严重阻塞，单字吐字间隙（TPOT）大幅飙升，产生明显的卡顿感。"
    "为了解决这个问题，vLLM 引入了 Chunked Prefill（分块预填充）技术。"
    "通过设定 max-num-batched-tokens 预算，引擎会将超级长的高耗时 Prefill 任务切分成多个小于等于预算的 Chunk（分块）。"
    "在一个调度时间步内，引擎既能够计算一个 Chunk 的前缀，又能将剩下的显存带宽和算力顺便分配给正在进行 Decode 的请求，"
    "真正实现了 Prefill 与 Decode 的混合调度（Piggybacking/Interleaving）！ "
) * 8

SHORT_PROMPT = "请详细阐述深度学习中 Transformer 架构的自注意力机制原理，并结合公式进行深度分析，写一篇800字左右的技术长文。"

_lock = threading.Lock()


def send_decode_request(worker_id, results_list):
    """Worker sending short prompt demanding continuous decoding (measuring TPOT)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"[{worker_id}] {SHORT_PROMPT}"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.3,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    request_start = time.perf_counter()
    last_time = None
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
                    if last_time is not None:
                        tpots.append((now - last_time) * 1000)
                    last_time = now
                    tokens_generated += 1
    except Exception as e:
        print(f"   ⚠️ [Decode Worker {worker_id}] Error: {e}")
        return

    e2e_ms = (time.perf_counter() - request_start) * 1000

    # Discard the first few inter-token intervals (prefill tail noise)
    stable_tpots = tpots[TPOT_WARMUP_TOKENS:] if len(tpots) > TPOT_WARMUP_TOKENS else tpots

    gen_tokens = completion_tokens if completion_tokens else tokens_generated

    with _lock:
        results_list.append({
            "type": "decode",
            "worker_id": worker_id,
            "tpots": stable_tpots,
            "gen_tokens": gen_tokens,
            "e2e_ms": e2e_ms,
        })


def send_prefill_request(worker_id, results_list):
    """Worker sending long prompt demanding massive prefill (measuring TTFT)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"[{uuid.uuid4()}] {LONG_PROMPT_BASE}"},
            {"role": "user", "content": "请用一句话总结这段话的核心思想。"},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 10,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    request_start = time.perf_counter()
    ttft_ms = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"] and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - request_start) * 1000
    except Exception as e:
        print(f"   ⚠️ [Prefill Worker {worker_id}] Error: {e}")
        return

    e2e_ms = (time.perf_counter() - request_start) * 1000

    with _lock:
        results_list.append({
            "type": "prefill",
            "worker_id": worker_id,
            "ttft_ms": ttft_ms,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": completion_tokens,
            "e2e_ms": e2e_ms,
        })


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


def warmup():
    print("🔄 Sending warmup request...")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        print("   ✅ Warmup complete.\n")
    except Exception as e:
        print(f"   ⚠️ Warmup failed ({e})\n")


def run_one_round(round_id, decoders, prefillers):
    """Run a single collision round. Returns the raw results list."""
    results_list = []
    total_start = time.perf_counter()

    print(f"\n── Round {round_id} ──")
    print(f"   🚀 [0.0s] Launching {decoders} Decode Workers...")

    with ThreadPoolExecutor(max_workers=decoders + prefillers) as executor:
        decode_futures = [
            executor.submit(send_decode_request, i + 1, results_list)
            for i in range(decoders)
        ]
        # Let decoders stream for 0.5 seconds to establish steady-state decoding
        time.sleep(0.5)

        print(f"   💥 [0.5s] Injecting {prefillers} Long Prefill requests!")
        prefill_futures = [
            executor.submit(send_prefill_request, i + 1, results_list)
            for i in range(prefillers)
        ]

        for f in decode_futures + prefill_futures:
            f.result()

    total_duration = time.perf_counter() - total_start
    return results_list, total_duration


def main():
    parser = argparse.ArgumentParser(description="Chunked Prefill Test")
    parser.add_argument("--budget", type=int, default=4096,
                        help="Current max-num-batched-tokens budget")
    parser.add_argument("--decoders", type=int, default=4,
                        help="Number of concurrent decode requests")
    parser.add_argument("--prefillers", type=int, default=2,
                        help="Number of concurrent prefill requests")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help="Number of rounds to repeat for statistical stability")
    args = parser.parse_args()

    print("=" * 70)
    print(f"   vLLM Serving Lab - Experiment 2: Chunked Prefill (Budget={args.budget})")
    print("=" * 70)
    print(f"ℹ️  Configuration:")
    print(f"    - Decode Workers    : {args.decoders} (streaming ~300 tokens each)")
    print(f"    - Prefill Workers   : {args.prefillers} (~2000 token prompt each)")
    print(f"    - Rounds            : {args.rounds}")
    print(f"    - TPOT warmup skip  : first {TPOT_WARMUP_TOKENS} inter-token intervals")
    print("=" * 70)

    warmup()

    # ── Run multiple rounds ───────────────────────────────────
    all_tpots = []
    all_ttfts = []
    all_e2es = []
    total_gen_tokens = 0
    total_duration = 0.0

    for r in range(1, args.rounds + 1):
        results, duration = run_one_round(r, args.decoders, args.prefillers)
        total_duration += duration

        for res in results:
            all_e2es.append(res["e2e_ms"])
            if res["type"] == "decode":
                all_tpots.extend(res["tpots"])
                total_gen_tokens += res["gen_tokens"]
            elif res["type"] == "prefill":
                if res["ttft_ms"] is not None:
                    all_ttfts.append(res["ttft_ms"])
                total_gen_tokens += res["gen_tokens"]

        # Short pause between rounds
        if r < args.rounds:
            time.sleep(1)

    # ── Aggregate statistics ──────────────────────────────────
    mean_ttft = statistics.mean(all_ttfts) if all_ttfts else 0.0
    p99_ttft = percentile(all_ttfts, 99)

    mean_tpot = statistics.mean(all_tpots) if all_tpots else 0.0
    p95_tpot = percentile(all_tpots, 95)
    p99_tpot = percentile(all_tpots, 99)

    p99_e2e = percentile(all_e2es, 99)

    gen_tps = total_gen_tokens / total_duration if total_duration > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"📈 Chunked Prefill Results (max-num-batched-tokens = {args.budget})")
    print(f"   ({args.rounds} rounds × {args.decoders}D + {args.prefillers}P)")
    print("-" * 70)
    print(f" 🔴 Long Request TTFT  (Mean)  : {mean_ttft:8.2f} ms")
    print(f" 🔴 Long Request TTFT  (P99)   : {p99_ttft:8.2f} ms")
    print("-" * 70)
    print(f" 🟢 Short Request TPOT (Mean)  : {mean_tpot:8.2f} ms")
    print(f" 🟢 Short Request TPOT (P95)   : {p95_tpot:8.2f} ms")
    print(f" 🟢 Short Request TPOT (P99)   : {p99_tpot:8.2f} ms")
    print("-" * 70)
    print(f" 🔵 E2E Latency        (P99)   : {p99_e2e:8.2f} ms")
    print("-" * 70)
    print(f" ⚙️  Total Gen Tokens           : {total_gen_tokens}")
    print(f" 🚀 Generation Throughput       : {gen_tps:8.1f} tokens/s")
    print("=" * 70)

    # ── Save JSON report ──────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_file = os.path.join(RESULT_DIR, f"budget_{args.budget}.json")

    report_data = {
        "experiment": "chunked_prefill",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "max_num_batched_tokens": args.budget,
        "rounds": args.rounds,
        "decoders_count": args.decoders,
        "prefillers_count": args.prefillers,
        "tpot_warmup_skip": TPOT_WARMUP_TOKENS,
        "long_request_ttft_mean_ms": round(mean_ttft, 2),
        "long_request_ttft_p99_ms": round(p99_ttft, 2),
        "long_request_ttft_each_ms": [round(t, 2) for t in all_ttfts],
        "short_request_tpot_mean_ms": round(mean_tpot, 2),
        "short_request_tpot_p95_ms": round(p95_tpot, 2),
        "short_request_tpot_p99_ms": round(p99_tpot, 2),
        "short_request_tpot_samples": len(all_tpots),
        "e2e_latency_p99_ms": round(p99_e2e, 2),
        "total_gen_tokens": total_gen_tokens,
        "total_duration_sec": round(total_duration, 2),
        "generation_throughput_tps": round(gen_tps, 2),
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {result_file}\n")


if __name__ == "__main__":
    main()
