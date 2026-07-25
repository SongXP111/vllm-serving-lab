#!/usr/bin/env python3
"""
vLLM Serving Lab - Experiment 1: Prefix Caching Demonstration
Tests TTFT difference between first request (cold cache) and subsequent requests (warm cache).

Uses streaming mode to measure true Time-To-First-Token (TTFT) instead of E2E latency.
"""
import urllib.request
import urllib.error
import json
import time
import os
import statistics
import uuid
from datetime import datetime, timezone

import os
PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}/v1/chat/completions")
MODEL = "qwen3-8b-awq"
RESULT_DIR = "results/prefix-cache"

WARM_ROUNDS = 5  # number of warm-cache requests for statistical significance

# 构建一个约 1000 字的详细系统设定，并在开头加入动态 UUID 扰动，确保每次执行测试时都能触发真正的冷启动 (Cold Prefill)
LONG_SYSTEM_PROMPT = f"[{uuid.uuid4()}] 你是一个精通分布式系统、AI底层算子优化与大模型服务架构的首席架构师。" + (
    "在大语言模型（LLM）的线上推理服务中，Prefix Caching（前缀缓存）是一种极其关键的算力与显存优化手段。"
    "在智能体（Agent）工作流、多轮对话历史复用、长文档知识库问答以及固定 System Prompt 设定等工业级场景下，"
    "绝大多数并发请求都会携带一段极其冗长且完全一致的背景前缀。"
    "传统的 LLM 推理引擎在处理每一个新到来的请求时，即使它的前缀与前一秒刚处理完的请求完全一致，"
    "也会愚蠢地从零开始对整段 Prompt 进行完整的 Prefill（预填充）矩阵计算，重新生成自注意力机制所需的 Key-Value (KV) Cache。"
    "这种重复计算不仅极大地消耗了宝贵的 GPU Tensor Core 算力，导致首字延迟（TTFT - Time to First Token）急剧飙升，"
    "更是对显存带宽（Memory Bandwidth）的巨大浪费。"
    "而 vLLM 创新的 Prefix Caching 技术，巧妙借用了操作系统中 PagedAttention（分页内存管理）和 Radix Tree（基数树）的数据结构设计。"
    "当引擎完成任意一段前缀的计算后，会将对应的物理 KV Cache 块保持在显存池中，并在基数树里维护该 Token 序列的哈希映射。 "
    "一旦新的请求到来，引擎就能通过对输入 Token 进行快速树搜索，匹配到最大公共前缀。 "
    "命中缓存后，直接零复制复用物理 KV Cache，完全跳过前缀部分的所有 Transformer 运算，实现瞬间吐字！ "
) * 6

# User questions — each request uses the same system prompt but different user content
QUESTIONS = [
    "请简述什么是 KV Cache。",
    "请简述什么是 PagedAttention。",
    "请简述什么是 Continuous Batching。",
    "请简述什么是 Tensor Parallelism。",
    "请简述什么是 Speculative Decoding。",
    "请简述什么是 FlashAttention。",
    "请简述什么是 Rotary Position Embedding。",
]


def send_request_streaming(question: str, label: str, timeout: int = 60):
    """Send a streaming request and measure true TTFT (time to first content token).

    Returns (ttft_ms, prompt_tokens, answer_preview) or raises on failure.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": LONG_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": 16,
    }

    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start_time = time.perf_counter()
    ttft_time = None
    content_pieces = []
    prompt_tokens = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])

                # Capture usage from the final chunk (stream_options.include_usage)
                usage = chunk.get("usage")
                if usage and "prompt_tokens" in usage:
                    prompt_tokens = usage["prompt_tokens"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    if ttft_time is None:
                        ttft_time = time.perf_counter()
                    content_pieces.append(delta["content"])

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}") from e

    if ttft_time is None:
        raise RuntimeError("No content token received in streaming response")

    ttft_ms = (ttft_time - start_time) * 1000
    answer = "".join(content_pieces).strip().replace("\n", " ")[:30]
    print(f"  [{label:<14}] TTFT: {ttft_ms:7.2f} ms | Prompt tokens: {prompt_tokens or '?'} | Preview: {answer}...")
    return ttft_ms, prompt_tokens


def warmup():
    """Send a short throwaway request to warm up CUDA kernels and memory allocators."""
    print("🔄 Sending warmup request (eliminates one-time CUDA overhead)...")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            response.read()
        print("   ✅ Warmup done.\n")
    except Exception as e:
        print(f"   ⚠️  Warmup failed ({e}), continuing anyway...\n")


def main():
    print("=" * 60)
    print("    vLLM Serving Lab - Experiment 1: Prefix Caching")
    print("=" * 60)
    print(f"ℹ️  System Prompt Length : {len(LONG_SYSTEM_PROMPT)} chars")
    print(f"ℹ️  Warm-cache rounds   : {WARM_ROUNDS}")
    print(f"ℹ️  Measurement method  : Streaming TTFT (first content chunk)\n")

    # ── Warmup ────────────────────────────────────────────────
    warmup()

    # ── 1. Cold Cache Request ─────────────────────────────────
    print("1️⃣  Cold Cache (full prefill, no cached KV blocks)")
    ttft_cold, prompt_tokens = send_request_streaming(QUESTIONS[0], "Cold Cache")

    # ── 2. Warm Cache Requests ────────────────────────────────
    print(f"\n2️⃣  Warm Cache ({WARM_ROUNDS} rounds, should hit Radix Tree)")
    warm_ttfts = []
    for i in range(WARM_ROUNDS):
        q = QUESTIONS[(i + 1) % len(QUESTIONS)]
        ttft, _ = send_request_streaming(q, f"Warm #{i+1}")
        warm_ttfts.append(ttft)

    # ── Statistics ────────────────────────────────────────────
    avg_warm = statistics.mean(warm_ttfts)
    std_warm = statistics.stdev(warm_ttfts) if len(warm_ttfts) >= 2 else 0.0
    speedup = ttft_cold / avg_warm if avg_warm > 0 else 0

    print("\n" + "=" * 60)
    print("📈 Prefix Caching Experiment Results:")
    print(f"   Prompt tokens (actual)  : {prompt_tokens or 'N/A'}")
    print(f"   ❄️  Cold TTFT            : {ttft_cold:7.2f} ms")
    print(f"   🔥 Warm TTFT (mean±std) : {avg_warm:7.2f} ± {std_warm:.2f} ms")
    print(f"   🚀 Speedup              : {speedup:.1f}x faster")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_DIR, "summary.json")

    summary_data = {
        "experiment": "prefix_caching",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "system_prompt_chars": len(LONG_SYSTEM_PROMPT),
        "prompt_tokens": prompt_tokens,
        "measurement_method": "streaming_ttft",
        "warm_rounds": WARM_ROUNDS,
        "ttft_cold_ms": round(ttft_cold, 2),
        "ttft_warm_each_ms": [round(t, 2) for t in warm_ttfts],
        "ttft_warm_mean_ms": round(avg_warm, 2),
        "ttft_warm_std_ms": round(std_warm, 2),
        "speedup_factor": round(speedup, 2),
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {result_path}")


if __name__ == "__main__":
    main()
