#!/usr/bin/env python3
"""
vLLM Serving Lab - PyTorch Profiler Generator & Kernel Inspector

Profiles transformer inference operations (GEMM, Self-Attention, KV Cache indexing),
outputs operator breakdown table, CPU vs CUDA time, and exports Chrome tracing JSON.
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def profile_inference_kernels():
    try:
        import torch
    except ImportError:
        print("⚠️  PyTorch is not installed in the local Python environment.")
        print("   Run this inside the vLLM container or install torch: pip install torch")
        return

    profile_dir = "results/profiles"
    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(profile_dir, "torch_trace.json")

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"🖥️ Profiling on device: {device}")

    # Simulated lightweight transformer layer GEMM + Attention
    hidden_dim = 2048
    batch_size = 4
    seq_len = 128
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    w_q = torch.randn(hidden_dim, hidden_dim, device=device)
    w_k = torch.randn(hidden_dim, hidden_dim, device=device)
    w_v = torch.randn(hidden_dim, hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        _ = torch.matmul(x, w_q)
    if has_cuda:
        torch.cuda.synchronize()

    # Profile run
    activities = [torch.profiler.ProfilerActivity.CPU]
    if has_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            q = torch.matmul(x, w_q)
            k = torch.matmul(x, w_k)
            v = torch.matmul(x, w_v)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (hidden_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            if has_cuda:
                torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"✅ Chrome trace exported to: {trace_path}")
    print("   Open this trace in: https://ui.perfetto.dev or chrome://tracing\n")

    # Print Top 10 expensive operations
    key_metric = "cuda_time_total" if has_cuda else "cpu_time_total"
    print(prof.key_averages().table(sort_by=key_metric, row_limit=10))

if __name__ == "__main__":
    profile_inference_kernels()
