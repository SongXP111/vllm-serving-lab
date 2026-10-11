#!/bin/bash
# vLLM Serving Lab - Multi-Level Profiling Suite
# Level 1: Macro Service Metrics (Queue vs Prefill vs Decode)
# Level 2: PyTorch Profiler (Operator breakdown & CPU-GPU sync)
# Level 3: Nsight Systems (Kernel launch, Memory memcpy, Streams)
# Level 4: Nsight Compute (SM throughput & Roofline model)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"
PROFILE_DIR="${PROJECT_ROOT}/results/profiles"
mkdir -p "${PROFILE_DIR}"

MODE="${1:---macro}"

echo "===================================================================="
echo "          vLLM Serving Lab - Profiling Suite (${MODE})"
echo "===================================================================="

if [ "$MODE" = "--macro" ] || [ "$MODE" = "-m" ]; then
    echo "📊 [Level 1: Macro Service Metrics Profiling]"
    echo "Querying Prometheus /metrics on port ${VLLM_PORT}..."
    
    python3 -c "
import urllib.request, re, sys

url = 'http://localhost:${VLLM_PORT}/metrics'
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        text = resp.read().decode('utf-8')
except Exception as e:
    print(f'❌ Cannot connect to vLLM metrics ({e}). Make sure vLLM is running.')
    sys.exit(1)

metrics = {}
patterns = {
    'queue_time_sum': r'vllm:request_queue_time_seconds_sum\{?[^}]*\}?\s+([\d\.e\+\-]+)',
    'queue_time_count': r'vllm:request_queue_time_seconds_count\{?[^}]*\}?\s+([\d\.e\+\-]+)',
    'ttft_sum': r'vllm:time_to_first_token_seconds_sum\{?[^}]*\}?\s+([\d\.e\+\-]+)',
    'ttft_count': r'vllm:time_to_first_token_seconds_count\{?[^}]*\}?\s+([\d\.e\+\-]+)',
    'tpot_sum': r'vllm:time_per_output_token_seconds_sum\{?[^}]*\}?\s+([\d\.e\+\-]+)',
    'tpot_count': r'vllm:time_per_output_token_seconds_count\{?[^}]*\}?\s+([\d\.e\+\-]+)',
    'gpu_cache': r'vllm:gpu_cache_usage_perc\{?[^}]*\}?\s+([\d\.e\+\-]+)',
}

for k, pat in patterns.items():
    m = re.search(pat, text)
    metrics[k] = float(m.group(1)) if m else 0.0

avg_queue_ms = (metrics['queue_time_sum'] / metrics['queue_time_count'] * 1000) if metrics['queue_time_count'] > 0 else 0.0
avg_ttft_ms = (metrics['ttft_sum'] / metrics['ttft_count'] * 1000) if metrics['ttft_count'] > 0 else 0.0
avg_tpot_ms = (metrics['tpot_sum'] / metrics['tpot_count'] * 1000) if metrics['tpot_count'] > 0 else 0.0
gpu_cache_pct = metrics['gpu_cache'] * 100

print(f'   • Avg Queue Time : {avg_queue_ms:.2f} ms')
print(f'   • Avg TTFT (Prefill): {avg_ttft_ms:.2f} ms')
print(f'   • Avg TPOT (Decode) : {avg_tpot_ms:.2f} ms')
print(f'   • KV Cache Usage    : {gpu_cache_pct:.1f}%')
print('')
print('🔍 Triangulation Analysis:')
if avg_queue_ms > (avg_ttft_ms * 2) and avg_queue_ms > 50:
    print('   ⚠️  Primary Bottleneck: [SCHEDULER QUEUE CONGESTION]')
    print('      Action: Increase max-num-seqs, apply Priority scheduling, or scale worker instances.')
elif avg_ttft_ms > 500:
    print('   ⚠️  Primary Bottleneck: [PREFILL / COMPUTE-BOUND]')
    print('      Action: Enable Chunked Prefill, enable Prefix Caching, or route long prompts to PD Prefill worker.')
elif avg_tpot_ms > 20:
    print('   ⚠️  Primary Bottleneck: [DECODE / MEMORY-BANDWIDTH-BOUND]')
    print('      Action: Apply model weight quantization (AWQ/FP8), enable CUDA Graph, or quantize KV Cache.')
else:
    print('   ✅ Service is operating within balanced SLA envelope.')
"

elif [ "$MODE" = "--torch" ] || [ "$MODE" = "-t" ]; then
    echo "🔥 [Level 2: PyTorch Profiler Micro-profiling]"
    echo "To profile a vLLM serving step with PyTorch Profiler:"
    echo "1. Set VLLM_TORCH_PROFILER_DIR=/results/profiles in .env or compose.yaml"
    echo "2. Send request with header 'X-Profile: true' or trigger via /start_profile endpoint"
    echo "3. Open generated trace in https://ui.perfetto.dev or chrome://tracing"
    echo ""
    echo "Running local simulated PyTorch GEMM & Attention profile demo..."
    python3 "${SCRIPT_DIR}/torch_profiler.py"

elif [ "$MODE" = "--nsys" ] || [ "$MODE" = "-n" ]; then
    echo "⚡ [Level 3: NVIDIA Nsight Systems Profiling Guide & Wrapper]"
    echo "Nsight Systems down-drills GPU kernel launch gaps, H2D/D2H memory transfers, and CPU-GPU synchronization."
    echo ""
    echo "📌 Command to trace vLLM inside docker container:"
    echo "   docker exec -it vllm-server nsys profile \\"
    echo "     --trace=cuda,nvtx,osrt \\"
    echo "     --output=/results/profiles/vllm_trace \\"
    echo "     --duration=20 \\"
    echo "     --force-overwrite=true"
    echo ""
    echo "📌 What to look for in Nsight Systems timeline:"
    echo "   1. White gaps between small kernels -> CPU Kernel Launch Overhead (Fix: CUDA Graph)"
    echo "   2. cudaStreamSynchronize stalls -> Host waiting on GPU sync (Fix: asynchronous execution)"
    echo "   3. Red memory copy blocks during generation -> KV cache CPU swapping / thrashing (Fix: FP8 KV Cache)"

elif [ "$MODE" = "--ncu" ]; then
    echo "🔬 [Level 4: NVIDIA Nsight Compute Kernel Profiling Guide]"
    echo "Nsight Compute inspects single-kernel roofline model and SM core efficiency."
    echo ""
    echo "📌 Profiling PagedAttention or GEMV kernel:"
    echo "   ncu --set full --kernel-name regex:.*paged_attention.* -o /results/profiles/ncu_paged_att ..."
    echo "📌 Metric Focus:"
    echo "   • Memory Workload Analysis (SOL % of Memory Bandwidth)"
    echo "   • SM Compute SOL % (Tensor Core vs ALU utilization)"

else
    echo "Usage: bash scripts/profile_service.sh [--macro|--torch|--nsys|--ncu]"
fi
