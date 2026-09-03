#!/bin/bash
# vLLM Serving Lab - Unified One-Click Benchmark Suite
set -e

print_usage() {
    echo "===================================================================="
    echo "       vLLM Serving Lab - Unified Benchmark Suite"
    echo "===================================================================="
    echo "Usage: bash scripts/run_all_benchmarks.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --baseline     Run baseline workload benchmarks (short, prefill, decode)"
    echo "  -1, --exp1         Run Experiment 1: Prefix Caching ON vs OFF"
    echo "  -2, --exp2         Run Experiment 2: Chunked Prefill (budgets 2048, 4096, 8192)"
    echo "  -3, --exp3         Run Experiment 3: Quantization (3B-BF16, 7B-AWQ, 7B-FP8)"
    echo "  -4, --exp4         Run Experiment 4: Scheduling Policy (FCFS vs Priority)"
    echo "  -5, --exp5         Run Experiment 5: Speculative Decoding (Disabled vs N-gram)"
    echo "  -6, --exp6         Run Experiment 6: CUDA Graph (CUDA Graph vs Eager Mode)"
    echo "  -7, --exp7         Run Experiment 7: KV Cache Quantization (Auto vs FP8)"
    echo "  -8, --exp8         Run Experiment 8: CPU Offloading & Swap Space"
    echo "  -9, --exp9         Run Experiment 9: Prefill-Decode Disaggregation (PD Architecture)"
    echo "  -a, --all          Run ALL baseline and experiments sequentially"
    echo "  -h, --help         Show this help message"
    echo "===================================================================="
}

if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

RUN_BASELINE=false
RUN_EXP1=false
RUN_EXP2=false
RUN_EXP3=false
RUN_EXP4=false
RUN_EXP5=false
RUN_EXP6=false
RUN_EXP7=false
RUN_EXP8=false
RUN_EXP9=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--baseline) RUN_BASELINE=true; shift ;;
        -1|--exp1)     RUN_EXP1=true; shift ;;
        -2|--exp2)     RUN_EXP2=true; shift ;;
        -3|--exp3)     RUN_EXP3=true; shift ;;
        -4|--exp4)     RUN_EXP4=true; shift ;;
        -5|--exp5)     RUN_EXP5=true; shift ;;
        -6|--exp6)     RUN_EXP6=true; shift ;;
        -7|--exp7)     RUN_EXP7=true; shift ;;
        -8|--exp8)     RUN_EXP8=true; shift ;;
        -9|--exp9)     RUN_EXP9=true; shift ;;
        -a|--all)
            RUN_BASELINE=true
            RUN_EXP1=true
            RUN_EXP2=true
            RUN_EXP3=true
            RUN_EXP4=true
            RUN_EXP5=true
            RUN_EXP6=true
            RUN_EXP7=true
            RUN_EXP8=true
            RUN_EXP9=true
            shift
            ;;
        -h|--help)     print_usage; exit 0 ;;
        *)             echo "❌ Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

echo "===================================================================="
echo "🚀 Starting vLLM Serving Lab Benchmark Execution"
echo "===================================================================="

if [ "$RUN_BASELINE" = true ]; then
    echo ""
    echo "▶️  Executing Baseline Benchmarks..."
    bash scripts/benchmark.sh
fi

if [ "$RUN_EXP1" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 1: Prefix Caching..."
    bash scripts/run_exp1_prefix_caching.sh
fi

if [ "$RUN_EXP2" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 2: Chunked Prefill..."
    bash scripts/run_exp2_chunked_prefill.sh
fi

if [ "$RUN_EXP3" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 3: Quantization Comparison..."
    bash scripts/run_exp3_quantization.sh
fi

if [ "$RUN_EXP4" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 4: Scheduling Policy (FCFS vs Priority)..."
    bash scripts/run_exp4_scheduling_policy.sh
fi

if [ "$RUN_EXP5" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 5: Speculative Decoding (N-gram)..."
    bash scripts/run_exp5_speculative_decoding.sh
fi

if [ "$RUN_EXP6" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 6: CUDA Graph (Enabled vs Eager Mode)..."
    bash scripts/run_exp6_cuda_graph.sh
fi

if [ "$RUN_EXP7" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 7: KV Cache Quantization (Auto vs FP8)..."
    bash scripts/run_exp7_kv_cache_quant.sh
fi

if [ "$RUN_EXP8" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 8: CPU Offloading & Swap Space..."
    bash scripts/run_exp8_cpu_offload.sh
fi

if [ "$RUN_EXP9" = true ]; then
    echo ""
    echo "▶️  Executing Experiment 9: Prefill-Decode Disaggregation..."
    bash scripts/run_exp9_disaggregation.sh
fi

echo ""
echo "===================================================================="
echo "🎉 All selected benchmarks completed successfully!"
echo "   📁 Check the results/ directory for detailed JSON reports."
echo "===================================================================="
