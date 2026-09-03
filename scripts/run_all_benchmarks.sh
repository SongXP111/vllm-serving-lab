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

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--baseline) RUN_BASELINE=true; shift ;;
        -1|--exp1)     RUN_EXP1=true; shift ;;
        -2|--exp2)     RUN_EXP2=true; shift ;;
        -3|--exp3)     RUN_EXP3=true; shift ;;
        -4|--exp4)     RUN_EXP4=true; shift ;;
        -a|--all)
            RUN_BASELINE=true
            RUN_EXP1=true
            RUN_EXP2=true
            RUN_EXP3=true
            RUN_EXP4=true
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
    echo "▶️  [1/5] Executing Baseline Benchmarks..."
    bash scripts/benchmark.sh
fi

if [ "$RUN_EXP1" = true ]; then
    echo ""
    echo "▶️  [2/5] Executing Experiment 1: Prefix Caching..."
    bash scripts/run_exp1_prefix_caching.sh
fi

if [ "$RUN_EXP2" = true ]; then
    echo ""
    echo "▶️  [3/5] Executing Experiment 2: Chunked Prefill..."
    bash scripts/run_exp2_chunked_prefill.sh
fi

if [ "$RUN_EXP3" = true ]; then
    echo ""
    echo "▶️  [4/5] Executing Experiment 3: Quantization Comparison..."
    bash scripts/run_exp3_quantization.sh
fi

if [ "$RUN_EXP4" = true ]; then
    echo ""
    echo "▶️  [5/5] Executing Experiment 4: Scheduling Policy (FCFS vs Priority)..."
    bash scripts/run_exp4_scheduling_policy.sh
fi

echo ""
echo "===================================================================="
echo "🎉 All selected benchmarks completed successfully!"
echo "   📁 Check the results/ directory for detailed JSON reports."
echo "===================================================================="
