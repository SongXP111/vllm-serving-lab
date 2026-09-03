#!/bin/bash
# vLLM Serving Lab - Experiment 6: CUDA/HIP Graph vs Eager Mode Automated Suite
set -e

echo "===================================================================="
echo "   vLLM Serving Lab - Experiment 6: CUDA/HIP Graph Automated Suite"
echo "===================================================================="
echo "Comparing: CUDA Graph Enabled (Default) vs Eager Mode (--enforce-eager)"
echo "This test restarts the vLLM container for each mode and measures decode latency (TPOT)."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

CONFIGS=(
    "cuda_graph|"
    "eager_mode|--enforce-eager"
)

MAX_WAIT_ATTEMPTS=80
TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r LABEL EAGER_FLAG <<< "${CONFIGS[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [$((i+1))/${TOTAL}] Setting ENFORCE_EAGER_FLAG='${EAGER_FLAG}' (${LABEL})..."
    echo "--------------------------------------------------------------------"

    export ENFORCE_EAGER_FLAG="${EAGER_FLAG}"

    # Verify compose interpolation
    RESOLVED=$(docker compose config 2>/dev/null | grep -o "\-\-enforce-eager" | head -1 || true)
    echo "   ✅ Resolved compose arg: ${RESOLVED:-'(CUDA Graph Enabled)'}"

    docker compose up -d --force-recreate vllm

    echo "⏳ Waiting for vLLM container to become healthy (CUDA Graph capture may take ~20s)..."
    ATTEMPTS=0
    until curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        ATTEMPTS=$((ATTEMPTS + 1))
        if [ ${ATTEMPTS} -ge ${MAX_WAIT_ATTEMPTS} ]; then
            echo ""
            echo "❌ ERROR: vLLM failed to become healthy after $((MAX_WAIT_ATTEMPTS * 3))s. Aborting."
            docker logs vllm-server --tail 30
            exit 1
        fi
        sleep 3
        echo -n "."
    done
    echo " ✅ Server is UP and Healthy!"

    echo "🚀 [$((i+1))/${TOTAL}] Running CUDA Graph benchmark for ${LABEL}..."
    python3 tests/test_cuda_graph.py --state "${LABEL}"

    echo "✅ State '${LABEL}' completed."
    sleep 2
done

# ── Cross-state comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 CUDA Graph (Enabled vs Eager Mode) Comparison Summary"
echo "===================================================================="
printf "%-14s  %12s  %14s  %14s  %12s\n" \
    "Execution Mode" "VRAM Used" "c=1 TPOT Mean" "c=4 TPOT Mean" "c=1 Speedup"
printf "%-14s  %12s  %14s  %14s  %12s\n" \
    "--------------" "-----------" "--------------" "--------------" "------------"

EAGER_C1_TPOT=0
for LABEL in "eager_mode" "cuda_graph"; do
    FILE="results/cuda-graph/${LABEL}.json"
    if [ -f "${FILE}" ]; then
        VRAM=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('gpu_memory_used_mib', 0)} MiB\")")
        C1_TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); r=d['results'][0]; print(f\"{r['mean_tpot_ms']:.1f} ms\")")
        C4_TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); r=d['results'][1]; print(f\"{r['mean_tpot_ms']:.1f} ms\")")
        
        if [ "$LABEL" = "eager_mode" ]; then
            EAGER_C1_TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(d['results'][0]['mean_tpot_ms'])")
            SPEEDUP="1.0x (base)"
        else
            SPEEDUP=$(python3 -c "
import json
d = json.load(open('${FILE}'))
cg_tpot = d['results'][0]['mean_tpot_ms']
if cg_tpot > 0 and ${EAGER_C1_TPOT} > 0:
    print(f\"{${EAGER_C1_TPOT} / cg_tpot:.2f}x\")
else:
    print('N/A')
")
        fi

        printf "%-14s  %12s  %14s  %14s  %12s\n" \
            "${LABEL}" "${VRAM}" "${C1_TPOT}" "${C4_TPOT}" "${SPEEDUP}"
    else
        printf "%-14s  %12s\n" "${LABEL}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 CUDA Graph comparison suite completed!"
echo "   📁 Detailed JSON reports: results/cuda-graph/"
echo ""
echo "   💡 Key Insights:"
echo "      1. In Eager Mode, Python/CPU dispatches dozens of kernels per token, creating CPU bottlenecks."
echo "      2. CUDA Graph captures the entire decode loop into a single GPU graph execution."
echo "      3. Notice how CUDA Graph significantly lowers TPOT (especially at c=1), while taking slight VRAM."
echo "===================================================================="
