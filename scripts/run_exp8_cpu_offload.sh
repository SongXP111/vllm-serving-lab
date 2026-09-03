#!/bin/bash
# vLLM Serving Lab - Experiment 8: CPU Offloading & Swap Space Automated Suite
set -e

echo "===================================================================="
echo "   vLLM Serving Lab - Experiment 8: CPU Offload & Swap Space Suite"
echo "===================================================================="
echo "Comparing: Swap Disabled (--swap-space 0) vs Swap Enabled (--swap-space 4)"
echo "This test restarts the vLLM container for each configuration to verify memory overflow protection."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

CONFIGS=(
    "swap_disabled|0"
    "swap_enabled|4"
)

MAX_WAIT_ATTEMPTS=80
TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r LABEL SWAP_GB <<< "${CONFIGS[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [$((i+1))/${TOTAL}] Setting SWAP_SPACE_GB='${SWAP_GB}' (${LABEL})..."
    echo "--------------------------------------------------------------------"

    export SWAP_SPACE_GB="${SWAP_GB}"

    # Verify compose interpolation
    RESOLVED=$(docker compose config 2>/dev/null | grep -o "\-\-swap-space [^ ]*" | head -1 || true)
    echo "   ✅ Resolved compose arg: ${RESOLVED:-'(default)'}"

    docker compose up -d --force-recreate vllm

    echo "⏳ Waiting for vLLM container to become healthy..."
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

    echo "🚀 [$((i+1))/${TOTAL}] Running CPU Offload/Swap benchmark for ${LABEL}..."
    python3 tests/test_cpu_offload.py --state "${LABEL}"

    echo "✅ State '${LABEL}' completed."
    sleep 2
done

# ── Cross-state comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 CPU Offloading & Swap Space Comparison Summary"
echo "===================================================================="
printf "%-15s  %14s  %15s  %15s  %14s\n" \
    "State" "CPU Blocks" "Peak GPU Cache" "Peak CPU Swap" "Success Rate"
printf "%-15s  %14s  %15s  %15s  %14s\n" \
    "---------------" "--------------" "---------------" "---------------" "--------------"

for LABEL in "swap_disabled" "swap_enabled"; do
    FILE="results/cpu-offload/${LABEL}.json"
    if [ -f "${FILE}" ]; then
        CPU_B=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('num_cpu_blocks', 0)} blocks\")")
        GPU_C=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('gpu_cache_usage_peak_perc', 0):.1f}%\")")
        CPU_C=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('cpu_cache_usage_peak_perc', 0):.1f}%\")")
        SR=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('success_rate_pct', 0):.1f}%\")")
        printf "%-15s  %14s  %15s  %15s  %14s\n" \
            "${LABEL}" "${CPU_B}" "${GPU_C}" "${CPU_C}" "${SR}"
    else
        printf "%-15s  %14s\n" "${LABEL}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 CPU Offloading & Swap Space comparison suite completed!"
echo "   📁 Detailed JSON reports: results/cpu-offload/"
echo ""
echo "   💡 Key Insights:"
echo "      1. CPU Swap Space provides a memory safety net during KV Cache bursts."
echo "      2. Preempted blocks are staged to host RAM instead of aborting the connection."
echo "      3. When memory pressure subsides, swapped blocks migrate back via PCIe seamlessly."
echo "===================================================================="
