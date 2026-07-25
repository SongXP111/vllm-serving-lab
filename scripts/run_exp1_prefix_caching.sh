#!/bin/bash
# vLLM Serving Lab - Experiment 1: Prefix Caching Automated Suite
set -e

echo "===================================================================="
echo "    vLLM Serving Lab - Experiment 1: Prefix Caching Automated Suite"
echo "===================================================================="
echo "Testing states: Enabled (--enable-prefix-caching) vs Disabled (--no-enable-prefix-caching)"
echo "This test will restart the vLLM container for each state and measure TTFT reduction."
echo "===================================================================="

FLAGS=("--enable-prefix-caching" "--no-enable-prefix-caching")
LABELS=("enabled" "disabled")
MAX_WAIT_ATTEMPTS=80

for i in "${!FLAGS[@]}"; do
    FLAG="${FLAGS[$i]}"
    LABEL="${LABELS[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [1/3] Setting PREFIX_CACHING_FLAG='${FLAG}' (${LABEL}) and recreating vLLM..."
    echo "--------------------------------------------------------------------"
    
    export PREFIX_CACHING_FLAG="${FLAG}"
    docker compose up -d --force-recreate vllm
    
    echo "⏳ Waiting for vLLM container to become healthy..."
    ATTEMPTS=0
    until curl -s -f http://localhost:8000/health > /dev/null 2>&1; do
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
    
    echo "🚀 [2/3] Running sequential Prefix Caching TTFT test..."
    python3 scripts/test_prefix_caching.py
    
    # Move or copy the summary.json to a specific filename for comparison
    cp results/prefix-cache/summary.json "results/prefix-cache/${LABEL}.json"
    
    echo "✅ State '${LABEL}' completed."
    sleep 2
done

# ── Cross-state comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 Prefix Caching (ON vs OFF) Comparison Summary"
echo "===================================================================="
printf "%-10s  %12s  %15s  %10s\n" \
    "State" "Cold TTFT" "Warm TTFT (Mean)" "Speedup"
printf "%-10s  %12s  %15s  %10s\n" \
    "----------" "------------" "---------------" "----------"

for LABEL in "enabled" "disabled"; do
    FILE="results/prefix-cache/${LABEL}.json"
    if [ -f "${FILE}" ]; then
        COLD=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['ttft_cold_ms']:.1f} ms\")")
        WARM=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['ttft_warm_mean_ms']:.1f} ms\")")
        SPEEDUP=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['speedup_factor']:.1f}x\")")
        printf "%-10s  %12s  %15s  %10s\n" \
            "${LABEL}" "${COLD}" "${WARM}" "${SPEEDUP}"
    else
        printf "%-10s  %12s\n" "${LABEL}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 Prefix Caching automated comparison completed!"
echo "   📁 Detailed JSON reports: results/prefix-cache/enabled.json & disabled.json"
echo "   💡 Notice how Disabled state shows ~1.0x speedup (recomputing full prefill every time)"
echo "   💡 Enabled state shows ~20x speedup (Radix Tree zero-copy cache hit)"
echo "===================================================================="
