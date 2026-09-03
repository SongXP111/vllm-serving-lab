#!/bin/bash
# vLLM Serving Lab - Experiment 5: Speculative Decoding Automated Suite
set -e

echo "===================================================================="
echo "   vLLM Serving Lab - Experiment 5: Speculative Decoding Suite"
echo "===================================================================="
echo "Comparing: Disabled (Standard Autoregressive) vs Enabled (N-gram Speculation)"
echo "This test restarts the vLLM container for each state and measures TPOT acceleration."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

CONFIGS=(
    "disabled|"
    "enabled|{\"method\": \"ngram\", \"num_speculative_tokens\": 4, \"prompt_lookup_max\": 4}"
)

MAX_WAIT_ATTEMPTS=80
TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r LABEL SPEC_CFG <<< "${CONFIGS[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [$((i+1))/${TOTAL}] Configuring Speculative Decoding (${LABEL})..."
    echo "--------------------------------------------------------------------"

    export SPECULATIVE_CONFIG="${SPEC_CFG}"

    # Verify compose interpolation
    RESOLVED=$(docker compose config 2>/dev/null | grep -o "\-\-speculative-config [^ ]*" | head -1 || true)
    echo "   ✅ Resolved compose arg: ${RESOLVED:-'(disabled)'}"

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

    echo "🚀 [$((i+1))/${TOTAL}] Running Speculative Decoding benchmark for ${LABEL}..."
    python3 tests/test_speculative_decoding.py --state "${LABEL}"

    echo "✅ State '${LABEL}' completed."
    sleep 2
done

# ── Cross-state comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 Speculative Decoding (Disabled vs Enabled) Comparison Summary"
echo "===================================================================="
printf "%-12s  %18s  %16s  %12s\n" \
    "State" "Mean TPOT (Latency)" "Throughput" "Speedup"
printf "%-12s  %18s  %16s  %12s\n" \
    "------------" "-------------------" "----------------" "------------"

DIS_TPOT=0
for LABEL in "disabled" "enabled"; do
    FILE="results/speculative-decoding/${LABEL}.json"
    if [ -f "${FILE}" ]; then
        TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['overall_tpot_mean_ms']:.1f} ms\")")
        TPS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['overall_throughput_tps']:.1f} tok/s\")")
        
        if [ "$LABEL" = "disabled" ]; then
            DIS_TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(d['overall_tpot_mean_ms'])")
            SPEEDUP="1.0x (base)"
        else
            SPEEDUP=$(python3 -c "
import json
d = json.load(open('${FILE}'))
en_tpot = d['overall_tpot_mean_ms']
if en_tpot > 0 and ${DIS_TPOT} > 0:
    print(f\"{${DIS_TPOT} / en_tpot:.2f}x\")
else:
    print('N/A')
")
        fi

        printf "%-12s  %18s  %16s  %12s\n" \
            "${LABEL}" "${TPOT}" "${TPS}" "${SPEEDUP}"
    else
        printf "%-12s  %18s\n" "${LABEL}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 Speculative decoding comparison suite completed!"
echo "   📁 Detailed JSON reports: results/speculative-decoding/"
echo ""
echo "   💡 Key Insights:"
echo "      1. Speculative Decoding produces multiple tokens per verification step."
echo "      2. For repetitive / structured tasks (Code, JSON), TPOT drops significantly."
echo "      3. N-gram speculation achieves this with 0 MB extra GPU VRAM footprint!"
echo "===================================================================="
