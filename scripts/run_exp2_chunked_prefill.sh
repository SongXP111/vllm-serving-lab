#!/bin/bash
# vLLM Serving Lab - Experiment 2: Chunked Prefill Automated Suite
set -e

echo "===================================================================="
echo "    vLLM Serving Lab - Experiment 2: Chunked Prefill Automated Suite"
echo "===================================================================="
echo "Testing budgets: 2048, 4096, 8192"
echo "This test will restart the vLLM container for each budget and run the collision test."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

BUDGETS=(2048 4096 8192)
MAX_WAIT_ATTEMPTS=80   # max health-check attempts (80 × 3s = 240s timeout)

for BUDGET in "${BUDGETS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [1/3] Setting MAX_BATCHED_TOKENS=${BUDGET} and recreating vLLM container..."
    echo "--------------------------------------------------------------------"

    export MAX_BATCHED_TOKENS=${BUDGET}

    # Verify compose interpolation
    RESOLVED=$(docker compose config | grep -o "max-num-batched-tokens [0-9]*" | head -1 || true)
    echo "   ✅ Resolved compose arg: ${RESOLVED:-'(unable to parse)'}"

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

    echo "🚀 [2/3] Running simultaneous Long Prefill + Ongoing Decode collision test..."
    python3 tests/test_chunked_prefill.py --budget ${BUDGET}

    echo "✅ Budget ${BUDGET} completed."
    sleep 2
done

# ── Cross-budget comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 Cross-Budget Comparison Summary"
echo "===================================================================="
printf "%-8s  %10s  %10s  %10s  %10s  %12s\n" \
    "Budget" "TTFT Mean" "TTFT P99" "TPOT Mean" "TPOT P99" "Throughput"
printf "%-8s  %10s  %10s  %10s  %10s  %12s\n" \
    "------" "---------" "--------" "---------" "--------" "----------"

for BUDGET in "${BUDGETS[@]}"; do
    FILE="results/chunked-prefill/budget_${BUDGET}.json"
    if [ -f "${FILE}" ]; then
        TTFT_MEAN=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['long_request_ttft_mean_ms']:.1f} ms\")")
        TTFT_P99=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['long_request_ttft_p99_ms']:.1f} ms\")")
        TPOT_MEAN=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['short_request_tpot_mean_ms']:.1f} ms\")")
        TPOT_P99=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['short_request_tpot_p99_ms']:.1f} ms\")")
        TPS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['generation_throughput_tps']:.0f} tok/s\")")
        printf "%-8s  %10s  %10s  %10s  %10s  %12s\n" \
            "${BUDGET}" "${TTFT_MEAN}" "${TTFT_P99}" "${TPOT_MEAN}" "${TPOT_P99}" "${TPS}"
    else
        printf "%-8s  %10s\n" "${BUDGET}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 All Chunked Prefill experiments completed!"
echo "   📁 Detailed JSON reports: results/chunked-prefill/"
echo "   💡 Smaller budget → lower TPOT (better decode latency)"
echo "   💡 Larger budget  → lower TTFT (better prefill throughput)"
echo "===================================================================="
