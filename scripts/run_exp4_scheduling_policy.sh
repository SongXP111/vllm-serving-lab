#!/bin/bash
# vLLM Serving Lab - Experiment 4: Scheduling Policy Automated Suite (FCFS vs Priority)
set -e

echo "===================================================================="
echo "   vLLM Serving Lab - Experiment 4: Scheduling Policy Automated Suite"
echo "===================================================================="
echo "Comparing: FCFS (First-Come-First-Served) vs Priority (Priority-based Queueing)"
echo "This test restarts the vLLM container for each policy and tests VIP request preemption."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

POLICIES=("fcfs" "priority")
MAX_WAIT_ATTEMPTS=80
TOTAL=${#POLICIES[@]}

for i in "${!POLICIES[@]}"; do
    POLICY="${POLICIES[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [$((i+1))/${TOTAL}] Setting SCHEDULING_POLICY='${POLICY}' and recreating vLLM..."
    echo "--------------------------------------------------------------------"

    export SCHEDULING_POLICY="${POLICY}"

    # Verify compose interpolation
    RESOLVED=$(docker compose config 2>/dev/null | grep -o "\-\-scheduling-policy [^ ]*" | head -1 || true)
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

    echo "🚀 [$((i+1))/${TOTAL}] Running queue congestion & VIP preemption test for ${POLICY}..."
    python3 tests/test_scheduling_policy.py --policy "${POLICY}"

    echo "✅ Policy '${POLICY}' completed."
    sleep 2
done

# ── Cross-policy comparison summary ──────────────────────
echo ""
echo "===================================================================="
echo "📊 Scheduling Policy (FCFS vs Priority) Comparison Summary"
echo "===================================================================="
printf "%-10s  %14s  %14s  %16s  %12s\n" \
    "Policy" "VIP TTFT Mean" "VIP E2E Mean" "Batch E2E Mean" "Throughput"
printf "%-10s  %14s  %14s  %16s  %12s\n" \
    "----------" "--------------" "--------------" "----------------" "------------"

for POLICY in "${POLICIES[@]}"; do
    FILE="results/scheduling-policy/${POLICY}.json"
    if [ -f "${FILE}" ]; then
        VIP_TTFT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['vip_ttft_mean_ms']:.1f} ms\")")
        VIP_E2E=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['vip_e2e_mean_ms']:.1f} ms\")")
        BATCH_E2E=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['batch_e2e_mean_ms']:.1f} ms\")")
        TPS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['overall_generation_throughput_tps']:.1f} tok/s\")")
        printf "%-10s  %14s  %14s  %16s  %12s\n" \
            "${POLICY}" "${VIP_TTFT}" "${VIP_E2E}" "${BATCH_E2E}" "${TPS}"
    else
        printf "%-10s  %14s\n" "${POLICY}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 Scheduling policy comparison suite completed!"
echo "   📁 Detailed JSON reports: results/scheduling-policy/"
echo ""
echo "   💡 Key Insights:"
echo "      1. Under FCFS, VIP requests get stuck behind all ongoing batch jobs (Head-of-Line Blocking)."
echo "      2. Under Priority, VIP requests (priority=0) cut in line ahead of batch tasks (priority=10)."
echo "      3. Notice how Priority scheduling drastically slashes VIP TTFT with minimal throughput penalty."
echo "===================================================================="
