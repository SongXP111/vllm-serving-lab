#!/bin/bash
# vLLM Serving Lab - Experiment 9: Prefill-Decode Disaggregation Automated Suite
set -e

echo "===================================================================="
echo "   vLLM Serving Lab - Experiment 9: PD Disaggregation Automated Suite"
echo "===================================================================="
echo "Comparing: Monolithic Architecture vs Disaggregated (Prefill + Decode)"
echo "Evaluates how splitting Prefill from Decode isolates interactive latency from prefill bursts."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

# 1. First test current Monolithic instance
echo ""
echo "--------------------------------------------------------------------"
echo "🔄 [1/2] Running Benchmark on Monolithic Architecture..."
echo "--------------------------------------------------------------------"
if ! curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "Starting monolithic vLLM instance..."
    docker compose up -d vllm
    until curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do sleep 3; done
fi

python3 tests/test_disaggregation.py --arch monolithic
echo "✅ Monolithic baseline benchmark completed."

# 2. Start Disaggregated Cluster
echo ""
echo "--------------------------------------------------------------------"
echo "🔄 [2/2] Launching Disaggregated Architecture (Prefill: 8100 + Decode: 8200)..."
echo "--------------------------------------------------------------------"

# Stop monolithic instance to avoid port conflict
docker compose stop vllm > /dev/null 2>&1 || true

# Start disaggregated compose services
docker compose -f compose.disaggregated.yaml up -d

echo "⏳ Waiting for Prefill (8100) and Decode (8200) workers to become healthy..."
ATTEMPTS=0
until curl -s -f "http://localhost:8100/health" > /dev/null 2>&1 && curl -s -f "http://localhost:8200/health" > /dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ ${ATTEMPTS} -ge 80 ]; then
        echo ""
        echo "❌ ERROR: Disaggregated workers failed to become healthy. Aborting."
        docker compose -f compose.disaggregated.yaml logs --tail 30
        docker compose -f compose.disaggregated.yaml down
        docker compose up -d vllm
        exit 1
    fi
    sleep 3
    echo -n "."
done
echo " ✅ Both Prefill and Decode workers are Healthy!"

# Start lightweight Gateway Router in background
ROUTER_PORT=8000 python3 clients/disagg_router.py &
ROUTER_PID=$!
sleep 1

# Ensure cleanup on script exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up disaggregated router and containers..."
    kill ${ROUTER_PID} 2>/dev/null || true
    docker compose -f compose.disaggregated.yaml down > /dev/null 2>&1 || true
    echo "Restoring monolithic vLLM..."
    docker compose up -d vllm > /dev/null 2>&1 || true
}
trap cleanup EXIT

echo "🚀 Running Benchmark on Disaggregated Architecture..."
python3 tests/test_disaggregation.py --arch disaggregated
echo "✅ Disaggregated benchmark completed."

# ── Cross-architecture comparison ────────────────────────
echo ""
echo "===================================================================="
echo "📊 Prefill-Decode Disaggregation (Monolithic vs Disaggregated) Summary"
echo "===================================================================="
printf "%-16s  %18s  %16s  %16s\n" \
    "Architecture" "Prefill Burst (ms)" "Decode TPOT (ms)" "Decode Jitter (ms)"
printf "%-16s  %18s  %16s  %16s\n" \
    "----------------" "------------------" "----------------" "------------------"

for ARCH in "monolithic" "disaggregated"; do
    FILE="results/disaggregation/${ARCH}.json"
    if [ -f "${FILE}" ]; then
        PREFILL=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('heavy_prefill_mean_ms', 0):.1f} ms\")")
        TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('interactive_decode_tpot_mean_ms', 0):.1f} ms\")")
        JITTER=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('interactive_decode_jitter_std_ms', 0):.2f} ms\")")
        printf "%-16s  %18s  %16s  %16s\n" \
            "${ARCH}" "${PREFILL}" "${TPOT}" "${JITTER}"
    else
        printf "%-16s  %18s\n" "${ARCH}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 PD Disaggregation comparison suite completed!"
echo "   📁 Detailed JSON reports: results/disaggregation/"
echo ""
echo "   💡 Key Insights:"
echo "      1. In Monolithic serving, heavy prefills stall running decode batches (TPOT spikes)."
echo "      2. Disaggregation isolates compute-heavy prefill from bandwidth-bound decode."
echo "      3. Notice how Disaggregation slashes Decode Jitter (Standard Deviation of TPOT)!"
echo "===================================================================="
