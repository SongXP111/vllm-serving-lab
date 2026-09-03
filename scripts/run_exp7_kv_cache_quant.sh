#!/bin/bash
# vLLM Serving Lab - Experiment 7: KV Cache Quantization Automated Suite (Auto vs FP8)
set -e

echo "===================================================================="
echo "   vLLM Serving Lab - Experiment 7: KV Cache Quantization Suite"
echo "===================================================================="
echo "Comparing: Auto (BF16/FP16 KV Cache) vs FP8 (fp8_e4m3 KV Cache)"
echo "This test restarts the vLLM container for each precision and evaluates memory expansion."
echo "===================================================================="

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
VLLM_PORT="${VLLM_PORT:-8000}"

CONFIGS=(
    "auto"
    "fp8_e4m3"
)

MAX_WAIT_ATTEMPTS=80
TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    DTYPE="${CONFIGS[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [$((i+1))/${TOTAL}] Setting KV_CACHE_DTYPE='${DTYPE}'..."
    echo "--------------------------------------------------------------------"

    export KV_CACHE_DTYPE="${DTYPE}"

    # Verify compose interpolation
    RESOLVED=$(docker compose config 2>/dev/null | grep -o "\-\-kv-cache-dtype [^ ]*" | head -1 || true)
    echo "   ✅ Resolved compose arg: ${RESOLVED:-'(auto)'}"

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

    echo "🚀 [$((i+1))/${TOTAL}] Running KV Cache benchmark for ${DTYPE}..."
    python3 tests/test_kv_cache_quant.py --dtype "${DTYPE}"

    echo "✅ Precision '${DTYPE}' completed."
    sleep 2
done

# ── Cross-state comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 KV Cache Quantization (Auto vs FP8) Comparison Summary"
echo "===================================================================="
printf "%-12s  %14s  %16s  %14s  %12s\n" \
    "KV Dtype" "GPU Blocks" "Token Capacity" "Throughput" "Expansion"
printf "%-12s  %14s  %16s  %14s  %12s\n" \
    "----------" "--------------" "----------------" "--------------" "------------"

AUTO_BLOCKS=0
for DTYPE in "auto" "fp8_e4m3"; do
    FILE="results/kv-cache-quant/${DTYPE}.json"
    if [ -f "${FILE}" ]; then
        BLOCKS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('num_gpu_blocks', 0)} blocks\")")
        CAPACITY=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('kv_cache_token_capacity', 0):,} tok\")")
        TPS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d.get('generation_throughput_tps', 0):.1f} tok/s\")")
        
        if [ "$DTYPE" = "auto" ]; then
            AUTO_BLOCKS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(d.get('num_gpu_blocks', 1))")
            EXPANSION="1.0x (base)"
        else
            EXPANSION=$(python3 -c "
import json
d = json.load(open('${FILE}'))
fp8_b = d.get('num_gpu_blocks', 0)
if fp8_b > 0 and ${AUTO_BLOCKS} > 0:
    print(f\"{fp8_b / ${AUTO_BLOCKS}:.2f}x\")
else:
    print('N/A')
")
        fi

        printf "%-12s  %14s  %16s  %14s  %12s\n" \
            "${DTYPE}" "${BLOCKS}" "${CAPACITY}" "${TPS}" "${EXPANSION}"
    else
        printf "%-12s  %14s\n" "${DTYPE}" "(no data)"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 KV cache quantization comparison suite completed!"
echo "   📁 Detailed JSON reports: results/kv-cache-quant/"
echo ""
echo "   💡 Key Insights:"
echo "      1. FP8 KV Cache halves the memory required per token (from 2 bytes to 1 byte)."
echo "      2. Available KV Cache Blocks approximately double (~1.9x - 2.0x capacity)."
echo "      3. Enables substantially larger concurrency and longer context before OOM."
echo "===================================================================="
