#!/bin/bash
# vLLM Serving Lab - Experiment 3: Quantization Comparison Automated Suite
set -e

echo "===================================================================="
echo "    vLLM Serving Lab - Experiment 3: Quantization Comparison Suite"
echo "===================================================================="
echo "Comparing: Qwen/Qwen3-4B (BF16), Qwen/Qwen3-4B-AWQ, Qwen/Qwen3-8B-AWQ"
echo "This test restarts the vLLM container for each model and evaluates memory and speed."
echo "===================================================================="

# ── Model configurations ──────────────────────────────────
#   Format: HF_MODEL_ID | SERVED_NAME | LABEL | DTYPE
CONFIGS=(
    "Qwen/Qwen3-4B|qwen3-4b|4B-BF16|bfloat16"
    "Qwen/Qwen3-4B-AWQ|qwen3-4b-awq|4B-AWQ|auto"
    "Qwen/Qwen3-8B-AWQ|qwen3-8b-awq|8B-AWQ|auto"
)

MAX_WAIT_ATTEMPTS=120   # max health-check attempts (120 × 3s = 360s timeout for HF downloads)
TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r MODEL SERVED LABEL DTYPE <<< "${CONFIGS[$i]}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🔄 [$((i+1))/${TOTAL}] Setting VLLM_MODEL='${MODEL}' (${LABEL}, dtype=${DTYPE})"
    echo "--------------------------------------------------------------------"

    export VLLM_MODEL="${MODEL}"
    export VLLM_SERVED_MODEL="${SERVED}"
    export VLLM_DTYPE="${DTYPE}"

    # Verify compose interpolation
    RESOLVED=$(docker compose config 2>/dev/null | grep -o "\-\-model [^ ]*" | head -1 || true)
    echo "   ✅ Resolved: ${RESOLVED:-'(unable to parse)'}"

    docker compose up -d --force-recreate vllm

    echo "⏳ Waiting for vLLM container to become healthy (may download weights on first run)..."
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

    echo "🚀 [$((i+1))/${TOTAL}] Running Quantization benchmark for ${LABEL}..."
    python3 tests/test_quantization.py --model-label "${LABEL}" --model-name "${SERVED}"

    echo "✅ Model '${LABEL}' completed."
    sleep 2
done

# ── Cross-model comparison ───────────────────────────────
echo ""
echo "===================================================================="
echo "📊 Quantization Comparison Summary (AWQ vs BF16)"
echo "===================================================================="

# Extract all labels for iteration
LABELS=()
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r _ _ LABEL _ <<< "${cfg}"
    LABELS+=("${LABEL}")
done

printf "%-10s  %10s  %12s  %12s  %10s  %10s  %10s  %12s\n" \
    "Model" "VRAM" "Weights" "KV Tokens" "TTFT Mean" "TPOT Mean" "TPOT P95" "Throughput"
printf "%-10s  %10s  %12s  %12s  %10s  %10s  %10s  %12s\n" \
    "------" "------" "-------" "---------" "---------" "---------" "--------" "----------"

for LABEL in "${LABELS[@]}"; do
    FILE="results/quantization/${LABEL}.json"
    if [ -f "${FILE}" ]; then
        VRAM=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['gpu_memory_used_mib']} MiB\")")
        WGHT=$(python3 -c "import json; d=json.load(open('${FILE}')); v=d.get('model_weights_memory_mib',0); print(f'{v} MiB' if v>0 else 'N/A')")
        KVT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['kv_cache_tokens_capacity']} tok\")")
        TTFT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['ttft_mean_ms']:.1f} ms\")")
        TPOT=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['tpot_mean_ms']:.1f} ms\")")
        TPOT95=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['tpot_p95_ms']:.1f} ms\")")
        TPS=$(python3 -c "import json; d=json.load(open('${FILE}')); print(f\"{d['generation_throughput_tps']:.0f} tok/s\")")
        printf "%-10s  %10s  %12s  %12s  %10s  %10s  %10s  %12s\n" \
            "${LABEL}" "${VRAM}" "${WGHT}" "${KVT}" "${TTFT}" "${TPOT}" "${TPOT95}" "${TPS}"
    else
        printf "%-10s  %10s\n" "${LABEL}" "(no data)"
    fi
done

# ── Quality comparison ────────────────────────────────────
echo ""
echo "--------------------------------------------------------------------"
echo "🧠 Quality Sample Comparison"
echo "--------------------------------------------------------------------"

for LABEL in "${LABELS[@]}"; do
    FILE="results/quantization/${LABEL}.json"
    if [ -f "${FILE}" ]; then
        echo ""
        echo "📝 ${LABEL}:"
        python3 -c "
import json
d = json.load(open('${FILE}'))
a = d.get('quality_sample_answer', 'N/A')
# Show first 200 chars
print(f'   {a[:200]}...' if len(a) > 200 else f'   {a}')
"
    fi
done

echo ""
echo "===================================================================="
echo "🎉 Quantization comparison suite completed!"
echo "   📁 Detailed JSON reports: results/quantization/"
echo ""
echo "   💡 Key Conclusions:"
echo "      1. AWQ primarily solves GPU memory capacity — smaller weights → more KV cache."
echo "      2. AWQ is NOT guaranteed faster than BF16 on all hardware / batch sizes."
echo "      3. Compare KV cache tokens capacity to see the real memory benefit."
echo "===================================================================="
