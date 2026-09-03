#!/bin/bash
# vLLM Serving Lab - Benchmark Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi

MODEL_NAME="${VLLM_SERVED_MODEL:-qwen2.5-7b-instruct-fp8}"
TOKENIZER_NAME="${VLLM_MODEL:-RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic}"
# NOTE: BASE_URL is the container-internal address (used by docker exec).
# Do NOT change this to match VLLM_PORT in .env — inside the container it's always 8000.
BASE_URL="http://127.0.0.1:8000"
ENDPOINT="/v1/chat/completions"
CONCURRENCIES=(1 2 4 8)
NUM_PROMPTS=32

echo "=================================================="
echo "    Running vLLM Serving Lab Benchmarks"
echo "    Model: ${MODEL_NAME}"
echo "=================================================="

# 检查服务是否存在
if ! curl -s -f ${BASE_URL}/v1/models > /dev/null; then
    echo "❌ 错误: vLLM 服务未就绪，请先执行 bash scripts/start.sh 或 docker compose up -d"
    exit 1
fi

run_test() {
    local workload_name=$1
    local input_len=$2
    local output_len=$3
    local result_prefix=$4

    echo ""
    echo "--------------------------------------------------"
    echo "📊 Workload: ${workload_name} (Input: ${input_len}, Output: ${output_len})"
    echo "--------------------------------------------------"

    for c in "${CONCURRENCIES[@]}"; do
        echo "⏳ Running concurrency = ${c}..."
        local out_dir="/results/${result_prefix}/in${input_len}_out${output_len}_c${c}"
        
        # 在容器内部调用 vllm bench serve
        docker exec vllm-server vllm bench serve \
          --backend openai-chat \
          --base-url ${BASE_URL} \
          --endpoint ${ENDPOINT} \
          --model ${MODEL_NAME} \
          --tokenizer ${TOKENIZER_NAME} \
          --dataset-name random \
          --num-prompts ${NUM_PROMPTS} \
          --random-input-len ${input_len} \
          --random-output-len ${output_len} \
          --max-concurrency ${c} \
          --request-rate inf \
          --save-result \
          --save-detailed \
          --result-dir ${out_dir} > /dev/null
        
        echo "✅ Concurrency ${c} completed. Results saved to ${out_dir}"
    done
}

# 1. 常规短对话 (Short Chat)
run_test "Short Chat" 128 128 "baseline"

# 2. 长首字预载 (Long Prefill) - 测试 TTFT
run_test "Long Prefill" 2048 128 "baseline"

# 3. 重度生成 (Decode Heavy) - 测试 TPOT / ITL
run_test "Decode Heavy" 128 512 "baseline"

echo ""
echo "=================================================="
echo "🎉 所有基线压测已完成！请查看 results/baseline 目录下的报告，或前往 Grafana 查看历史曲线。"
echo "=================================================="
