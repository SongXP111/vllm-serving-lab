#!/bin/bash
# vLLM Serving Lab - Comprehensive Benchmark Suite
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi

MODEL_NAME="${VLLM_SERVED_MODEL:-qwen2.5-7b-instruct-fp8}"
TOKENIZER_NAME="${VLLM_MODEL:-RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic}"
MODEL_DTYPE="${VLLM_DTYPE:-FP8 (Dynamic)}"
REQUEST_RATE="${BENCHMARK_REQUEST_RATE:-inf}"
BURSTINESS="${BENCHMARK_BURSTINESS:-1.0}"
CONCURRENCIES=(1 2 4 8)
NUM_PROMPTS=32

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rate|-r)        REQUEST_RATE="$2"; shift 2 ;;
        --burstiness|-b)  BURSTINESS="$2"; shift 2 ;;
        --prompts|-p)     NUM_PROMPTS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/benchmark.sh [OPTIONS]"
            echo "  -r, --rate <inf|N>       Arrival rate in RPS (default: inf = saturated burst; number = Poisson)"
            echo "  -b, --burstiness <1.0>   Traffic burstiness factor (1.0 = standard Poisson process)"
            echo "  -p, --prompts <N>        Total prompts per test (default: 32)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# NOTE: BASE_URL is container-internal address (always 8000 inside vllm container)
BASE_URL="http://127.0.0.1:8000"
ENDPOINT="/v1/chat/completions"

# 1. 检查服务存活
if ! curl -s -f "http://localhost:${VLLM_PORT:-8000}/v1/models" > /dev/null 2>&1; then
    echo "❌ 错误: vLLM 服务未就绪，请先执行 bash scripts/start.sh 或 docker compose up -d"
    exit 1
fi

# 2. 探查并固定 GPU 型号与驱动版本
GPU_NAME=$(docker exec vllm-server nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
DRIVER_VERSION=$(docker exec vllm-server nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

if [ "$REQUEST_RATE" = "inf" ]; then
    DISTRIB_TEXT="Saturated Stress (Closed-loop / inf RPS)"
    RATE_ARGS=(--request-rate inf)
else
    DISTRIB_TEXT="Poisson Arrival Process (${REQUEST_RATE} RPS, burstiness=${BURSTINESS})"
    RATE_ARGS=(--request-rate "${REQUEST_RATE}" --burstiness "${BURSTINESS}")
fi

echo "===================================================================="
echo "          vLLM Serving Lab - Comprehensive Benchmark Suite"
echo "===================================================================="
echo "🖥️  GPU Hardware    : ${GPU_NAME} (Driver: ${DRIVER_VERSION})"
echo "🤖 Model & Precision: ${MODEL_NAME} (Dtype: ${MODEL_DTYPE})"
echo "📈 Traffic Arrival  : ${DISTRIB_TEXT}"
echo "👥 Concurrency Grid : ${CONCURRENCIES[*]}"
echo "📦 Prompts Per Run  : ${NUM_PROMPTS}"
echo "===================================================================="

run_test() {
    local workload_name=$1
    local input_len=$2
    local output_len=$3
    local result_prefix=$4

    echo ""
    echo "--------------------------------------------------------------------"
    echo "📊 Workload: ${workload_name} (Input: ${input_len}, Output: ${output_len})"
    echo "--------------------------------------------------------------------"

    for c in "${CONCURRENCIES[@]}"; do
        echo "⏳ [c=${c}] Executing bench serve (Num Prompts: ${NUM_PROMPTS})..."
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
          "${RATE_ARGS[@]}" \
          --save-result \
          --save-detailed \
          --result-dir ${out_dir} > /dev/null

        # 实时快速解析单轮结果
        local host_dir="${PROJECT_ROOT}/results/${result_prefix}/in${input_len}_out${output_len}_c${c}"
        local latest_json=$(ls -t "${host_dir}"/*.json 2>/dev/null | head -1 || true)
        if [ -n "${latest_json}" ]; then
            python3 -c "
import json
d = json.load(open('${latest_json}'))
tps = d.get('output_throughput', 0.0)
ttft_p99 = d.get('p99_ttft_ms', 0.0)
comp = d.get('completed', 0)
fail = d.get('failed', 0)
err = (fail / (comp + fail) * 100) if (comp + fail) > 0 else 0.0
print(f'   ✅ Concurrency {c} Done: Throughput={tps:.1f} tok/s | P99 TTFT={ttft_p99:.1f} ms | Error Rate={err:.1f}% ({fail}/{comp+fail})')
"
        else
            echo "   ✅ Concurrency ${c} completed."
        fi
    done
}

# 1. 常规短对话 (Short Chat - 128 / 128)
run_test "Short Chat" 128 128 "baseline"

# 2. 长首字预载 (Long Prefill - 2048 / 128) - 重点测试预填充算力与 TTFT 尾延迟
run_test "Long Prefill" 2048 128 "baseline"

# 3. 重度生成 (Decode Heavy - 128 / 512) - 重点测试显存带宽与 TPOT 尾延迟
run_test "Decode Heavy" 128 512 "baseline"

# 4. 执行综合 SLA 汇总与报告生成
python3 "${SCRIPT_DIR}/parse_benchmark_results.py" \
    --result-dir "${PROJECT_ROOT}/results/baseline" \
    --gpu-name "${GPU_NAME}" \
    --driver-version "${DRIVER_VERSION}" \
    --model-name "${MODEL_NAME}" \
    --dtype "${MODEL_DTYPE}" \
    --request-rate "${REQUEST_RATE}" \
    --burstiness "${BURSTINESS}"

echo "===================================================================="
echo "🎉 所有固定矩阵基线压测已圆满完成！"
echo "   📁 汇总 SLA 报告：results/baseline/summary.json"
echo "   📈 实时监控历史：http://localhost:${GRAFANA_PORT:-3000}"
echo "===================================================================="
