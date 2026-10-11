#!/bin/bash
# vLLM Serving Lab - Automated Performance Regression Test Suite
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then set -a; source .env; set +a; fi

echo "===================================================================="
echo "      vLLM Serving Lab - Automated Performance Regression Suite"
echo "===================================================================="
echo "Runs critical test workloads and asserts against golden baseline SLA."
echo "===================================================================="

# Check if vLLM service is healthy
if ! curl -s -f "http://localhost:${VLLM_PORT:-8000}/health" > /dev/null 2>&1; then
    echo "❌ Error: vLLM server is not healthy at http://localhost:${VLLM_PORT:-8000}."
    echo "   Please start the service with: bash scripts/start.sh"
    exit 1
fi

REGRESSION_DIR="${PROJECT_ROOT}/results/regression"
mkdir -p "${REGRESSION_DIR}"

# Run benchmarks into results/regression
echo "🚀 [1/3] Executing fixed regression workloads..."
docker exec vllm-server vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model "${VLLM_SERVED_MODEL:-qwen2.5-7b-instruct-fp8}" \
  --tokenizer "${VLLM_MODEL:-RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic}" \
  --dataset-name random \
  --num-prompts 32 \
  --random-input-len 136 \
  --random-output-len 128 \
  --max-concurrency 4 \
  --request-rate inf \
  --save-result \
  --save-detailed \
  --result-dir /results/regression/in136_out128_c4 > /dev/null

echo "🚀 [2/3] Parsing regression SLA summary..."
python3 scripts/parse_benchmark_results.py \
  --result-dir "${REGRESSION_DIR}" \
  --gpu-name "NVIDIA GeForce RTX 5080 Laptop GPU" \
  --driver-version "572.16" \
  --model-name "${VLLM_SERVED_MODEL:-qwen2.5-7b-instruct-fp8}" \
  --dtype "${VLLM_DTYPE:-FP8 (Dynamic)}"

echo "🚀 [3/3] Asserting regression against Golden Baseline (results/baseline/summary.json)..."
python3 scripts/check_regression.py \
  --baseline results/baseline/summary.json \
  --current results/regression/summary.json \
  --ttft-tol 15.0 \
  --tpot-tol 10.0 \
  --p99-e2e-tol 20.0 \
  --output results/regression/regression_report.json

echo "🎉 Performance Regression Suite Finished Successfully!"
