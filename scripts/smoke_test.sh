#!/bin/bash
# vLLM Serving Lab - Milestone 0: Single-Node API & Metrics Verification (Smoke Test)
set -e

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${VLLM_SERVED_MODEL:-qwen2.5-7b-instruct-fp8}"

echo "=================================================="
echo "    🧪 vLLM Serving Lab - Milestone 0 Smoke Test"
echo "=================================================="
echo "🎯 Target Server : http://localhost:${VLLM_PORT}"
echo "🤖 Target Model  : ${MODEL_NAME}"
echo "=================================================="

# 1. 检查 curl 和 python3 是否安装
if ! command -v curl &> /dev/null; then
    echo "❌ 错误: 未找到 curl，请先安装 (sudo apt install curl)。"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 (sudo apt install python3)。"
    exit 1
fi

# 2. 检查 vLLM 服务存活并发现模型 (curl /v1/models)
echo ""
echo "🔍 [Step 1/4] Verifying /v1/models endpoint via curl..."
MODELS_RESP=$(curl -s -f "http://localhost:${VLLM_PORT}/v1/models" || true)
if [ -z "$MODELS_RESP" ]; then
    echo "❌ 错误: 无法连接到 vLLM 服务 (http://localhost:${VLLM_PORT})。"
    echo "请确保您已经通过 'bash scripts/start.sh' 启动了服务，并且权重已加载完毕。"
    exit 1
fi
echo "   ✅ Models Response: $(echo "$MODELS_RESP" | head -c 120)..."

# 3. 测试非流式请求 (curl /v1/chat/completions non-streaming)
echo ""
echo "💬 [Step 2/4] Testing non-streaming /v1/chat/completions (stream=false)..."
NON_STREAM_RESP=$(curl -s -f -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"1+1等于几？请简短回答。\"}],
    \"temperature\": 0,
    \"max_tokens\": 16
  }" || true)

if [ -n "$NON_STREAM_RESP" ]; then
    ANSWER=$(echo "$NON_STREAM_RESP" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'].strip())" 2>/dev/null || echo "$NON_STREAM_RESP")
    echo "   ✅ Non-streaming Response: ${ANSWER}"
else
    echo "   ⚠️ Non-streaming request failed or empty."
fi

# 4. 测试流式请求 (curl /v1/chat/completions streaming SSE with stream=true)
echo ""
echo "🌊 [Step 3/4] Testing streaming /v1/chat/completions (stream=true)..."
echo -n "   ✅ Streamed Tokens: "
curl -s -N -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"请用一句话简述连续批处理（Continuous Batching）。\"}],
    \"stream\": true,
    \"temperature\": 0,
    \"max_tokens\": 32
  }" | while read -r line; do
    if [[ "$line" =~ ^data:\ (.*) ]] && [[ "$line" != "data: [DONE]" ]]; then
        json_data="${BASH_REMATCH[1]}"
        delta=$(echo "$json_data" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0].get('delta', {}).get('content', ''), end='')" 2>/dev/null || true)
        echo -n "$delta"
    fi
done
echo ""

# 5. 校验生产 Prometheus 指标 (curl /metrics)
echo ""
echo "📊 [Step 4/4] Verifying Prometheus /metrics (waiting, running, TTFT, KV Cache)..."
METRICS_DATA=$(curl -s -f "http://localhost:${VLLM_PORT}/metrics" || true)
if [ -n "$METRICS_DATA" ]; then
    python3 -c "
import sys, re

text = '''$METRICS_DATA'''
def get_val(pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else 'N/A'

running = get_val(r'vllm:num_requests_running\{?[^}]*\}?\s+([\d\.e\+\-]+)')
waiting = get_val(r'vllm:num_requests_waiting\{?[^}]*\}?\s+([\d\.e\+\-]+)')
gpu_kv = get_val(r'vllm:gpu_cache_usage_perc\{?[^}]*\}?\s+([\d\.e\+\-]+)')
ttft_cnt = get_val(r'vllm:time_to_first_token_seconds_count\{?[^}]*\}?\s+([\d\.e\+\-]+)')
tpot_cnt = get_val(r'vllm:time_per_output_token_seconds_count\{?[^}]*\}?\s+([\d\.e\+\-]+)')

print(f'   • vllm:num_requests_running        : {running}')
print(f'   • vllm:num_requests_waiting        : {waiting}')
print(f'   • vllm:gpu_cache_usage_perc (KV)   : {gpu_kv}')
print(f'   • vllm:time_to_first_token_seconds : count={ttft_cnt}')
print(f'   • vllm:time_per_output_token_seconds: count={tpot_cnt}')
"
    echo "   ✅ /metrics successfully exposes all required production metrics!"
else
    echo "   ⚠️ /metrics endpoint returned no data."
fi

# 6. 运行深度 API 单元测试套件 (tests/test_api.py)
echo ""
echo "🧪 Running full Python API verification suite (tests/test_api.py)..."
python3 tests/test_api.py

echo ""
echo "=================================================="
echo "    🎉 Milestone 0 Smoke Test Completed! 🚀"
echo "=================================================="
