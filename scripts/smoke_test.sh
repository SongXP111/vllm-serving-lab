#!/bin/bash
# vLLM Serving Lab - Smoke Test Script
set -e

# ── Resolve project root & load .env for port config ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

VLLM_PORT="${VLLM_PORT:-8000}"

echo "=================================================="
echo "    Running vLLM Serving Lab Smoke Test"
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

# 2. 检查 vLLM 服务是否存活
echo "Checking if vLLM server is reachable at http://localhost:${VLLM_PORT}..."
if ! curl -s -f "http://localhost:${VLLM_PORT}/v1/models" > /dev/null; then
    echo "❌ 错误: 无法连接到 vLLM 服务 (http://localhost:${VLLM_PORT})。"
    echo "请确保您已经通过 'bash scripts/start.sh' 启动了服务，并且模型已经加载完毕。"
    exit 1
fi
echo "✅ vLLM 服务连接成功！"
echo ""

# 3. 运行详细的 Python API 测试脚本
echo "Running comprehensive Python API tests (tests/test_api.py)..."
python3 tests/test_api.py

echo ""
echo "=================================================="
echo "    Smoke Test Completed Successfully! 🚀"
echo "=================================================="
