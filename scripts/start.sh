#!/bin/bash
# vLLM Serving Lab - One-Click Start Script
set -e

# ── Load port config from .env (if available) ────────────
VLLM_PORT="${VLLM_PORT:-8000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

# Source .env if present (docker compose does this automatically,
# but we also need the port values for health-check and display)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    # Export only known safe variables (no eval of arbitrary code)
    set -a
    source .env
    set +a
fi

# Re-read after sourcing
VLLM_PORT="${VLLM_PORT:-8000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

echo "===================================================================="
echo "          vLLM Serving Lab - One-Click Start"
echo "===================================================================="

# Check if .env exists, otherwise warn and use defaults
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "ℹ️  No .env file found. Using default values from .env.example..."
    echo "   (To customize tokens or API keys, run: cp .env.example .env)"
fi

echo "🚀 Starting vLLM, Prometheus, and Grafana containers in background..."
docker compose up -d

echo ""
echo "⏳ Waiting for vLLM server to become healthy (first load may take minutes if downloading model)..."
MAX_WAIT_ATTEMPTS=120
ATTEMPTS=0
until curl -s -f "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ ${ATTEMPTS} -ge ${MAX_WAIT_ATTEMPTS} ]; then
        echo ""
        echo "❌ ERROR: vLLM failed to become healthy after $((MAX_WAIT_ATTEMPTS * 3))s."
        echo "   Check container logs using: docker compose logs --tail 50 vllm"
        exit 1
    fi
    sleep 3
    echo -n "."
done

echo ""
echo "✅ All services are UP and Healthy!"
echo "===================================================================="
echo "🌐 Endpoints Available:"
echo "   - vLLM API Server : http://localhost:${VLLM_PORT}/v1/chat/completions"
echo "   - vLLM Metrics    : http://localhost:${VLLM_PORT}/metrics"
echo "   - Prometheus      : http://localhost:${PROMETHEUS_PORT}"
echo "   - Grafana         : http://localhost:${GRAFANA_PORT} (default: admin/admin)"
echo "===================================================================="
echo "💡 Useful commands:"
echo "   - View real-time logs : docker compose logs -f vllm"
echo "   - Run smoke test      : bash scripts/smoke_test.sh"
echo "   - Run all benchmarks  : bash scripts/run_all_benchmarks.sh --all"
echo "   - Stop all services   : docker compose down"
echo "===================================================================="
