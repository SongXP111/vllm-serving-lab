"""Gateway configuration for M1 Step 1 (Single Backend Async Proxy + SSE)."""
from dataclasses import dataclass
import os


@dataclass
class GatewayConfig:
    """Configuration for the async LLM Serving Gateway.

    Timeout dimensions (mapped to LLM inference physics):
    - connect_timeout: TCP handshake timeout with backend vLLM.
    - write_timeout: Timeout for sending prompt payload to backend.
    - read_timeout: Max silence interval between consecutive chunks
      (bounds TTFT before first token, and bounds TPOT/jitter during streaming).
    - pool_timeout: Max wait time to acquire a connection from the httpx pool.
    """

    backend_url: str = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8000")
    connect_timeout: float = float(os.environ.get("GATEWAY_CONNECT_TIMEOUT", "3.0"))
    read_timeout: float = float(os.environ.get("GATEWAY_READ_TIMEOUT", "30.0"))
    write_timeout: float = float(os.environ.get("GATEWAY_WRITE_TIMEOUT", "5.0"))
    pool_timeout: float = float(os.environ.get("GATEWAY_POOL_TIMEOUT", "3.0"))
    max_connections: int = int(os.environ.get("GATEWAY_MAX_CONNECTIONS", "100"))
    max_keepalive_connections: int = int(
        os.environ.get("GATEWAY_MAX_KEEPALIVE_CONNECTIONS", "20")
    )
    api_key: str = os.environ.get("VLLM_API_KEY", "")
