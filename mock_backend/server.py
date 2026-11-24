"""Containerized Mock vLLM Replica for Step 5 Kubernetes Lifecycle Drills.

Supports realistic LLM backend behaviors inside Kubernetes without requiring a GPU:
1. Configurable model cold start (`STARTUP_DELAY_SEC`): `/startupz` and `/readyz`
   return 503 until simulated model weights finish loading, while `/healthz`
   (liveness) stays 200 so Kubernetes never mis-kills a slow-starting replica.
2. Independent Liveness (`/healthz`) vs Readiness (`/readyz`):
   - `/healthz`: lightweight process liveness check (never fails under load!).
   - `/readyz`: returns 200 only when model loaded, not draining, and not toggled
     unready via `/admin/readiness`.
3. Graceful Streaming Drain (`POST /admin/drain` from `preStop` + `SIGTERM`):
   - Immediately stops admitting new requests (`/readyz` -> 503, new completions -> 503).
   - Allows all currently in-flight SSE streams to finish emitting `[DONE]` before
     process exit within `terminationGracePeriodSeconds`.
4. OpenAI-compatible `/v1/models`, `/v1/chat/completions` (streaming & non-streaming),
   and Prometheus `/metrics` (`vllm:num_requests_running`, `vllm:prefix_cache_*`).
"""
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("mock_vllm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

POD_NAME = os.environ.get("POD_NAME", "mock-vllm-local")
APP_VERSION = os.environ.get("APP_VERSION", "v1")
SERVED_MODEL = os.environ.get("SERVED_MODEL", "mock-qwen2.5-7b-fp8")
STARTUP_DELAY_SEC = float(os.environ.get("STARTUP_DELAY_SEC", "2.0"))
DEFAULT_TOKEN_DELAY_MS = float(os.environ.get("TOKEN_DELAY_MS", "35.0"))
PREFILL_MISS_MS = float(os.environ.get("PREFILL_MISS_MS", "45.0"))
PREFILL_HIT_MS = float(os.environ.get("PREFILL_HIT_MS", "8.0"))
KV_CACHE_CAPACITY = int(os.environ.get("KV_CACHE_CAPACITY_BLOCKS", "512"))

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s\w]", re.UNICODE)


class MockReplicaRuntime:
    def __init__(self, startup_delay_sec: float = STARTUP_DELAY_SEC) -> None:
        self.pod_name = POD_NAME
        self.app_version = APP_VERSION
        self.served_model = SERVED_MODEL
        self.startup_delay_sec = max(0.0, startup_delay_sec)
        self.boot_ts = time.monotonic()
        self.started = self.startup_delay_sec == 0.0
        self.draining = False
        self.reject_new_on_drain = False
        self.forced_unready = False
        self.in_flight = 0
        self.total_completed = 0
        self.total_rejected_draining = 0
        self.prefix_cache_queries = 0.0
        self.prefix_cache_hits = 0.0
        self.kv_cache: "OrderedDict[str, float]" = OrderedDict()
        self._drained_event = asyncio.Event()
        self._drained_event.set()

    def mark_draining(self, reason: str = "sigterm", reject_new: bool = True) -> None:
        changed = (not self.draining) or (reject_new and not self.reject_new_on_drain)
        self.draining = True
        if reject_new:
            self.reject_new_on_drain = True
        if changed:
            logger.info(
                "Pod %s (%s) entering DRAINING state (reason=%s, reject_new=%s, in_flight=%d)",
                self.pod_name,
                self.app_version,
                reason,
                self.reject_new_on_drain,
                self.in_flight,
            )

    def acquire_stream(self) -> None:
        self.in_flight += 1
        self._drained_event.clear()

    def release_stream(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.total_completed += 1
        if self.in_flight == 0:
            self._drained_event.set()

    async def wait_for_drain(self, timeout_sec: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._drained_event.wait(), timeout=timeout_sec)
            return True
        except asyncio.TimeoutError:
            return False

    def check_and_record_prefix(self, messages: List[Dict[str, Any]]) -> float:
        text = "\n".join(
            f"{m.get('role', '')}:{m.get('content', '')}"
            for m in messages
            if isinstance(m, dict)
        )
        tokens = _TOKEN_RE.findall(text)
        num_blocks = len(tokens) // 16
        if num_blocks == 0:
            return 0.0
        hashes: List[str] = []
        parent = hashlib.blake2b(self.served_model.encode("utf-8"), digest_size=8).hexdigest()
        for idx in range(num_blocks):
            chunk = " ".join(tokens[idx * 16 : (idx + 1) * 16])
            h = hashlib.blake2b(f"{parent}|{chunk}".encode("utf-8"), digest_size=8).hexdigest()
            hashes.append(h)
            parent = h

        hits = 0
        for h in hashes:
            if h in self.kv_cache:
                hits += 1
            else:
                break
        self.prefix_cache_queries += num_blocks * 16
        self.prefix_cache_hits += hits * 16

        now = time.monotonic()
        for h in reversed(hashes):
            self.kv_cache[h] = now
            self.kv_cache.move_to_end(h)
        while len(self.kv_cache) > KV_CACHE_CAPACITY:
            self.kv_cache.popitem(last=False)

        return hits / num_blocks


def create_mock_replica_app(startup_delay_sec: float = STARTUP_DELAY_SEC) -> FastAPI:
    rt = MockReplicaRuntime(startup_delay_sec=startup_delay_sec)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rt = rt
        # NOTE: Do NOT override SIGTERM via loop.add_signal_handler here — doing so
        # replaces Uvicorn's native graceful shutdown handler and causes the Pod to
        # hang until terminationGracePeriodSeconds SIGKILL! Instead, K8s preStop
        # invokes `/admin/drain`, and then Uvicorn handles SIGTERM natively while
        # draining active HTTP/SSE streams up to `--timeout-graceful-shutdown`.

        async def load_model_weights():
            if rt.startup_delay_sec > 0:
                logger.info(
                    "Pod %s (%s) loading simulated model weights (%.1fs)...",
                    rt.pod_name,
                    rt.app_version,
                    rt.startup_delay_sec,
                )
                await asyncio.sleep(rt.startup_delay_sec)
            rt.started = True
            logger.info(
                "Pod %s (%s) model loaded and READY!", rt.pod_name, rt.app_version
            )

        loader_task = asyncio.create_task(load_model_weights())
        try:
            yield
        finally:
            loader_task.cancel()
            rt.mark_draining("lifespan_shutdown", reject_new=True)
            if rt.in_flight > 0:
                logger.info(
                    "Pod %s draining %d in-flight requests before exit...",
                    rt.pod_name,
                    rt.in_flight,
                )
                await rt.wait_for_drain(timeout_sec=30.0)
            logger.info("Pod %s shutdown complete.", rt.pod_name)

    app = FastAPI(title="Mock vLLM Replica", version=APP_VERSION, lifespan=lifespan)

    @app.get("/startupz")
    async def startupz() -> Response:
        elapsed = round(time.monotonic() - rt.boot_ts, 2)
        if rt.started:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "started",
                    "pod": rt.pod_name,
                    "version": rt.app_version,
                    "elapsed_sec": elapsed,
                },
            )
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading_weights",
                "pod": rt.pod_name,
                "elapsed_sec": elapsed,
            },
        )

    @app.get("/healthz")
    @app.get("/health")
    async def healthz() -> Dict[str, Any]:
        """Lightweight liveness endpoint (NEVER coupled to inference load)."""
        return {
            "status": "alive",
            "pod": rt.pod_name,
            "version": rt.app_version,
        }

    @app.get("/readyz")
    async def readyz() -> Response:
        """Readiness endpoint: 200 iff started, not draining, and not forced unready."""
        if not rt.started:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "model_loading", "pod": rt.pod_name},
            )
        if rt.draining:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "draining",
                    "pod": rt.pod_name,
                    "in_flight": rt.in_flight,
                },
            )
        if rt.forced_unready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "forced_unready", "pod": rt.pod_name},
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "pod": rt.pod_name,
                "version": rt.app_version,
                "in_flight": rt.in_flight,
            },
        )

    @app.post("/admin/drain")
    async def admin_drain(reject_new: bool = True) -> Dict[str, Any]:
        """Invoked by Kubernetes `preStop` hook before SIGTERM.

        - `reject_new=false` (Phase 1 preStop grace): flips `/readyz -> 503` immediately
          so Endpoints/DNS stops advertising this Pod, while still serving stray requests
          arriving during the ~1s asynchronous DNS/Endpoints propagation window.
        - `reject_new=true` (Phase 2 preStop / hard drain): rejects any further new
          requests with HTTP 503 while existing SSE streams finish draining cleanly.
        """
        rt.mark_draining(
            "prestop_grace" if not reject_new else "prestop_drain",
            reject_new=reject_new,
        )
        return {
            "status": "draining",
            "reject_new": rt.reject_new_on_drain,
            "pod": rt.pod_name,
            "in_flight": rt.in_flight,
        }

    @app.post("/admin/readiness")
    async def admin_readiness(request: Request) -> Dict[str, Any]:
        """Toggle readiness for Fault Drill 3 (proves摘流 without restarting container)."""
        payload = await request.json()
        ready = bool(payload.get("ready", True))
        rt.forced_unready = not ready
        logger.info(
            "Pod %s forced_unready set to %s", rt.pod_name, rt.forced_unready
        )
        return {
            "pod": rt.pod_name,
            "ready": ready,
            "forced_unready": rt.forced_unready,
        }

    @app.get("/v1/models")
    async def models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": rt.served_model, "object": "model", "owned_by": rt.pod_name}],
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        kv_perc = min(1.0, len(rt.kv_cache) / max(1, KV_CACHE_CAPACITY))
        lines = [
            f"vllm:num_requests_running {float(rt.in_flight)}",
            "vllm:num_requests_waiting 0.0",
            f"vllm:gpu_cache_usage_perc {kv_perc:.4f}",
            f"vllm:prefix_cache_queries_total {rt.prefix_cache_queries:.1f}",
            f"vllm:prefix_cache_hits_total {rt.prefix_cache_hits:.1f}",
        ]
        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if (
            not rt.started
            or (rt.draining and rt.reject_new_on_drain)
            or rt.forced_unready
        ):
            rt.total_rejected_draining += 1
            reason = (
                "replica_draining"
                if (rt.draining and rt.reject_new_on_drain)
                else ("model_loading" if not rt.started else "replica_unready")
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"Replica {rt.pod_name} unavailable ({reason})",
                        "type": reason,
                        "pod": rt.pod_name,
                    }
                },
            )

        body = await request.json()
        is_stream = bool(body.get("stream", False))
        max_tokens = max(1, min(64, int(body.get("max_tokens", 12))))
        token_delay_ms = float(body.get("token_delay_ms", DEFAULT_TOKEN_DELAY_MS))
        messages = body.get("messages", [])
        hit_ratio = rt.check_and_record_prefix(messages)
        prefill_ms = PREFILL_HIT_MS + (1.0 - hit_ratio) * (PREFILL_MISS_MS - PREFILL_HIT_MS)
        comp_id = f"chatcmpl-{rt.pod_name}-{rt.app_version}-{uuid.uuid4().hex[:8]}"

        rt.acquire_stream()

        if not is_stream:
            try:
                await asyncio.sleep((prefill_ms + max_tokens * token_delay_ms) / 1000.0)
                return JSONResponse(
                    status_code=200,
                    headers={"X-Backend-Pod": rt.pod_name, "X-Backend-Version": rt.app_version},
                    content={
                        "id": comp_id,
                        "object": "chat.completion",
                        "model": rt.served_model,
                        "pod": rt.pod_name,
                        "version": rt.app_version,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"Response from {rt.pod_name} ({rt.app_version})",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            finally:
                rt.release_stream()

        async def sse_gen() -> AsyncIterator[bytes]:
            try:
                await asyncio.sleep(prefill_ms / 1000.0)
                for idx in range(max_tokens):
                    if await request.is_disconnected():
                        logger.info("Client disconnected on %s (%s)", rt.pod_name, comp_id)
                        break
                    chunk_obj = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "model": rt.served_model,
                        "pod": rt.pod_name,
                        "version": rt.app_version,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": (
                                        f"[{rt.pod_name}:{rt.app_version}] "
                                        if idx == 0
                                        else f"t{idx} "
                                    )
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_obj)}\n\n".encode("utf-8")
                    if idx < max_tokens - 1:
                        await asyncio.sleep(token_delay_ms / 1000.0)
                yield b"data: [DONE]\n\n"
            finally:
                rt.release_stream()

        return StreamingResponse(
            sse_gen(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "X-Backend-Pod": rt.pod_name,
                "X-Backend-Version": rt.app_version,
                "Cache-Control": "no-cache",
            },
        )

    return app


app = create_mock_replica_app()
