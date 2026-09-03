#!/usr/bin/env python3
"""
vLLM Serving Lab - Prefill-Decode Disaggregation Gateway Router

A lightweight, high-performance HTTP proxy that disaggregates LLM traffic:
  - Long Prefill / Batch requests -> Routed to Prefill Worker (Port 8100)
  - Interactive / Short Chat requests -> Routed to Decode Worker (Port 8200)
"""
import http.server
import socketserver
import urllib.request
import urllib.error
import json
import os
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "8000"))
PREFILL_URL = os.environ.get("PREFILL_URL", "http://localhost:8100")
DECODE_URL = os.environ.get("DECODE_URL", "http://localhost:8200")
PREFILL_THRESHOLD_CHARS = 800  # Requests with prompts longer than this route to Prefill

stats = {
    "total_requests": 0,
    "routed_to_prefill": 0,
    "routed_to_decode": 0,
    "errors": 0,
}
stats_lock = threading.Lock()


class DisaggProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/stats":
            self._handle_stats()
        elif self.path.startswith("/v1/models"):
            # Forward model query to decode worker
            self._forward_simple("GET", f"{DECODE_URL}{self.path}")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/v1/chat/completions") or self.path.startswith("/v1/completions"):
            self._handle_chat_completion()
        else:
            self._forward_simple("POST", f"{DECODE_URL}{self.path}")

    def _handle_health(self):
        prefill_ok = self._check_backend(f"{PREFILL_URL}/health")
        decode_ok = self._check_backend(f"{DECODE_URL}/health")
        status_code = 200 if (prefill_ok and decode_ok) else 503
        body = json.dumps({
            "status": "healthy" if status_code == 200 else "degraded",
            "prefill_worker": "up" if prefill_ok else "down",
            "decode_worker": "up" if decode_ok else "down",
        }).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_stats(self):
        with stats_lock:
            data = json.dumps(stats).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_chat_completion(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)

        # Parse request to determine optimal target worker
        target_url = DECODE_URL
        worker_label = "decode"

        try:
            payload = json.loads(post_data.decode("utf-8"))
            # Check prompt length
            messages = payload.get("messages", [])
            prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
            
            # Explicit override in extra_body or character length threshold
            extra_body = payload.get("extra_body", {})
            force_worker = extra_body.get("pd_worker") if isinstance(extra_body, dict) else None

            if force_worker == "prefill" or len(prompt_text) >= PREFILL_THRESHOLD_CHARS:
                target_url = PREFILL_URL
                worker_label = "prefill"
        except Exception:
            pass

        with stats_lock:
            stats["total_requests"] += 1
            if worker_label == "prefill":
                stats["routed_to_prefill"] += 1
            else:
                stats["routed_to_decode"] += 1

        dest_url = f"{target_url}{self.path}"
        self._forward_stream(dest_url, post_data)

    def _forward_stream(self, dest_url: str, body: bytes):
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        headers["Host"] = urllib.parse.urlparse(dest_url).netloc

        req = urllib.request.Request(dest_url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.end_headers()

                # Stream response chunks transparently
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            with stats_lock:
                stats["errors"] += 1
            err_body = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)

    def _forward_simple(self, method: str, dest_url: str):
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host",)}
        req = urllib.request.Request(dest_url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            err = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def _check_backend(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def log_message(self, format, *args):
        # Silence routine request logging
        pass


def run_router():
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", ROUTER_PORT), DisaggProxyHandler)
    print("=" * 70)
    print("   vLLM Serving Lab - Prefill-Decode Disaggregation Gateway")
    print(f"   Listening on    : http://0.0.0.0:{ROUTER_PORT}")
    print(f"   Prefill Worker  : {PREFILL_URL} (for inputs >= {PREFILL_THRESHOLD_CHARS} chars)")
    print(f"   Decode Worker   : {DECODE_URL} (for interactive stream decoding)")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down PD router.")
        server.shutdown()


if __name__ == "__main__":
    run_router()
